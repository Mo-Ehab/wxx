"""FFT and complex-valued primitives backed by WebGPU.

Complex tensors are stored as interleaved ``vec2<f32>`` values (real, imag)
and carry ``tensor.type == "c64"``.  Their logical shape omits the trailing
complex component, so a spectrum of length ``K`` has shape ``(..., K)`` and
``tensor.nbytes == numel * 8``.

The FFT is a radix-2 Cooley-Tukey implementation.  It uses one kernel launch
per butterfly stage so that arbitrarily large power-of-two transform sizes are
supported (the intermediate buffer lives in global memory rather than in a
fixed-size workgroup array).
"""

import math
import struct

import numpy as np
import wgpu

from ..backend.device import get_buffer_pool, get_cmd_buffer, get_ctx, get_device
from .. import validation
from ..backend.dispatch import _dispatch_1d, broadcast_shapes
from ..shaders import walk_shader_files
from ..Tensor import Tensor

cmd_buf = get_cmd_buffer()
buffer_pool = get_buffer_pool()
device = get_device()
gpu_ctx = get_ctx()

shaders = walk_shader_files()

WORKGROUP_SIZE = 256
MAX_RANK = 6


def _prod(shape):
    return int(math.prod(shape)) if len(shape) else 1


def _check_pow2(n):
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"FFT length must be a positive power of two, got {n}")


def _new_buffer(nbytes):
    return buffer_pool.create_buffer(
        size=max(8, nbytes),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )


def _scratch_buffer(nbytes):
    """Allocate a non-pooled scratch buffer (freed on flush, not cached)."""
    buf = device.create_buffer(
        size=max(8, (nbytes + 3) & ~3),
        usage=(
            wgpu.BufferUsage.STORAGE
            | wgpu.BufferUsage.COPY_SRC
            | wgpu.BufferUsage.COPY_DST
        ),
    )
    buf._no_cache = True
    return buf


def _meta_buffer(data, label, usage=wgpu.BufferUsage.UNIFORM):
    buf = buffer_pool.create_buffer_with_data(
        data=data, usage=usage, label=label
    )
    cmd_buf.keep_alive(buf)
    return buf


def _complex_tensor(shape, buffer, requires_grad=False):
    return Tensor(shape, buffer, requires_grad=requires_grad, dtype="c64")


def from_numpy_complex(array):
    """Create a complex64 tensor from a numpy array."""
    array = np.asarray(array)
    if not np.iscomplexobj(array):
        array = array.astype(np.float32).astype(np.complex64)
    else:
        array = array.astype(np.complex64, copy=False)

    interleaved = np.empty(array.shape + (2,), dtype=np.float32)
    interleaved[..., 0] = array.real
    interleaved[..., 1] = array.imag

    buffer = buffer_pool.create_buffer_with_data(
        data=interleaved.tobytes(),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        label="from_numpy_complex",
    )
    return _complex_tensor(list(array.shape), buffer)


def complex_weights(weights):
    """Real weights as a 1-D complex tensor (imaginary part = 0)."""
    weights = np.asarray(weights, dtype=np.float32)
    return from_numpy_complex(weights.astype(np.complex64))


def _broadcast_strides(tensor, out_shape):
    rank = len(out_shape)
    pad = rank - len(tensor.shape)
    if pad < 0:
        raise ValueError(
            f"cannot broadcast shape {tuple(tensor.shape)} to {tuple(out_shape)}"
        )

    strides = []
    for i in range(rank):
        if i < pad:
            strides.append(0)
            continue
        dim = tensor.shape[i - pad]
        out_dim = out_shape[i]
        if dim == out_dim:
            strides.append(tensor.stride[i - pad])
        elif dim == 1:
            strides.append(0)
        else:
            raise ValueError(
                f"cannot broadcast shape {tuple(tensor.shape)} to {tuple(out_shape)}"
            )
    return tuple(strides)


def _pack_complex_binary_meta(
    numel, rank, offset_a, offset_b, out_shape, strides_a, strides_b,
    conj_a=0, conj_b=0,
):
    padded_shape = list(out_shape) + [0] * (MAX_RANK - rank)
    padded_a = list(strides_a) + [0] * (MAX_RANK - rank)
    padded_b = list(strides_b) + [0] * (MAX_RANK - rank)
    values = (
        [numel, rank, offset_a, offset_b]
        + padded_shape
        + padded_a
        + padded_b
        + [conj_a, conj_b]
    )
    return struct.pack(f"<{len(values)}I", *values)


def _complex_binary_meta_buffer(tensor_a, tensor_b, out_shape, label, conj_a=0, conj_b=0):
    rank = len(out_shape)
    if rank > MAX_RANK:
        raise ValueError(f"Maximum rank supported by complex kernels is {MAX_RANK}")
    strides_a = _broadcast_strides(tensor_a, out_shape)
    strides_b = _broadcast_strides(tensor_b, out_shape)
    data = _pack_complex_binary_meta(
        _prod(out_shape),
        rank,
        tensor_a.offset,
        tensor_b.offset,
        out_shape,
        strides_a,
        strides_b,
        conj_a,
        conj_b,
    )
    return _meta_buffer(data, label, wgpu.BufferUsage.STORAGE)


def complex_mul(tensor_a, tensor_b, conj_a=False, conj_b=False):
    """Broadcasting elementwise complex multiplication.

    ``conj_a`` / ``conj_b`` conjugate an operand inside the kernel, which lets
    autograd express ``grad * conj(x)`` without a separate conjugate dispatch.
    """
    if tensor_a.type != "c64" or tensor_b.type != "c64":
        raise TypeError("complex_mul expects complex64 tensors")

    out_shape, _, _ = broadcast_shapes(tensor_a.shape, tensor_b.shape)
    numel = _prod(out_shape)

    pipeline = gpu_ctx.get_pipeline(
        "complex_mul", shaders["helpers/complex_mul.wgsl"]
    )
    meta_buf = _complex_binary_meta_buffer(
        tensor_a,
        tensor_b,
        out_shape,
        "complex_mul meta",
        conj_a=1 if conj_a else 0,
        conj_b=1 if conj_b else 0,
    )
    out_buf = _new_buffer(numel * 8)

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor_a.buffer}},
            {"binding": 1, "resource": {"buffer": tensor_b.buffer}},
            {"binding": 2, "resource": {"buffer": out_buf}},
            {"binding": 3, "resource": {"buffer": meta_buf}},
        ],
    )

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    _dispatch_1d(pass_enc, numel)
    pass_enc.end()

    return _complex_tensor(
        out_shape,
        out_buf,
        requires_grad=tensor_a.requires_grad or tensor_b.requires_grad,
    )


def complex_add_inplace(tensor_a, tensor_b):
    """``tensor_a += tensor_b`` for complex tensors (used by autograd)."""
    if tensor_a.type != "c64" or tensor_b.type != "c64":
        raise TypeError("complex_add_inplace expects complex64 tensors")

    out_shape = tuple(tensor_a.shape)
    numel = _prod(out_shape)

    pipeline = gpu_ctx.get_pipeline(
        "complex_add_inplace", shaders["helpers/complex_add_inplace.wgsl"]
    )
    meta_buf = _complex_binary_meta_buffer(
        tensor_a, tensor_b, out_shape, "complex_add meta"
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor_a.buffer}},
            {"binding": 1, "resource": {"buffer": tensor_b.buffer}},
            {"binding": 2, "resource": {"buffer": meta_buf}},
        ],
    )

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    _dispatch_1d(pass_enc, numel)
    pass_enc.end()
    return tensor_a


def complex_conj(tensor1):
    if tensor1.type != "c64":
        raise TypeError("complex_conj expects a complex64 tensor")

    numel = _prod(tensor1.shape)
    rank = len(tensor1.shape)
    if rank > MAX_RANK:
        raise ValueError(f"Maximum rank supported by complex kernels is {MAX_RANK}")

    pipeline = gpu_ctx.get_pipeline(
        "complex_conj", shaders["helpers/complex_conj.wgsl"]
    )

    values = (
        [numel, rank, tensor1.offset, 0]
        + list(tensor1.shape)
        + [0] * (MAX_RANK - rank)
        + list(tensor1.stride)
        + [0] * (MAX_RANK - rank)
    )
    meta_buf = _meta_buffer(
        struct.pack(f"<{len(values)}I", *values),
        "complex_conj meta",
        wgpu.BufferUsage.STORAGE,
    )
    out_buf = _new_buffer(numel * 8)

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor1.buffer}},
            {"binding": 1, "resource": {"buffer": out_buf}},
            {"binding": 2, "resource": {"buffer": meta_buf}},
        ],
    )

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    _dispatch_1d(pass_enc, numel)
    pass_enc.end()

    return _complex_tensor(
        tensor1.shape, out_buf, requires_grad=tensor1.requires_grad
    )


def complex_contiguous(tensor1):
    if tensor1.is_contiguous():
        return tensor1
    if tensor1.type != "c64":
        raise TypeError("complex_contiguous expects a complex64 tensor")

    numel = _prod(tensor1.shape)
    rank = len(tensor1.shape)
    if rank > MAX_RANK:
        raise ValueError(f"Maximum rank supported by complex kernels is {MAX_RANK}")

    pipeline = gpu_ctx.get_pipeline(
        "complex_contiguous", shaders["helpers/complex_contiguous.wgsl"]
    )

    values = (
        [numel, rank, tensor1.offset, 0]
        + list(tensor1.shape)
        + [0] * (MAX_RANK - rank)
        + list(tensor1.stride)
        + [0] * (MAX_RANK - rank)
    )
    meta_buf = _meta_buffer(
        struct.pack(f"<{len(values)}I", *values),
        "complex_contiguous meta",
        wgpu.BufferUsage.STORAGE,
    )
    out_buf = _new_buffer(numel * 8)

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor1.buffer}},
            {"binding": 1, "resource": {"buffer": out_buf}},
            {"binding": 2, "resource": {"buffer": meta_buf}},
        ],
    )

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    _dispatch_1d(pass_enc, numel)
    pass_enc.end()

    out = _complex_tensor(
        tensor1.shape, out_buf, requires_grad=tensor1.requires_grad
    )
    return out


def complex_pad(tensor1, out_len):
    """Copy a contiguous complex tensor, zero-padding/truncating the last dim."""
    if tensor1.type != "c64":
        raise TypeError("complex_pad expects a complex64 tensor")
    if not tensor1.is_contiguous():
        tensor1 = complex_contiguous(tensor1)

    out_len = int(out_len)
    in_len = tensor1.shape[-1]
    batch_count = _prod(tensor1.shape[:-1])

    pipeline = gpu_ctx.get_pipeline(
        "complex_pad", shaders["helpers/complex_pad.wgsl"]
    )
    meta_buf = _meta_buffer(
        struct.pack("<4I", batch_count, in_len, out_len, 0), "complex_pad meta"
    )
    out_buf = _new_buffer(batch_count * out_len * 8)

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor1.buffer}},
            {"binding": 1, "resource": {"buffer": out_buf}},
            {"binding": 2, "resource": {"buffer": meta_buf}},
        ],
    )

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    _dispatch_1d(pass_enc, batch_count * out_len)
    pass_enc.end()

    out_shape = list(tensor1.shape)
    out_shape[-1] = out_len
    return _complex_tensor(
        out_shape, out_buf, requires_grad=tensor1.requires_grad
    )


# Single-workgroup transforms are used for small sizes.  One workgroup holds
# the whole row in workgroup memory, so the host issues one dispatch instead
# of ``log2(n) + 2``.  Larger transforms fall back to the staged kernels.
FFT_WORKGROUP_MAX = 1024


def _fft_meta_buffer(batch_count, n, log2_n, in_len, out_len, label):
    return _meta_buffer(
        struct.pack("<8I", batch_count, n, log2_n, in_len, out_len, 0, 0, 0),
        label,
    )


def _rfft_workgroup(tensor1, n, shape, batch_count, in_len, log2_n):
    out_len = n // 2 + 1
    out_buf = _new_buffer(batch_count * out_len * 8)
    pipeline = gpu_ctx.get_pipeline(
        "fft_wg_real", shaders["helpers/fft_wg_real.wgsl"]
    )
    meta_buf = _fft_meta_buffer(
        batch_count, n, log2_n, in_len, out_len, "rfft wg meta"
    )
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor1.buffer}},
            {"binding": 1, "resource": {"buffer": out_buf}},
            {"binding": 2, "resource": {"buffer": meta_buf}},
        ],
    )
    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    pass_enc.dispatch_workgroups(batch_count)
    pass_enc.end()

    out_shape = list(shape)
    out_shape[-1] = out_len
    out = _complex_tensor(out_shape, out_buf, requires_grad=tensor1.requires_grad)
    return out, n


def _irfft_workgroup(tensor1, n, shape, batch_count, in_len, log2_n):
    out_buf = _new_buffer(batch_count * n * 4)
    pipeline = gpu_ctx.get_pipeline(
        "fft_wg_inverse", shaders["helpers/fft_wg_inverse.wgsl"]
    )
    meta_buf = _fft_meta_buffer(batch_count, n, log2_n, in_len, n, "irfft wg meta")
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor1.buffer}},
            {"binding": 1, "resource": {"buffer": out_buf}},
            {"binding": 2, "resource": {"buffer": meta_buf}},
        ],
    )
    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    pass_enc.dispatch_workgroups(batch_count)
    pass_enc.end()

    out_shape = list(shape)
    out_shape[-1] = n
    return Tensor(out_shape, out_buf, requires_grad=tensor1.requires_grad)


def _run_fft_stages(z_buf, batch_count, n, log2_n, inverse):
    pipeline = gpu_ctx.get_pipeline("fft_stage", shaders["helpers/fft_stage.wgsl"])
    for stage in range(log2_n):
        meta_buf = _meta_buffer(
            struct.pack(
                "<8I",
                batch_count,
                n,
                log2_n,
                stage,
                1 if inverse else 0,
                0,
                0,
                0,
            ),
            f"fft stage {stage} meta",
        )
        bind_group = device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": z_buf}},
                {"binding": 1, "resource": {"buffer": meta_buf}},
            ],
        )
        pass_enc = cmd_buf.begin_compute_pass()
        pass_enc.set_pipeline(pipeline)
        pass_enc.set_bind_group(0, bind_group)
        _dispatch_1d(pass_enc, batch_count * (n // 2))
        pass_enc.end()


def rfft(tensor1, n=None):
    """Real FFT along the last dimension.

    Returns ``(spectrum, n)`` where ``spectrum`` is a complex64 tensor with
    shape ``(..., n // 2 + 1)``.  ``n`` may be larger than the input length
    (zero padding) or smaller (truncation).
    """
    if tensor1.type == "c64":
        raise TypeError("rfft expects a real-valued tensor")

    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()

    validation.check_rank(tensor1.shape, min_rank=1, name="rfft input")
    shape = list(tensor1.shape)
    in_len = shape[-1]
    n = in_len if n is None else int(n)
    _check_pow2(n)

    batch_count = _prod(shape[:-1])
    log2_n = int(math.log2(n))

    if n <= FFT_WORKGROUP_MAX:
        return _rfft_workgroup(tensor1, n, shape, batch_count, in_len, log2_n)

    # 1. real input -> bit-reversed complex buffer
    z_buf = _scratch_buffer(batch_count * n * 8)
    pipeline = gpu_ctx.get_pipeline(
        "fft_bitrev_real", shaders["helpers/fft_bitrev_real.wgsl"]
    )
    meta_buf = _meta_buffer(
        struct.pack("<4I", batch_count, n, log2_n, in_len),
        "rfft bitrev meta",
    )
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor1.buffer}},
            {"binding": 1, "resource": {"buffer": z_buf}},
            {"binding": 2, "resource": {"buffer": meta_buf}},
        ],
    )
    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    _dispatch_1d(pass_enc, batch_count * n)
    pass_enc.end()

    # 2. butterflies
    _run_fft_stages(z_buf, batch_count, n, log2_n, inverse=False)

    # 3. copy the non-redundant half into a compact contiguous buffer
    out_len = n // 2 + 1
    out_buf = _new_buffer(batch_count * out_len * 8)
    pipeline = gpu_ctx.get_pipeline(
        "fft_copy_out", shaders["helpers/fft_copy_out.wgsl"]
    )
    meta_buf = _meta_buffer(
        struct.pack("<4I", batch_count, n, out_len, 0), "rfft copy meta"
    )
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": z_buf}},
            {"binding": 1, "resource": {"buffer": out_buf}},
            {"binding": 2, "resource": {"buffer": meta_buf}},
        ],
    )
    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    _dispatch_1d(pass_enc, batch_count * out_len)
    pass_enc.end()

    out_shape = list(shape)
    out_shape[-1] = out_len
    out = _complex_tensor(
        out_shape, out_buf, requires_grad=tensor1.requires_grad
    )
    return out, n


def irfft(tensor1, n=None):
    """Inverse real FFT along the last dimension.

    ``tensor1`` is a complex one-sided spectrum.  The result is a real tensor
    with last dimension ``n``.  By default ``n = 2 * (in_len - 1)``.
    """
    if tensor1.type != "c64":
        raise TypeError("irfft expects a complex64 tensor")

    if not tensor1.is_contiguous():
        tensor1 = complex_contiguous(tensor1)

    validation.check_rank(tensor1.shape, min_rank=1, name="irfft input")
    shape = list(tensor1.shape)
    in_len = shape[-1]
    n = 2 * (in_len - 1) if n is None else int(n)
    _check_pow2(n)
    if n < 2:
        raise ValueError("irfft length must be at least 2")

    batch_count = _prod(shape[:-1])
    log2_n = int(math.log2(n))

    if n <= FFT_WORKGROUP_MAX:
        return _irfft_workgroup(tensor1, n, shape, batch_count, in_len, log2_n)

    # 1. one-sided spectrum -> full Hermitian spectrum in bit-reversed order
    z_buf = _scratch_buffer(batch_count * n * 8)
    pipeline = gpu_ctx.get_pipeline(
        "fft_bitrev_complex", shaders["helpers/fft_bitrev_complex.wgsl"]
    )
    meta_buf = _meta_buffer(
        struct.pack("<4I", batch_count, n, log2_n, in_len),
        "irfft bitrev meta",
    )
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor1.buffer}},
            {"binding": 1, "resource": {"buffer": z_buf}},
            {"binding": 2, "resource": {"buffer": meta_buf}},
        ],
    )
    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    _dispatch_1d(pass_enc, batch_count * n)
    pass_enc.end()

    # 2. inverse butterflies
    _run_fft_stages(z_buf, batch_count, n, log2_n, inverse=True)

    # 3. real part / n
    out_buf = _new_buffer(batch_count * n * 4)
    pipeline = gpu_ctx.get_pipeline(
        "ifft_write_real", shaders["helpers/ifft_write_real.wgsl"]
    )
    meta_buf = _meta_buffer(
        struct.pack("<IIfI", batch_count, n, 1.0 / float(n), 0),
        "irfft write meta",
    )
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": z_buf}},
            {"binding": 1, "resource": {"buffer": out_buf}},
            {"binding": 2, "resource": {"buffer": meta_buf}},
        ],
    )
    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    _dispatch_1d(pass_enc, batch_count * n)
    pass_enc.end()

    out_shape = list(shape)
    out_shape[-1] = n
    return Tensor(
        out_shape, out_buf, requires_grad=tensor1.requires_grad
    )


__all__ = [
    "rfft",
    "irfft",
    "complex_mul",
    "complex_add_inplace",
    "complex_conj",
    "complex_contiguous",
    "complex_pad",
    "complex_weights",
    "from_numpy_complex",
]
