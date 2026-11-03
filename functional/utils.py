"""Tensor utilities: assignment, contiguous, permute, reshape, init helpers."""

import math
import struct

import numpy as np
import wgpu

from wxx import autograd
from wxx.backend.dispatch import _dispatch_1d

from .. import validation
from ..backend.device import get_buffer_pool, get_cmd_buffer, get_ctx, get_device
from ..shaders import walk_shader_files
from ..Tensor import Tensor

cmd_buf = get_cmd_buffer()
buffer_pool = get_buffer_pool()
device = get_device()
gpu_ctx = get_ctx()

shaders = walk_shader_files()

WORKGROUP_SIZE = 256
MAX_WORKGROUPS_PER_DIMENSION = 65535



def assign(tensor1, val):
    """Copy ``val`` into ``tensor1`` (element-wise or a scalar fill)."""
    pipeline = gpu_ctx.get_pipeline("assign", shaders["helpers/assign.wgsl"])

    rank = validation.check_rank(tensor1.shape, name="assign target")
    numel = validation.numel(tensor1.shape)
    padded_shape = list(tensor1.shape) + [0] * (6 - rank)
    padded_strides = list(tensor1.stride) + [0] * (6 - rank)
    if isinstance(val, (float, int)):
        u32_list = (
            [numel, rank, 1, float(val), tensor1.offset, 0, 0, 0]
            + padded_shape
            + padded_strides
            + [0] * 6
        )

        src = buffer_pool.create_buffer(size=16, usage=wgpu.BufferUsage.STORAGE)
        cmd_buf.keep_alive(src)

        meta_data = struct.pack("<3If22I", *u32_list)

        meta_buf = buffer_pool.create_buffer_with_data(
            data=meta_data, usage=wgpu.BufferUsage.STORAGE, label="assign meta"
        )
    else:
        if not isinstance(val, Tensor):
            raise TypeError(
                f"assign expects a Tensor or a scalar, got {type(val).__name__}"
            )
        if tuple(val.shape) != tuple(tensor1.shape):
            raise ValueError(
                f"assign shape mismatch: target {tuple(tensor1.shape)} vs "
                f"source {tuple(val.shape)}"
            )
        if val.type != tensor1.type:
            raise TypeError(
                f"assign dtype mismatch: target {tensor1.type!r} vs "
                f"source {val.type!r}"
            )
        padded_strides_src = list(val.stride) + [0] * (6 - rank)

        u32_list = (
            [numel, rank, 0, 0, tensor1.offset, 0, 0, 0]
            + padded_shape
            + padded_strides
            + padded_strides_src
        )

        src = val.buffer
        meta_data = struct.pack("<3If22I", *u32_list)

        meta_buf = buffer_pool.create_buffer_with_data(
            data=meta_data, usage=wgpu.BufferUsage.STORAGE, label="assign meta"
        )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor1.buffer}},
            {"binding": 1, "resource": {"buffer": src}},
            {"binding": 2, "resource": {"buffer": meta_buf}},
        ],
    )

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    _dispatch_1d(pass_enc, numel)
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)

    return tensor1


def permute(tensor1, *axes):
    """Permute the dimensions of ``tensor1`` according to ``axes`` (view)."""
    if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
        axes = tuple(axes[0])
    rank = len(tensor1.shape)
    if len(axes) != rank:
        raise ValueError(
            f"permute expects {rank} axes for shape {tuple(tensor1.shape)}, "
            f"got {len(axes)}"
        )
    axes = validation.normalize_dims(axes, rank, name="axes", allow_duplicates=False)

    new_shape = tuple(tensor1.shape[i] for i in axes)
    new_stride = tuple(tensor1.stride[i] for i in axes)

    out = Tensor(
        shape=new_shape,
        buffer=tensor1.buffer,
        stride=new_stride,
        offset=tensor1.offset,
        requires_grad=tensor1.requires_grad,
        dtype=tensor1.type,
    )
    out._base = tensor1
    return out


def reshape(tensor1, new_shape):
    """Return a tensor with the same elements laid out as ``new_shape``."""
    total = validation.numel(tensor1.shape)
    new_shape = validation.infer_shape(new_shape, total, name="new shape")

    if tensor1.is_contiguous():
        out_buf = tensor1.buffer
        offset = tensor1.offset
        base = tensor1
    else:
        contiguous_tensor = tensor1.contiguous()
        out_buf = contiguous_tensor.buffer
        offset = contiguous_tensor.offset
        base = contiguous_tensor

    out = Tensor(
        shape=new_shape,
        buffer=out_buf,
        offset=offset,
        requires_grad=tensor1.requires_grad,
        dtype=tensor1.type,
    )
    out._base = base
    return out


def contiguous(tensor1):
    """Return a contiguous copy of ``tensor1`` (no-op if already contiguous)."""
    if tensor1.is_contiguous():
        return tensor1
    if getattr(tensor1, "type", "f32") == "c64":
        from .fft import complex_contiguous

        return complex_contiguous(tensor1)
    pipeline = gpu_ctx.get_pipeline("contiguous", shaders["helpers/contiguous.wgsl"])

    rank = validation.check_rank(tensor1.shape, name="contiguous input")

    numel = validation.numel(tensor1.shape)

    padded_shape = list(tensor1.shape) + [0] * (6 - rank)
    padded_strides = list(tensor1.stride) + [0] * (6 - rank)

    u32_list = (
        [numel, rank, tensor1.offset, 0] + padded_shape + padded_strides + [0, 0, 0, 0]
    )

    meta_data = struct.pack(f"<{len(u32_list)}I", *u32_list)

    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.STORAGE, label="contiguous meta"
    )

    buf_out = buffer_pool.create_buffer(
        size=numel * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor1.buffer}},
            {"binding": 1, "resource": {"buffer": buf_out}},
            {"binding": 2, "resource": {"buffer": meta_buf}},
        ],
    )

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    _dispatch_1d(pass_enc, numel)
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)

    out = Tensor(
        buffer=buf_out, shape=tensor1.shape, requires_grad=tensor1.requires_grad
    )

    return out








def matmul_add(tensor1, tensor2, tensor3, transpose_A=False, transpose_B=False):
    """Batched ``A @ B (+ bias)`` with optional transposes of the last two dims."""
    validation.check_rank(tensor1.shape, min_rank=2, name="matmul left operand")
    validation.check_rank(tensor2.shape, min_rank=2, name="matmul right operand")
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    if not tensor2.is_contiguous():
        tensor2 = tensor2.contiguous()

    pipeline = gpu_ctx.get_pipeline("matmul", shaders["matmul.wgsl"])

    if transpose_A:
        M, K = tensor1.shape[-1], tensor1.shape[-2]
    else:
        M, K = tensor1.shape[-2], tensor1.shape[-1]

    if transpose_B:
        K_B, N = tensor2.shape[-1], tensor2.shape[-2]
    else:
        K_B, N = tensor2.shape[-2], tensor2.shape[-1]
    if K != K_B:
        raise ValueError(f"Inner dimension mismatch: {K} vs {K_B}")

    batch_shape_A = tensor1.shape[:-2]
    batch_shape_B = tensor2.shape[:-2]

    if batch_shape_A and batch_shape_B and batch_shape_A != batch_shape_B:
        raise ValueError(
            f"Batch dimensions must match or be unbatched: {batch_shape_A} vs {batch_shape_B}"
        )

    out_batch_shape = batch_shape_A or batch_shape_B
    batch_count = int(math.prod(out_batch_shape)) if out_batch_shape else 1
    stride_A_batch = M * K if batch_shape_A else 0
    stride_B_batch = K * N if batch_shape_B else 0
    stride_C_batch = M * N

    # Detect bias mode and stride
    bias_mode = 0
    stride_bias_batch = 0

    if tensor3 is not None and tensor3.shape is not None:
        if not tensor3.is_contiguous():
            tensor3 = tensor3.contiguous()
        bias_shape = tuple(tensor3.shape)
        if bias_shape == (N,):
            bias_mode = 1  # 1D Vector (N,)
        elif bias_shape == (M, N):
            bias_mode = 2  # 2D Matrix (M, N)
        elif bias_shape == tuple(out_batch_shape) + (M, N):
            bias_mode = 3  # Batched Tensor (B, M, N)
            stride_bias_batch = M * N
        else:
            raise ValueError(
                f"Bias shape {bias_shape} is not broadcastable to {tuple(out_batch_shape) + (M, N)}"
            )
    out_shape = out_batch_shape + [M, N]
    out_size = batch_count * M * N

    u32_list = [
        M,
        N,
        K,
        batch_count,
        stride_A_batch,
        stride_B_batch,
        stride_C_batch,
        stride_bias_batch,
        bias_mode,
        1 if transpose_A else 0,
        1 if transpose_B else 0,
        0,
    ]
    meta_data = struct.pack(f"<{len(u32_list)}I", *u32_list)

    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.UNIFORM, label="matmul add meta"
    )

    buf_out = buffer_pool.create_buffer(
        size=out_size * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )

    buf_bias = tensor1.buffer if tensor3 is None else tensor3.buffer

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor1.buffer}},
            {"binding": 1, "resource": {"buffer": tensor2.buffer}},
            {"binding": 2, "resource": {"buffer": buf_bias}},
            {"binding": 3, "resource": {"buffer": buf_out}},
            {"binding": 4, "resource": {"buffer": meta_buf}},
        ],
    )

    tile_dim = 64

    # Calculate workgroup dispatches (works for 100352 x 8 and all large shapes)
    dispatch_x = (N + tile_dim - 1) // tile_dim  # (8 + 63) // 64 = 1
    dispatch_y = min(
        (M + tile_dim - 1) // tile_dim, 64
    )  # Cap grid at 64; grid-stride processes all 1568 tiles
    dispatch_z = batch_count

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    pass_enc.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z)
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)

    return Tensor(
        out_shape,
        buf_out,
        requires_grad=(
            tensor1.requires_grad
            or tensor2.requires_grad
            or (tensor3 is not None and tensor3.requires_grad)
        ),
    )









def _equal_impl(tensor1, tensor2, mode, number=0):
    """Run the shared equality kernel.

    ``mode`` is ``0`` (f32/f32), ``1`` (u32/u32), ``2`` (f32/scalar) or
    ``3`` (u32/scalar).  For scalar modes ``tensor2`` may be ``None``; the
    first input buffer is bound as a safe placeholder for the unused slot.
    """
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    if tensor2 is not None and not tensor2.is_contiguous():
        tensor2 = tensor2.contiguous()

    pipeline = gpu_ctx.get_pipeline("equal", shaders["helpers/equal.wgsl"])

    meta_buf = buffer_pool.create_buffer_with_data(
        data=struct.pack("<4I", mode, int(number) & 0xFFFFFFFF, 0, 0),
        usage=wgpu.BufferUsage.UNIFORM,
        label="equal meta",
    )

    buf_out = buffer_pool.create_buffer(
        size=tensor1.nbytes,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    buf_b = tensor2.buffer if tensor2 is not None else tensor1.buffer

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor1.buffer}},
            {"binding": 1, "resource": {"buffer": buf_b}},
            {"binding": 2, "resource": {"buffer": buf_out}},
            {"binding": 3, "resource": {"buffer": meta_buf}},
        ],
    )

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    _dispatch_1d(pass_enc, tensor1.nbytes // 4)
    pass_enc.end()
    cmd_buf.keep_alive(meta_buf)

    return Tensor(tensor1.shape, buf_out, requires_grad=(tensor1.requires_grad))


def equal(tensor1, tensor2):
    """Elementwise equality, returning ``1.0``/``0.0`` as ``f32``."""
    if isinstance(tensor2, (int, float)):
        return equal_number(tensor1, tensor2)

    if tensor1.type not in ("f32", "u32") or tensor1.type != tensor2.type:
        raise TypeError(
            f"equal expects matching f32 or u32 tensors, got "
            f"{tensor1.type!r} and {tensor2.type!r}"
        )
    return _equal_impl(tensor1, tensor2, 0 if tensor1.type == "f32" else 1)


def equal_number(tensor1, number):
    """Compare every element of ``tensor1`` against a scalar."""
    if tensor1.type == "f32":
        bits = struct.unpack("<I", struct.pack("<f", float(number)))[0]
        return _equal_impl(tensor1, None, 2, bits)
    if tensor1.type == "u32":
        return _equal_impl(tensor1, None, 3, int(number))
    raise TypeError(f"equal_number expects an f32 or u32 tensor, got {tensor1.type!r}")

def masked_fill(tensor1, number):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()

    pipeline = gpu_ctx.get_pipeline("maskedfill", shaders["helpers/maskedfill.wgsl"])

    u32_list = [number, 0, 0, 0]
    meta_data = struct.pack("<4f", *u32_list)

    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.STORAGE, label="masked fill meta"
    )



    buf_out = buffer_pool.create_buffer(
        size=tensor1.nbytes,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )



    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor1.buffer}},
            {"binding": 1, "resource": {"buffer": buf_out}},
            {"binding": 2, "resource": {"buffer": meta_buf}},
        ],
    )

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    _dispatch_1d(pass_enc, tensor1.nbytes // 4)
    pass_enc.end()
    cmd_buf.keep_alive(meta_buf)

    return Tensor(tensor1.shape, buf_out, requires_grad=(tensor1.requires_grad))


def linspace(start, end, steps):
    """Evenly spaced 1-D tensor with `steps` values in `[start, end]`."""
    values = np.linspace(start, end, int(steps), dtype=np.float32)
    return Tensor((int(steps),), data=values.tobytes())


def ones_like(tensor1):
    return Tensor(tensor1.shape, value=1)


def arange(start, shape):
    values = []
    if isinstance(shape, int):
        shape = (shape,)
    if isinstance(shape, list):
        shape = tuple(shape)

    for i in range(math.prod(shape)):
        values += [(i % shape[-1]) + start]

    values = np.array(values).astype(dtype=np.float32).data

    return Tensor(shape, data=values)


def normal(mu, sigma, shape):
    rng = np.random.default_rng()

    if isinstance(shape, int):
        shape = (shape,)
    if isinstance(shape, list):
        shape = tuple(shape)

    values = rng.normal(mu, sigma, size=math.prod(shape)).astype(dtype=np.float32).data

    return Tensor(shape, data=values)


def uniform(a, b, shape):
    rng = np.random.default_rng()

    if isinstance(shape, int):
        shape = (shape,)
    if isinstance(shape, list):
        shape = tuple(shape)
    values = rng.uniform(a, b, size=math.prod(shape)).astype(dtype=np.float32).data

    return Tensor(shape, data=values)


def from_numpy(array, dtype=np.float32):
    """Create a tensor from a numpy array.

    ``dtype`` may be ``np.float32`` (default) or ``np.uint32``.  For complex
    data use :func:`from_numpy_complex` instead.
    """
    if dtype == np.uint32:
        array = np.ascontiguousarray(array, dtype=np.uint32)
        buffer = buffer_pool.create_buffer_with_data(
            data=memoryview(array),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
            label="from_numpy",
        )
        tensor = Tensor(list(array.shape), buffer)
        tensor.type = "u32"
        tensor.nbytes = int(array.nbytes)
        return tensor

    array = np.asarray(array, dtype=np.float32, order="C")
    buffer = buffer_pool.create_buffer_with_data(
        data=memoryview(array),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        label="from_numpy"
    )

    tensor = Tensor(list(array.shape), buffer)

    return tensor


def from_tensor(array, type=np.float32):
    """Create a tensor from a torch tensor or numpy array."""
    if hasattr(array, "detach"):
        array = array.detach().cpu().contiguous().numpy()
    array = np.asarray(array, dtype=type, order="C")

    if type == np.uint32:
        return from_numpy(array, dtype=np.uint32)

    buffer = buffer_pool.create_buffer_with_data(
        data=memoryview(array),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        label="from_tensor"
    )

    tensor = Tensor(list(array.shape), buffer)
    tensor.type = "f32" if type == np.float32 else tensor.type
    return tensor


def concat(tensors, dim):

    return autograd.concat(*tensors).forward(tensors, dim)


def _zero_init(tensor):
    tensor[:] = 0


def _xavier_normal_init(tensor, fan_in, fan_out):
    sigma = math.sqrt((2) / (fan_in + fan_out))
    n = normal(0, sigma, tensor.shape)
    assign(tensor, n)
    n.destroy()
    del n


def _xavier_uniform_init(tensor, fan_in, fan_out):
    a = math.sqrt((6) / (fan_in + fan_out))
    n = uniform(-a, a, tensor.shape)
    assign(tensor, n)
    n.destroy()
    del n


def _kaiming_normal_init(tensor, fan_in):
    sigma = math.sqrt((2) / (fan_in))
    n = normal(0, sigma, tensor.shape)
    assign(tensor, n)
    n.destroy()
    del n


def _kaiming_uniform_init(tensor, fan_in):
    a = math.sqrt((6) / (fan_in))
    n = uniform(-a, a, tensor.shape)
    assign(tensor, n)
    n.destroy()
    del n


def scatter_zeros(grad, out_shape, dim, start):
    """Scatter ``grad`` into a zero-filled tensor of ``out_shape`` along ``dim``.

    Single-dispatch replacement for the ``Tensor(shape, value=0)`` + ``assign``
    pair that the backward passes of ``narrow`` / basic slicing use.  The
    zero-fill happens in the kernel, so no host-side zero array is uploaded.
    """
    out_shape = list(validation.normalize_shape(out_shape, name="out_shape"))
    rank = validation.check_rank(out_shape, name="scatter_zeros output")
    dim = validation.normalize_dim(dim, rank, name="dim")
    length = grad.shape[dim]
    if not 0 <= start <= out_shape[dim] - length:
        raise IndexError(
            f"scatter_zeros bounds out of range: start={start}, length={length}, "
            f"dim {dim} has size {out_shape[dim]}"
        )
    for axis, (actual, expected) in enumerate(zip(grad.shape, out_shape)):
        if axis != dim and actual != expected:
            raise ValueError(
                f"scatter_zeros gradient shape {tuple(grad.shape)} does not fit "
                f"output shape {tuple(out_shape)}"
            )
    if any(s < 0 for s in grad.stride):
        grad = grad.contiguous()

    pipeline = gpu_ctx.get_pipeline(
        "scatter_zeros", shaders["helpers/scatter_zeros.wgsl"]
    )

    out_numel = int(math.prod(out_shape))
    padded_out_shape = out_shape + [0] * (6 - rank)
    padded_strides = list(grad.stride) + [0] * (6 - rank)
    meta_data = struct.pack(
        "<20I",
        out_numel,
        rank,
        dim,
        start,
        length,
        grad.offset,
        *padded_out_shape,
        *padded_strides,
        0,
        0,
    )
    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.STORAGE, label="scatter_zeros meta"
    )
    out_buf = buffer_pool.create_buffer(
        size=max(4, out_numel * 4),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": grad.buffer}},
            {"binding": 1, "resource": {"buffer": out_buf}},
            {"binding": 2, "resource": {"buffer": meta_buf}},
        ],
    )
    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    _dispatch_1d(pass_enc, out_numel)
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)
    return Tensor(out_shape, out_buf)
