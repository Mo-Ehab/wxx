"""Elementwise binary/unary pointwise operations."""
import math
import struct

import wgpu

from ..backend.dispatch import (
    _dispatch_1d,
    _unary_op,
    broadcast_shapes,
    compute_broadcast_strides,
    supports_strided,
    view_broadcast_strides,
)

from .. import validation

from ..backend.device import get_buffer_pool, get_cmd_buffer, get_ctx, get_device
from ..shaders import walk_shader_files
from ..Tensor import Tensor

cmd_buf = get_cmd_buffer()
buffer_pool = get_buffer_pool()
device = get_device()
gpu_ctx = get_ctx()

shaders = walk_shader_files()

def pointwise(tensor1, tensor2, op):
    if not isinstance(tensor2, Tensor):
        return scalarpointwise(tensor1, tensor2, op)
    # Strided views are addressed directly by the kernel (offset + strides), so
    # only layouts the shader cannot express need materialising.
    if not supports_strided(tensor1):
        tensor1 = tensor1.contiguous()
    if not supports_strided(tensor2):
        tensor2 = tensor2.contiguous()

    pipeline = gpu_ctx.get_pipeline("pointwise", shaders["pointwise/pointwise.wgsl"])

    if tensor1.type != "f32" or tensor2.type != "f32":
        raise TypeError(
            f"pointwise expects f32 tensors, got {tensor1.type!r} and "
            f"{tensor2.type!r}"
        )
    out_shape, pad_A, pad_B = broadcast_shapes(tensor1.shape, tensor2.shape)
    rank = validation.check_rank(out_shape, name="pointwise output")

    strides_A = view_broadcast_strides(tensor1, out_shape)
    strides_B = view_broadcast_strides(tensor2, out_shape)

    numel = int(math.prod(out_shape))

    # Pad shape and stride lists to length 6 for WGSL array<u32, 6>
    padded_out_shape = list(out_shape) + [0] * (6 - rank)
    padded_strides_A = list(strides_A) + [0] * (6 - rank)
    padded_strides_B = list(strides_B) + [0] * (6 - rank)

    u32_list = (
        [numel, rank, op, tensor1.offset, tensor2.offset, 0]
        + padded_out_shape
        + padded_strides_A
        + padded_strides_B
    )
    meta_data = struct.pack(f"<{len(u32_list)}I", *u32_list)

    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.STORAGE, label="pointwise meta"
    )

    buf_out = buffer_pool.create_buffer(
        size=numel * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor1.buffer}},
            {"binding": 1, "resource": {"buffer": tensor2.buffer}},
            {"binding": 2, "resource": {"buffer": buf_out}},
            {"binding": 3, "resource": {"buffer": meta_buf}},
        ],
    )

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    _dispatch_1d(pass_enc, numel)
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)

    return Tensor(
        out_shape,
        buf_out,
        requires_grad=(tensor1.requires_grad or tensor2.requires_grad),
    )


def scalarpointwise(tensor1, value, op):
    if tensor1.type != "f32":
        raise TypeError(f"scalarpointwise expects f32 input, got {tensor1.type!r}")
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()

    pipeline = gpu_ctx.get_pipeline(
        "scalarpointwise", shaders["pointwise/scalarpointwise.wgsl"]
    )

    numel = int(math.prod(tensor1.shape))

    u32_list = [numel, int(op), value, 0]
    meta_data = struct.pack("IIfI", *u32_list)

    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.STORAGE, label="scalarpointwise meta"
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

    return Tensor(tensor1.shape, buf_out, requires_grad=(tensor1.requires_grad))


def clamp(tensor1, min, max):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()

    pipeline = gpu_ctx.get_pipeline("clamp", shaders["helpers/clamp.wgsl"])

    numel = int(math.prod(tensor1.shape))

    u32_list = [float(min), float(max), 0, 0]
    meta_data = struct.pack("ffII", *u32_list)

    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.STORAGE, label="clamp meta"
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

    return Tensor(tensor1.shape, buf_out, requires_grad=(tensor1.requires_grad))


def clampbackwards(tensor1, min, max):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()

    pipeline = gpu_ctx.get_pipeline(
        "clampbackwards", shaders["helpers/clampbackwards.wgsl"]
    )

    numel = int(math.prod(tensor1.shape))

    u32_list = [float(min), float(max), 0, 0]
    meta_data = struct.pack("ffII", *u32_list)

    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.STORAGE, label="clamp dx meta"
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

    return Tensor(tensor1.shape, buf_out, requires_grad=(tensor1.requires_grad))


def _pointwise(tensor1, tensor2, op):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()

    if not tensor2.is_contiguous():
        tensor2 = tensor2.contiguous()

    pipeline = gpu_ctx.get_pipeline("_pointwise", shaders["pointwise/_pointwise.wgsl"])

    if tensor1.type != "f32" or tensor2.type != "f32":
        raise TypeError(
            f"_pointwise expects f32 tensors, got {tensor1.type!r} and "
            f"{tensor2.type!r}"
        )
    out_shape, pad_A, pad_B = broadcast_shapes(tensor1.shape, tensor2.shape)
    rank = validation.check_rank(out_shape, name="_pointwise output")

    strides_A = compute_broadcast_strides(pad_A, out_shape)
    strides_B = compute_broadcast_strides(pad_B, out_shape)

    numel = int(math.prod(out_shape))

    # Pad shape and stride lists to length 6 for WGSL array<u32, 6>
    padded_out_shape = list(out_shape) + [0] * (6 - rank)
    padded_strides_A = list(strides_A) + [0] * (6 - rank)
    padded_strides_B = list(strides_B) + [0] * (6 - rank)

    u32_list = (
        [numel, rank, op, 0] + padded_out_shape + padded_strides_A + padded_strides_B
    )
    meta_data = struct.pack(f"<{len(u32_list)}I", *u32_list)

    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.STORAGE, label="_pointwise meta"
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor1.buffer}},
            {"binding": 1, "resource": {"buffer": tensor2.buffer}},
            {"binding": 2, "resource": {"buffer": meta_buf}},
        ],
    )

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    _dispatch_1d(pass_enc, numel)
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)



def negate(tensor1):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    pipeline = gpu_ctx.get_pipeline("negate", shaders["pointwise/negate.wgsl"])

    return _unary_op(tensor1, pipeline)


def abs(tensor1):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    pipeline = gpu_ctx.get_pipeline("abs", shaders["pointwise/abs.wgsl"])

    return _unary_op(tensor1, pipeline)


def sign(tensor1):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    pipeline = gpu_ctx.get_pipeline("sign", shaders["pointwise/sign.wgsl"])

    return _unary_op(tensor1, pipeline)


def square(tensor1):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    pipeline = gpu_ctx.get_pipeline("square", shaders["pointwise/square.wgsl"])

    return _unary_op(tensor1, pipeline)


def exp(tensor1):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    pipeline = gpu_ctx.get_pipeline("exp", shaders["pointwise/exp.wgsl"])

    return _unary_op(tensor1, pipeline)


def sqrt(tensor1):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    pipeline = gpu_ctx.get_pipeline("sqrt", shaders["pointwise/sqrt.wgsl"])

    return _unary_op(tensor1, pipeline)


def rsqrt(tensor1):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    pipeline = gpu_ctx.get_pipeline("rsqrt", shaders["pointwise/rsqrt.wgsl"])

    return _unary_op(tensor1, pipeline)


def log(tensor1):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    pipeline = gpu_ctx.get_pipeline("log", shaders["pointwise/log.wgsl"])

    return _unary_op(tensor1, pipeline)


def decay_window(decay, stage, length):
    """Causal exponential window ``exp(-|decay[stage, d]| * t)``.

    Returns a ``(channels, length)`` tensor.  Fused into one dispatch so the
    ``abs -> mul -> negate -> exp`` chain (and its backward graph) disappears.
    """
    if decay.type != "f32":
        raise TypeError(f"decay_window expects f32 decay, got {decay.type!r}")
    if not decay.is_contiguous():
        decay = decay.contiguous()
    if len(decay.shape) != 2:
        raise ValueError(f"decay_window expects a 2-D decay, got {decay.shape}")
    rows, channels = decay.shape
    if not isinstance(length, int) or length < 1:
        raise ValueError(f"decay_window length must be a positive int, got {length!r}")
    if not (0 <= stage < rows):
        raise IndexError(f"decay_window stage {stage} out of range for {rows} rows")

    pipeline = gpu_ctx.get_pipeline(
        "decay_window", shaders["helpers/decay_window.wgsl"]
    )
    meta_buf = buffer_pool.create_buffer_with_data(
        data=struct.pack("<4I", rows, channels, length, stage),
        usage=wgpu.BufferUsage.UNIFORM,
        label="decay_window meta",
    )
    out_buf = buffer_pool.create_buffer(
        size=channels * length * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": decay.buffer}},
            {"binding": 1, "resource": {"buffer": out_buf}},
            {"binding": 2, "resource": {"buffer": meta_buf}},
        ],
    )
    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    _dispatch_1d(pass_enc, channels * length)
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)
    return Tensor(
        (channels, length), out_buf, requires_grad=decay.requires_grad
    )


def decay_window_backward(grad, decay, stage, length):
    """Backward of :func:`decay_window`.

    Returns a ``(rows, channels)`` gradient for the whole decay parameter:
    the kernel writes zeros in every stage row except ``stage``, so the
    narrow/scatter kernel is fused in as well.
    """
    if not grad.is_contiguous():
        grad = grad.contiguous()
    if decay.type != "f32":
        raise TypeError(f"decay_window expects f32 decay, got {decay.type!r}")
    if not decay.is_contiguous():
        decay = decay.contiguous()
    if len(decay.shape) != 2:
        raise ValueError(f"decay_window expects a 2-D decay, got {decay.shape}")
    rows, channels = decay.shape
    if not isinstance(length, int) or length < 1:
        raise ValueError(f"decay_window length must be a positive int, got {length!r}")
    if not (0 <= stage < rows):
        raise IndexError(f"decay_window stage {stage} out of range for {rows} rows")

    pipeline = gpu_ctx.get_pipeline(
        "decay_window_backward",
        shaders["helpers/decay_window_backward.wgsl"],
    )
    meta_buf = buffer_pool.create_buffer_with_data(
        data=struct.pack("<4I", rows, channels, length, stage),
        usage=wgpu.BufferUsage.UNIFORM,
        label="decay_window bwd meta",
    )
    out_buf = buffer_pool.create_buffer(
        size=rows * channels * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": grad.buffer}},
            {"binding": 1, "resource": {"buffer": decay.buffer}},
            {"binding": 2, "resource": {"buffer": out_buf}},
            {"binding": 3, "resource": {"buffer": meta_buf}},
        ],
    )
    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    _dispatch_1d(pass_enc, rows * channels)
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)
    return Tensor((rows, channels), out_buf)


def mul_backward(grad, tensor_a, tensor_b):
    """Return ``(grad * b, grad * a)`` in a single dispatch.

    Only valid when all three tensors have identical, contiguous shapes; the
    caller falls back to two pointwise kernels otherwise.
    """
    numel = int(math.prod(grad.shape))
    pipeline = gpu_ctx.get_pipeline(
        "mul_backward", shaders["helpers/mul_backward.wgsl"]
    )
    meta_buf = buffer_pool.create_buffer_with_data(
        data=struct.pack("<4I", numel, 0, 0, 0),
        usage=wgpu.BufferUsage.UNIFORM,
        label="mul_backward meta",
    )
    da_buf = buffer_pool.create_buffer(
        size=numel * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    db_buf = buffer_pool.create_buffer(
        size=numel * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": grad.buffer}},
            {"binding": 1, "resource": {"buffer": tensor_a.buffer}},
            {"binding": 2, "resource": {"buffer": tensor_b.buffer}},
            {"binding": 3, "resource": {"buffer": da_buf}},
            {"binding": 4, "resource": {"buffer": db_buf}},
            {"binding": 5, "resource": {"buffer": meta_buf}},
        ],
    )
    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    _dispatch_1d(pass_enc, numel)
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)
    shape = list(grad.shape)
    return Tensor(shape, da_buf), Tensor(shape, db_buf)
