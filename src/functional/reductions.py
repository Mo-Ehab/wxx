"""Reduction operations (sum, mean, max, min)."""
import math
import struct

import wgpu

from .. import validation
from ..backend.dispatch import _dispatch_1d, _get_reduce_dims

from ..backend.device import get_buffer_pool, get_cmd_buffer, get_ctx, get_device
from ..shaders import walk_shader_files
from ..Tensor import Tensor

cmd_buf = get_cmd_buffer()
buffer_pool = get_buffer_pool()
device = get_device()
gpu_ctx = get_ctx()

shaders = walk_shader_files()

pythonmin = min


WORKGROUP_SIZE = 256
MAX_WORKGROUPS_PER_DIMENSION = 65535

def sum(tensor1, axis=None, keepdims=False, *, keepdim=None):
    if keepdim is not None:
        keepdims = keepdim

    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    pipeline = gpu_ctx.get_pipeline("reduce", shaders["helpers/reduce.wgsl"])

    outer, reduce_size, inner, out_shape = _get_reduce_dims(tensor1.shape, axis)
    if keepdims:
        if axis is None:
            out_shape = (1,) * len(tensor1.shape)
        else:
            reduced = set(validation.normalize_dims(axis, len(tensor1.shape)))
            out_shape = tuple(
                1 if dim in reduced else size
                for dim, size in enumerate(tensor1.shape)
            )
    total_outputs = outer * inner

    u32_list = [outer, reduce_size, inner, 0]
    meta_data = struct.pack(f"<{len(u32_list)}I", *u32_list)
    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.UNIFORM, label="sum meta"
    )

    out_bytes = total_outputs * 4
    buf_out = buffer_pool.create_buffer(
        size=out_bytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
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
    if outer * inner:
        pass_enc.dispatch_workgroups(
            pythonmin(outer * inner, MAX_WORKGROUPS_PER_DIMENSION), 1, 1
        )
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)

    out = Tensor(out_shape, buffer=buf_out)

    return out


def sumsq(tensor1, axis=None, keepdims=False, *, keepdim=None):
    """Sum of squares along ``axis`` (fused, no temporary tensor).

    Equivalent to ``sum(tensor1 * tensor1, ...)`` but computes the square in
    the reduction kernel.  Used by :func:`wxx.nn.clip_grad_norm_` so the
    gradient norm can be accumulated entirely on the GPU.
    """
    if keepdim is not None:
        keepdims = keepdim

    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    pipeline = gpu_ctx.get_pipeline("reduce", shaders["helpers/reduce.wgsl"])

    outer, reduce_size, inner, out_shape = _get_reduce_dims(tensor1.shape, axis)
    if keepdims:
        if axis is None:
            out_shape = (1,) * len(tensor1.shape)
        else:
            reduced = set(validation.normalize_dims(axis, len(tensor1.shape)))
            out_shape = tuple(
                1 if dim in reduced else size
                for dim, size in enumerate(tensor1.shape)
            )
    total_outputs = outer * inner

    # mode 2 = fused sum of squares
    u32_list = [outer, reduce_size, inner, 2]
    meta_data = struct.pack(f"<{len(u32_list)}I", *u32_list)
    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.UNIFORM, label="sumsq meta"
    )

    out_bytes = total_outputs * 4
    buf_out = buffer_pool.create_buffer(
        size=out_bytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
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
    if total_outputs:
        pass_enc.dispatch_workgroups(
            pythonmin(total_outputs, MAX_WORKGROUPS_PER_DIMENSION), 1, 1
        )
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)

    return Tensor(out_shape, buffer=buf_out)


def mean(tensor1, axis=None, keepdims=False, *, keepdim=None):
    if keepdim is not None:
        keepdims = keepdim

    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    pipeline = gpu_ctx.get_pipeline("reduce", shaders["helpers/reduce.wgsl"])

    outer, reduce_size, inner, out_shape = _get_reduce_dims(tensor1.shape, axis)
    if keepdims:
        if axis is None:
            out_shape = (1,) * len(tensor1.shape)
        else:
            reduced = set(validation.normalize_dims(axis, len(tensor1.shape)))
            out_shape = tuple(
                1 if dim in reduced else size
                for dim, size in enumerate(tensor1.shape)
            )
    total_outputs = outer * inner

    u32_list = [outer, reduce_size, inner, 1]
    meta_data = struct.pack(f"<{len(u32_list)}I", *u32_list)
    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.UNIFORM, label="mean meta"
    )

    out_bytes = total_outputs * 4
    buf_out = buffer_pool.create_buffer(
        size=out_bytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
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
    if total_outputs:
        pass_enc.dispatch_workgroups(
            pythonmin(total_outputs, MAX_WORKGROUPS_PER_DIMENSION), 1, 1
        )
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)

    out = Tensor(out_shape, buffer=buf_out)

    return out


def maxmin_backward(grad_output, indices, input_shape, axis=None):
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()
    if not indices.is_contiguous():
        indices = indices.contiguous()

    outer, reduce_size, inner, _ = _get_reduce_dims(input_shape, axis)
    numel = int(math.prod(input_shape))
    pipeline = gpu_ctx.get_pipeline(
        "maxmin_backward", shaders["helpers/maxmin_backward.wgsl"]
    )
    meta_buf = buffer_pool.create_buffer_with_data(
        data=struct.pack("<4I", outer, reduce_size, inner, numel),
        usage=wgpu.BufferUsage.UNIFORM,
        label="maxmin dx meta"
    )
    buf_out = buffer_pool.create_buffer(
        size=numel * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": grad_output.buffer}},
            {"binding": 1, "resource": {"buffer": indices.buffer}},
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

    return Tensor(input_shape, buffer=buf_out)



def _minmax(tensor1, axis, keepdims, op):
    """Shared implementation of :func:`max` and :func:`min`.

    ``op`` is ``0`` for max and ``1`` for min; it is forwarded to the fused
    ``reduce_minmax`` kernel so only one shader/pipeline is needed.
    """
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    pipeline = gpu_ctx.get_pipeline(
        "reduce_minmax", shaders["helpers/reduce_minmax.wgsl"]
    )

    outer, reduce_size, inner, out_shape = _get_reduce_dims(tensor1.shape, axis)
    total_outputs = outer * inner

    u32_list = [outer, reduce_size, inner, op]
    meta_data = struct.pack(f"<{len(u32_list)}I", *u32_list)
    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.UNIFORM, label="reduce_minmax meta"
    )

    out_bytes = total_outputs * 4
    buf_out = buffer_pool.create_buffer(
        size=out_bytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )

    buf_out_2 = buffer_pool.create_buffer(
        size=out_bytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor1.buffer}},
            {"binding": 1, "resource": {"buffer": buf_out}},
            {"binding": 2, "resource": {"buffer": buf_out_2}},
            {"binding": 3, "resource": {"buffer": meta_buf}},
        ],
    )

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    pass_enc.dispatch_workgroups(total_outputs, 1, 1)
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)

    out = Tensor(out_shape, buffer=buf_out)
    out_2 = Tensor(out_shape, buffer=buf_out_2)
    out_2.type = "u32"

    if keepdims:
        if axis is None:
            out_shape = (1,) * len(tensor1.shape)
        else:
            reduced = set(validation.normalize_dims(axis, len(tensor1.shape)))
            out_shape = tuple(
                1 if dim in reduced else size
                for dim, size in enumerate(tensor1.shape)
            )
        for result in (out, out_2):
            result.shape = list(out_shape)
            result.stride = result.calc_contiguous_strides()

    return out, out_2


def max(tensor1, axis=None, keepdims=False, *, keepdim=None):
    if keepdim is not None:
        keepdims = keepdim
    return _minmax(tensor1, axis, keepdims, 0)


def var(tensor1, axis=None, keepdims=False, *, keepdim=None):
    """Variance along ``axis`` (population, ddof=0)."""
    from .pointwise import pointwise

    if keepdim is not None:
        keepdims = keepdim
    if tensor1.type != "f32":
        raise TypeError(f"var expects an f32 tensor, got {tensor1.type!r}")
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    mean_t = mean(tensor1, axis, keepdims=True)
    diff = pointwise(tensor1, mean_t, 1)
    return mean(pointwise(diff, diff, 2), axis, keepdims)


def std(tensor1, axis=None, keepdims=False, *, keepdim=None):
    """Standard deviation along ``axis`` (population, ddof=0)."""
    from .pointwise import sqrt

    if keepdim is not None:
        keepdims = keepdim
    return sqrt(var(tensor1, axis=axis, keepdims=keepdims))


def min(tensor1, axis=None, keepdims=False, *, keepdim=None):
    if keepdim is not None:
        keepdims = keepdim
    return _minmax(tensor1, axis, keepdims, 1)
