"""Normalization primitives (RMSNorm, LayerNorm, BatchNorm)."""
import math
import struct

import wgpu

from ..backend.dispatch import _dispatch_1d, _get_reduce_dims

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


def rmsnorm(tensor1, gamma, axis=None):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    if not gamma.is_contiguous():
        gamma = gamma.contiguous()

    pipeline = gpu_ctx.get_pipeline("rmsnorm", shaders["layers/rmsnorm.wgsl"])

    outer, reduce_size, inner, out_shape = _get_reduce_dims(tensor1.shape, axis)

    total_outputs = outer * inner * reduce_size

    u32_list = [outer, reduce_size, inner, 1e-6]
    meta_data = struct.pack("<IIIf", *u32_list)
    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.UNIFORM, label="rmsnorm meta"
    )

    out_bytes = total_outputs * 4
    buf_out = buffer_pool.create_buffer(
        size=out_bytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )
    buf_irms = buffer_pool.create_buffer(
        size=inner * outer * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor1.buffer}},
            {"binding": 1, "resource": {"buffer": gamma.buffer}},
            {"binding": 2, "resource": {"buffer": buf_out}},
            {"binding": 3, "resource": {"buffer": buf_irms}},
            {"binding": 4, "resource": {"buffer": meta_buf}},
        ],
    )

    # Cap X-dispatch to 65,535 max
    dispatch_x = min(inner, 65535)
    dispatch_y = min(outer, 65535)

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    pass_enc.dispatch_workgroups(dispatch_x, dispatch_y, 1)
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)

    irms_shape = out_shape

    out = Tensor(tensor1.shape, buffer=buf_out)
    irms = Tensor(irms_shape, buffer=buf_irms)

    return out, irms


def meanvarnorm(tensor1, gamma, beta, axis=None):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    if not gamma.is_contiguous():
        gamma = gamma.contiguous()

    pipeline = gpu_ctx.get_pipeline("meanvarnorm", shaders["layers/meanvarnorm.wgsl"])

    outer, reduce_size, inner, out_shape = _get_reduce_dims(tensor1.shape, axis)

    total_outputs = outer * inner * reduce_size

    u32_list = [outer, reduce_size, inner, 1e-5]
    meta_data = struct.pack("<IIIf", *u32_list)
    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.UNIFORM, label="meanvarnorm meta"
    )

    out_bytes = total_outputs * 4
    buf_out = buffer_pool.create_buffer(
        size=out_bytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )
    buf_mean = buffer_pool.create_buffer(
        size=inner * outer * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    buf_std = buffer_pool.create_buffer(
        size=inner * outer * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor1.buffer}},
            {"binding": 1, "resource": {"buffer": gamma.buffer}},
            {"binding": 2, "resource": {"buffer": beta.buffer}},
            {"binding": 3, "resource": {"buffer": buf_out}},
            {"binding": 4, "resource": {"buffer": buf_mean}},
            {"binding": 5, "resource": {"buffer": buf_std}},
            {"binding": 6, "resource": {"buffer": meta_buf}},
        ],
    )

    # Cap X-dispatch to 65,535 max
    dispatch_x = min(inner, 65535)
    dispatch_y = min(outer, 65535)

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    pass_enc.dispatch_workgroups(dispatch_x, dispatch_y, 1)
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)

    o_shape = out_shape

    out = Tensor(tensor1.shape, buffer=buf_out)
    mean = Tensor(o_shape, buffer=buf_mean)
    std = Tensor(o_shape, buffer=buf_std)

    return out, mean, std


def batchnorm(tensor1, gamma, beta, running_mean, running_std, training=False):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    if not gamma.is_contiguous():
        gamma = gamma.contiguous()

    B, C, H, W = tensor1.shape

    pipeline = gpu_ctx.get_pipeline("batchnorm", shaders["layers/batchnorm.wgsl"])
    u32_list = [B, C, H, W, 1e-5, 0.1, training, 0]

    meta_data = struct.pack("<IIIIffII", *u32_list)
    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.UNIFORM, label="batchnorm meta"
    )

    out_bytes = math.prod(tensor1.shape) * 4

    buf_out = buffer_pool.create_buffer(
        size=out_bytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )

    buf_saved_mean = buffer_pool.create_buffer(
        size=C * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )

    buf_saved_std = buffer_pool.create_buffer(
        size=C * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor1.buffer}},
            {"binding": 1, "resource": {"buffer": gamma.buffer}},
            {"binding": 2, "resource": {"buffer": beta.buffer}},
            {"binding": 3, "resource": {"buffer": running_mean.buffer}},
            {"binding": 4, "resource": {"buffer": running_std.buffer}},
            {"binding": 5, "resource": {"buffer": buf_out}},
            {"binding": 6, "resource": {"buffer": buf_saved_mean}},
            {"binding": 7, "resource": {"buffer": buf_saved_std}},
            {"binding": 8, "resource": {"buffer": meta_buf}},
        ],
    )

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    pass_enc.dispatch_workgroups(C, 1, 1)
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)

    out = Tensor(tensor1.shape, buffer=buf_out)

    mean = Tensor(C, buffer=buf_saved_mean)
    std = Tensor(C, buffer=buf_saved_std)

    return out, mean, std


def batchnormdx(grad, tensor1, gamma, dgamma, dbeta, saved_mean, saved_std):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    if not gamma.is_contiguous():
        gamma = gamma.contiguous()

    pipeline = gpu_ctx.get_pipeline("batchnormdx", shaders["layers/batchnormdx.wgsl"])

    B, C, H, W = tensor1.shape

    u32_list = [B, C, H, W, 1e-5, 0.1, 0, 0]

    meta_data = struct.pack("<IIIIffII", *u32_list)
    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.UNIFORM, label="batchnorm dx meta"
    )

    out_bytes = math.prod(tensor1.shape) * 4

    buf_out = buffer_pool.create_buffer(
        size=out_bytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": grad.buffer}},
            {"binding": 1, "resource": {"buffer": tensor1.buffer}},
            {"binding": 2, "resource": {"buffer": gamma.buffer}},
            {"binding": 3, "resource": {"buffer": dgamma.buffer}},
            {"binding": 4, "resource": {"buffer": dbeta.buffer}},
            {"binding": 5, "resource": {"buffer": saved_mean.buffer}},
            {"binding": 6, "resource": {"buffer": saved_std.buffer}},
            {"binding": 7, "resource": {"buffer": buf_out}},
            {"binding": 8, "resource": {"buffer": meta_buf}},
        ],
    )

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    _dispatch_1d(pass_enc, B * C * H * W)
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)

    out = Tensor(tensor1.shape, buffer=buf_out)

    return out


def batchnormdg(grad, tensor1, saved_mean, saved_std):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()

    B, C, H, W = tensor1.shape

    pipeline = gpu_ctx.get_pipeline("batchnormdg", shaders["layers/batchnormdg.wgsl"])
    u32_list = [B, C, H, W, 1e-5, 0.1, 0, 0]

    meta_data = struct.pack("<IIIIffII", *u32_list)
    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.UNIFORM, label="batchnorm dg meta"
    )

    buf_dg = buffer_pool.create_buffer(
        size=C * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )
    buf_db = buffer_pool.create_buffer(
        size=C * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": grad.buffer}},
            {"binding": 1, "resource": {"buffer": tensor1.buffer}},
            {"binding": 2, "resource": {"buffer": saved_mean.buffer}},
            {"binding": 3, "resource": {"buffer": saved_std.buffer}},
            {"binding": 4, "resource": {"buffer": buf_dg}},
            {"binding": 5, "resource": {"buffer": buf_db}},
            {"binding": 6, "resource": {"buffer": meta_buf}},
        ],
    )

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    pass_enc.dispatch_workgroups(C, 1, 1)
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)

    dg = Tensor(C, buffer=buf_dg)
    db = Tensor(C, buffer=buf_db)

    return dg, db


def meanvarnormdx(grad, tensor1, gamma, mean, var, axis=None):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    if not gamma.is_contiguous():
        gamma = gamma.contiguous()

    pipeline = gpu_ctx.get_pipeline(
        "meanvarnormdx", shaders["layers/meanvarnormdx.wgsl"]
    )

    outer, reduce_size, inner, out_shape = _get_reduce_dims(tensor1.shape, axis)

    total_outputs = outer * inner * reduce_size

    u32_list = [outer, reduce_size, inner, 1e-7]
    meta_data = struct.pack("<IIIf", *u32_list)
    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.UNIFORM, label="meanvarnorm dx meta"
    )

    out_bytes = total_outputs * 4
    buf_out = buffer_pool.create_buffer(
        size=out_bytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": grad.buffer}},
            {"binding": 1, "resource": {"buffer": tensor1.buffer}},
            {"binding": 2, "resource": {"buffer": gamma.buffer}},
            {"binding": 3, "resource": {"buffer": mean.buffer}},
            {"binding": 4, "resource": {"buffer": var.buffer}},
            {"binding": 5, "resource": {"buffer": buf_out}},
            {"binding": 6, "resource": {"buffer": meta_buf}},
        ],
    )

    dispatch_x = min(inner, 65535)
    dispatch_y = min(outer, 65535)

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    pass_enc.dispatch_workgroups(dispatch_x, dispatch_y, 1)
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)

    out = Tensor(tensor1.shape, buffer=buf_out)

    return out


def meanvarnormdg(grad, tensor1, mean, var, axis=None):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()

    pipeline = gpu_ctx.get_pipeline(
        "meanvarnormdg", shaders["layers/meanvarnormdg.wgsl"]
    )

    outer, reduce_size, inner, out_shape = _get_reduce_dims(tensor1.shape, axis)

    u32_list = [outer, reduce_size, inner, 1e-7]
    meta_data = struct.pack("<IIIf", *u32_list)
    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.UNIFORM, label="meanvarnorm dg meta"
    )

    buf_dg = buffer_pool.create_buffer(
        size=reduce_size * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )
    buf_db = buffer_pool.create_buffer(
        size=reduce_size * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": grad.buffer}},
            {"binding": 1, "resource": {"buffer": tensor1.buffer}},
            {"binding": 2, "resource": {"buffer": mean.buffer}},
            {"binding": 3, "resource": {"buffer": var.buffer}},
            {"binding": 4, "resource": {"buffer": buf_dg}},
            {"binding": 5, "resource": {"buffer": buf_db}},
            {"binding": 6, "resource": {"buffer": meta_buf}},
        ],
    )

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    if reduce_size:
        pass_enc.dispatch_workgroups(
            min(reduce_size, MAX_WORKGROUPS_PER_DIMENSION), 1, 1
        )
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)

    dg = Tensor(reduce_size, buffer=buf_dg)
    db = Tensor(reduce_size, buffer=buf_db)

    return dg, db


def rmsnormdx(grad, tensor1, gamma, irms, axis=None):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    if not gamma.is_contiguous():
        gamma = gamma.contiguous()

    pipeline = gpu_ctx.get_pipeline("rmsnormdx", shaders["layers/rmsnormdx.wgsl"])

    outer, reduce_size, inner, out_shape = _get_reduce_dims(tensor1.shape, axis)

    total_outputs = outer * inner * reduce_size

    u32_list = [outer, reduce_size, inner, 1e-7]
    meta_data = struct.pack("<IIIf", *u32_list)
    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.UNIFORM, label="rmsnorm dx meta"
    )

    out_bytes = total_outputs * 4
    buf_out = buffer_pool.create_buffer(
        size=out_bytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": grad.buffer}},
            {"binding": 1, "resource": {"buffer": tensor1.buffer}},
            {"binding": 2, "resource": {"buffer": gamma.buffer}},
            {"binding": 3, "resource": {"buffer": irms.buffer}},
            {"binding": 4, "resource": {"buffer": buf_out}},
            {"binding": 5, "resource": {"buffer": meta_buf}},
        ],
    )

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    if outer * inner:
        pass_enc.dispatch_workgroups(
            min(outer * inner, MAX_WORKGROUPS_PER_DIMENSION), 1, 1
        )
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)

    out = Tensor(tensor1.shape, buffer=buf_out)

    return out


def rmsnormdg(grad, tensor1, gamma, irms, axis=None):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    if not gamma.is_contiguous():
        gamma = gamma.contiguous()

    pipeline = gpu_ctx.get_pipeline("rmsnormdg", shaders["layers/rmsnormdg.wgsl"])

    outer, reduce_size, inner, out_shape = _get_reduce_dims(tensor1.shape, axis)

    total_outputs = reduce_size

    u32_list = [outer, reduce_size, inner, 1e-7]
    meta_data = struct.pack("<IIIf", *u32_list)
    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.UNIFORM, label="rmsnorm dg meta"
    )

    out_bytes = total_outputs * 4
    buf_out = buffer_pool.create_buffer(
        size=out_bytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": grad.buffer}},
            {"binding": 1, "resource": {"buffer": tensor1.buffer}},
            {"binding": 2, "resource": {"buffer": irms.buffer}},
            {"binding": 3, "resource": {"buffer": buf_out}},
            {"binding": 4, "resource": {"buffer": meta_buf}},
        ],
    )

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    if reduce_size:
        pass_enc.dispatch_workgroups(
            min(reduce_size, MAX_WORKGROUPS_PER_DIMENSION), 1, 1
        )
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)

    dg = Tensor(gamma.shape, buffer=buf_out)

    return dg
