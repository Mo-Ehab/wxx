"""Elementwise activation functions and their backward helpers."""
import math
import struct

import wgpu

from ..backend.dispatch import _dispatch_1d, _unary_op

from .. import validation

from ..backend.device import get_buffer_pool, get_cmd_buffer, get_ctx, get_device
from ..shaders import walk_shader_files
from ..Tensor import Tensor

cmd_buf = get_cmd_buffer()
buffer_pool = get_buffer_pool()
device = get_device()
gpu_ctx = get_ctx()

shaders = walk_shader_files()

def relu(tensor1):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    pipeline = gpu_ctx.get_pipeline("relu", shaders["activations/relu.wgsl"])

    return _unary_op(tensor1, pipeline)


def sin(tensor1):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    pipeline = gpu_ctx.get_pipeline("sin", shaders["activations/sin.wgsl"])

    return _unary_op(tensor1, pipeline)


def sinback(tensor1):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    pipeline = gpu_ctx.get_pipeline("sinback", shaders["activations/sinback.wgsl"])

    return _unary_op(tensor1, pipeline)


def gelu(tensor1):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    pipeline = gpu_ctx.get_pipeline("gelu", shaders["activations/gelu.wgsl"])

    return _unary_op(tensor1, pipeline)


def sigmoid(tensor1):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    pipeline = gpu_ctx.get_pipeline("sigmoid", shaders["activations/sigmoid.wgsl"])

    return _unary_op(tensor1, pipeline)


def selu(tensor1):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    pipeline = gpu_ctx.get_pipeline("selu", shaders["activations/selu.wgsl"])

    return _unary_op(tensor1, pipeline)


def relubackwards(tensor1):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    pipeline = gpu_ctx.get_pipeline(
        "relubackwards", shaders["activations/relubackwards.wgsl"]
    )

    return _unary_op(tensor1, pipeline)


def gelubackwards(tensor1):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    pipeline = gpu_ctx.get_pipeline(
        "gelubackwards", shaders["activations/gelubackwards.wgsl"]
    )

    return _unary_op(tensor1, pipeline)


def sigmoidbackwards(tensor1):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    pipeline = gpu_ctx.get_pipeline(
        "sigmoidbackwards", shaders["activations/sigmoidbackwards.wgsl"]
    )

    return _unary_op(tensor1, pipeline)


def selubackwards(tensor1):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    pipeline = gpu_ctx.get_pipeline(
        "selubackwards", shaders["activations/selubackwards.wgsl"]
    )

    return _unary_op(tensor1, pipeline)


def softmax(tensor1, dim):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()

    pipeline = gpu_ctx.get_pipeline("softmax", shaders["activations/softmax.wgsl"])

    rank = validation.check_rank(tensor1.shape, name="softmax input")
    dim = validation.normalize_dim(dim, rank, name="dim")

    if tensor1.type != "f32":
        raise TypeError(f"softmax expects f32 input, got {tensor1.type!r}")

    numel = int(math.prod(tensor1.shape))

    padded_out_shape = list(tensor1.shape) + [0] * (8 - rank)
    padded_strides = list(tensor1.stride) + [0] * (8 - rank)

    u32_list = [numel, rank, dim, 0] + padded_out_shape + padded_strides

    meta_data = struct.pack(f"<{len(u32_list)}I", *u32_list)

    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.STORAGE, label="Softmax meta"
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
    _dispatch_1d(pass_enc, numel // tensor1.shape[dim])
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)

    return Tensor(tensor1.shape, buf_out, requires_grad=(tensor1.requires_grad))
