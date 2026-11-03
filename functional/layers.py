import math
import random
import struct

import wgpu

from ..backend.dispatch import _dispatch_1d

from ..backend.device import get_buffer_pool, get_cmd_buffer, get_ctx, get_device
from ..shaders import walk_shader_files
from ..Tensor import Tensor

cmd_buf = get_cmd_buffer()
buffer_pool = get_buffer_pool()
device = get_device()
gpu_ctx = get_ctx()

shaders = walk_shader_files()


def adamw(
    tensor1,
    grad,
    exp_avg,
    exp_avg_sq,
    step,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    bc1,
    bc2,
):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()

    pipeline = gpu_ctx.get_pipeline("adamw", shaders["optimizers/adamw.wgsl"])

    numel = int(math.prod(tensor1.shape))

    # Pack struct (8 fields = 32 bytes, matching 16-byte WGSL alignment rules)
    meta_bytes = struct.pack(
        "<Ifffffff", numel, lr, beta1, beta2, eps, weight_decay, bc1, bc2
    )
    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_bytes, usage=wgpu.BufferUsage.STORAGE, label= "adamW"
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor1.buffer}},
            {"binding": 1, "resource": {"buffer": grad.buffer}},
            {"binding": 2, "resource": {"buffer": exp_avg.buffer}},
            {"binding": 3, "resource": {"buffer": exp_avg_sq.buffer}},
            {"binding": 4, "resource": {"buffer": meta_buf}},
        ],
    )

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    _dispatch_1d(pass_enc, numel)
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)





def dropout(tensor1, p):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()

    pipeline = gpu_ctx.get_pipeline("dropout", shaders["layers/dropout.wgsl"])

    numel = math.prod(tensor1.shape)
    scale = 1 / (1 - p)
    r = random.randint(0, 2**32 - 1)
    meta = [numel, r, p, scale]

    meta_data = struct.pack("<IIff", *meta)

    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.UNIFORM, label="dropout meta"
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

    out = Tensor(tensor1.shape, buffer=buf_out)

    return out, r


def dropoutdx(tensor1, p, seed):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()

    pipeline = gpu_ctx.get_pipeline("dropoutdx", shaders["layers/dropoutdx.wgsl"])

    numel = math.prod(tensor1.shape)
    scale = 1 / (1 - p)
    meta = [numel, seed, p, scale]

    meta_data = struct.pack("<IIff", *meta)

    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.UNIFORM, label="dropout dx meta"
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

    grad = Tensor(tensor1.shape, buffer=buf_out)

    return grad



def embed(tensor1, tensor2, n_token, embed_dim):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    if not tensor2.is_contiguous():
        tensor2 = tensor2.contiguous()


    pipeline = gpu_ctx.get_pipeline("embed", shaders["layers/embed.wgsl"])


    meta = [n_token, embed_dim, 0, 0]

    meta_data = struct.pack("<IIII", *meta)

    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.UNIFORM, label="dropout meta"
    )
    buf_out = buffer_pool.create_buffer(
        size=n_token * embed_dim * 4,
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

    # One thread per output element (num_tokens * embed_dim), not per token.
    _dispatch_1d(pass_enc, n_token * embed_dim)
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)


    out = Tensor((tensor1.shape[0], tensor1.shape[1], embed_dim), buffer=buf_out)

    return out


def embeddx(tensor1, tensor2, n_token, embed_dim, vocab_size):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()

    pipeline = gpu_ctx.get_pipeline("embeddx", shaders["layers/embeddx.wgsl"])



    meta = [n_token, embed_dim, vocab_size, 0]

    meta_data = struct.pack("<IIII", *meta)

    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.UNIFORM, label="dropout dx meta"
    )
    buf_out = buffer_pool.create_buffer(
        size=vocab_size*embed_dim *4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    grad = Tensor((vocab_size, embed_dim), buffer=buf_out)
    from .utils import assign

    assign(grad, 0.0)

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor2.buffer}},
            {"binding": 1, "resource": {"buffer": tensor1.buffer}},
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


    return grad
