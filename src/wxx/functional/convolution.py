"""2-D convolution and pooling primitives."""
import struct

import wgpu

from .. import validation
from ..backend.dispatch import _dispatch_1d, _dispatch_tiled_rows

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


def maxpool2d(tensor1, kernel, stride, padding):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()

    pipeline = gpu_ctx.get_pipeline("maxpool2d", shaders["layers/maxpool2d.wgsl"])

    B, C, H_in, W_in = tensor1.shape
    KH, KW = kernel
    sh, sw = stride
    ph, pw = padding

    H_out = (H_in + 2 * ph * (KH - 1) - 1) // sh + 1
    W_out = (W_in + 2 * pw * (KW - 1) - 1) // sw + 1

    u32_list = [B, C, H_in, W_in, H_out, W_out, KH, KW, sh, sw, ph, pw]

    meta_data = struct.pack("<IIIIIIIIIIII", *u32_list)
    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.UNIFORM, label="Maxpool2d meta"
    )

    out_bytes = B * C * H_out * W_out * 4

    buf_out = buffer_pool.create_buffer(
        size=out_bytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )

    buf_indicies = buffer_pool.create_buffer(
        size=out_bytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor1.buffer}},
            {"binding": 1, "resource": {"buffer": buf_out}},
            {"binding": 2, "resource": {"buffer": buf_indicies}},
            {"binding": 3, "resource": {"buffer": meta_buf}},
        ],
    )

    total_out = B * C * H_out * W_out
    workgroup_count = (total_out + 255) // 256
    dispatch_x = min(workgroup_count, 65535)

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    pass_enc.dispatch_workgroups(dispatch_x, 1, 1)
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)

    out = Tensor((B, C, H_out, W_out), buffer=buf_out)
    indicies = Tensor((B, C, H_out, W_out), buffer=buf_indicies)

    return out, indicies


def avgpool2d(tensor1, kernel, stride, padding):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()

    pipeline = gpu_ctx.get_pipeline("avgpool2d", shaders["layers/avgpool2d.wgsl"])

    B, C, H_in, W_in = tensor1.shape
    KH, KW = kernel
    sh, sw = stride
    ph, pw = padding

    H_out = (H_in + 2 * ph * (KH - 1) - 1) // sh + 1
    W_out = (W_in + 2 * pw * (KW - 1) - 1) // sw + 1

    u32_list = [B, C, H_in, W_in, H_out, W_out, KH, KW, sh, sw, ph, pw]

    meta_data = struct.pack("<IIIIIIIIIIII", *u32_list)
    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.UNIFORM, label="avgpool2d meta"
    )

    out_bytes = B * C * H_out * W_out * 4

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

    total_out = B * C * H_out * W_out
    workgroup_count = (total_out + 255) // 256
    dispatch_x = min(workgroup_count, 65535)

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    pass_enc.dispatch_workgroups(dispatch_x, 1, 1)
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)

    out = Tensor((B, C, H_out, W_out), buffer=buf_out)

    return out


def avgpool2ddx(grad, tensor1, kernel, stride, padding):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()

    pipeline = gpu_ctx.get_pipeline("avgpool2ddx", shaders["layers/avgpool2ddx.wgsl"])

    B, C, H_in, W_in = tensor1.shape
    KH, KW = kernel
    sh, sw = stride
    ph, pw = padding

    H_out = (H_in + 2 * ph * (KH - 1) - 1) // sh + 1
    W_out = (W_in + 2 * pw * (KW - 1) - 1) // sw + 1

    u32_list = [B, C, H_in, W_in, H_out, W_out, KH, KW, sh, sw, ph, pw]

    meta_data = struct.pack("<IIIIIIIIIIII", *u32_list)
    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.UNIFORM, label="avgpool2d dx meta"
    )

    out_bytes = B * C * H_in * W_in * 4

    buf_out = buffer_pool.create_buffer(
        size=out_bytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": grad.buffer}},
            {"binding": 1, "resource": {"buffer": buf_out}},
            {"binding": 2, "resource": {"buffer": meta_buf}},
        ],
    )

    total_out = B * C * H_out * W_out
    workgroup_count = (total_out + 255) // 256
    dispatch_x = min(workgroup_count, 65535)

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    pass_enc.dispatch_workgroups(dispatch_x, 1, 1)
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)

    out = Tensor((B, C, H_in, W_in), buffer=buf_out)

    return out


def _normalize_output_size(output_size):
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    if len(output_size) != 2:
        raise ValueError(f"output_size must be an int or 2-tuple, got {output_size!r}")
    h_out, w_out = (int(output_size[0]), int(output_size[1]))
    if h_out < 1 or w_out < 1:
        raise ValueError(f"output_size entries must be positive, got {output_size!r}")
    return h_out, w_out


def adaptive_avgpool2d(tensor1, output_size):
    """Adaptive average pooling over the last two dimensions.

    Each output cell averages the input window
    ``[floor(o * in / out), ceil((o + 1) * in / out))`` independently per
    spatial axis, matching PyTorch's ``AdaptiveAvgPool2d``.
    """
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    if tensor1.type != "f32":
        raise TypeError(f"adaptive_avgpool2d expects f32 input, got {tensor1.type!r}")
    if validation.check_rank(tensor1.shape, name="adaptive_avgpool2d input") != 4:
        raise ValueError(
            f"adaptive_avgpool2d expects (B, C, H, W), got {tensor1.shape}"
        )

    h_out, w_out = _normalize_output_size(output_size)
    batch, channels, h_in, w_in = tensor1.shape

    pipeline = gpu_ctx.get_pipeline(
        "adaptive_avgpool2d", shaders["layers/adaptive_avgpool2d.wgsl"]
    )
    meta_buf = buffer_pool.create_buffer_with_data(
        data=struct.pack("<8I", batch, channels, h_in, w_in, h_out, w_out, 0, 0),
        usage=wgpu.BufferUsage.UNIFORM,
        label="adaptive_avgpool2d meta",
    )

    buf_out = buffer_pool.create_buffer(
        size=batch * channels * h_out * w_out * 4,
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
    _dispatch_1d(pass_enc, batch * channels * h_out * w_out)
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)
    return Tensor(
        (batch, channels, h_out, w_out),
        buffer=buf_out,
        requires_grad=tensor1.requires_grad,
    )


def adaptive_avgpool2ddx(grad, input_shape, output_size):
    """Backward of :func:`adaptive_avgpool2d`."""
    if not grad.is_contiguous():
        grad = grad.contiguous()
    if grad.type != "f32":
        raise TypeError(
            f"adaptive_avgpool2ddx expects f32 grad, got {grad.type!r}"
        )
    if validation.check_rank(input_shape, name="adaptive_avgpool2ddx input") != 4:
        raise ValueError(
            f"adaptive_avgpool2ddx expects a 4-D input shape, got {input_shape}"
        )

    h_out, w_out = _normalize_output_size(output_size)
    batch, channels, h_in, w_in = input_shape
    expected = (batch, channels, h_out, w_out)
    if tuple(grad.shape) != expected:
        raise ValueError(
            f"adaptive_avgpool2ddx grad shape {tuple(grad.shape)} does not match "
            f"expected {expected}"
        )

    pipeline = gpu_ctx.get_pipeline(
        "adaptive_avgpool2ddx", shaders["layers/adaptive_avgpool2ddx.wgsl"]
    )
    meta_buf = buffer_pool.create_buffer_with_data(
        data=struct.pack("<8I", batch, channels, h_in, w_in, h_out, w_out, 0, 0),
        usage=wgpu.BufferUsage.UNIFORM,
        label="adaptive_avgpool2ddx meta",
    )

    buf_out = buffer_pool.create_buffer(
        size=batch * channels * h_in * w_in * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": grad.buffer}},
            {"binding": 1, "resource": {"buffer": buf_out}},
            {"binding": 2, "resource": {"buffer": meta_buf}},
        ],
    )

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    _dispatch_1d(pass_enc, batch * channels * h_in * w_in)
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)
    return Tensor(input_shape, buffer=buf_out)


def maxpool2ddx(grad, tensor1, indicies, kernel, stride, padding):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()

    pipeline = gpu_ctx.get_pipeline("maxpool2ddx", shaders["layers/maxpool2ddx.wgsl"])

    B, C, H_in, W_in = tensor1.shape
    KH, KW = kernel
    sh, sw = stride
    ph, pw = padding

    H_out = (H_in + 2 * ph * (KH - 1) - 1) // sh + 1
    W_out = (W_in + 2 * pw * (KW - 1) - 1) // sw + 1

    u32_list = [B, C, H_in, W_in, H_out, W_out, KH, KW, sh, sw, ph, pw]

    meta_data = struct.pack("<IIIIIIIIIIII", *u32_list)
    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.UNIFORM, label= "maxpool2d dx meta"
    )

    out_bytes = B * C * H_in * W_in * 4

    buf_out = buffer_pool.create_buffer(
        size=out_bytes, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": grad.buffer}},
            {"binding": 1, "resource": {"buffer": indicies.buffer}},
            {"binding": 2, "resource": {"buffer": buf_out}},
            {"binding": 3, "resource": {"buffer": meta_buf}},
        ],
    )

    total_out = B * C * H_out * W_out
    workgroup_count = (total_out + 255) // 256
    dispatch_x = min(workgroup_count, 65535)

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    pass_enc.dispatch_workgroups(dispatch_x, 1, 1)
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)

    out = Tensor((B, C, H_in, W_in), buffer=buf_out)

    return out




def conv2d(tensor1, W, stride, padding, dilation, bias=None):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    if not W.is_contiguous():
        W = W.contiguous()
    if bias is not None:
        if not bias.is_contiguous():
            bias = bias.contiguous()
        hb = 1
        buf_bias = bias.buffer
    else:
        buf_bias = tensor1.buffer
        hb = 0

    pipeline = gpu_ctx.get_pipeline("conv2d", shaders["layers/conv2d.wgsl"])

    B, C_in, H_in, W_in = tensor1.shape
    C_out, _, KH, KW = W.shape
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    H_out = (H_in + 2 * ph - dh * (KH - 1) - 1) // sh + 1
    W_out = (W_in + 2 * pw - dw * (KW - 1) - 1) // sw + 1

    u32_list = (
        [B, C_in, H_in, W_in]
        + [C_out, H_out, W_out]
        + [KH, KW]
        + [sh, sw]
        + [ph, pw]
        + [dh, dw]
        + [hb]
    )

    meta_data = struct.pack(f"<{len(u32_list)}I", *u32_list)

    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.STORAGE, label="conv2d meta"
    )

    buf_out = buffer_pool.create_buffer(
        size=C_out * H_out * W_out * B * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor1.buffer}},
            {"binding": 1, "resource": {"buffer": W.buffer}},
            {"binding": 2, "resource": {"buffer": buf_bias}},
            {"binding": 3, "resource": {"buffer": buf_out}},
            {"binding": 4, "resource": {"buffer": meta_buf}},
        ],
    )

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    output_rows = B * H_out * W_out
    _dispatch_tiled_rows(pass_enc, output_rows, C_out)
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)

    out = Tensor(
        buffer=buf_out,
        shape=(B, C_out, H_out, W_out),
        requires_grad=tensor1.requires_grad,
    )

    return out


def conv2ddx(tensor1, W, stride, padding, dilation, Xshape, wshape):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    if not W.is_contiguous():
        W = W.contiguous()

    pipeline = gpu_ctx.get_pipeline("conv2ddx", shaders["layers/conv2ddx.wgsl"])

    B, C_in, H_in, W_in = Xshape
    C_out, _, KH, KW = wshape
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    H_out = (H_in + 2 * ph - dh * (KH - 1) - 1) // sh + 1
    W_out = (W_in + 2 * pw - dw * (KW - 1) - 1) // sw + 1
    u32_list = (
        [B, C_in, H_in, W_in]
        + [C_out, H_out, W_out]
        + [KH, KW]
        + [sh, sw]
        + [ph, pw]
        + [dh, dw]
        + [0]
    )

    meta_data = struct.pack(f"<{len(u32_list)}I", *u32_list)

    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.STORAGE, label="conv2d dx meta"
    )

    buf_out = buffer_pool.create_buffer(
        size=B * C_in * H_in * W_in * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor1.buffer}},
            {"binding": 1, "resource": {"buffer": W.buffer}},
            {"binding": 2, "resource": {"buffer": buf_out}},
            {"binding": 3, "resource": {"buffer": meta_buf}},
        ],
    )

    output_rows = B * H_in * W_in

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)

    _dispatch_tiled_rows(pass_enc, output_rows, C_in)

    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)

    out = Tensor(
        buffer=buf_out, shape=(B, C_in, H_in, W_in), requires_grad=tensor1.requires_grad
    )
    return out


def conv2ddw(tensor1, X, stride, padding, dilation, Xshape, wshape):
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()

    pipeline = gpu_ctx.get_pipeline("conv2ddw", shaders["layers/conv2ddw.wgsl"])

    B, C_in, H_in, W_in = Xshape
    C_out, _, KH, KW = wshape
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    H_out = (H_in + 2 * ph - dh * (KH - 1) - 1) // sh + 1
    W_out = (W_in + 2 * pw - dw * (KW - 1) - 1) // sw + 1
    filter_size = C_in * KH * KW
    total_weights = C_out * filter_size
    total_spatial = B * H_out * W_out

    filter_workgroups = (filter_size + 15) // 16
    filter_workgroups_z = (
        filter_workgroups + MAX_WORKGROUPS_PER_DIMENSION - 1
    ) // MAX_WORKGROUPS_PER_DIMENSION
    max_split_count = max(1, MAX_WORKGROUPS_PER_DIMENSION // filter_workgroups_z)
    split_size = 2048
    split_count = (total_spatial + split_size - 1) // split_size
    if split_count > max_split_count:
        split_count = max_split_count
        split_size = (total_spatial + split_count - 1) // split_count

    partial_size = split_count * total_weights * 4
    if (
        split_count > 1
        and partial_size > device.limits["max-storage-buffer-binding-size"]
    ):
        split_count = 1
        split_size = total_spatial

    u32_list = (
        [B, C_in, H_in, W_in]
        + [C_out, H_out, W_out]
        + [KH, KW]
        + [sh, sw]
        + [ph, pw]
        + [dh, dw]
        + [0, split_count, split_size]
    )

    meta_data = struct.pack(f"<{len(u32_list)}I", *u32_list)
    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.STORAGE, label="conv2d dw meta"
    )

    buf_out = buffer_pool.create_buffer(
        size=total_weights * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    buf_work = buf_out
    if split_count > 1:
        buf_work = buffer_pool.create_buffer(
            size=partial_size,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor1.buffer}},
            {"binding": 1, "resource": {"buffer": X.buffer}},
            {"binding": 2, "resource": {"buffer": buf_work}},
            {"binding": 3, "resource": {"buffer": meta_buf}},
        ],
    )

    dispatch_x = min(filter_workgroups, MAX_WORKGROUPS_PER_DIMENSION)
    dispatch_y = (C_out + 15) // 16
    dispatch_z = split_count * filter_workgroups_z

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    pass_enc.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z)

    if split_count > 1:
        reduce_pipeline = gpu_ctx.get_pipeline(
            "conv2ddw_reduce", shaders["layers/conv2ddw_reduce.wgsl"]
        )
        reduce_meta = buffer_pool.create_buffer_with_data(
            data=struct.pack("<2I", total_weights, split_count),
            usage=wgpu.BufferUsage.STORAGE,
            label="conv2d dw meta"
        )
        reduce_bind_group = device.create_bind_group(
            layout=reduce_pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": buf_work}},
                {"binding": 1, "resource": {"buffer": buf_out}},
                {"binding": 2, "resource": {"buffer": reduce_meta}},
            ],
        )
        pass_enc.set_pipeline(reduce_pipeline)
        pass_enc.set_bind_group(0, reduce_bind_group)
        _dispatch_1d(pass_enc, total_weights)
        cmd_buf.keep_alive(reduce_meta)
        cmd_buf.defer_destroy(buf_work)

    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)

    out = Tensor(
        buffer=buf_out, shape=(C_out, C_in, KH, KW), requires_grad=tensor1.requires_grad
    )
    return out


def conv2dtranspose(tensor1, W, stride, padding, dilation, output_padding, bias=None):
    """Transposed (fractionally strided) 2-D convolution.

    ``W`` has shape ``(C_in, C_out, KH, KW)`` and the output spatial size is
    ``(H_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1``.
    """
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    if not W.is_contiguous():
        W = W.contiguous()
    if bias is not None:
        if not bias.is_contiguous():
            bias = bias.contiguous()
        hb = 1
        buf_bias = bias.buffer
    else:
        buf_bias = tensor1.buffer
        hb = 0

    pipeline = gpu_ctx.get_pipeline(
        "conv2dtranspose", shaders["layers/conv2dtranspose.wgsl"]
    )

    B, C_in, H_in, W_in = tensor1.shape
    weight_in_c, C_out, KH, KW = W.shape
    if weight_in_c != C_in:
        raise ValueError(
            f"conv2dtranspose weight expects {C_in} input channels, got {weight_in_c}"
        )
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    oph, opw = output_padding
    if not (0 <= oph < max(sh, dh)) or not (0 <= opw < max(sw, dw)):
        raise ValueError(
            f"output_padding {(oph, opw)} must be >= 0 and smaller than "
            f"max(stride, dilation) {(max(sh, dh), max(sw, dw))}"
        )
    H_out = (H_in - 1) * sh - 2 * ph + dh * (KH - 1) + oph + 1
    W_out = (W_in - 1) * sw - 2 * pw + dw * (KW - 1) + opw + 1
    if H_out < 1 or W_out < 1:
        raise ValueError(
            f"conv2dtranspose produces an empty output ({H_out}, {W_out})"
        )

    u32_list = (
        [B, C_in, H_in, W_in]
        + [C_out, H_out, W_out]
        + [KH, KW]
        + [sh, sw]
        + [ph, pw]
        + [dh, dw]
        + [hb]
    )

    meta_data = struct.pack(f"<{len(u32_list)}I", *u32_list)

    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.STORAGE, label="conv2dtranspose meta"
    )

    buf_out = buffer_pool.create_buffer(
        size=B * C_out * H_out * W_out * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor1.buffer}},
            {"binding": 1, "resource": {"buffer": W.buffer}},
            {"binding": 2, "resource": {"buffer": buf_bias}},
            {"binding": 3, "resource": {"buffer": buf_out}},
            {"binding": 4, "resource": {"buffer": meta_buf}},
        ],
    )

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    _dispatch_tiled_rows(pass_enc, B * H_out * W_out, C_out)
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)

    return Tensor(
        buffer=buf_out,
        shape=(B, C_out, H_out, W_out),
        requires_grad=tensor1.requires_grad,
    )


def conv2dtranspose_dx(tensor1, W, stride, padding, dilation, Xshape, wshape, out_shape):
    """Gradient of :func:`conv2dtranspose` with respect to its input."""
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    if not W.is_contiguous():
        W = W.contiguous()

    pipeline = gpu_ctx.get_pipeline(
        "conv2dtranspose_dx", shaders["layers/conv2dtranspose_dx.wgsl"]
    )

    B, C_in, H_in, W_in = Xshape
    _, C_out, KH, KW = wshape
    _, _, H_out, W_out = out_shape
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    u32_list = (
        [B, C_in, H_in, W_in]
        + [C_out, H_out, W_out]
        + [KH, KW]
        + [sh, sw]
        + [ph, pw]
        + [dh, dw]
        + [0]
    )

    meta_data = struct.pack(f"<{len(u32_list)}I", *u32_list)

    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.STORAGE, label="conv2dtranspose dx meta"
    )

    buf_out = buffer_pool.create_buffer(
        size=B * C_in * H_in * W_in * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor1.buffer}},
            {"binding": 1, "resource": {"buffer": W.buffer}},
            {"binding": 2, "resource": {"buffer": buf_out}},
            {"binding": 3, "resource": {"buffer": meta_buf}},
        ],
    )

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    _dispatch_tiled_rows(pass_enc, B * H_in * W_in, C_in)
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)

    return Tensor(
        buffer=buf_out,
        shape=(B, C_in, H_in, W_in),
        requires_grad=tensor1.requires_grad,
    )


def conv2dtranspose_dw(tensor1, X, stride, padding, dilation, out_shape, wshape):
    """Gradient of :func:`conv2dtranspose` with respect to its weight."""
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    if not X.is_contiguous():
        X = X.contiguous()

    pipeline = gpu_ctx.get_pipeline(
        "conv2dtranspose_dw", shaders["layers/conv2dtranspose_dw.wgsl"]
    )

    B, C_in, H_in, W_in = X.shape
    _, C_out, KH, KW = wshape
    _, _, H_out, W_out = out_shape
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    filter_size = C_in * KH * KW
    total_weights = C_out * filter_size
    total_spatial = B * H_out * W_out

    filter_workgroups = (filter_size + 15) // 16
    filter_workgroups_z = (
        filter_workgroups + MAX_WORKGROUPS_PER_DIMENSION - 1
    ) // MAX_WORKGROUPS_PER_DIMENSION
    max_split_count = max(1, MAX_WORKGROUPS_PER_DIMENSION // filter_workgroups_z)
    split_size = 2048
    split_count = (total_spatial + split_size - 1) // split_size
    if split_count > max_split_count:
        split_count = max_split_count
        split_size = (total_spatial + split_count - 1) // split_count

    partial_size = split_count * total_weights * 4
    if (
        split_count > 1
        and partial_size > device.limits["max-storage-buffer-binding-size"]
    ):
        split_count = 1
        split_size = total_spatial

    u32_list = (
        [B, C_in, H_in, W_in]
        + [C_out, H_out, W_out]
        + [KH, KW]
        + [sh, sw]
        + [ph, pw]
        + [dh, dw]
        + [0, split_count, split_size]
    )

    meta_data = struct.pack(f"<{len(u32_list)}I", *u32_list)
    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.STORAGE, label="conv2dtranspose dw meta"
    )

    buf_out = buffer_pool.create_buffer(
        size=total_weights * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    buf_work = buf_out
    if split_count > 1:
        buf_work = buffer_pool.create_buffer(
            size=partial_size,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor1.buffer}},
            {"binding": 1, "resource": {"buffer": X.buffer}},
            {"binding": 2, "resource": {"buffer": buf_work}},
            {"binding": 3, "resource": {"buffer": meta_buf}},
        ],
    )

    dispatch_x = min(filter_workgroups, MAX_WORKGROUPS_PER_DIMENSION)
    dispatch_y = (C_out + 15) // 16
    dispatch_z = split_count * filter_workgroups_z

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    pass_enc.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z)

    if split_count > 1:
        reduce_pipeline = gpu_ctx.get_pipeline(
            "conv2ddw_reduce", shaders["layers/conv2ddw_reduce.wgsl"]
        )
        reduce_meta = buffer_pool.create_buffer_with_data(
            data=struct.pack("<2I", total_weights, split_count),
            usage=wgpu.BufferUsage.STORAGE,
            label="conv2dtranspose dw reduce meta",
        )
        reduce_bind_group = device.create_bind_group(
            layout=reduce_pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": buf_work}},
                {"binding": 1, "resource": {"buffer": buf_out}},
                {"binding": 2, "resource": {"buffer": reduce_meta}},
            ],
        )
        pass_enc.set_pipeline(reduce_pipeline)
        pass_enc.set_bind_group(0, reduce_bind_group)
        _dispatch_1d(pass_enc, total_weights)
        cmd_buf.keep_alive(reduce_meta)
        cmd_buf.defer_destroy(buf_work)

    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)

    return Tensor(
        buffer=buf_out,
        shape=(C_in, C_out, KH, KW),
        requires_grad=tensor1.requires_grad,
    )


def causal_conv1d(tensor1, weight, bias):
    """Depthwise causal 1-D convolution over ``(B, D, L)`` inputs.

    Computes ``out[b, d, n] = bias[d] + sum_k weight[d, k] * x[b, d, n - k]``
    in a single dispatch.  ``tensor1`` may be a strided view (for example the
    ``(B, L, D) -> (B, D, L)`` transpose), so no ``contiguous()`` copy is
    needed.  ``weight`` has shape ``(D, K)`` and ``bias`` shape ``(D,)``.
    """
    if len(tensor1.shape) != 3:
        raise ValueError(f"causal_conv1d expects a 3-D input, got {tensor1.shape}")
    if len(weight.shape) != 2:
        raise ValueError(f"causal_conv1d weight must be (D, K), got {weight.shape}")
    if any(s < 0 for s in tensor1.stride):
        tensor1 = tensor1.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if not bias.is_contiguous():
        bias = bias.contiguous()

    batch, channels, length = tensor1.shape
    kernel = weight.shape[1]

    pipeline = gpu_ctx.get_pipeline(
        "causal_conv1d", shaders["helpers/causal_conv1d.wgsl"]
    )

    meta_data = struct.pack(
        "<8I",
        batch,
        channels,
        length,
        kernel,
        tensor1.offset,
        tensor1.stride[0],
        tensor1.stride[1],
        tensor1.stride[2],
    )
    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.UNIFORM, label="causal_conv1d meta"
    )

    out_buf = buffer_pool.create_buffer(
        size=batch * channels * length * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor1.buffer}},
            {"binding": 1, "resource": {"buffer": weight.buffer}},
            {"binding": 2, "resource": {"buffer": bias.buffer}},
            {"binding": 3, "resource": {"buffer": out_buf}},
            {"binding": 4, "resource": {"buffer": meta_buf}},
        ],
    )

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    _dispatch_1d(pass_enc, batch * channels * length)
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)

    return Tensor(
        (batch, channels, length),
        out_buf,
        requires_grad=(tensor1.requires_grad or weight.requires_grad),
    )


def causal_conv1d_backward(grad_output, tensor1, weight, bias, need_weight_grad=True):
    """Backward pass of :func:`causal_conv1d` in a single dispatch.

    Returns ``(dX, dWeight, dBias)``.  When ``need_weight_grad`` is false the
    kernel only computes ``dX`` and the weight/bias outputs are empty.
    """
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()
    if any(s < 0 for s in tensor1.stride):
        tensor1 = tensor1.contiguous()

    batch, channels, length = tensor1.shape
    kernel = weight.shape[1]

    pipeline = gpu_ctx.get_pipeline(
        "causal_conv1d_backward",
        shaders["helpers/causal_conv1d_backward.wgsl"],
    )

    total_dx = batch * channels * length
    total_dw = channels * kernel

    meta_data = struct.pack(
        "<8I",
        batch,
        channels,
        length,
        kernel,
        total_dx,
        total_dw,
        1 if need_weight_grad else 0,
        0,
    )
    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data,
        usage=wgpu.BufferUsage.UNIFORM,
        label="causal_conv1d bwd meta",
    )
    view_data = struct.pack(
        "<4I",
        tensor1.offset,
        tensor1.stride[0],
        tensor1.stride[1],
        tensor1.stride[2],
    )
    view_buf = buffer_pool.create_buffer_with_data(
        data=view_data,
        usage=wgpu.BufferUsage.UNIFORM,
        label="causal_conv1d bwd view",
    )

    dx_buf = buffer_pool.create_buffer(
        size=total_dx * 4,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    dw_buf = buffer_pool.create_buffer(
        size=max(4, total_dw * 4),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )
    db_buf = buffer_pool.create_buffer(
        size=max(4, channels * 4),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": grad_output.buffer}},
            {"binding": 1, "resource": {"buffer": tensor1.buffer}},
            {"binding": 2, "resource": {"buffer": weight.buffer}},
            {"binding": 3, "resource": {"buffer": dx_buf}},
            {"binding": 4, "resource": {"buffer": dw_buf}},
            {"binding": 5, "resource": {"buffer": db_buf}},
            {"binding": 6, "resource": {"buffer": meta_buf}},
            {"binding": 7, "resource": {"buffer": view_buf}},
        ],
    )

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    total = total_dx + (total_dw + channels if need_weight_grad else 0)
    _dispatch_1d(pass_enc, total)
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)
    cmd_buf.keep_alive(view_buf)

    dX = Tensor((batch, channels, length), dx_buf)
    dW = Tensor((channels, kernel), dw_buf) if need_weight_grad else None
    dB = Tensor((channels,), db_buf) if need_weight_grad else None
    return dX, dW, dB
