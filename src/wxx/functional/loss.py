"""Cross-entropy loss kernel."""

import struct

import wgpu

from .. import validation
from ..backend.device import get_buffer_pool, get_cmd_buffer, get_ctx, get_device
from ..shaders import walk_shader_files
from ..Tensor import Tensor

cmd_buf = get_cmd_buffer()
buffer_pool = get_buffer_pool()
device = get_device()
gpu_ctx = get_ctx()

shaders = walk_shader_files()


def cceloss(tensor1, tensor2):
    """Softmax cross-entropy over ``(B, K)`` logits and ``(B,)`` targets.

    Returns ``(loss, grad)`` where ``loss`` has shape ``(B, 1)`` and ``grad``
    has the logits' shape.  Targets must be ``u32`` class indices.
    """
    validation.check_rank(tensor1.shape, min_rank=2, max_rank=2, name="logits")
    validation.check_rank(tensor2.shape, min_rank=1, max_rank=1, name="targets")
    if tensor1.type != "f32":
        raise TypeError(f"cceloss expects f32 logits, got {tensor1.type!r}")
    if tensor2.type != "u32":
        raise TypeError(f"cceloss expects u32 targets, got {tensor2.type!r}")
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    if not tensor2.is_contiguous():
        tensor2 = tensor2.contiguous()

    B, K = tensor1.shape
    if tensor2.shape[0] != B:
        raise ValueError(
            f"cceloss expects one target per row, got {tensor2.shape[0]} "
            f"targets for {B} rows"
        )
    if B < 1 or K < 1:
        raise ValueError(f"cceloss expects non-empty logits, got {(B, K)}")

    # The kernel indexes Z[row][target] directly: a target outside [0, K) is an
    # out-of-bounds GPU read (undefined values, not a clean zero). Check on the
    # host so it fails loudly instead of silently returning garbage. This costs
    # one small readback of the B ids, so the check is unconditional: it is the
    # only way the mistake is reported at all, which matters because it is fast
    # to reproduce (e.g. passing hidden states of width D_MODEL as logits while
    # the targets are token ids in the full vocabulary range). Targets are
    # indices, so nothing about them is secret.
    target_array = tensor2.numpy().reshape(-1)
    target_max = int(target_array.max())
    target_min = int(target_array.min())
    if target_max >= K:
        raise ValueError(
            f"cceloss targets must be in [0, {K}), got max target {target_max}"
        )
    if target_min < 0:  # unreachable for a u32 buffer; kept for clarity
        raise ValueError(f"cceloss targets must be non-negative, got {target_min}")

    pipeline = gpu_ctx.get_pipeline("cceloss", shaders["loss/cceloss.wgsl"])

    loss_buf = buffer_pool.create_buffer(
        size=B * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )
    grad_buf = buffer_pool.create_buffer(
        size=B * K * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
    )

    meta_data = struct.pack("<2I", B, K)
    meta_buf = buffer_pool.create_buffer_with_data(
        data=meta_data, usage=wgpu.BufferUsage.UNIFORM, label="cceloss meta"
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor1.buffer}},
            {"binding": 1, "resource": {"buffer": tensor2.buffer}},
            {"binding": 2, "resource": {"buffer": loss_buf}},
            {"binding": 3, "resource": {"buffer": grad_buf}},
            {"binding": 4, "resource": {"buffer": meta_buf}},
        ],
    )

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)

    pass_enc.dispatch_workgroups(B, 1, 1)
    pass_enc.end()

    cmd_buf.keep_alive(meta_buf)

    loss_tensor = Tensor((B, 1), buffer=loss_buf, requires_grad=tensor1.requires_grad)
    grad_tensor = Tensor(tensor1.shape, buffer=grad_buf)

    return loss_tensor, grad_tensor
