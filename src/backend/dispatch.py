
import wgpu

from .. import functional, validation
from ..backend.device import get_buffer_pool, get_cmd_buffer, get_ctx, get_device
from ..Tensor import Tensor

cmd_buf = get_cmd_buffer()
buffer_pool = get_buffer_pool()
device = get_device()
gpu_ctx = get_ctx()

WORKGROUP_SIZE = 256
MAX_WORKGROUPS_PER_DIMENSION = 65535


def _sum_to_shape(gradient, target_shape):
    """Reduce ``gradient`` to ``target_shape`` (the adjoint of broadcasting)."""
    target_shape = validation.normalize_shape(target_shape, name="target_shape")
    if len(gradient.shape) < len(target_shape):
        raise ValueError(
            f"cannot reduce gradient {tuple(gradient.shape)} to "
            f"{target_shape}: target has higher rank"
        )
    validation.check_broadcastable(target_shape, gradient.shape, name="target_shape")

    while len(gradient.shape) > len(target_shape):
        gradient = functional.sum(gradient, axis=0)

    for axis, size in enumerate(target_shape):
        if size == 1 and gradient.shape[axis] != 1:
            gradient = functional.sum(gradient, axis=axis, keepdims=True)

    if tuple(gradient.shape) != target_shape:
        gradient = functional.reshape(gradient, target_shape)

    return gradient

def _dispatch_1d(compute_pass, element_count):
    workgroup_count = (element_count + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE
    if workgroup_count:
        compute_pass.dispatch_workgroups(
            min(workgroup_count, MAX_WORKGROUPS_PER_DIMENSION)
        )


def _dispatch_tiled_rows(compute_pass, row_count, column_count):
    row_workgroups = (row_count + 15) // 16
    column_workgroups = (column_count + 15) // 16
    if row_workgroups and column_workgroups:
        compute_pass.dispatch_workgroups(
            min(row_workgroups, MAX_WORKGROUPS_PER_DIMENSION),
            column_workgroups,
            (row_workgroups + MAX_WORKGROUPS_PER_DIMENSION - 1)
            // MAX_WORKGROUPS_PER_DIMENSION,
        )


def _unary_op(tensor1, pipeline):
    """Run a one-input, one-output kernel.

    Non-contiguous inputs are materialised here so individual ops cannot forget
    to do it.  The callers may still pass a contiguous view (the common case is
    a no-op check).
    """
    if not tensor1.is_contiguous():
        tensor1 = tensor1.contiguous()
    validation.check_rank(tensor1.shape, name="unary op input")

    buf_out = buffer_pool.create_buffer(
        size=tensor1.nbytes,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": tensor1.buffer}},
            {"binding": 1, "resource": {"buffer": buf_out}},
        ],
    )

    pass_enc = cmd_buf.begin_compute_pass()
    pass_enc.set_pipeline(pipeline)
    pass_enc.set_bind_group(0, bind_group)
    _dispatch_1d(pass_enc, tensor1.nbytes // 4)
    pass_enc.end()

    return Tensor(tensor1.shape, buf_out, requires_grad=(tensor1.requires_grad))


def broadcast_shapes(shape_a, shape_b):
    """Broadcast two shapes using NumPy/PyTorch rules."""
    shape_a = validation.normalize_shape(shape_a, name="shape_a")
    shape_b = validation.normalize_shape(shape_b, name="shape_b")
    rank_a, rank_b = len(shape_a), len(shape_b)
    max_rank = max(rank_a, rank_b)
    validation.check_rank(shape_a, max_rank=validation.MAX_RANK, name="shape_a")
    validation.check_rank(shape_b, max_rank=validation.MAX_RANK, name="shape_b")

    pad_a = (1,) * (max_rank - rank_a) + shape_a
    pad_b = (1,) * (max_rank - rank_b) + shape_b

    out_shape = []
    for dim_a, dim_b in zip(pad_a, pad_b):
        if dim_a == dim_b:
            out_shape.append(dim_a)
        elif dim_a == 1:
            out_shape.append(dim_b)
        elif dim_b == 1:
            out_shape.append(dim_a)
        else:
            raise ValueError(
                f"cannot broadcast shapes {tuple(shape_a)} and {tuple(shape_b)}"
            )

    return tuple(out_shape), pad_a, pad_b


def compute_broadcast_strides(padded_shape, out_shape):
    if len(padded_shape) != len(out_shape):
        raise ValueError(
            f"padded shape {tuple(padded_shape)} and output shape "
            f"{tuple(out_shape)} must have the same rank"
        )
    rank = len(out_shape)
    strides = [0] * rank
    acc_stride = 1

    for i in range(rank - 1, -1, -1):
        if padded_shape[i] == 1:
            strides[i] = 0
        else:
            strides[i] = acc_stride

        acc_stride *= padded_shape[i]

    return tuple(strides)


def view_broadcast_strides(tensor, out_shape):
    """Strides that map ``out_shape`` onto ``tensor``'s (possibly strided) view.

    Unlike :func:`compute_broadcast_strides`, this reads the tensor's own
    ``stride`` so that non-contiguous views can be consumed directly instead of
    being materialised with ``contiguous()`` first.  Dimensions of size one
    broadcast (stride zero), and missing leading dimensions are treated as
    broadcast.
    """
    validation.check_broadcastable(
        tensor.shape, out_shape, name="pointwise operand"
    )
    rank = len(out_shape)
    pad = rank - len(tensor.shape)
    strides = [0] * rank
    for i in range(rank):
        src = i - pad
        if src < 0 or tensor.shape[src] == 1:
            strides[i] = 0
        else:
            strides[i] = tensor.stride[src]
    return tuple(strides)


def supports_strided(tensor):
    """True when a kernel can address ``tensor`` through its stride/offset."""
    return (
        len(tensor.shape) <= validation.MAX_RANK
        and all(step >= 0 for step in tensor.stride)
    )


def _get_reduce_dims(shape, axis=None):
    """Map a reduction axis (or axes) onto ``(outer, reduce, inner)``."""
    shape = validation.normalize_shape(shape, name="reduction input")
    ndim = len(shape)
    if not ndim:
        raise ValueError("cannot reduce a 0-d tensor")

    # 1. Total reduction across all dimensions.
    if axis is None:
        return 1, validation.numel(shape), 1, (1,)

    # 2. Normalize axis (int, list or tuple) to sorted non-negative indices.
    axes = validation.normalize_dims(
        axis, ndim, name="axis", allow_duplicates=False
    )
    axes = tuple(sorted(axes))
    if not axes:
        raise ValueError("axis must contain at least one dimension")

    # 3. Only a contiguous range of axes can be expressed as one (outer,
    # reduce, inner) view without an explicit permute.
    if not all(axes[i] == axes[i - 1] + 1 for i in range(1, len(axes))):
        raise ValueError(
            f"non-contiguous multi-axis reduction {tuple(axes)} on shape "
            f"{shape} requires permuting the tensor first"
        )

    first_axis, last_axis = axes[0], axes[-1]
    outer_size = (
        validation.numel(shape[:first_axis]) if first_axis > 0 else 1
    )
    reduce_size = validation.numel(shape[first_axis : last_axis + 1])
    inner_size = (
        validation.numel(shape[last_axis + 1 :])
        if last_axis < ndim - 1
        else 1
    )

    out_shape = tuple(size for d, size in enumerate(shape) if d not in axes)
    if not out_shape:
        out_shape = (1,)
    return outer_size, reduce_size, inner_size, out_shape


def _expand_grad_output(tensor1, shape, axis, keepdims=False):
    """Restore reduced dimensions so a grad can broadcast back to ``shape``."""
    ndim = len(shape)

    if keepdims:
        return tensor1

    if axis is None:
        expanded_shape = (1,) * ndim
        return tensor1.reshape(expanded_shape)

    axes_to_expand = set(
        validation.normalize_dims(axis, ndim, name="axis", allow_duplicates=False)
    )

    expanded_shape = []
    grad_dim_index = 0
    grad_shape = tensor1.shape

    for i in range(ndim):
        if i in axes_to_expand:
            expanded_shape.append(1)
        else:
            if grad_dim_index < len(grad_shape):
                expanded_shape.append(grad_shape[grad_dim_index])
                grad_dim_index += 1
            else:
                expanded_shape.append(1)

    return tensor1.reshape(tuple(expanded_shape))
