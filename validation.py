"""Cheap shape, dimension and layout validators shared by ``wxx`` ops.

Every helper here inspects metadata only (``shape`` / ``stride`` / ``offset``).
No buffer is read, no kernel is dispatched and no GPU synchronisation happens,
so calling them on hot paths costs O(rank) Python work and cannot change
numerical results.

The module deliberately has **no wxx imports** so that ``Tensor`` and the
backend can use it without creating import cycles.
"""

import math

MAX_RANK = 6
"""Highest rank addressable by the WebGPU kernels."""


def numel(shape):
    """Number of elements described by ``shape``."""
    return int(math.prod(shape)) if len(shape) else 1


def normalize_shape(shape, *, name="shape", allow_infer=False):
    """Return ``shape`` as a tuple of non-negative ints.

    If ``allow_infer`` is true a single ``-1`` entry is kept for
    :func:`infer_shape` to resolve.
    """
    if isinstance(shape, int):
        shape = (shape,)
    try:
        items = tuple(shape)
    except TypeError:
        raise TypeError(
            f"{name} must be an int or an iterable of ints, "
            f"got {type(shape).__name__}"
        ) from None

    inferred = 0
    for i, dim in enumerate(items):
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise TypeError(f"{name}[{i}] must be an int, got {type(dim).__name__}")
        if dim == -1 and allow_infer:
            inferred += 1
            if inferred > 1:
                raise ValueError(f"{name} may contain at most one -1, got {items}")
        elif dim < 0:
            raise ValueError(f"{name}[{i}] must be non-negative, got {dim}")
    return items


def infer_shape(shape, total, *, name="shape"):
    """Resolve an optional ``-1`` in ``shape`` and check the element count."""
    shape = normalize_shape(shape, name=name, allow_infer=True)
    if -1 in shape:
        index = shape.index(-1)
        known = 1
        for i, dim in enumerate(shape):
            if i != index:
                known *= dim
        if known == 0 or total % known != 0:
            raise ValueError(f"cannot infer {name} {shape} for {total} elements")
        shape = shape[:index] + (total // known,) + shape[index + 1 :]

    if numel(shape) != total:
        raise ValueError(
            f"{name} {shape} describes {numel(shape)} elements, "
            f"expected {total}"
        )
    return shape


def normalize_dim(dim, rank, *, name="dim"):
    """Normalise one dimension index, allowing negative values."""
    if isinstance(dim, bool) or not isinstance(dim, int):
        raise TypeError(f"{name} must be an int, got {type(dim).__name__}")
    if rank == 0:
        raise IndexError(f"{name}={dim} is invalid for a 0-d tensor")
    normalized = dim + rank if dim < 0 else dim
    if not 0 <= normalized < rank:
        raise IndexError(
            f"{name}={dim} is out of range for a tensor of rank {rank}"
        )
    return normalized


def normalize_dims(dims, rank, *, name="dims", allow_duplicates=True):
    """Normalise an int or iterable of dimension indices to a tuple."""
    if isinstance(dims, int) and not isinstance(dims, bool):
        dims = (dims,)
    try:
        items = tuple(dims)
    except TypeError:
        raise TypeError(
            f"{name} must be an int or an iterable of ints, "
            f"got {type(dims).__name__}"
        ) from None

    normalized = tuple(
        normalize_dim(d, rank, name=f"{name}[{i}]")
        for i, d in enumerate(items)
    )
    if not allow_duplicates and len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} contains duplicate dimensions: {normalized}")
    return normalized


def check_rank(shape, *, min_rank=0, max_rank=MAX_RANK, name="tensor"):
    """Check the rank of ``shape`` and return it."""
    rank = len(shape)
    if rank < min_rank or (max_rank is not None and rank > max_rank):
        if max_rank is None:
            expected = f"at least {min_rank}"
        elif min_rank == max_rank:
            expected = str(min_rank)
        else:
            expected = f"{min_rank}..{max_rank}"
        raise ValueError(f"{name} must have rank {expected}, got {rank}")
    return rank


def check_same_shape(*shapes, name="tensors"):
    """Require every shape to be identical and return the common shape."""
    if not shapes:
        raise ValueError(f"{name} must contain at least one shape")
    first = tuple(shapes[0])
    for i, shape in enumerate(shapes[1:], 1):
        if tuple(shape) != first:
            raise ValueError(
                f"{name} must have the same shape, got {first} and {tuple(shape)} "
                f"(at index {i})"
            )
    return first


def check_broadcastable(shape, out_shape, *, name="tensor"):
    """Check that ``shape`` can broadcast to ``out_shape``."""
    if len(shape) > len(out_shape):
        raise ValueError(
            f"{name} of shape {tuple(shape)} cannot broadcast to {tuple(out_shape)}"
        )
    pad = len(out_shape) - len(shape)
    for i, dim in enumerate(shape):
        expected = out_shape[i + pad]
        if dim != expected and dim != 1:
            raise ValueError(
                f"{name} of shape {tuple(shape)} cannot broadcast to "
                f"{tuple(out_shape)}"
            )
    return tuple(out_shape)


def contiguous_strides(shape):
    """Row-major (C-order) strides for ``shape``."""
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def is_contiguous_layout(shape, stride):
    """True when ``stride`` describes a compact row-major layout.

    Singleton dimensions are allowed an arbitrary stride, matching PyTorch.
    """
    expected = 1
    for size, step in zip(reversed(shape), reversed(stride)):
        if size > 1 and step != expected:
            return False
        expected *= size
    return True


def check_contiguous(tensor, *, name="tensor"):
    """Raise unless ``tensor`` is contiguous (used by kernels without strides)."""
    if not tensor.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous, got shape={tuple(tensor.shape)} "
            f"stride={tuple(tensor.stride)} offset={tensor.offset}"
        )
    return tensor


def check_no_negative_stride(tensor, *, name="tensor"):
    """Kernels address memory with unsigned strides, so reject negatives."""
    if any(step < 0 for step in tensor.stride):
        raise ValueError(
            f"{name} has a negative stride {tuple(tensor.stride)}; "
            "call .contiguous() first"
        )
    return tensor


__all__ = [
    "MAX_RANK",
    "numel",
    "normalize_shape",
    "infer_shape",
    "normalize_dim",
    "normalize_dims",
    "check_rank",
    "check_same_shape",
    "check_broadcastable",
    "contiguous_strides",
    "is_contiguous_layout",
    "check_contiguous",
    "check_no_negative_stride",
]
