"""Autograd functions for FFT, complex ops, slicing, and indexing."""

import math

from .. import Tensor, functional, validation
from ..backend import context
from ..backend.dispatch import _sum_to_shape
from .function import Function


def _single_slice(index, shape):
    """Return ``(dim, start)`` when ``index`` is one unit-step slice.

    The remaining axes must be full slices (no integer indexing), which is the
    common ``x[..., :n]`` / ``chunk`` pattern that can be served by the fused
    ``scatter_zeros`` kernel.  Returns ``None`` for anything more complex.
    """
    if not isinstance(index, tuple):
        index = (index,)
    if len(index) > len(shape):
        return None
    index = index + (slice(None),) * (len(shape) - len(index))

    found = None
    for axis, item in enumerate(index):
        if not isinstance(item, slice):
            return None
        start, stop, step = item.indices(shape[axis])
        if step != 1:
            return None
        if start == 0 and stop == shape[axis]:
            continue
        if found is not None:
            return None
        found = (axis, start)
    return found



class matmul(Function):
    def forward(self, tensor1, tensor2, transpose_a=False, transpose_b=False):
        out = functional.matmul_add(
            tensor1, tensor2, None, transpose_A=transpose_a, transpose_B=transpose_b
        )
        out.requires_grad = tensor1.requires_grad or tensor2.requires_grad

        if out.requires_grad and context.MODE == 0:
            out.creator = self
            self.tensor1 = tensor1
            self.tensor2 = tensor2
            self.transpose_a = transpose_a
            self.transpose_b = transpose_b

        return out

    def backward(self, grad_output):
        if self.needs_input_grad[0]:
            if self.transpose_a:
                grad_A = functional.matmul_add(
                    self.tensor2,
                    grad_output,
                    None,
                    self.transpose_b,
                    True,
                )
            else:
                grad_A = functional.matmul_add(
                    grad_output,
                    self.tensor2,
                    None,
                    False,
                    not self.transpose_b,
                )
            grad_A = _sum_to_shape(grad_A, self.inputs[0].shape)
        else:
            grad_A = None

        if self.needs_input_grad[1]:
            if self.transpose_b:
                grad_B = functional.matmul_add(
                    grad_output,
                    self.tensor1,
                    None,
                    True,
                    self.transpose_a,
                )
            else:
                grad_B = functional.matmul_add(
                    self.tensor1,
                    grad_output,
                    None,
                    not self.transpose_a,
                    False,
                )
            grad_B = _sum_to_shape(grad_B, self.inputs[1].shape)
        else:
            grad_B = None

        self.tensor1 = None
        self.tensor2 = None

        return grad_A, grad_B

class matmuladd(Function):
    def forward(self, tensor1, tensor2, tensor3):
        out = functional.matmul_add(tensor1, tensor2, tensor3)
        out.requires_grad = any(self.needs_input_grad)

        if out.requires_grad and context.MODE == 0:
            out.creator = self
            self.tensor1 = tensor1
            self.tensor2 = tensor2
        return out

    def backward(self, grad_output):
        grad_A = (
            _sum_to_shape(
                functional.matmul_add(grad_output, self.tensor2, None, False, True),
                self.inputs[0].shape,
            )
            if self.needs_input_grad[0]
            else None
        )
        grad_B = (
            _sum_to_shape(
                functional.matmul_add(self.tensor1, grad_output, None, True, False),
                self.inputs[1].shape,
            )
            if self.needs_input_grad[1]
            else None
        )
        if self.needs_input_grad[2]:
            grad_C = _sum_to_shape(grad_output, self.inputs[2].shape)
        else:
            grad_C = None

        self.tensor1 = None
        self.tensor2 = None

        return grad_A, grad_B, grad_C




class flatten(Function):
    def forward(self, tensor1, start_dim=0, end_dim=-1):
        self.shape = list(tensor1.shape)
        ndim = len(self.shape)
        start_dim = validation.normalize_dim(start_dim, ndim, name="start_dim")
        end_dim = validation.normalize_dim(end_dim, ndim, name="end_dim")
        if start_dim > end_dim:
            raise ValueError(
                f"start_dim ({start_dim}) must be <= end_dim ({end_dim})"
            )

        if start_dim == 0 and end_dim == ndim - 1:
            new_shape = (validation.numel(self.shape),)
        else:
            flattened_dim = int(
                math.prod(self.shape[start_dim : end_dim + 1])
            )
            new_shape = (
                self.shape[:start_dim]
                + [flattened_dim]
                + self.shape[end_dim + 1 :]
            )

        out = tensor1.reshape(new_shape)

        if out.requires_grad and context.MODE == 0:
            out.creator = self
        return out

    def backward(self, grad_output):
        grad_A = grad_output.reshape(self.shape)
        return grad_A



class permute(Function):
    def forward(self, tensor1, axes):
        # Accept both ``permute(0, 2, 1)`` and ``permute((0, 2, 1))`` and store
        # the flattened, normalised axes so backward can invert them.
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        else:
            axes = tuple(axes)
        axes = validation.normalize_dims(
            axes, len(tensor1.shape), name="axes", allow_duplicates=False
        )

        self.needs_input_grad = [tensor1.requires_grad]

        out = functional.permute(tensor1, *axes)

        out.requires_grad = tensor1.requires_grad
        if out.requires_grad and context.MODE == 0:
            self.axes = axes
            self.tensor1 = tensor1
            out.creator = self

        return out

    def backward(self, grad_output):
        if not self.needs_input_grad[0]:
            return None

        inv_axes = [0] * len(self.axes)
        for i, axis in enumerate(self.axes):
            inv_axes[axis] = i

        grad_A = functional.permute(grad_output, *inv_axes)
        self.tensor1 = None

        return grad_A

class reshape(Function):
    def forward(self, tensor1, shape):

        out = functional.reshape(tensor1, shape)
        out.requires_grad = tensor1.requires_grad

        if out.requires_grad and context.MODE == 0:
            self.tensor1 = tensor1
            out.creator = self

        return out

    def backward(self, grad_output):
        if not self.needs_input_grad[0]:
            return None

        grad_A = functional.reshape(grad_output, self.tensor1.shape)

        self.tensor1 = None
        return grad_A

class narrow(Function):
    def forward(self, tensor1, dim, start, length):

        out = tensor1.narrow(dim, start, length)
        out.requires_grad = tensor1.requires_grad

        if out.requires_grad and context.MODE == 0:
            self.tensor1 = tensor1
            self.dim = dim % len(tensor1.shape)
            self.start = start
            self.length = length
            out.creator = self
        return out

    def backward(self, grad_output):

        if not self.needs_input_grad[0]:
            return None

        if getattr(self.tensor1, "type", "f32") == "f32" and len(self.tensor1.shape) <= 6:
            dX = functional.scatter_zeros(
                grad_output, self.tensor1.shape, self.dim, self.start
            )
        else:
            dX = Tensor(
                self.tensor1.shape,
                value=0,
                requires_grad=False,
                dtype=self.tensor1.type,
            )
            functional.assign(
                dX.narrow(self.dim, self.start, self.length), grad_output
            )

        self.tensor1 = None
        return dX


class diagonal(Function):
    """Differentiable :func:`functional.diagonal` (a strided view)."""

    def forward(self, tensor1, offset, dim1, dim2):
        out = functional.diagonal(tensor1, offset, dim1, dim2)
        out.requires_grad = tensor1.requires_grad

        if out.requires_grad and context.MODE == 0:
            self.tensor1 = tensor1
            self.offset = offset
            self.dim1 = dim1
            self.dim2 = dim2
            out.creator = self
        return out

    def backward(self, grad_output):
        if not self.needs_input_grad[0]:
            return None

        # Scatter through a diagonal view of fresh zeros: one dispatch, no
        # host round-trip, and independent of the input's layout.
        dX = Tensor(
            self.tensor1.shape,
            value=0,
            requires_grad=False,
            dtype=self.tensor1.type,
        )
        functional.assign(
            functional.diagonal(dX, self.offset, self.dim1, self.dim2), grad_output
        )
        self.tensor1 = None
        return dX


class unfold(Function):
    """Differentiable :func:`functional.unfold` (a sliding-window view)."""

    def forward(self, tensor1, dimension, size, step):
        out = functional.unfold(tensor1, dimension, size, step)
        out.requires_grad = tensor1.requires_grad

        if out.requires_grad and context.MODE == 0:
            self.input_shape = tuple(tensor1.shape)
            self.dimension = dimension
            self.size = size
            self.step = step
            out.creator = self
        return out

    def backward(self, grad_output):
        if not self.needs_input_grad[0]:
            return None

        dX = functional.unfolddx(
            grad_output,
            self.input_shape,
            self.dimension,
            self.size,
            self.step,
        )
        self.input_shape = None
        return dX


class getitem(Function):
    """Differentiable basic slicing (slices and integer indices)."""

    def forward(self, tensor1, index):
        out = tensor1._basic_getitem(index)
        out.requires_grad = tensor1.requires_grad

        if out.requires_grad and context.MODE == 0:
            self.tensor1 = tensor1
            self.index = index
            out.creator = self
        return out

    def backward(self, grad_output):
        if not self.needs_input_grad[0]:
            return None

        target = _single_slice(self.index, self.tensor1.shape)
        if target is not None and getattr(self.tensor1, "type", "f32") == "f32":
            # Fused zero-fill + scatter (no host-side zero array, one dispatch).
            dim, start = target
            dX = functional.scatter_zeros(
                grad_output, self.tensor1.shape, dim, start
            )
        else:
            dX = Tensor(
                self.tensor1.shape,
                value=0,
                requires_grad=False,
                dtype=self.tensor1.type,
            )
            functional.assign(dX[self.index], grad_output)

        self.tensor1 = None
        return dX

class concat(Function):
    def forward(self, tensors, dim=-1):
        if not tensors:
            raise ValueError("concat expects at least one tensor")
        rank = len(tensors[0].shape)
        dim = validation.normalize_dim(dim, rank, name="dim")
        base_shape = list(tensors[0].shape)

        for i, tensor in enumerate(tensors):
            if len(tensor.shape) != rank:
                raise ValueError(
                    f"concat expects every tensor to have rank {rank}, "
                    f"got {len(tensor.shape)} at index {i}"
                )
            if tensor.type != tensors[0].type:
                raise TypeError(
                    f"concat expects matching dtypes, got {tensors[0].type!r} "
                    f"and {tensor.type!r} at index {i}"
                )
            for axis, (size, expected) in enumerate(
                zip(tensor.shape, base_shape)
            ):
                if axis != dim and size != expected:
                    raise ValueError(
                        f"concat shapes must match except on dim {dim}, "
                        f"got {tuple(tensor.shape)} at index {i} and "
                        f"{tuple(base_shape)}"
                    )

        self.starts = []
        self.lengths = []

        offset = 0
        for tensor in tensors:
            length = tensor.shape[dim]
            self.starts.append(offset)
            self.lengths.append(length)
            offset += length

        out_shape = list(base_shape)
        out_shape[dim] = offset

        out = Tensor(out_shape)
        out.requires_grad = any(tensor.requires_grad for tensor in tensors)

        for tensor, start, length in zip(tensors, self.starts, self.lengths):
            dest_slice = out.narrow(dim, start, length)
            tensor_contig = (
                tensor if tensor.is_contiguous() else tensor.contiguous()
            )
            functional.assign(dest_slice, tensor_contig)

        self.dim = dim
        if out.requires_grad and context.MODE == 0:
            self.tensors = tensors
            out.creator = self
        return out

    def backward(self, grad_output):
        input_grads = []
        for i, (t, start, length) in enumerate(
            zip(self.inputs, self.starts, self.lengths)
        ):
            if not self.needs_input_grad[i]:
                input_grads.append(None)
                continue

            grad_i = grad_output.narrow(self.dim, start, length)

            if not grad_i.is_contiguous():
                grad_i = grad_i.contiguous()

            input_grads.append(grad_i)

        self.tensors = None

        return input_grads

class unsqueeze(Function):
    """Insert one or more size-1 dimensions (a stride-preserving view)."""

    def forward(self, tensor1, dim):
        self.tensor1shape = list(tensor1.shape)
        rank = len(tensor1.shape)

        if dim is None:
            dims = (rank,)
        elif isinstance(dim, int) and not isinstance(dim, bool):
            dims = (dim,)
        else:
            dims = tuple(dim)

        final_rank = rank + len(dims)
        normalized = sorted(
            validation.normalize_dim(d, final_rank, name="dim") for d in dims
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"duplicate unsqueeze dimensions: {normalized}")

        new_shape = list(tensor1.shape)
        new_stride = list(tensor1.stride)
        for axis in normalized:
            new_shape.insert(axis, 1)
            new_stride.insert(axis, 0)

        out = Tensor(
            shape=new_shape,
            buffer=tensor1.buffer,
            stride=new_stride,
            offset=tensor1.offset,
            requires_grad=tensor1.requires_grad,
            dtype=tensor1.type,
        )
        out._base = tensor1

        if out.requires_grad and context.MODE == 0:
            out.creator = self
        return out

    def backward(self, grad_output):
        if not self.needs_input_grad[0]:
            return None
        return functional.reshape(grad_output, self.tensor1shape)


class squeeze(Function):
    """Remove size-1 dimensions (a stride-preserving view).

    Like ``torch.squeeze``, a requested dimension whose size is not 1 is left
    untouched instead of raising.
    """

    def forward(self, tensor1, dim):
        self.tensor1shape = list(tensor1.shape)
        rank = len(tensor1.shape)

        if dim is None:
            removed = {i for i, size in enumerate(tensor1.shape) if size == 1}
        else:
            if isinstance(dim, int) and not isinstance(dim, bool):
                dims = (dim,)
            else:
                dims = tuple(dim)
            normalized = {
                validation.normalize_dim(d, rank, name="dim") for d in dims
            }
            removed = {d for d in normalized if tensor1.shape[d] == 1}

        keep = [axis for axis in range(rank) if axis not in removed]
        new_shape = [tensor1.shape[axis] for axis in keep]
        new_stride = [tensor1.stride[axis] for axis in keep]

        out = Tensor(
            shape=new_shape,
            buffer=tensor1.buffer,
            stride=new_stride,
            offset=tensor1.offset,
            requires_grad=tensor1.requires_grad,
            dtype=tensor1.type,
        )
        out._base = tensor1

        if out.requires_grad and context.MODE == 0:
            out.creator = self
        return out

    def backward(self, grad_output):
        if not self.needs_input_grad[0]:
            return None
        return functional.reshape(grad_output, self.tensor1shape)



def _rfft_backward_weights(n):
    if n == 1:
        return [1.0]
    return [1.0] + [0.5] * (n // 2 - 1) + [1.0]


def _irfft_backward_weights(n):
    if n == 1:
        return [1.0]
    return [1.0] + [2.0] * (n // 2 - 1) + [1.0]


def _match_last_dim_real(tensor, target):
    """Make ``tensor``'s last dimension equal ``target`` (pad with zeros)."""
    current = tensor.shape[-1]
    if current == target:
        return tensor
    if target < current:
        return tensor.narrow(-1, 0, target)

    out = Tensor(list(tensor.shape[:-1]) + [target], value=0)
    functional.assign(out.narrow(-1, 0, current), tensor)
    return out


def _sum_to_shape_complex(gradient, target_shape):
    """Reduce a complex gradient to ``target_shape`` (broadcast adjoint)."""
    target_shape = tuple(target_shape)
    if tuple(gradient.shape) == target_shape:
        return gradient

    # A complex tensor of shape S is bit-identical to a real tensor of shape
    # S + (2,), so we can reuse the regular real-valued reduction kernels.
    # Complex strides count ``vec2`` elements, so they must be doubled when
    # reinterpreted as float strides.
    if not gradient.is_contiguous():
        gradient = functional.complex_contiguous(gradient)

    real = Tensor(
        buffer=gradient.buffer,
        shape=list(gradient.shape) + [2],
        stride=tuple(s * 2 for s in gradient.stride) + (1,),
        offset=gradient.offset * 2,
        requires_grad=False,
        dtype="f32",
    )
    real._base = gradient

    reduced = _sum_to_shape(real, target_shape + (2,))
    if not reduced.is_contiguous():
        reduced = functional.contiguous(reduced)

    out = Tensor(
        buffer=reduced.buffer,
        shape=list(target_shape),
        stride=tuple(s // 2 for s in reduced.stride[:-1]),
        offset=reduced.offset // 2,
        requires_grad=gradient.requires_grad,
        dtype="c64",
    )
    out._base = reduced
    return out


class rfft(Function):
    def forward(self, tensor1, n=None):

        out, self.N = functional.rfft(tensor1, n)
        self.in_len = tensor1.shape[-1]
        self.in_shape = list(tensor1.shape)
        out.requires_grad = tensor1.requires_grad
        if out.requires_grad and context.MODE == 0:
            out.creator = self

        return out

    def backward(self, grad_output):
        if not self.needs_input_grad[0]:
            return None

        weights = functional.complex_weights(
            _rfft_backward_weights(self.N)
        )
        scaled_grad = functional.complex_mul(grad_output, weights)

        grad_A = functional.irfft(scaled_grad, self.N)
        grad_A = grad_A * float(self.N)

        return _match_last_dim_real(grad_A, self.in_len)


class irfft(Function):
    def forward(self, tensor1, n=None):

        out = functional.irfft(tensor1, n)
        self.N = (
            2 * (tensor1.shape[-1] - 1) if n is None else int(n)
        )
        self.in_len = tensor1.shape[-1]
        self.in_shape = list(tensor1.shape)
        out.requires_grad = tensor1.requires_grad
        if out.requires_grad and context.MODE == 0:
            out.creator = self

        return out

    def backward(self, grad_output):
        if not self.needs_input_grad[0]:
            return None

        spectrum, _ = functional.rfft(grad_output, self.N)
        weights = functional.complex_weights(
            [w / float(self.N) for w in _irfft_backward_weights(self.N)]
        )
        grad_A = functional.complex_mul(spectrum, weights)

        # The input may contain fewer/more one-sided frequencies than the
        # canonical N/2 + 1; match its original shape.
        return functional.complex_pad(grad_A, self.in_len)


class complex_mul(Function):
    def forward(self, tensor1, tensor2):

        out = functional.complex_mul(tensor1, tensor2)
        out.requires_grad = tensor1.requires_grad or tensor2.requires_grad

        if out.requires_grad and context.MODE == 0:
            self.tensor1 = tensor1
            self.tensor2 = tensor2
            out.creator = self

        return out

    def backward(self, grad_output):
        if self.needs_input_grad[0]:
            grad_A = functional.complex_mul(
                grad_output, self.tensor2, conj_b=True
            )
            grad_A = _sum_to_shape_complex(grad_A, self.inputs[0].shape)
        else:
            grad_A = None

        if self.needs_input_grad[1]:
            grad_B = functional.complex_mul(
                grad_output, self.tensor1, conj_b=True
            )
            grad_B = _sum_to_shape_complex(grad_B, self.inputs[1].shape)
        else:
            grad_B = None

        self.tensor1 = None
        self.tensor2 = None
        return grad_A, grad_B


class complex_conj(Function):
    def forward(self, tensor1):

        out = functional.complex_conj(tensor1)
        out.requires_grad = tensor1.requires_grad
        if out.requires_grad and context.MODE == 0:
            out.creator = self

        return out

    def backward(self, grad_output):
        if not self.needs_input_grad[0]:
            return None
        return functional.complex_conj(grad_output)
