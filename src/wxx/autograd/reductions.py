

import math

from .. import functional
from . import Function
from ..backend import context
from ..backend.dispatch import _expand_grad_output


class softmax(Function):
    def forward(self, tensor1, dim):

        if dim == -1:
            dim = len(tensor1.shape) - 1
        self.dim = dim
        out = functional.softmax(tensor1, dim)
        out.requires_grad = tensor1.requires_grad
        if out.requires_grad and context.MODE == 0:
            out.creator = self
            self.hold_output(out)

        return out

    def backward(self, grad_output):
        out = self.held_output()
        dot_product = (grad_output * out).sum(axis=self.dim).unsqueeze(self.dim)

        grad_A = functional.pointwise(
            out, functional.pointwise(grad_output, dot_product, 1), 2
        )

        self.dim = None

        return grad_A


class sum(Function):
    def forward(self, tensor1, axis=None, keepdims=False, *, keepdim=None):
        if keepdim is not None:
            keepdims = keepdim

        self.tensor1 = tensor1
        out = functional.sum(tensor1, axis, keepdims)
        out.requires_grad = tensor1.requires_grad

        self.axis = axis
        self.keepdims = keepdims

        if out.requires_grad and context.MODE == 0:
            self.tensor1 = tensor1
            out.creator = self
        return out

    def backward(self, grad_output):
        grad_output = _expand_grad_output(
            grad_output,
            shape=self.tensor1.shape,
            axis=self.axis,
            keepdims=self.keepdims,
        )

        grad_A = (
            functional.pointwise(functional.ones_like(self.tensor1), grad_output, 2)
            if self.needs_input_grad[0]
            else None
        )

        self.tensor1 = None

        return grad_A



class mean(Function):
    def forward(self, tensor1, axis=None, keepdims=False, *, keepdim=None):
        if keepdim is not None:
            keepdims = keepdim

        self.numel_in = int(math.prod(tensor1.shape))
        out = functional.mean(tensor1, axis, keepdims)
        out.requires_grad = tensor1.requires_grad

        self.axis = axis
        self.keepdims = keepdims

        if out.requires_grad and context.MODE == 0:
            self.tensor1 = tensor1
            out.creator = self
        return out

    def backward(self, grad_output):
        self.numel_out = int(math.prod(grad_output.shape))

        grad_output = _expand_grad_output(
            grad_output,
            shape=self.tensor1.shape,
            axis=self.axis,
            keepdims=self.keepdims,
        )

        N = self.numel_in // self.numel_out

        grad_A = (
            functional.pointwise(
                functional.pointwise(
                    functional.ones_like(self.tensor1), grad_output, 2
                ),
                (1 / N),
                2,
            )
            if self.needs_input_grad[0]
            else None
        )

        self.tensor1 = None

        return grad_A

class var(Function):
    """Population variance (``ddof=0``) with an analytic backward."""

    def forward(self, tensor1, axis=None, keepdims=False, *, keepdim=None):
        if keepdim is not None:
            keepdims = keepdim

        self.numel_in = int(math.prod(tensor1.shape))
        out = functional.var(tensor1, axis, keepdims)
        out.requires_grad = tensor1.requires_grad

        self.axis = axis
        self.keepdims = keepdims

        if out.requires_grad and context.MODE == 0:
            self.tensor1 = tensor1
            self.mean = functional.mean(tensor1, axis, keepdims=True)
            out.creator = self
        return out

    def backward(self, grad_output):
        if not self.needs_input_grad[0]:
            return None

        grad_output = _expand_grad_output(
            grad_output,
            shape=self.tensor1.shape,
            axis=self.axis,
            keepdims=self.keepdims,
        )

        self.numel_out = int(math.prod(grad_output.shape))
        N = self.numel_in // self.numel_out

        diff = functional.pointwise(self.tensor1, self.mean, 1)
        grad_A = functional.pointwise(
            functional.pointwise(diff, grad_output, 2), 2.0 / N, 2
        )

        self.tensor1 = None
        self.mean = None
        return grad_A


class std(Function):
    """Population standard deviation with an analytic backward."""

    def forward(self, tensor1, axis=None, keepdims=False, *, keepdim=None):
        if keepdim is not None:
            keepdims = keepdim

        self.numel_in = int(math.prod(tensor1.shape))
        out = functional.std(tensor1, axis, keepdims)
        out.requires_grad = tensor1.requires_grad

        self.axis = axis
        self.keepdims = keepdims

        if out.requires_grad and context.MODE == 0:
            self.tensor1 = tensor1
            self.mean = functional.mean(tensor1, axis, keepdims=True)
            self.std = functional.std(tensor1, axis, keepdims=True)
            out.creator = self
        return out

    def backward(self, grad_output):
        if not self.needs_input_grad[0]:
            return None

        grad_output = _expand_grad_output(
            grad_output,
            shape=self.tensor1.shape,
            axis=self.axis,
            keepdims=self.keepdims,
        )

        self.numel_out = int(math.prod(grad_output.shape))
        N = self.numel_in // self.numel_out

        diff = functional.pointwise(self.tensor1, self.mean, 1)
        denom = functional.pointwise(self.std, float(N), 2)
        grad_A = functional.pointwise(
            diff, functional.pointwise(grad_output, denom, 3), 2
        )

        self.tensor1 = None
        self.mean = None
        self.std = None
        return grad_A


class max(Function):
    def forward(self, tensor1, axis=None, keepdims=False, *, keepdim=None):
        if keepdim is not None:
            keepdims = keepdim

        self.numel_in = int(math.prod(tensor1.shape))

        out, indices = functional.max(tensor1, axis, keepdims)
        out.requires_grad = tensor1.requires_grad

        self.axis = axis
        self.keepdims = keepdims

        if out.requires_grad and context.MODE == 0:
            self.input_shape = tuple(tensor1.shape)
            self.indices = indices
            out.creator = self
        return out

    def backward(self, grad_output):
        if not self.needs_input_grad[0]:
            return None

        grad_input = functional.maxmin_backward(
            grad_output, self.indices, self.input_shape, self.axis
        )
        self.indices = None
        return grad_input

class min(Function):
    def forward(self, tensor1, axis=None, keepdims=False, *, keepdim=None):
        if keepdim is not None:
            keepdims = keepdim

        self.numel_in = int(math.prod(tensor1.shape))

        out, indices = functional.min(tensor1, axis, keepdims)
        out.requires_grad = tensor1.requires_grad

        self.axis = axis
        self.keepdims = keepdims

        if out.requires_grad and context.MODE == 0:
            self.input_shape = tuple(tensor1.shape)
            self.indices = indices

            out.creator = self
        return out

    def backward(self, grad_output):
        if not self.needs_input_grad[0]:
            return None

        grad_input = functional.maxmin_backward(
            grad_output, self.indices, self.input_shape, self.axis
        )
        self.indices = None
        return grad_input


class argmax(Function):
    def forward(self, tensor1, axis=None, keepdims=False, *, keepdim=None):
        if keepdim is not None:
            keepdims = keepdim

        self.numel_in = int(math.prod(tensor1.shape))

        out = functional.max(tensor1, axis, keepdims)[1]

        return out

    def backward(self, grad_output):
        return None

class argmin(Function):
    def forward(self, tensor1, axis=None, keepdims=False, *, keepdim=None):
        if keepdim is not None:
            keepdims = keepdim

        self.numel_in = int(math.prod(tensor1.shape))

        out = functional.min(tensor1, axis, keepdims)[1]

        return out

    def backward(self, grad_output):
        return None
