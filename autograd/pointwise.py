
from .. import functional

from ..backend import context
from ..backend.dispatch import _sum_to_shape

from .function import Function


class add(Function):
    def forward(self, tensor1, tensor2):
        out = functional.pointwise(tensor1, tensor2, 0)
        out.requires_grad = tensor1.requires_grad or tensor2.requires_grad
        if out.requires_grad and context.MODE == 0:
            out.creator = self
        return out

    def backward(self, grad_output):
        grad_A = (
            _sum_to_shape(grad_output, self.inputs[0].shape)
            if self.needs_input_grad[0]
            else None
        )
        grad_B = (
            _sum_to_shape(grad_output, self.inputs[1].shape)
            if self.needs_input_grad[1]
            else None
        )
        return grad_A, grad_B

class addscaler(Function):
    def forward(self, tensor1, float):
        out = functional.scalarpointwise(tensor1, float, 0)
        out.requires_grad = tensor1.requires_grad

        if out.requires_grad and context.MODE == 0:
            out.creator = self
        return out

    def backward(self, grad_output):
        grad_A = grad_output if self.needs_input_grad[0] else None
        return grad_A

class sub(Function):
    def forward(self, tensor1, tensor2):
        out = functional.pointwise(tensor1, tensor2, 1)
        out.requires_grad = tensor1.requires_grad or tensor2.requires_grad

        if out.requires_grad and context.MODE == 0:
            out.creator = self
        return out

    def backward(self, grad_output):
        grad_A = (
            _sum_to_shape(grad_output, self.inputs[0].shape)
            if self.needs_input_grad[0]
            else None
        )
        grad_B = (
            _sum_to_shape(functional.negate(grad_output), self.inputs[1].shape)
            if self.needs_input_grad[1]
            else None
        )

        return grad_A, grad_B

class subscaler(Function):
    def forward(self, tensor1, float):
        out = functional.scalarpointwise(tensor1, float, 1)
        out.requires_grad = tensor1.requires_grad
        if out.requires_grad and context.MODE == 0:
            out.creator = self
        return out

    def backward(self, grad_output):
        grad_A = grad_output if self.needs_input_grad[0] else None
        return grad_A

class mul(Function):
    def forward(self, tensor1, tensor2):
        out = functional.pointwise(tensor1, tensor2, 2)
        out.requires_grad = tensor1.requires_grad or tensor2.requires_grad

        if out.requires_grad and context.MODE == 0:
            self.tensor1 = tensor1
            self.tensor2 = tensor2
            out.creator = self
        return out

    def backward(self, grad_output):
        shapes_match = (
            tuple(grad_output.shape)
            == tuple(self.tensor1.shape)
            == tuple(self.tensor2.shape)
        )
        if (
            shapes_match
            and self.needs_input_grad[0]
            and self.needs_input_grad[1]
            and grad_output.is_contiguous()
            and self.tensor1.is_contiguous()
            and self.tensor2.is_contiguous()
        ):
            grad_A, grad_B = functional.mul_backward(
                grad_output, self.tensor1, self.tensor2
            )
        else:
            grad_A = (
                _sum_to_shape(grad_output * self.tensor2, self.inputs[0].shape)
                if self.needs_input_grad[0]
                else None
            )
            grad_B = (
                _sum_to_shape(grad_output * self.tensor1, self.inputs[1].shape)
                if self.needs_input_grad[1]
                else None
            )

        self.tensor1 = None
        self.tensor2 = None

        return grad_A, grad_B

class mulscaler(Function):
    def forward(self, tensor1, float):
        out = functional.scalarpointwise(tensor1, float, 2)
        out.requires_grad = tensor1.requires_grad

        self.float = float
        if out.requires_grad and context.MODE == 0:
            out.creator = self
        return out

    def backward(self, grad_output):
        grad_A = grad_output * self.float if self.needs_input_grad[0] else None
        return grad_A

class powscaler(Function):
    def forward(self, tensor1, float):
        out = functional.scalarpointwise(tensor1, float, 3)
        out.requires_grad = tensor1.requires_grad

        if out.requires_grad and context.MODE == 0:
            self.float = float
            self.tensor1 = tensor1
            out.creator = self
        return out

    def backward(self, grad_output):
        grad_output = (
            functional.pointwise(
                grad_output,
                functional.scalarpointwise(
                    functional.scalarpointwise(self.tensor1, self.float - 1, 3),
                    self.float,
                    2,
                ),
                2,
            )
            if self.needs_input_grad[0]
            else None
        )

        grad_A = grad_output
        self.tensor1 = None
        return grad_A

class div(Function):
    def forward(self, tensor1, tensor2):
        out = functional.pointwise(tensor1, tensor2, 3)
        out.requires_grad = tensor1.requires_grad or tensor2.requires_grad

        if out.requires_grad and context.MODE == 0:
            out.creator = self
            self.tensor1 = tensor1
            self.tensor2 = tensor2
        return out

    def backward(self, grad_output):
        grad_A = (
            _sum_to_shape(
                functional.pointwise(grad_output, self.tensor2, 3),
                self.inputs[0].shape,
            )
            if self.needs_input_grad[0]
            else None
        )
        grad_B = (
            _sum_to_shape(
                functional.pointwise(
                    -functional.pointwise(grad_output, self.tensor1, 2),
                    functional.square(self.tensor2),
                    3,
                ),
                self.inputs[1].shape,
            )
            if self.needs_input_grad[1]
            else None
        )
        self.tensor1 = None
        self.tensor2 = None
        return grad_A, grad_B


class relu(Function):
    def forward(self, tensor1):
        out = functional.relu(tensor1)
        out.requires_grad = tensor1.requires_grad

        if out.requires_grad and context.MODE == 0:
            out.creator = self
            self.tensor1 = tensor1
        return out

    def backward(self, grad_output):
        grad_A = (
            functional.pointwise(
                functional.relubackwards(self.tensor1), grad_output, 2
            )
            if self.needs_input_grad[0]
            else None
        )

        self.tensor1 = None

        return grad_A

class sin(Function):
    def forward(self, tensor1):
        out = functional.sin(tensor1)
        out.requires_grad = tensor1.requires_grad

        if out.requires_grad and context.MODE == 0:
            out.creator = self
            self.tensor1 = tensor1
        return out

    def backward(self, grad_output):
        grad_A = (
            functional.pointwise(
                functional.sinback(self.tensor1), grad_output, 2
            )
            if self.needs_input_grad[0]
            else None
        )

        self.tensor1 = None

        return grad_A


class gelu(Function):
    def forward(self, tensor1):
        out = functional.gelu(tensor1)
        out.requires_grad = tensor1.requires_grad

        if out.requires_grad and context.MODE == 0:
            out.creator = self
            self.tensor1 = tensor1

        return out

    def backward(self, grad_output):
        grad_A = (
            functional.pointwise(
                functional.gelubackwards(self.tensor1), grad_output, 2
            )
            if self.needs_input_grad[0]
            else None
        )

        self.tensor1 = None

        return grad_A

class sigmoid(Function):
    def forward(self, tensor1):
        out = functional.sigmoid(tensor1)
        out.requires_grad = tensor1.requires_grad

        if out.requires_grad and context.MODE == 0:
            out.creator = self
            self.out = out
        return out

    def backward(self, grad_output):
        grad_A = (
            functional.pointwise(
                functional.sigmoidbackwards(self.out), grad_output, 2
            )
            if self.needs_input_grad[0]
            else None
        )

        self.out = None

        return grad_A

class selu(Function):
    def forward(self, tensor1):
        out = functional.selu(tensor1)
        out.requires_grad = tensor1.requires_grad
        if out.requires_grad and context.MODE == 0:
            self.tensor1 = tensor1

            out.creator = self
        return out

    def backward(self, grad_output):
        grad_A = (
            functional.pointwise(
                functional.selubackwards(self.tensor1), grad_output, 2
            )
            if self.needs_input_grad[0]
            else None
        )

        self.tensor1 = None

        return grad_A



class negate(Function):
    def forward(self, tensor1):
        out = functional.negate(tensor1)
        out.requires_grad = tensor1.requires_grad
        if out.requires_grad and context.MODE == 0:
            out.creator = self
        return out

    def backward(self, grad_output):
        grad_A = (
            functional.negate(grad_output) if self.needs_input_grad[0] else None
        )
        return grad_A

class abs(Function):
    def forward(self, tensor1):
        out = functional.abs(tensor1)
        out.requires_grad = tensor1.requires_grad
        if out.requires_grad and context.MODE == 0:
            out.creator = self
            self.tensor1 = tensor1

        return out

    def backward(self, grad_output):
        grad_A = (
            functional.pointwise(grad_output, functional.sign(self.tensor1), 2)
            if self.needs_input_grad[0]
            else None
        )
        self.tensor1 = None

        return grad_A

class sign(Function):
    def forward(self, tensor1):
        out = functional.sign(tensor1)
        out.requires_grad = tensor1.requires_grad
        if out.requires_grad and context.MODE == 0:
            out.creator = self
        return out

    def backward(self, grad_output):
        return None

class square(Function):
    def forward(self, tensor1):
        out = functional.square(tensor1)
        out.requires_grad = tensor1.requires_grad
        if out.requires_grad and context.MODE == 0:
            out.creator = self
            self.tensor1 = tensor1

        return out

    def backward(self, grad_output):
        if self.needs_input_grad[0]:
            grad_output = functional.pointwise(
                grad_output, functional.scalarpointwise(self.tensor1, 2, 2), 2
            )
            grad_output.requires_grad = True
            grad_A = grad_output
        else:
            grad_A = None
        self.tensor1 = None
        return grad_A

class exp(Function):
    def forward(self, tensor1):
        out = functional.exp(tensor1)
        out.requires_grad = tensor1.requires_grad
        self.out = out
        if out.requires_grad and context.MODE == 0:
            out.creator = self
        return out

    def backward(self, grad_output):
        if self.needs_input_grad[0]:
            grad_output = functional.pointwise(grad_output, self.out, 2)
            grad_output.requires_grad = True
            grad_A = grad_output
        else:
            grad_A = None
        self.out = None
        return grad_A

class sqrt(Function):
    def forward(self, tensor1):
        out = functional.sqrt(tensor1)
        out.requires_grad = tensor1.requires_grad

        if out.requires_grad and context.MODE == 0:
            self.out = out
            out.creator = self
        return out

    def backward(self, grad_output):
        if self.needs_input_grad[0]:
            grad_output = functional.pointwise(
                grad_output, functional.scalarpointwise(self.out, 2, 2), 3
            )
            grad_output.requires_grad = True
            grad_A = grad_output
        else:
            grad_A = None
        self.out = None
        return grad_A

class rsqrt(Function):
    def forward(self, tensor1):
        out = functional.rsqrt(tensor1)
        out.requires_grad = tensor1.requires_grad

        if out.requires_grad and context.MODE == 0:
            out.creator = self
            self.tensor1 = tensor1
        return out

    def backward(self, grad_output):
        if self.needs_input_grad[0]:
            grad_output = functional.pointwise(
                functional.scalarpointwise(self.tensor1, -1.5, 3),
                functional.scalarpointwise(grad_output, -0.5, 2),
                2,
            )
            grad_output.requires_grad = True
            grad_A = grad_output
        else:
            grad_A = None
        self.tensor1 = None

        return grad_A

class log(Function):
    def forward(self, tensor1):
        out = functional.log(tensor1)
        out.requires_grad = tensor1.requires_grad
        if out.requires_grad and context.MODE == 0:
            out.creator = self
            self.tensor1 = tensor1

        return out

    def backward(self, grad_output):
        if self.needs_input_grad[0]:
            grad_output = functional.pointwise(grad_output, self.tensor1, 3)
            grad_output.requires_grad = True
            grad_A = grad_output
        else:
            grad_A = None
        self.tensor1 = None
        return grad_A

class clamp(Function):
    def forward(self, tensor1, min, max):
        out = functional.clamp(tensor1, min, max)
        out.requires_grad = tensor1.requires_grad
        self.min = min
        self.max = max

        if out.requires_grad and context.MODE == 0:
            self.tensor1 = tensor1
            out.creator = self
        return out

    def backward(self, grad_output):
        if self.needs_input_grad[0]:
            grad_output = functional.pointwise(
                grad_output,
                functional.clampbackwards(self.tensor1, self.min, self.max),
                2,
            )
            grad_output.requires_grad = True
            grad_A = grad_output
        else:
            grad_A = None
        self.tensor1 = None
        return grad_A

class decay_window(Function):
    """Causal exponential decay window with a fused backward scatter."""

    def forward(self, decay, stage, length):
        out = functional.decay_window(decay, stage, length)
        out.requires_grad = decay.requires_grad
        if out.requires_grad and context.MODE == 0:
            self.decay = decay
            self.stage = stage
            self.length = length
            out.creator = self
        return out

    def backward(self, grad_output):
        if not self.needs_input_grad[0]:
            return None
        d_decay = functional.decay_window_backward(
            grad_output, self.decay, self.stage, self.length
        )
        self.decay = None
        return d_decay
