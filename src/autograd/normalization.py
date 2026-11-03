from .. import functional
from . import Function
from ..backend import context


class rmsnorm(Function):
    def forward(self, tensor1, gamma, axis=None):

        out, irms = functional.rmsnorm(tensor1, gamma, axis)
        out.requires_grad = any([tensor1.requires_grad, gamma.requires_grad])
        self.axis = axis

        if out.requires_grad and context.MODE == 0:
            self.tensor1 = tensor1
            self.irms = irms
            self.gamma = gamma
            out.creator = self

        return out

    def backward(self, grad_output):

        grad_A = functional.rmsnormdx(
            grad_output, self.tensor1, self.gamma, self.irms, self.axis
        )
        grad_g = functional.rmsnormdg(
            grad_output, self.tensor1, self.gamma, self.irms, self.axis
        )
        self.tensor1 = None
        self.irms = None
        self.gamma = None

        return grad_A, grad_g

class meanvarnorm(Function):
    def forward(self, tensor1, gamma, beta, axis=None):

        out, mean, var = functional.meanvarnorm(tensor1, gamma, beta, axis)
        out.requires_grad = any(
            [tensor1.requires_grad, gamma.requires_grad, beta.requires_grad]
        )
        self.axis = axis

        if out.requires_grad and context.MODE == 0:
            self.tensor1 = tensor1
            self.gamma = gamma
            self.mean = mean
            self.var = var
            out.creator = self

        return out

    def backward(self, grad_output):

        grad_A = functional.meanvarnormdx(
            grad_output, self.tensor1, self.gamma, self.mean, self.var, self.axis
        )

        grad_g, grad_b = functional.meanvarnormdg(
            grad_output, self.tensor1, self.mean, self.var, self.axis
        )
        self.tensor1 = None
        self.mean = None
        self.var = None

        return grad_A, grad_g, grad_b

class batchnorm(Function):
    def forward(
        self, tensor1, gamma, beta, running_mean, running_var, training=False
    ):

        out, mean, var = functional.batchnorm(
            tensor1, gamma, beta, running_mean, running_var, training
        )
        out.requires_grad = any(
            [tensor1.requires_grad, gamma.requires_grad, beta.requires_grad]
        )

        if out.requires_grad and context.MODE == 0:
            self.tensor1 = tensor1
            self.gamma = gamma
            self.mean = mean
            self.var = var
            out.creator = self

        return out

    def backward(self, grad_output):

        grad_g, grad_b = functional.batchnormdg(
            grad_output, self.tensor1, self.mean, self.var
        )
        grad_A = functional.batchnormdx(
            grad_output,
            self.tensor1,
            self.gamma,
            grad_g,
            grad_b,
            self.mean,
            self.var,
        )

        self.tensor1 = None
        self.mean = None
        self.var = None

        return grad_A, grad_g, grad_b
