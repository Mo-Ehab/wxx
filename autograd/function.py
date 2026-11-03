
from .. import Tensor
from ..backend import context

class Function:
    """Function that supports autograd"""

    def __init__(self, *inputs):
        if context.MODE == 0:
            self.inputs = [input if not None else None for input in inputs]
            self.needs_input_grad = [
                input.requires_grad if isinstance(input, Tensor) and not None else False
                for input in inputs
            ]
        else:
            self.inputs = [input if not None else None for input in inputs]
            self.needs_input_grad = [
                False if isinstance(input, Tensor) and not None else False
                for input in inputs
            ]

        NotImplemented

    def forward(self, *args):
        NotImplemented

    def backward(self, grad_output):
        NotImplemented
