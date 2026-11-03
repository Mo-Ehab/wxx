from .. import functional
from . import Function
from ..backend import context


class Conv2D(Function):
    def forward(
        self, tensor1, W, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1)
    ):

        self.Xshape = tensor1.shape
        self.wshape = W.shape
        self.tensor1 = tensor1
        self.W = W
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        out = functional.conv2d(tensor1, W, stride, padding, dilation, bias)
        out.requires_grad = any(self.needs_input_grad)
        if out.requires_grad and context.MODE == 0:
            out.creator = self
        return out

    def backward(self, grad_output):

        dw = functional.conv2ddw(
            grad_output,
            self.tensor1,
            self.stride,
            self.padding,
            self.dilation,
            self.Xshape,
            self.wshape,
        )
        dx = functional.conv2ddx(
            grad_output,
            self.W,
            self.stride,
            self.padding,
            self.dilation,
            self.Xshape,
            self.wshape,
        )

        self.W = None
        self.tensor1 = None

        if self.bias is None:
            self.bias = None
            return dx, dw
        if len(grad_output.shape) == 4:
            db = grad_output.sum(axis=(0, 2, 3))
        else:
            db = grad_output.sum(axis=0)
        self.bias = None
        return dx, dw, db

class maxpooling2d(Function):
    def forward(self, tensor1, kernel, stride, padding):

        out, indices = functional.maxpool2d(tensor1, kernel, stride, padding)
        out.requires_grad = tensor1.requires_grad

        if out.requires_grad and context.MODE == 0:
            self.input_shape = tuple(tensor1.shape)
            self.tensor1 = tensor1
            self.indices = indices
            self.kernel = kernel
            self.stride = stride
            self.padding = padding
            out.creator = self
        return out

    def backward(self, grad_output):
        if not self.needs_input_grad[0]:
            return None

        grad_input = functional.maxpool2ddx(
            grad_output,
            self.tensor1,
            self.indices,
            self.kernel,
            self.stride,
            self.padding,
        )

        self.tensor1 = None
        self.indices = None

        return grad_input

class avgpooling2d(Function):
    def forward(self, tensor1, kernel, stride, padding):

        out = functional.avgpool2d(tensor1, kernel, stride, padding)
        out.requires_grad = tensor1.requires_grad

        if out.requires_grad and context.MODE == 0:
            self.input_shape = tuple(tensor1.shape)
            self.tensor1 = tensor1
            self.kernel = kernel
            self.stride = stride
            self.padding = padding
            out.creator = self
        return out

    def backward(self, grad_output):
        if not self.needs_input_grad[0]:
            return None

        grad_input = functional.avgpool2ddx(
            grad_output, self.tensor1, self.kernel, self.stride, self.padding
        )

        self.tensor1 = None
        self.indices = None

        return grad_input

class adaptive_avgpooling2d(Function):
    def forward(self, tensor1, output_size):
        out = functional.adaptive_avgpool2d(tensor1, output_size)
        out.requires_grad = tensor1.requires_grad

        if out.requires_grad and context.MODE == 0:
            self.input_shape = tuple(tensor1.shape)
            self.output_size = output_size
            out.creator = self
        return out

    def backward(self, grad_output):
        if not self.needs_input_grad[0]:
            return None
        grad_input = functional.adaptive_avgpool2ddx(
            grad_output, self.input_shape, self.output_size
        )
        self.input_shape = None
        return grad_input


class causal_conv1d(Function):
    """Depthwise causal 1-D convolution (fused forward + backward kernels)."""

    def forward(self, tensor1, weight, bias):
        out = functional.causal_conv1d(tensor1, weight, bias)
        out.requires_grad = any(self.needs_input_grad)
        if out.requires_grad and context.MODE == 0:
            self.tensor1 = tensor1
            self.weight = weight
            self.bias = bias
            out.creator = self
        return out

    def backward(self, grad_output):
        need_weight = self.needs_input_grad[1] or self.needs_input_grad[2]
        dx, dw, db = functional.causal_conv1d_backward(
            grad_output,
            self.tensor1,
            self.weight,
            self.bias,
            need_weight_grad=need_weight,
        )

        self.tensor1 = None
        self.weight = None
        self.bias = None

        return (
            dx if self.needs_input_grad[0] else None,
            dw if self.needs_input_grad[1] else None,
            db if self.needs_input_grad[2] else None,
        )
