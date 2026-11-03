from .. import functional
from . import Function
from ..backend import context


class CrossEntropyLoss(Function):
    def forward(self, tensor1, tensor2):
        loss, grad = functional.cceloss(tensor1, tensor2)

        loss.requires_grad = any(self.needs_input_grad)
        if loss.requires_grad and context.MODE == 0:
            loss.creator = self
            self.grad = grad

        return loss

    def backward(self, grad_output):
        grad = functional.pointwise(self.grad, grad_output, 2)
        self.grad = None
        return grad

class Dropout(Function):
    def forward(self, tensor1, p=0.2):
        out, seed = functional.dropout(tensor1, p)

        out.requires_grad = tensor1.requires_grad

        if out.requires_grad and context.MODE == 0:
            self.p = p
            self.seed = seed
            out.creator = self
        return out

    def backward(self, grad_output):
        if not self.needs_input_grad:
            return None
        grad = functional.dropoutdx(grad_output, self.p, self.seed)

        self.p = None
        self.seed = None
        return grad


class Embedding(Function):
    def forward(self, tensor1, tensor2, n_token, embed_dim, vocab_size):
        out = functional.embed(tensor1, tensor2, n_token, embed_dim)

        out.requires_grad = any([tensor1.requires_grad, tensor2.requires_grad])

        if out.requires_grad and context.MODE == 0:
            self.tensor1 = tensor1
            self.n_token = n_token
            self.embed_dim = embed_dim
            self.vocab_size = vocab_size

            out.creator = self
        return out


    def backward(self, grad_output):

        if not self.needs_input_grad:
            return None, None

        grad = functional.embeddx(
            grad_output, self.tensor1, self.n_token, self.embed_dim, self.vocab_size
        )

        self.tensor1 = None

        return None, grad
