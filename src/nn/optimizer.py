


from ..backend.device import get_device, get_cmd_buffer
from .. import functional


device = get_device()
cmd_encoder = get_cmd_buffer()
class Optimizer:
    def __init__(self, parameters):
        self.parameters = list(parameters)

    def step(self):
        pass

    def zero_grad(self, wait=False):
        for _, weight in self.parameters:
            if weight.grad is not None:
                weight.grad = None

        cmd_encoder.flush(wait=wait)




class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01):
        super().__init__( parameters)

        self.lr = lr

    def step(self):
        for parameter, weight in self.parameters:
            functional._pointwise(weight, - self.lr * weight.grad , 0)


class AdamW(Optimizer):
    def __init__(self, parameters, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-7, weight_decay=1e-4):
        from ..Tensor import Tensor

        super().__init__(parameters)
        self.params = self.parameters
        self.lr = float(lr)
        self.beta1, self.beta2 = float(beta1), float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.step_count = 0


        self.state = []
        for _, p in self.params:
            self.state.append({
                "exp_avg": Tensor(p.shape, value=0),
                "exp_avg_sq": Tensor(p.shape, value=0)
            })
        self.lr = lr

    def step(self):
        self.step_count += 1

        bc1 = 1.0 - (self.beta1 ** self.step_count)
        bc2 = 1.0 - (self.beta2 ** self.step_count)


        for (_, weight), state in zip(self.params, self.state):

            if weight.grad is None:
                continue

            functional.adamw(
                weight,
                weight.grad,
                state["exp_avg"],
                state["exp_avg_sq"],
                self.step_count,
                self.lr,
                self.beta1,
                self.beta2,
                self.eps,
                self.weight_decay,
                bc1,
                bc2,
            )
