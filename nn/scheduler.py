"""Learning-rate schedulers.

The optimizers in :mod:`wxx.nn` expose a scalar ``lr`` attribute, so every
scheduler here updates ``optimizer.lr`` directly.  Calling :meth:`step` once
per optimizer update follows the usual PyTorch ordering::

    scheduler = WarmupCosineLR(optim, warmup_steps=100, total_steps=5000)
    for batch in loader:
        ...
        optim.step()
        scheduler.step()
"""

import math


class LRScheduler:
    """Base class for scalar learning-rate schedules.

    ``base_lr`` is captured from ``optimizer.lr`` when the scheduler is
    constructed.  The constructor advances to epoch/step ``0`` (and writes the
    corresponding learning rate), matching :mod:`torch.optim.lr_scheduler`.
    """

    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        self.base_lr = float(optimizer.lr)
        self.last_epoch = last_epoch
        self._last_lr = self.base_lr
        self.step()

    def get_lr(self):
        """Return the learning rate for ``self.last_epoch``."""
        return self.base_lr

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = int(epoch)
        self._last_lr = float(self.get_lr())
        self.optimizer.lr = self._last_lr
        return self._last_lr

    def get_last_lr(self):
        """Return the learning rate written by the most recent ``step``."""
        return self._last_lr

    def state_dict(self):
        return {
            "base_lr": self.base_lr,
            "last_epoch": self.last_epoch,
            "_last_lr": self._last_lr,
        }

    def load_state_dict(self, state):
        self.base_lr = float(state["base_lr"])
        self.last_epoch = int(state["last_epoch"])
        self._last_lr = float(state["_last_lr"])
        self.optimizer.lr = self._last_lr


class LambdaLR(LRScheduler):
    """Multiply the base learning rate by ``lr_lambda(step)``."""

    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        if not callable(lr_lambda):
            raise TypeError("lr_lambda must be callable")
        self.lr_lambda = lr_lambda
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return self.base_lr * float(self.lr_lambda(self.last_epoch))


class StepLR(LRScheduler):
    """Decay by ``gamma`` every ``step_size`` steps."""

    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        if step_size < 1:
            raise ValueError(f"step_size must be >= 1, got {step_size}")
        self.step_size = int(step_size)
        self.gamma = float(gamma)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return self.base_lr * self.gamma ** (self.last_epoch // self.step_size)


class ExponentialLR(LRScheduler):
    """Decay by ``gamma`` on every step."""

    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = float(gamma)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return self.base_lr * self.gamma**self.last_epoch


class CosineAnnealingLR(LRScheduler):
    """Cosine anneal from ``base_lr`` to ``eta_min`` over ``T_max`` steps."""

    def __init__(self, optimizer, T_max, eta_min=0.0, last_epoch=-1):
        if T_max < 1:
            raise ValueError(f"T_max must be >= 1, got {T_max}")
        self.T_max = int(T_max)
        self.eta_min = float(eta_min)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = min(self.last_epoch, self.T_max)
        factor = 0.5 * (1.0 + math.cos(math.pi * t / self.T_max))
        return self.eta_min + (self.base_lr - self.eta_min) * factor


class WarmupCosineLR(LRScheduler):
    """Linear warmup followed by cosine decay to ``min_lr``.

    With ``warmup_steps=100`` the first update runs at ``base_lr / 100`` and
    the 100th update reaches ``base_lr``; subsequent updates follow a true
    cosine (in radians) down to ``min_lr`` at ``total_steps``.
    """

    def __init__(
        self, optimizer, warmup_steps, total_steps, min_lr=0.0, last_epoch=-1
    ):
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if total_steps < 1:
            raise ValueError(f"total_steps must be >= 1, got {total_steps}")
        if total_steps <= warmup_steps:
            raise ValueError(
                "total_steps must be greater than warmup_steps "
                f"({total_steps} <= {warmup_steps})"
            )
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps

        if self.warmup_steps:
            t = step - self.warmup_steps + 1
            span = self.total_steps - self.warmup_steps
        else:
            t = step
            span = self.total_steps - 1
        progress = min(1.0, t / max(1, span))
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * factor


__all__ = [
    "LRScheduler",
    "LambdaLR",
    "StepLR",
    "ExponentialLR",
    "CosineAnnealingLR",
    "WarmupCosineLR",
]
