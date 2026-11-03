import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from common import from_np, to_np, assert_close, rng

from wxx import functional
from wxx.nn.utils import _clip_grad_norm_cpu, _grad_pairs, clip_grad_norm_
from wxx.nn.module import Module, Linear
from wxx.nn.optimizer import AdamW


def _make_module(seed=0):
    """A couple of parameters with deterministic gradients."""
    class Net(Module):
        def __init__(self):
            super().__init__()
            self.a = Linear(4, 6)
            self.b = Linear(6, 2)

        def forward(self, x):
            return self.b(self.a(x))

    net = Net()
    x = from_np(rng(seed).standard_normal((3, 4)))
    out = net(x)
    out.sum().backward()
    return net


def test_sumsq_matches_numpy():
    x = rng(0).standard_normal((7, 13)).astype(np.float32)
    t = from_np(x)
    assert_close(to_np(functional.sumsq(t)), np.array([np.sum(x * x)]), atol=1e-3)


def test_clip_grad_norm_matches_cpu():
    net = _make_module()
    pairs = _grad_pairs(net.parameters())
    snapshot = [(p, p.grad.numpy().copy()) for p, _ in pairs]

    n_cpu = _clip_grad_norm_cpu(_grad_pairs(net.parameters()), 0.5, 2.0)
    cpu_grads = {id(p): p.grad.numpy().copy() for p, _ in pairs}

    # restore the original gradients and run the GPU path
    for p, arr in snapshot:
        functional.assign(p.grad, from_np(arr))
    n_gpu = clip_grad_norm_(net.parameters(), 0.5)

    assert abs(n_cpu - n_gpu) < 1e-3, (n_cpu, n_gpu)
    for p, _ in pairs:
        assert_close(p.grad.numpy(), cpu_grads[id(p)], atol=1e-5)


def test_clip_grad_norm_noop_below_max():
    net = _make_module(seed=1)
    pairs = _grad_pairs(net.parameters())
    before = {id(p): p.grad.numpy().copy() for p, _ in pairs}

    norm = clip_grad_norm_(net.parameters(), max_norm=1e6)
    assert norm > 0
    # Gradients are unchanged, and no gradient tensor was replaced.
    for p, _ in pairs:
        assert_close(p.grad.numpy(), before[id(p)], atol=1e-6)


def test_clip_grad_norm_inf_fallback():
    net = _make_module(seed=2)
    pairs = _grad_pairs(net.parameters())
    snapshot = [(p, p.grad.numpy().copy()) for p, _ in pairs]

    n_cpu = _clip_grad_norm_cpu(_grad_pairs(net.parameters()), 0.5, float("inf"))
    cpu_grads = {id(p): p.grad.numpy().copy() for p, _ in pairs}

    for p, arr in snapshot:
        functional.assign(p.grad, from_np(arr))
    n_gpu = clip_grad_norm_(net.parameters(), 0.5, float("inf"))

    assert abs(n_cpu - n_gpu) < 1e-4, (n_cpu, n_gpu)
    for p, _ in pairs:
        assert_close(p.grad.numpy(), cpu_grads[id(p)], atol=1e-6)


TESTS = [
    ("sumsq_matches_numpy", test_sumsq_matches_numpy),
    ("clip_grad_norm_matches_cpu", test_clip_grad_norm_matches_cpu),
    ("clip_grad_norm_noop_below_max", test_clip_grad_norm_noop_below_max),
    ("clip_grad_norm_inf_fallback", test_clip_grad_norm_inf_fallback),
]


if __name__ == "__main__":
    from common import run_all
    run_all(TESTS)
