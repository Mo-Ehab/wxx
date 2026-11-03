"""The fast bind-group path must be numerically identical to wgpu-py's.

``wxx.backend.bindgroup`` replaces ``device.create_bind_group`` with a version
that reuses pre-allocated cffi structs.  These tests run the same ops through
both implementations and compare the results bit for bit.
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import wgpu

from common import from_np, assert_close, rng

from wxx import functional as F
from wxx.nn.module import Linear
from wxx.nn.optimizer import AdamW
from wxx.backend.device import get_device, get_ctx


def _reference_create_bind_group():
    """wgpu-py's original implementation, bound to our device."""
    from wgpu.backends.wgpu_native._api import GPUDevice

    return GPUDevice.create_bind_group.__get__(get_device())


def _with_path(fast, fn):
    """Run ``fn`` with the fast (or reference) bind-group implementation."""
    device = get_device()
    original = _reference_create_bind_group()
    previous = device.create_bind_group
    device.create_bind_group = previous if fast else original
    try:
        return fn()
    finally:
        device.create_bind_group = previous


def test_fast_bind_groups_installed():
    from wgpu.backends.wgpu_native._api import GPUDevice

    ctx = get_ctx()
    assert ctx.fast_bind_groups is True, "fast bind groups were not installed"
    bound = get_device().create_bind_group
    # The fast path installs a plain function on the instance; the reference is
    # the bound class method.
    assert getattr(bound, "__func__", None) is not GPUDevice.create_bind_group


def test_pointwise_matches_reference():
    a = rng(0).standard_normal((7, 5)).astype(np.float32)
    b = rng(1).standard_normal((7, 5)).astype(np.float32)

    def run():
        x, y = from_np(a), from_np(b)
        out = (x * y + x) / 2.0
        return out.numpy()

    fast = _with_path(True, run)
    slow = _with_path(False, run)
    assert_close(fast, slow, atol=0.0)


def test_broadcast_matches_reference():
    a = rng(2).standard_normal((3, 1, 5)).astype(np.float32)
    b = rng(3).standard_normal((1, 4, 5)).astype(np.float32)

    def run():
        x, y = from_np(a), from_np(b)
        return (x + y).numpy()

    assert_close(_with_path(True, run), _with_path(False, run), atol=0.0)


def test_strided_binding_matches_reference():
    """Transposed views are bound through offset/size/stride metadata."""

    def run():
        x = from_np(rng(4).standard_normal((4, 6)).astype(np.float32))
        w = from_np(rng(5).standard_normal((6, 3)).astype(np.float32))
        return (x @ w).numpy()

    assert_close(_with_path(True, run), _with_path(False, run), atol=0.0)


def test_reduction_matches_reference():
    a = rng(6).standard_normal((9, 4)).astype(np.float32)

    def run():
        return from_np(a).sum().numpy()

    assert_close(_with_path(True, run), _with_path(False, run), atol=0.0)


def test_optimizer_matches_reference():
    """AdamW touches one bind group per parameter per step."""

    def run():
        lin = Linear(4, 3)
        F.assign(lin.weight, from_np(rng(7).standard_normal((4, 3)).astype(np.float32)))
        x = from_np(rng(8).standard_normal((5, 4)).astype(np.float32))
        optim = AdamW(list(lin.parameters()), lr=0.05, weight_decay=0.01)
        for _ in range(3):
            optim.zero_grad(wait=False)
            lin(x).sum().backward()
            optim.step()
        return lin.weight.numpy()

    assert_close(_with_path(True, run), _with_path(False, run), atol=0.0)


TESTS = [
    ("fast_bind_groups_installed", test_fast_bind_groups_installed),
    ("pointwise_matches_reference", test_pointwise_matches_reference),
    ("broadcast_matches_reference", test_broadcast_matches_reference),
    ("strided_binding_matches_reference", test_strided_binding_matches_reference),
    ("reduction_matches_reference", test_reduction_matches_reference),
    ("optimizer_matches_reference", test_optimizer_matches_reference),
]


if __name__ == "__main__":
    from common import run_all

    run_all(TESTS)
