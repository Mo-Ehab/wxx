import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from common import from_np, to_np, assert_close, rng

from wxx import functional as F
from wxx.nn.module import Linear
from wxx import Tensor


def gradcheck_param(model_fn, param, idx, x, w, eps=1e-2):
    # model_fn() returns scalar loss tensor
    base = param.numpy().copy()
    a = base.copy(); a[idx] += eps
    F.assign(param, from_np(a))
    lp = model_fn().item()
    a = base.copy(); a[idx] -= eps
    F.assign(param, from_np(a))
    lm = model_fn().item()
    F.assign(param, from_np(base))
    param.grad = None
    model_fn().backward()
    analytic = float(param.grad.numpy()[idx])
    numeric = (lp - lm) / (2 * eps)
    rel = abs(analytic - numeric) / max(1e-4, abs(analytic) + abs(numeric))
    assert rel < 5e-2, (idx, analytic, numeric, rel)


def test_linear_gradcheck():
    lin = Linear(3, 2)
    x_np = rng(0).standard_normal((4, 3)).astype(np.float32)
    w_np = rng(1).standard_normal((4, 2)).astype(np.float32)
    x = from_np(x_np)
    w = from_np(w_np)

    def loss_fn():
        return (lin(x) * w).sum()

    loss_fn().backward()
    for idx in [(0, 0), (1, 1), (2, 0)]:
        gradcheck_param(loss_fn, lin.weight, idx, x, w)
    gradcheck_param(loss_fn, lin.bias, (1,), x, w)


def test_activation_gradcheck():
    a = from_np(rng(2).standard_normal((3, 4)).astype(np.float32))
    a.requires_grad = True
    w = from_np(rng(3).standard_normal((3, 4)).astype(np.float32))

    def loss_fn():
        return (a.gelu() * w).sum()

    loss_fn().backward()
    # finite diff on input
    base = a.numpy().copy()
    eps = 1e-3
    analytic = a.grad.numpy().copy()
    for flat in [0, 5, 11]:
        i, j = divmod(flat, 4)
        arr = base.copy(); arr[i, j] += eps
        F.assign(a, from_np(arr))
        lp = (a.gelu() * w).sum().item()
        arr = base.copy(); arr[i, j] -= eps
        F.assign(a, from_np(arr))
        a.requires_grad = True
        lm = (a.gelu() * w).sum().item()
        F.assign(a, from_np(base))
        a.requires_grad = True
        numeric = (lp - lm) / (2 * eps)
        rel = abs(analytic[i, j] - numeric) / max(1e-4, abs(analytic[i, j]) + abs(numeric))
        assert rel < 5e-2, (i, j, analytic[i, j], numeric, rel)


def _gradcheck_unary(fn, seed):
    """Finite-difference check of an elementwise op's input gradient."""
    a = from_np(rng(seed).standard_normal((3, 4)).astype(np.float32))
    a.requires_grad = True
    w = from_np(rng(seed + 1).standard_normal((3, 4)).astype(np.float32))

    (fn(a) * w).sum().backward()
    analytic = a.grad.numpy().copy()

    base = a.numpy().copy()
    eps = 1e-3
    for flat in [0, 5, 11]:
        i, j = divmod(flat, 4)
        arr = base.copy(); arr[i, j] += eps
        F.assign(a, from_np(arr))
        lp = (fn(a) * w).sum().item()
        arr = base.copy(); arr[i, j] -= eps
        F.assign(a, from_np(arr))
        lm = (fn(a) * w).sum().item()
        F.assign(a, from_np(base))
        numeric = (lp - lm) / (2 * eps)
        rel = abs(analytic[i, j] - numeric) / max(1e-4, abs(analytic[i, j]) + abs(numeric))
        assert rel < 5e-2, (i, j, analytic[i, j], numeric, rel)


def test_exp_sqrt_gradcheck():
    """``exp``/``sqrt`` backward used to call ``self.out`` as a function."""
    _gradcheck_unary(lambda t: t.exp(), 4)
    # sqrt needs a positive input; shift and clamp the weights to keep the
    # finite difference away from the singularity at 0.
    a = from_np(np.abs(rng(6).standard_normal((3, 4))).astype(np.float32) + 0.5)
    a.requires_grad = True
    w = from_np(rng(7).standard_normal((3, 4)).astype(np.float32))
    (a.sqrt() * w).sum().backward()
    analytic = a.grad.numpy().copy()
    base = a.numpy().copy()
    eps = 1e-3
    for flat in [0, 5, 11]:
        i, j = divmod(flat, 4)
        arr = base.copy(); arr[i, j] += eps
        F.assign(a, from_np(arr))
        lp = (a.sqrt() * w).sum().item()
        arr = base.copy(); arr[i, j] -= eps
        F.assign(a, from_np(arr))
        lm = (a.sqrt() * w).sum().item()
        F.assign(a, from_np(base))
        numeric = (lp - lm) / (2 * eps)
        rel = abs(analytic[i, j] - numeric) / max(1e-4, abs(analytic[i, j]) + abs(numeric))
        assert rel < 5e-2, (i, j, analytic[i, j], numeric, rel)


def test_slice_backward_scatter():
    """narrow/getitem backward fills zeros and scatters in one kernel."""
    # narrow along a middle dim, including a non-zero start
    x = from_np(rng(8).standard_normal((2, 5, 4)).astype(np.float32))
    x.requires_grad = True
    w = from_np(rng(9).standard_normal((2, 2, 4)).astype(np.float32))
    (x.slice(1, 1, 2) * w).sum().backward()
    ref = np.zeros((2, 5, 4), dtype=np.float32)
    ref[:, 1:3, :] = w.numpy()
    assert_close(x.grad.numpy(), ref)

    # basic slicing on the last dim: x[..., :3]
    y = from_np(rng(10).standard_normal((3, 6)).astype(np.float32))
    y.requires_grad = True
    g = from_np(rng(11).standard_normal((3, 3)).astype(np.float32))
    (y[:, :3] * g).sum().backward()
    ref = np.zeros((3, 6), dtype=np.float32)
    ref[:, :3] = g.numpy()
    assert_close(y.grad.numpy(), ref)

    # int indexing still falls back to the generic path
    z = from_np(rng(12).standard_normal((4, 5)).astype(np.float32))
    z.requires_grad = True
    h = from_np(rng(13).standard_normal((5,)).astype(np.float32))
    (z[2] * h).sum().backward()
    ref = np.zeros((4, 5), dtype=np.float32)
    ref[2] = h.numpy()
    assert_close(z.grad.numpy(), ref)


TESTS = [
    ("linear_gradcheck", test_linear_gradcheck),
    ("activation_gradcheck", test_activation_gradcheck),
    ("exp_sqrt_gradcheck", test_exp_sqrt_gradcheck),
    ("slice_backward_scatter", test_slice_backward_scatter),
]


if __name__ == "__main__":
    from common import run_all
    run_all(TESTS)
