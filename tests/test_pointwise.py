import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from common import from_np, to_np, assert_close, rng

from wxx import functional as F


def make():
    a = from_np(rng(0).standard_normal((5, 7)))
    b = from_np(rng(1).standard_normal((5, 7)))
    return a, b


def test_add_sub_mul_div():
    a, b = make()
    assert_close(to_np(a + b), to_np(a) + to_np(b))
    assert_close(to_np(a - b), to_np(a) - to_np(b))
    assert_close(to_np(a * b), to_np(a) * to_np(b))
    assert_close(to_np(a / b), to_np(a) / to_np(b))


def test_scalar_ops():
    a, _ = make()
    s = 2.5
    assert_close(to_np(a + s), to_np(a) + s)
    assert_close(to_np(a * s), to_np(a) * s)
    assert_close(to_np(a - s), to_np(a) - s)
    assert_close(to_np(a / s), to_np(a) / s)


def test_pow_and_reverse_scalars():
    """``pow``/``**`` and scalar-first ``*``, ``/``, ``**`` overloads."""
    arr = np.array(
        [[-2.0, -0.5, 0.5, 2.0], [0.25, 1.5, 3.0, 4.0]], dtype=np.float32
    )
    a = from_np(arr)

    # Tensor methods and both operand orders.
    assert_close(to_np(a.pow(3)), arr ** 3, atol=1e-5)
    assert_close(to_np(a ** 3), arr ** 3, atol=1e-5)
    assert_close(to_np(a ** 2), arr ** 2)          # square fast path
    assert_close(to_np(a ** -1), arr ** -1, atol=1e-5)

    assert_close(to_np(2 * a), 2.0 * arr)
    assert_close(to_np(a * 2), arr * 2.0)
    assert_close(to_np(2 / a), 2.0 / arr, atol=1e-6)
    assert_close(to_np(a / 2), arr / 2.0)

    assert_close(to_np(2 ** a), 2.0 ** arr, atol=1e-5)
    assert_close(to_np(2.5 ** a), 2.5 ** arr, atol=1e-5)

    # Negative bases: integer exponents stay real, fractional ones are NaN.
    neg = from_np(np.array([[-2.0, -3.0, 2.0, 3.0]], dtype=np.float32))
    assert_close(to_np(neg ** 3), np.array([[-8.0, -27.0, 8.0, 27.0]]))
    frac = to_np(neg ** 0.5)
    assert np.isnan(frac[0, :2]).all()
    assert_close(frac[0, 2:], np.sqrt(np.array([2.0, 3.0])))

    # Tensor exponents and non-positive bases are still rejected.
    for bad in (lambda: a ** a, lambda: 0 ** a, lambda: (-2) ** a):
        try:
            bad()
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid power overload was accepted")


def test_unary():
    a = from_np(np.abs(rng(0).standard_normal((5, 7))).astype(np.float32))
    assert_close(to_np(a.abs()), np.abs(to_np(a)))
    assert_close(to_np(a.exp()), np.exp(to_np(a)))
    assert_close(to_np(a.sqrt()), np.sqrt(to_np(a)))
    assert_close(to_np(a.square()), to_np(a) ** 2)
    assert_close(to_np(-a), -to_np(a))


def test_broadcast():
    a = from_np(rng(2).standard_normal((3, 1)))
    b = from_np(rng(3).standard_normal((1, 4)))
    c = a * b
    assert c.shape == [3, 4]
    assert_close(to_np(c), to_np(a) * to_np(b))


def test_strided_views():
    """Non-contiguous operands are addressed by offset/stride, no copies."""
    x = rng(4).standard_normal((4, 6, 8)).astype(np.float32)
    y = rng(5).standard_normal((4, 6, 8)).astype(np.float32)
    xt = from_np(x).permute(0, 2, 1)
    yt = from_np(y).permute(0, 2, 1)
    assert not xt.is_contiguous()

    assert_close(to_np(xt * yt), np.transpose(x, (0, 2, 1)) * np.transpose(y, (0, 2, 1)))

    # offset view (contiguous but non-zero base offset)
    assert_close(to_np(from_np(x)[1:] + from_np(y)[1:]), x[1:] + y[1:])

    # stepped slice
    assert_close(to_np(from_np(x)[:, ::2, :] * 3.0), x[:, ::2, :] * 3.0)

    # strided broadcast column
    col = from_np(x).permute(0, 2, 1)[:, :, :1]
    assert_close(
        to_np(col + from_np(y).permute(0, 2, 1)),
        np.transpose(x, (0, 2, 1))[:, :, :1] + np.transpose(y, (0, 2, 1)),
    )


def test_aliased_operands():
    """In-place accumulation must not bind one buffer as RW and RO."""
    # ``add`` hands the same grad object to both operands of ``x + x``.
    a = from_np(rng(20).standard_normal((3, 4)))
    a.requires_grad = True
    (a + a).sum().backward()
    assert_close(to_np(a.grad), np.full((3, 4), 2.0, dtype=np.float32))

    # The same object can also reach one tensor from two branches.
    b = from_np(rng(21).standard_normal((3, 4)))
    b.requires_grad = True
    ((b + 0.0) + (b + 0.0)).sum().backward()
    assert_close(to_np(b.grad), np.full((3, 4), 2.0, dtype=np.float32))

    # Directly aliased operands: A += A.
    c = from_np(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    F._pointwise(c, c, 0)
    assert_close(to_np(c), np.array([2.0, 4.0, 6.0], dtype=np.float32))


TESTS = [
    ("add_sub_mul_div", test_add_sub_mul_div),
    ("scalar_ops", test_scalar_ops),
    ("pow_and_reverse_scalars", test_pow_and_reverse_scalars),
    ("unary", test_unary),
    ("broadcast", test_broadcast),
    ("strided_views", test_strided_views),
    ("aliased_operands", test_aliased_operands),
]


if __name__ == "__main__":
    from common import run_all
    run_all(TESTS)
