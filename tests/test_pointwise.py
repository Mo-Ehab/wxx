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


TESTS = [
    ("add_sub_mul_div", test_add_sub_mul_div),
    ("scalar_ops", test_scalar_ops),
    ("unary", test_unary),
    ("broadcast", test_broadcast),
    ("strided_views", test_strided_views),
]


if __name__ == "__main__":
    from common import run_all
    run_all(TESTS)
