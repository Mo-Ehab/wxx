import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from common import from_np, to_np, assert_close, rng

from wxx.nn.module import Conv2D


def test_conv2d_forward():
    c = Conv2D(2, 3, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
    # init weights deterministically
    from wxx import functional as F
    F.assign(c.W, from_np(rng(0).standard_normal(c.W.shape).astype(np.float32) * 0.1))
    if c.bias is not None:
        F.assign(c.bias, from_np(np.zeros(c.bias.shape).astype(np.float32)))
    x = from_np(rng(1).standard_normal((2, 2, 8, 8)).astype(np.float32))
    out = c(x)
    assert out.shape == [2, 3, 8, 8]
    # just sanity: finite
    assert np.all(np.isfinite(to_np(out)))


def ref_causal_conv1d(x, w, b):
    B, D, L = x.shape
    K = w.shape[1]
    out = np.zeros_like(x)
    for k in range(K):
        out[:, :, k:] += w[:, k][None, :, None] * x[:, :, : L - k]
    return out + b[None, :, None]


def test_causal_conv1d_forward_backward():
    from wxx.functional.convolution import causal_conv1d, causal_conv1d_backward

    B, D, L, K = 3, 5, 9, 3
    x = rng(2).standard_normal((B, D, L)).astype(np.float32)
    w = rng(3).standard_normal((D, K)).astype(np.float32)
    b = rng(4).standard_normal((D,)).astype(np.float32)
    dy = rng(5).standard_normal((B, D, L)).astype(np.float32)

    out = causal_conv1d(from_np(x), from_np(w), from_np(b))
    dx, dw, db = causal_conv1d_backward(
        from_np(dy), from_np(x), from_np(w), from_np(b)
    )
    assert_close(to_np(out), ref_causal_conv1d(x, w, b))

    ref_dx = np.zeros_like(x)
    ref_dw = np.zeros_like(w)
    for k in range(K):
        ref_dx[:, :, : L - k] += w[:, k][None, :, None] * dy[:, :, k:]
        ref_dw[:, k] = np.einsum("bdl,bdl->d", dy[:, :, k:], x[:, :, : L - k])
    assert_close(to_np(dx), ref_dx)
    assert_close(to_np(dw), ref_dw)
    assert_close(to_np(db), dy.sum(axis=(0, 2)))


def test_causal_conv1d_strided_input():
    """A transposed (B, L, D) view is consumed without a contiguous copy."""
    from wxx.functional.convolution import causal_conv1d

    B, D, L, K = 2, 4, 7, 3
    x = rng(6).standard_normal((B, L, D)).astype(np.float32)
    w = rng(7).standard_normal((D, K)).astype(np.float32)
    b = rng(8).standard_normal((D,)).astype(np.float32)

    view = from_np(x).permute(0, 2, 1)
    assert not view.is_contiguous()
    out = causal_conv1d(view, from_np(w), from_np(b))
    assert_close(to_np(out), ref_causal_conv1d(np.transpose(x, (0, 2, 1)), w, b))


TESTS = [
    ("conv2d_forward", test_conv2d_forward),
    ("causal_conv1d_forward_backward", test_causal_conv1d_forward_backward),
    ("causal_conv1d_strided_input", test_causal_conv1d_strided_input),
]


if __name__ == "__main__":
    from common import run_all
    run_all(TESTS)
