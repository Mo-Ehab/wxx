import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from common import from_np, to_np, assert_close, rng

from wxx import functional as F
from wxx.nn.module import AdaptiveAvgPool2d


def ref_adaptive_avgpool2d(x, output_size):
    h_out, w_out = output_size
    b, c, h_in, w_in = x.shape
    out = np.zeros((b, c, h_out, w_out), dtype=np.float64)
    for oh in range(h_out):
        h0 = (oh * h_in) // h_out
        h1 = -(-((oh + 1) * h_in) // h_out)
        for ow in range(w_out):
            w0 = (ow * w_in) // w_out
            w1 = -(-((ow + 1) * w_in) // w_out)
            out[:, :, oh, ow] = x[:, :, h0:h1, w0:w1].mean(axis=(2, 3))
    return out


def ref_adaptive_avgpool2ddx(grad, input_shape, output_size):
    h_out, w_out = output_size
    b, c, h_in, w_in = input_shape
    dx = np.zeros(input_shape, dtype=np.float64)
    for oh in range(h_out):
        h0 = (oh * h_in) // h_out
        h1 = -(-((oh + 1) * h_in) // h_out)
        for ow in range(w_out):
            w0 = (ow * w_in) // w_out
            w1 = -(-((ow + 1) * w_in) // w_out)
            count = (h1 - h0) * (w1 - w0)
            dx[:, :, h0:h1, w0:w1] += (
                grad[:, :, oh, ow][:, :, None, None] / count
            )
    return dx


def test_adaptive_avgpool2d_forward():
    x = rng(0).standard_normal((2, 3, 5, 7)).astype(np.float32)
    for output_size in [(1, 1), (2, 3), (5, 7), (3, 4)]:
        out = F.adaptive_avgpool2d(from_np(x), output_size)
        assert out.shape == [2, 3, output_size[0], output_size[1]]
        assert_close(
            to_np(out),
            ref_adaptive_avgpool2d(x, output_size),
            msg=f"output_size={output_size}",
        )


def test_adaptive_avgpool2d_backward():
    x = rng(1).standard_normal((2, 2, 4, 6)).astype(np.float32)
    dy = rng(2).standard_normal((2, 2, 3, 4)).astype(np.float32)
    dx = F.adaptive_avgpool2ddx(from_np(dy), x.shape, (3, 4))
    assert_close(to_np(dx), ref_adaptive_avgpool2ddx(dy, x.shape, (3, 4)))


def test_adaptive_avgpool2d_module_backward():
    x = from_np(rng(3).standard_normal((2, 3, 8, 9)).astype(np.float32))
    x.requires_grad = True
    pool = AdaptiveAvgPool2d((2, 3))
    out = pool(x)
    out.sum().backward()
    assert x.grad is not None
    ref = ref_adaptive_avgpool2ddx(
        np.ones((2, 3, 2, 3), dtype=np.float32), (2, 3, 8, 9), (2, 3)
    )
    assert_close(to_np(x.grad), ref)


TESTS = [
    ("adaptive_avgpool2d_forward", test_adaptive_avgpool2d_forward),
    ("adaptive_avgpool2d_backward", test_adaptive_avgpool2d_backward),
    (
        "adaptive_avgpool2d_module_backward",
        test_adaptive_avgpool2d_module_backward,
    ),
]

if __name__ == "__main__":
    from common import run_all
    run_all(TESTS)
