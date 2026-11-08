import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from common import from_np, to_np, assert_close, rng

from wxx import functional as F


def test_relu_gelu_sigmoid():
    a = from_np(rng(0).standard_normal((5, 7)))
    assert_close(to_np(F.relu(a)), np.maximum(to_np(a), 0))
    assert_close(to_np(F.gelu(a)), (to_np(a) * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi)*(to_np(a)+0.044715*to_np(a)**3)))), atol=1e-3)
    assert_close(to_np(F.sigmoid(a)), 1/(1+np.exp(-to_np(a))), atol=1e-5)


def test_sin_exp_log():
    a = from_np(rng(1).standard_normal((4, 6)) * 0.5)
    assert_close(to_np(F.sin(a)), np.sin(to_np(a)), atol=1e-6)
    assert_close(to_np(F.cos(a)), np.cos(to_np(a)), atol=1e-6)
    assert_close(to_np(a.sin()), np.sin(to_np(a)), atol=1e-6)
    assert_close(to_np(a.cos()), np.cos(to_np(a)), atol=1e-6)
    assert_close(to_np(F.exp(a)), np.exp(to_np(a)), atol=1e-5)
    b = from_np(np.abs(rng(2).standard_normal((4, 6))) + 0.5)
    assert_close(to_np(F.log(b)), np.log(to_np(b)), atol=1e-5)


def test_softmax():
    a = from_np(rng(3).standard_normal((3, 5)))
    out = F.softmax(a, dim=-1)
    ref = np.exp(to_np(a)) / np.exp(to_np(a)).sum(axis=-1, keepdims=True)
    assert_close(to_np(out), ref, atol=1e-5)


TESTS = [
    ("relu_gelu_sigmoid", test_relu_gelu_sigmoid),
    ("sin_exp_log", test_sin_exp_log),
    ("softmax", test_softmax),
]


if __name__ == "__main__":
    from common import run_all
    run_all(TESTS)
