import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from common import from_np, to_np, assert_close, rng

from wxx import functional as F
from wxx import functional
from wxx.nn.module import LayerNorm, RMSNorm
from wxx import Tensor


def test_layernorm_module():
    import torch
    ln = LayerNorm(4)
    # set weights to 1/0 for comparability
    functional.assign(ln.weightg, from_np(np.ones(4)))
    functional.assign(ln.weightb, from_np(np.zeros(4)))
    x = from_np(rng(0).standard_normal((2, 3, 4)).astype(np.float32))
    out = ln(x)
    ref = (to_np(x) - to_np(x).mean(axis=-1, keepdims=True)) / np.sqrt(to_np(x).var(axis=-1, keepdims=True) + 1e-5)
    assert_close(to_np(out), ref, atol=1e-4)


def test_rmsnorm_module():
    rn = RMSNorm(4)
    functional.assign(rn.weight, from_np(np.ones(4)))
    x = from_np(rng(1).standard_normal((2, 4)).astype(np.float32))
    out = rn(x)
    ref = to_np(x) / np.sqrt((to_np(x) ** 2).mean(axis=-1, keepdims=True) + 1e-6)
    assert_close(to_np(out), ref, atol=1e-4)


TESTS = [
    ("layernorm_module", test_layernorm_module),
    ("rmsnorm_module", test_rmsnorm_module),
]


if __name__ == "__main__":
    from common import run_all
    run_all(TESTS)
