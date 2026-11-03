import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from common import from_np, to_np, assert_close, rng

from wxx import functional as F


def test_sum_mean():
    a = from_np(rng(0).standard_normal((4, 5)))
    assert_close(to_np(F.sum(a)), to_np(a).sum())
    assert_close(to_np(F.sum(a, axis=0)), to_np(a).sum(axis=0))
    assert_close(to_np(F.sum(a, axis=1, keepdims=True)), to_np(a).sum(axis=1, keepdims=True))
    assert_close(to_np(F.mean(a, axis=-1)), to_np(a).mean(axis=-1))


def test_max_min():
    a = from_np(rng(1).standard_normal((3, 4)))
    vals, idxs = F.max(a, axis=1)
    assert_close(to_np(vals), to_np(a).max(axis=1))
    assert np.array_equal(to_np(idxs).astype(np.int64), to_np(a).argmax(axis=1).astype(np.int64))
    vals2, idxs2 = F.min(a, axis=0)
    assert_close(to_np(vals2), to_np(a).min(axis=0))


TESTS = [
    ("sum_mean", test_sum_mean),
    ("max_min", test_max_min),
]


if __name__ == "__main__":
    from common import run_all
    run_all(TESTS)
