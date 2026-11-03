"""Shared helpers for the ``webgpu_test`` suite."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import numpy as np

from wxx import Tensor, functional
from wxx.functional.utils import from_numpy


def from_np(arr):
    """Create a float32 tensor from a numpy array."""
    return from_numpy(np.asarray(arr, dtype=np.float32))


def from_np_u32(arr):
    from wxx.Tensor import Tensor

    arr = np.ascontiguousarray(arr, dtype=np.uint32)
    t = Tensor(list(arr.shape), data=arr.tobytes())
    t.type = "u32"
    t.nbytes = arr.nbytes
    return t


def to_np(t):
    return t.numpy()


def assert_close(a, b, atol=1e-4, rtol=1e-4, msg=""):
    a = np.asarray(a)
    b = np.asarray(b)
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        maxerr = float(np.max(np.abs(a - b)))
        raise AssertionError(f"{msg} maxerr={maxerr:.3g} (atol={atol})")


def rng(seed=0):
    return np.random.default_rng(seed)


def run_all(tests):
    """Run a list of (name, fn) pairs, printing PASS/FAIL."""
    ok = 0
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
            ok += 1
        except Exception as e:
            print(f"FAIL  {name}: {repr(e)}")
            failed.append((name, e))
    print(f"\n{ok}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)
    return ok, failed
