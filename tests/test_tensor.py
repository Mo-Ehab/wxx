import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from common import from_np, to_np, assert_close, rng

from wxx import Tensor, functional


def test_basic_construction():
    t = Tensor((2, 3), value=5.0)
    assert t.shape == [2, 3]
    assert t.type == "f32"
    assert_close(to_np(t), np.full((2, 3), 5.0))


def test_from_numpy_roundtrip():
    a = rng(0).standard_normal((4, 5)).astype(np.float32)
    t = from_np(a)
    assert_close(to_np(t), a)


def test_arithmetic_broadcast():
    a = from_np(rng(1).standard_normal((3, 4)))
    b = from_np(rng(2).standard_normal((4,)))
    c = a + b
    assert c.shape == [3, 4]
    assert_close(to_np(c), to_np(a) + to_np(b))


def test_reshape_permute_contiguous():
    a = from_np(rng(3).standard_normal((2, 3, 4)))
    r = a.reshape((6, 4))
    assert r.shape == [6, 4]
    assert_close(to_np(r), to_np(a).reshape(6, 4))
    p = a.permute(2, 0, 1)
    assert p.shape == [4, 2, 3]
    assert_close(to_np(p), np.transpose(to_np(a), (2, 0, 1)))


def test_permute_tuple_argument_backward():
    """``permute`` accepts a single tuple of axes, and backward inverts it."""
    a = rng(9).standard_normal((2, 3, 4)).astype(np.float32)
    for axes in ((2, 0, 1), (-1, 0, 1)):
        x = from_np(a)
        x.requires_grad = True

        p = x.permute(axes)
        normalized = tuple(axis % a.ndim for axis in axes)
        assert p.shape == [a.shape[axis] for axis in normalized]
        assert_close(to_np(p), np.transpose(a, normalized))

        w = rng(10).standard_normal(p.shape).astype(np.float32)
        (p * from_np(w)).sum().backward()

        inv = [0] * a.ndim
        for i, axis in enumerate(normalized):
            inv[axis] = i
        assert_close(
            to_np(x.grad), np.transpose(w, inv), msg=f"axes={axes}"
        )


def test_slicing_narrow_chunk():
    a = from_np(rng(4).standard_normal((4, 6)))
    s = a[:, 2:5]
    assert s.shape == [4, 3]
    assert_close(to_np(s), to_np(a)[:, 2:5])
    n = a.narrow(1, 1, 4)
    assert n.shape == [4, 4]
    assert_close(to_np(n), to_np(a)[:, 1:5])
    chunks = a.chunk(3, dim=1)
    assert len(chunks) == 3
    for i, c in enumerate(chunks):
        assert_close(to_np(c), to_np(a)[:, i * 2:(i + 1) * 2])


def test_advanced_indexing():
    a = from_np(rng(5).standard_normal((5, 4)))
    idx = from_np(np.array([0, 2, 4])).astype(np.int32) if False else None
    # basic fancy via numpy indices
    sub = a[[0, 2, 4]]
    assert sub.shape == [3, 4]
    assert_close(to_np(sub), to_np(a)[[0, 2, 4]])


def test_complex_tensor_roundtrip():
    from wxx.functional.fft import from_numpy_complex
    a = (rng(6).standard_normal((3, 4)) + 1j * rng(7).standard_normal((3, 4))).astype(np.complex64)
    t = from_numpy_complex(a)
    assert t.type == "c64"
    assert t.shape == [3, 4]
    assert_close(to_np(t), a, atol=0)


def test_item_and_repr():
    t = Tensor((1,), value=7.0)
    assert abs(t.item() - 7.0) < 1e-6


def test_contiguous_view():
    a = from_np(rng(8).standard_normal((3, 4)))
    v = a[:, ::2]
    assert not v.is_contiguous()
    c = v.contiguous()
    assert c.is_contiguous()
    assert_close(to_np(c), to_np(v))


TESTS = [
    ("basic_construction", test_basic_construction),
    ("from_numpy_roundtrip", test_from_numpy_roundtrip),
    ("arithmetic_broadcast", test_arithmetic_broadcast),
    ("reshape_permute_contiguous", test_reshape_permute_contiguous),
    ("permute_tuple_argument_backward", test_permute_tuple_argument_backward),
    ("slicing_narrow_chunk", test_slicing_narrow_chunk),
    ("advanced_indexing", test_advanced_indexing),
    ("complex_tensor_roundtrip", test_complex_tensor_roundtrip),
    ("item_and_repr", test_item_and_repr),
    ("contiguous_view", test_contiguous_view),
]


if __name__ == "__main__":
    from common import run_all
    run_all(TESTS)
