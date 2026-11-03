import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from common import from_np, from_np_u32, to_np, assert_close, rng

from wxx import Tensor, functional, validation


def _raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    except Exception as e:
        raise AssertionError(f"expected {exc.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc.__name__}, got no exception")


def test_validation_normalize_dim():
    assert validation.normalize_dim(-1, 3) == 2
    assert validation.normalize_dim(1, 3) == 1
    assert _raises(lambda: validation.normalize_dim(5, 3), IndexError)
    assert _raises(lambda: validation.normalize_dim(0, 0), IndexError)


def test_tensor_reshape_errors():
    a = from_np(rng(0).standard_normal((2, 3)))
    assert _raises(lambda: a.reshape((7,)), ValueError)
    assert _raises(lambda: a.reshape((2, 2)), ValueError)
    # infer
    r = a.reshape((-1, 3))
    assert r.shape == [2, 3]


def test_tensor_permute_errors():
    a = from_np(rng(1).standard_normal((2, 3, 4)))
    assert _raises(lambda: a.permute(0, 1), ValueError)
    assert _raises(lambda: a.permute(0, 0, 1), ValueError)
    p = a.permute(2, 0, 1)
    assert p.shape == [4, 2, 3]


def test_tensor_transpose_unsqueeze_squeeze():
    a = from_np(rng(2).standard_normal((2, 1, 4)))
    # transpose
    t = a.transpose(0, 2)
    assert t.shape == [4, 1, 2]
    # unsqueeze
    u = a.unsqueeze(1)
    assert u.shape == [2, 1, 1, 4]
    # squeeze None removes all size-1
    s = a.squeeze()
    assert s.shape == [2, 4]
    # squeeze on non-1 dim is a no-op
    s2 = a.squeeze(2)
    assert s2.shape == [2, 1, 4]


def test_tensor_narrow_errors():
    a = from_np(rng(3).standard_normal((4, 6)))
    assert _raises(lambda: a.narrow(1, 5, 2), IndexError)
    assert _raises(lambda: a.narrow(1, 0, 0 - 1), (IndexError, ValueError, TypeError))
    n = a.narrow(1, 1, 4)
    assert n.shape == [4, 4]


def test_tensor_flatten():
    a = from_np(rng(4).standard_normal((2, 3, 4)))
    f = a.flatten(1, 2)
    assert f.shape == [2, 12]
    assert _raises(lambda: a.flatten(2, 1), ValueError)


def test_tensor_view_alias():
    a = from_np(rng(5).standard_normal((2, 3, 4)))
    v = a.view(6, 4)
    assert v.shape == [6, 4]


def test_strided_op_correctness():
    a = from_np(rng(6).standard_normal((4, 6)))
    v = a[:, ::2]  # non-contiguous
    assert not v.is_contiguous()
    # pointwise on strided view materialises correctly
    out = v + 1.0
    assert_close(to_np(out), to_np(v) + 1.0)
    # reshape of strided view copies
    r = v.reshape((-1,))
    assert_close(to_np(r), to_np(v).reshape(-1))
    # permute of strided view
    p = a.permute(1, 0)
    assert p.shape == [6, 4]
    assert_close(to_np(p), np.transpose(to_np(a), (1, 0)))


def test_matmul_shape_checks():
    a = from_np(rng(7).standard_normal((2, 3, 4)))
    b = from_np(rng(8).standard_normal((2, 4, 5)))
    c = a @ b
    assert c.shape == [2, 3, 5]
    # rank too low
    x = from_np(rng(9).standard_normal((4,)))
    y = from_np(rng(10).standard_normal((4, 5)))
    assert _raises(lambda: x @ y, ValueError)


def test_broadcast_shapes():
    a = from_np(rng(11).standard_normal((3, 1)))
    b = from_np(rng(12).standard_normal((1, 4)))
    c = a + b
    assert c.shape == [3, 4]
    # incompatible
    x = from_np(rng(13).standard_normal((3, 2)))
    y = from_np(rng(14).standard_normal((1, 4)))
    assert _raises(lambda: x + y, ValueError)


def test_assign_shape_checks():
    a = from_np(rng(15).standard_normal((2, 3)))
    b = from_np(rng(16).standard_normal((2, 4)))
    assert _raises(lambda: functional.assign(a, b), ValueError)
    # scalar fill ok
    functional.assign(a, 0.0)
    assert_close(to_np(a), np.zeros((2, 3)))


def test_cceloss_shape_checks():
    logits = from_np(rng(17).standard_normal((4, 5)))
    targets = from_np_u32(np.array([0, 1, 2, 3], dtype=np.uint32))
    loss, grad = functional.cceloss(logits, targets)
    assert loss.shape == [4, 1]
    assert grad.shape == [4, 5]
    # wrong target count
    bad_targets = from_np_u32(np.array([0, 1], dtype=np.uint32))
    assert _raises(lambda: functional.cceloss(logits, bad_targets), ValueError)
    # wrong logits rank
    assert _raises(lambda: functional.cceloss(from_np(rng(18).standard_normal((5,))), targets), ValueError)


def test_unsqueeze_squeeze_strides():
    a = from_np(rng(19).standard_normal((2, 3, 4)))
    a.requires_grad = True
    v = a[:, 1:, :]  # strided view
    u = v.unsqueeze(1)
    assert u.shape == [2, 1, 2, 4]
    # round-trip through backward
    t = u.sum()
    t.backward()
    assert a.grad is not None


def test_concat_checks():
    a = from_np(rng(20).standard_normal((2, 3)))
    b = from_np(rng(21).standard_normal((2, 4)))
    c = functional.concat([a, b], dim=1)
    assert c.shape == [2, 7]
    # mismatched non-concat dim
    d = from_np(rng(22).standard_normal((3, 3)))
    assert _raises(lambda: functional.concat([a, d], dim=1), ValueError)
    # empty
    assert _raises(lambda: functional.concat([], dim=0), ValueError)


def test_flatten_errors():
    a = from_np(rng(23).standard_normal((2, 3, 4)))
    assert _raises(lambda: a.flatten(2, 1), ValueError)


TESTS = [
    ("validation_normalize_dim", test_validation_normalize_dim),
    ("tensor_reshape_errors", test_tensor_reshape_errors),
    ("tensor_permute_errors", test_tensor_permute_errors),
    ("tensor_transpose_unsqueeze_squeeze", test_tensor_transpose_unsqueeze_squeeze),
    ("tensor_narrow_errors", test_tensor_narrow_errors),
    ("tensor_flatten", test_tensor_flatten),
    ("tensor_view_alias", test_tensor_view_alias),
    ("strided_op_correctness", test_strided_op_correctness),
    ("matmul_shape_checks", test_matmul_shape_checks),
    ("broadcast_shapes", test_broadcast_shapes),
    ("assign_shape_checks", test_assign_shape_checks),
    ("cceloss_shape_checks", test_cceloss_shape_checks),
    ("unsqueeze_squeeze_strides", test_unsqueeze_squeeze_strides),
    ("concat_checks", test_concat_checks),
    ("flatten_errors", test_flatten_errors),
]

if __name__ == "__main__":
    from common import run_all
    run_all(TESTS)
