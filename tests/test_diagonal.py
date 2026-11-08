import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from common import assert_close, from_np, from_np_u32, rng, to_np

from wxx import functional


def scatter_reference(shape, grad, offset, dim1, dim2):
    """Reference ``d/dinput`` of ``diagonal`` (zeros except the diagonal)."""
    rank = len(shape)
    dim1 %= rank
    dim2 %= rank
    out = np.zeros(shape, dtype=np.float32)
    length = grad.shape[-1]
    start1 = 0 if offset >= 0 else -offset
    start2 = offset if offset >= 0 else 0
    others = [axis for axis in range(rank) if axis not in (dim1, dim2)]
    lead_shape = tuple(shape[axis] for axis in others)
    for lead in np.ndindex(lead_shape):
        for k in range(length):
            index = [0] * rank
            for pos, axis in enumerate(others):
                index[axis] = lead[pos]
            index[dim1] = start1 + k
            index[dim2] = start2 + k
            out[tuple(index)] = grad[lead + (k,)]
    return out


def test_diagonal_matches_numpy():
    a = rng(0).standard_normal((3, 4)).astype(np.float32)
    x = from_np(a)

    for offset in range(-5, 6):
        d = x.diagonal(offset)
        expected = np.diagonal(a, offset)
        assert d.shape == [expected.shape[0]], (offset, d.shape, expected.shape)
        assert_close(to_np(d), expected, msg=f"offset={offset}")


def test_diagonal_default_is_main_diagonal():
    a = rng(1).standard_normal((4, 4)).astype(np.float32)
    x = from_np(a)
    assert_close(to_np(x.diagonal()), np.diag(a))


def test_diagonal_any_dims_match_numpy():
    a = rng(2).standard_normal((2, 3, 4, 5)).astype(np.float32)
    x = from_np(a)

    pairs = (
        (0, 1), (1, 0), (0, 2), (2, 0), (0, 3),
        (3, 0), (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2),
    )
    for dim1, dim2 in pairs:
        for offset in (-2, 0, 1):
            d = x.diagonal(offset, dim1, dim2)
            expected = np.diagonal(a, offset, axis1=dim1, axis2=dim2)
            assert d.shape == list(expected.shape), (dim1, dim2, offset, d.shape)
            assert_close(to_np(d), expected, msg=f"dims={dim1, dim2} offset={offset}")


def test_diagonal_negative_dims():
    a = rng(3).standard_normal((2, 3, 4)).astype(np.float32)
    x = from_np(a)
    assert_close(
        to_np(x.diagonal(0, -1, -2)), np.diagonal(a, 0, axis1=2, axis2=1)
    )
    assert_close(
        to_np(x.diagonal(1, -3, -1)), np.diagonal(a, 1, axis1=0, axis2=2)
    )


def test_diagonal_empty_offset():
    a = rng(4).standard_normal((2, 5)).astype(np.float32)
    x = from_np(a)
    for offset in (5, 100, -2, -50):
        d = x.diagonal(offset)
        assert d.shape == [0], (offset, d.shape)
        assert to_np(d).shape == (0,)

    b = rng(5).standard_normal((2, 3, 4)).astype(np.float32)
    y = from_np(b)
    assert y.diagonal(9, 0, 2).shape == [3, 0]
    assert y.diagonal(-9, 1, 2).shape == [2, 0]


def test_diagonal_is_a_view():
    a = rng(6).standard_normal((4, 6)).astype(np.float32)
    x = from_np(a)

    d = x.diagonal(-1)
    assert d.buffer is x.buffer, "diagonal must share storage"
    assert d._base is x
    assert list(d.stride) == [x.stride[0] + x.stride[1]]
    assert d.offset == x.offset + 1 * x.stride[0]

    # Writing through the view writes the matrix (standard view semantics).
    functional.assign(d, -1.0)
    expected = a.copy()
    expected[1, 0] = expected[2, 1] = expected[3, 2] = -1.0
    assert_close(to_np(x), expected)


def test_diagonal_view_over_any_dims():
    a = rng(7).standard_normal((2, 3, 4)).astype(np.float32)
    x = from_np(a)

    d = x.diagonal(1, 1, 2)  # shape (2, 3), picks [i, j, j + 1]
    assert d.shape == [2, 3]
    assert list(d.stride) == [x.stride[0], x.stride[1] + x.stride[2]]

    functional.assign(d, -2.0)
    expected = a.copy()
    for i in range(2):
        for j in range(3):
            expected[i, j, j + 1] = -2.0
    assert_close(to_np(x), expected)


def test_diagonal_strided_input():
    a = rng(8).standard_normal((4, 6)).astype(np.float32)
    x = from_np(a)

    slice_view = x[:, 1:5]
    assert not slice_view.is_contiguous()
    assert_close(to_np(slice_view.diagonal(1)), np.diagonal(a[:, 1:5], 1))

    transposed = x.permute(1, 0)
    assert_close(to_np(transposed.diagonal(-1)), np.diagonal(a.T, -1))


def test_diagonal_backward():
    a = rng(9).standard_normal((3, 4)).astype(np.float32)

    for offset in (-1, 0, 2):
        x = from_np(a)
        x.requires_grad = True

        d = x.diagonal(offset)
        weights = rng(10 + offset).standard_normal(tuple(d.shape)).astype(np.float32)
        (d * from_np(weights)).sum().backward()

        assert_close(
            to_np(x.grad), scatter_reference(a.shape, weights, offset, 0, 1),
            msg=f"offset={offset}",
        )


def test_diagonal_backward_any_dims():
    a = rng(20).standard_normal((2, 3, 4)).astype(np.float32)

    for dim1, dim2, offset in ((0, 2, 1), (2, 0, -1), (1, 2, 0)):
        x = from_np(a)
        x.requires_grad = True

        d = x.diagonal(offset, dim1, dim2)
        weights = rng(21 + dim1 * 3 + dim2).standard_normal(
            tuple(d.shape)
        ).astype(np.float32)
        (d * from_np(weights)).sum().backward()

        expected = scatter_reference(a.shape, weights, offset, dim1, dim2)
        assert_close(to_np(x.grad), expected, msg=f"dims={dim1, dim2} offset={offset}")


def test_diagonal_backward_through_permute():
    a = rng(30).standard_normal((4, 5)).astype(np.float32)
    base = from_np(a)
    base.requires_grad = True

    x = base.permute(1, 0)
    assert not x.is_contiguous()
    d = x.diagonal(-1)
    weights = rng(31).standard_normal(tuple(d.shape)).astype(np.float32)
    (d * from_np(weights)).sum().backward()

    expected_x = np.zeros((5, 4), dtype=np.float32)
    for i, value in enumerate(weights):
        expected_x[1 + i, i] = value
    assert_close(to_np(base.grad), expected_x.T)


def test_diagonal_preserves_dtype():
    ids = from_np_u32(np.arange(24, dtype=np.uint32).reshape(2, 3, 4))
    d = ids.diagonal(1, 1, 2)
    assert d.type == "u32"
    expected = np.diagonal(np.arange(24).reshape(2, 3, 4), 1, axis1=1, axis2=2)
    assert_close(to_np(d), expected)


def test_diagonal_rejects_bad_arguments():
    x = from_np(np.zeros((4,), dtype=np.float32))
    try:
        x.diagonal()
    except ValueError:
        pass
    else:
        raise AssertionError("diagonal accepted a 1-D tensor")

    x = from_np(np.zeros((3, 3, 3), dtype=np.float32))
    for bad in (1.5, "1", True, None):
        try:
            x.diagonal(bad)
        except TypeError:
            pass
        else:
            raise AssertionError(f"diagonal accepted offset={bad!r}")

    try:
        x.diagonal(0, 1, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("diagonal accepted dim1 == dim2")

    for dims in ((0, 3), (-4, 0)):
        try:
            x.diagonal(0, *dims)
        except (IndexError, ValueError):
            pass
        else:
            raise AssertionError(f"diagonal accepted dims={dims}")


TESTS = [
    ("diagonal_matches_numpy", test_diagonal_matches_numpy),
    ("diagonal_default_is_main_diagonal", test_diagonal_default_is_main_diagonal),
    ("diagonal_any_dims_match_numpy", test_diagonal_any_dims_match_numpy),
    ("diagonal_negative_dims", test_diagonal_negative_dims),
    ("diagonal_empty_offset", test_diagonal_empty_offset),
    ("diagonal_is_a_view", test_diagonal_is_a_view),
    ("diagonal_view_over_any_dims", test_diagonal_view_over_any_dims),
    ("diagonal_strided_input", test_diagonal_strided_input),
    ("diagonal_backward", test_diagonal_backward),
    ("diagonal_backward_any_dims", test_diagonal_backward_any_dims),
    ("diagonal_backward_through_permute", test_diagonal_backward_through_permute),
    ("diagonal_preserves_dtype", test_diagonal_preserves_dtype),
    ("diagonal_rejects_bad_arguments", test_diagonal_rejects_bad_arguments),
]


if __name__ == "__main__":
    from common import run_all
    run_all(TESTS)
