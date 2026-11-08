import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from common import assert_close, from_np, from_np_u32, rng, to_np


def ref_unfold(x, dim, size, step=1):
    dim %= x.ndim
    windows = (x.shape[dim] - size) // step + 1
    parts = []
    for w in range(windows):
        index = [slice(None)] * x.ndim
        index[dim] = slice(w * step, w * step + size)
        parts.append(x[tuple(index)])
    stacked = np.stack(parts, axis=dim)
    return np.moveaxis(stacked, dim + 1, -1)


def ref_unfolddx(grad, shape, dim, size, step=1):
    dim %= len(shape)
    dx = np.zeros(shape, dtype=np.float64)
    for out_index in np.ndindex(grad.shape):
        coords = list(out_index)
        window = coords[-1]
        in_coords = coords[:-1]
        in_coords[dim] = coords[dim] * step + window
        dx[tuple(in_coords)] += grad[out_index]
    return dx


def test_unfold_matches_numpy():
    a = rng(0).standard_normal((2, 3, 7, 5)).astype(np.float32)
    x = from_np(a)

    for dim in (0, 1, 2, 3, -1, -2):
        for size, step in ((1, 1), (2, 2), (3, 1), (3, 2), (4, 3)):
            actual_dim = dim % a.ndim
            if size > a.shape[actual_dim]:
                continue
            out = x.unfold(dim, size, step)
            expected = ref_unfold(a, actual_dim, size, step)
            assert out.shape == list(expected.shape), (dim, size, step, out.shape)
            assert_close(
                to_np(out), expected, msg=f"dim={dim} size={size} step={step}"
            )


def test_unfold_is_a_view():
    a = rng(1).standard_normal((3, 8)).astype(np.float32)
    x = from_np(a)

    out = x.unfold(1, 3, 2)
    assert out.buffer is x.buffer, "unfold must share storage"
    assert out._base is x
    assert list(out.stride) == [x.stride[0], x.stride[1] * 2, x.stride[1]]
    assert out.offset == x.offset
    assert out.shape == [3, 3, 3]
    assert_close(to_np(out), ref_unfold(a, 1, 3, 2))


def test_unfold_strided_input():
    a = rng(2).standard_normal((4, 6)).astype(np.float32)
    x = from_np(a)

    transposed = x.permute(1, 0)
    assert not transposed.is_contiguous()
    out = transposed.unfold(0, 3, 2)
    assert_close(to_np(out), ref_unfold(a.T, 0, 3, 2))

    sliced = x[:3, 1:6]
    out = sliced.unfold(1, 3, 1)
    assert_close(to_np(out), ref_unfold(a[:3, 1:6], 1, 3, 1))


def test_unfold_backward():
    a = rng(3).standard_normal((2, 3, 6)).astype(np.float32)

    for dim, size, step in ((0, 2, 1), (1, 3, 2), (2, 4, 1), (2, 3, 2)):
        x = from_np(a)
        x.requires_grad = True

        out = x.unfold(dim, size, step)
        weights = rng(4 + dim + size).standard_normal(
            tuple(out.shape)
        ).astype(np.float32)
        (out * from_np(weights)).sum().backward()

        expected = ref_unfolddx(weights, a.shape, dim, size, step)
        assert_close(
            to_np(x.grad), expected, msg=f"dim={dim} size={size} step={step}"
        )


def test_unfold_backward_through_permute():
    a = rng(5).standard_normal((3, 8)).astype(np.float32)
    base = from_np(a)
    base.requires_grad = True

    out = base.permute(1, 0).unfold(1, 3, 2)
    weights = rng(6).standard_normal(tuple(out.shape)).astype(np.float32)
    (out * from_np(weights)).sum().backward()

    expected = ref_unfolddx(weights, (8, 3), 1, 3, 2).T
    assert_close(to_np(base.grad), expected)


def test_unfold_preserves_dtype():
    ids = from_np_u32(np.arange(24, dtype=np.uint32).reshape(2, 3, 4))
    out = ids.unfold(2, 2, 1)
    assert out.type == "u32"
    expected = ref_unfold(np.arange(24, dtype=np.uint32).reshape(2, 3, 4), 2, 2, 1)
    assert_close(to_np(out), expected)


def test_unfold_rejects_bad_arguments():
    x = from_np(np.zeros((2, 5), dtype=np.float32))

    for size in (0, -1, 6):
        try:
            x.unfold(1, size)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unfold accepted size={size}")

    for step in (0, -1):
        try:
            x.unfold(1, 2, step)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unfold accepted step={step}")

    for bad in (1.5, "2", True):
        try:
            x.unfold(1, bad)
        except TypeError:
            pass
        else:
            raise AssertionError(f"unfold accepted size={bad!r}")

    for dim in (2, -3):
        try:
            x.unfold(dim, 2)
        except (IndexError, ValueError):
            pass
        else:
            raise AssertionError(f"unfold accepted dim={dim}")

    x6 = from_np(np.zeros((2, 2, 2, 2, 2, 2), dtype=np.float32))
    try:
        x6.unfold(0, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("unfold accepted a rank-6 input")


TESTS = [
    ("unfold_matches_numpy", test_unfold_matches_numpy),
    ("unfold_is_a_view", test_unfold_is_a_view),
    ("unfold_strided_input", test_unfold_strided_input),
    ("unfold_backward", test_unfold_backward),
    ("unfold_backward_through_permute", test_unfold_backward_through_permute),
    ("unfold_preserves_dtype", test_unfold_preserves_dtype),
    ("unfold_rejects_bad_arguments", test_unfold_rejects_bad_arguments),
]


if __name__ == "__main__":
    from common import run_all
    run_all(TESTS)
