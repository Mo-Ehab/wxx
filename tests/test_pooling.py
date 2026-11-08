import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from common import from_np, from_np_u32, to_np, assert_close, rng

from wxx import functional as F
from wxx.nn.module import (
    AdaptiveAvgPool2d,
    AvgPool1d,
    AvgPool2d,
    MaxPool1d,
    MaxPool2d,
)


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


def ref_avgpool2d(x, kernel, stride, padding):
    b, c, h, w = x.shape
    kh, kw = kernel
    sh, sw = stride
    ph, pw = padding
    h_out = (h + 2 * ph - kh) // sh + 1
    w_out = (w + 2 * pw - kw) // sw + 1
    out = np.zeros((b, c, h_out, w_out), dtype=np.float64)
    for oh in range(h_out):
        for ow in range(w_out):
            hs, ws = oh * sh - ph, ow * sw - pw
            window = x[
                :, :, max(0, hs) : min(h, hs + kh), max(0, ws) : min(w, ws + kw)
            ]
            if window.size:
                out[:, :, oh, ow] = window.mean(axis=(2, 3))
    return out


def ref_maxpool2d(x, kernel, stride, padding):
    b, c, h, w = x.shape
    kh, kw = kernel
    sh, sw = stride
    ph, pw = padding
    h_out = (h + 2 * ph - kh) // sh + 1
    w_out = (w + 2 * pw - kw) // sw + 1
    out = np.full((b, c, h_out, w_out), -1e38, dtype=np.float64)
    idx = np.zeros((b, c, h_out, w_out), dtype=np.uint32)
    for bi in range(b):
        for ci in range(c):
            for oh in range(h_out):
                for ow in range(w_out):
                    hs, ws = oh * sh - ph, ow * sw - pw
                    best, best_idx = -1e38, 0xFFFFFFFF
                    for i in range(kh):
                        ih = hs + i
                        if 0 <= ih < h:
                            for j in range(kw):
                                iw = ws + j
                                if 0 <= iw < w:
                                    flat = ((bi * c + ci) * h + ih) * w + iw
                                    value = x[bi, ci, ih, iw]
                                    if value > best:
                                        best, best_idx = value, flat
                    out[bi, ci, oh, ow] = best
                    idx[bi, ci, oh, ow] = best_idx
    return out, idx


def ref_avgpool2ddx(dy, x_shape, kernel, stride, padding):
    b, c, h, w = x_shape
    kh, kw = kernel
    sh, sw = stride
    ph, pw = padding
    dx = np.zeros(x_shape, dtype=np.float64)
    for oh in range(dy.shape[2]):
        for ow in range(dy.shape[3]):
            hs, ws = oh * sh - ph, ow * sw - pw
            h0, h1 = max(0, hs), min(h, hs + kh)
            w0, w1 = max(0, ws), min(w, ws + kw)
            count = (h1 - h0) * (w1 - w0)
            if count:
                dx[:, :, h0:h1, w0:w1] += dy[:, :, oh, ow][:, :, None, None] / count
    return dx


def ref_maxpool2ddx(dy, x_shape, idx):
    dx = np.zeros(x_shape, dtype=np.float64)
    b, c, h, w = x_shape
    for flat_index, value in zip(idx.reshape(-1), dy.reshape(-1)):
        if flat_index == 0xFFFFFFFF:
            continue
        bi = flat_index // (c * h * w)
        rem = flat_index % (c * h * w)
        ci = rem // (h * w)
        rem %= h * w
        dx[bi, ci, rem // w, rem % w] += value
    return dx


def ref_avgpool1d(x, kernel, stride, padding):
    b, c, length = x.shape
    length_out = (length + 2 * padding - kernel) // stride + 1
    out = np.zeros((b, c, length_out), dtype=np.float64)
    for i in range(length_out):
        start = i * stride - padding
        window = x[:, :, max(0, start) : min(length, start + kernel)]
        if window.size:
            out[:, :, i] = window.mean(axis=2)
    return out


def ref_maxpool1d(x, kernel, stride, padding):
    b, c, length = x.shape
    length_out = (length + 2 * padding - kernel) // stride + 1
    out = np.full((b, c, length_out), -1e38, dtype=np.float64)
    idx = np.zeros((b, c, length_out), dtype=np.uint32)
    for bi in range(b):
        for ci in range(c):
            for i in range(length_out):
                start = i * stride - padding
                best, best_idx = -1e38, 0xFFFFFFFF
                for k in range(kernel):
                    pos = start + k
                    if 0 <= pos < length:
                        flat = (bi * c + ci) * length + pos
                        value = x[bi, ci, pos]
                        if value > best:
                            best, best_idx = value, flat
                out[bi, ci, i] = best
                idx[bi, ci, i] = best_idx
    return out, idx


def ref_avgpool1ddx(dy, x_shape, kernel, stride, padding):
    b, c, length = x_shape
    dx = np.zeros(x_shape, dtype=np.float64)
    for i in range(dy.shape[2]):
        start = i * stride - padding
        lo, hi = max(0, start), min(length, start + kernel)
        count = hi - lo
        if count:
            dx[:, :, lo:hi] += dy[:, :, i][:, :, None] / count
    return dx


def ref_maxpool1ddx(dy, x_shape, idx):
    b, c, length = x_shape
    dx = np.zeros(x_shape, dtype=np.float64)
    for flat_index, value in zip(idx.reshape(-1), dy.reshape(-1)):
        if flat_index == 0xFFFFFFFF:
            continue
        bi = flat_index // (c * length)
        rem = flat_index % (c * length)
        dx[bi, rem // length, rem % length] += value
    return dx


# ---------------------------------------------------------------------------
# adaptive average pooling
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# average pooling
# ---------------------------------------------------------------------------


def test_avgpool2d_forward():
    x = rng(10).standard_normal((2, 3, 7, 8)).astype(np.float32)
    for kernel, stride, padding in (
        ((2, 2), (2, 2), (0, 0)),
        ((3, 2), (2, 3), (1, 2)),
        ((2, 3), (1, 1), (0, 1)),
    ):
        out = F.avgpool2d(from_np(x), kernel, stride, padding)
        expected = ref_avgpool2d(x, kernel, stride, padding)
        assert out.shape == list(expected.shape)
        assert_close(
            to_np(out), expected, msg=f"{kernel}/{stride}/{padding}"
        )


def test_avgpool2d_backward():
    x = rng(11).standard_normal((2, 2, 6, 5)).astype(np.float32)
    for kernel, stride, padding in (((2, 2), (2, 2), (0, 0)), ((3, 2), (1, 2), (1, 1))):
        out_shape = ref_avgpool2d(x, kernel, stride, padding).shape
        dy = rng(12).standard_normal(out_shape).astype(np.float32)
        expected = ref_avgpool2ddx(dy, x.shape, kernel, stride, padding)
        dx = F.avgpool2ddx(from_np(dy), from_np(x), kernel, stride, padding)
        assert_close(to_np(dx), expected, msg=f"{kernel}/{stride}/{padding}")


def test_avgpool2d_module_backward():
    x = from_np(rng(13).standard_normal((2, 2, 5, 6)).astype(np.float32))
    x.requires_grad = True
    out = AvgPool2d(2, stride=2)(x)
    out.sum().backward()
    ref = ref_avgpool2ddx(
        np.ones(tuple(out.shape), dtype=np.float32), (2, 2, 5, 6), (2, 2), (2, 2), (0, 0)
    )
    assert_close(to_np(x.grad), ref)


def test_avgpool1d_forward():
    x = rng(14).standard_normal((2, 3, 11)).astype(np.float32)
    for kernel, stride, padding in (
        (2, 2, 0),
        (3, 2, 1),
        (4, 1, 2),
        ((2,), (3,), (0,)),
    ):
        out = F.avgpool1d(from_np(x), kernel, stride, padding)
        k = kernel[0] if isinstance(kernel, tuple) else kernel
        s = stride[0] if isinstance(stride, tuple) else stride
        p = padding[0] if isinstance(padding, tuple) else padding
        expected = ref_avgpool1d(x, k, s, p)
        assert out.shape == list(expected.shape)
        assert_close(to_np(out), expected, msg=f"{kernel}/{stride}/{padding}")


def test_avgpool1d_backward():
    x = rng(15).standard_normal((2, 2, 9)).astype(np.float32)
    for kernel, stride, padding in ((2, 2, 0), (3, 2, 1), (4, 1, 2)):
        out_shape = ref_avgpool1d(x, kernel, stride, padding).shape
        dy = rng(16).standard_normal(out_shape).astype(np.float32)
        expected = ref_avgpool1ddx(dy, x.shape, kernel, stride, padding)
        dx = F.avgpool1ddx(from_np(dy), from_np(x), kernel, stride, padding)
        assert_close(to_np(dx), expected, msg=f"{kernel}/{stride}/{padding}")


def test_avgpool1d_module_backward():
    x = from_np(rng(17).standard_normal((2, 2, 10)).astype(np.float32))
    x.requires_grad = True
    out = AvgPool1d(3, stride=2, padding=1)(x)
    out.sum().backward()
    ref = ref_avgpool1ddx(
        np.ones(tuple(out.shape), dtype=np.float32), (2, 2, 10), 3, 2, 1
    )
    assert_close(to_np(x.grad), ref)


# ---------------------------------------------------------------------------
# max pooling
# ---------------------------------------------------------------------------


def test_maxpool2d_forward():
    x = rng(20).standard_normal((2, 3, 7, 8)).astype(np.float32)
    for kernel, stride, padding in (
        ((2, 2), (2, 2), (0, 0)),
        ((3, 2), (2, 3), (1, 2)),
        ((2, 3), (1, 1), (1, 1)),
    ):
        out = F.maxpool2d(from_np(x), kernel, stride, padding)
        expected, _ = ref_maxpool2d(x, kernel, stride, padding)
        assert out.shape == list(expected.shape)
        assert_close(to_np(out), expected, msg=f"{kernel}/{stride}/{padding}")


def test_maxpool2d_backward():
    x = rng(21).standard_normal((2, 2, 6, 7)).astype(np.float32)
    for kernel, stride, padding in (((2, 2), (2, 2), (0, 0)), ((3, 2), (1, 2), (1, 1))):
        out, idx = ref_maxpool2d(x, kernel, stride, padding)
        dy = rng(22).standard_normal(out.shape).astype(np.float32)
        expected = ref_maxpool2ddx(dy, x.shape, idx)
        dx = F.maxpool2ddx(
            from_np(dy), from_np(x), from_np_u32(idx), kernel, stride, padding
        )
        assert_close(to_np(dx), expected, msg=f"{kernel}/{stride}/{padding}")


def test_maxpool1d_forward():
    x = rng(23).standard_normal((2, 3, 11)).astype(np.float32)
    for kernel, stride, padding in ((2, 2, 0), (3, 2, 1), (4, 1, 2), ((2,), (3,), (0,))):
        out = F.maxpool1d(from_np(x), kernel, stride, padding)
        k = kernel[0] if isinstance(kernel, tuple) else kernel
        s = stride[0] if isinstance(stride, tuple) else stride
        p = padding[0] if isinstance(padding, tuple) else padding
        expected, _ = ref_maxpool1d(x, k, s, p)
        assert out.shape == list(expected.shape)
        assert_close(to_np(out), expected, msg=f"{kernel}/{stride}/{padding}")


def test_maxpool1d_backward():
    x = rng(24).standard_normal((2, 2, 9)).astype(np.float32)
    for kernel, stride, padding in ((2, 2, 0), (3, 2, 1), (4, 1, 2)):
        out, idx = ref_maxpool1d(x, kernel, stride, padding)
        dy = rng(25).standard_normal(out.shape).astype(np.float32)
        expected = ref_maxpool1ddx(dy, x.shape, idx)
        dx = F.maxpool1ddx(
            from_np(dy), from_np(x), from_np_u32(idx), kernel, stride, padding
        )
        assert_close(to_np(dx), expected, msg=f"{kernel}/{stride}/{padding}")


# ---------------------------------------------------------------------------
# return_indices and module gradients
# ---------------------------------------------------------------------------


def test_avgpool_return_indices():
    x1 = rng(30).standard_normal((2, 2, 9)).astype(np.float32)
    out, idx = F.avgpool1d(from_np(x1), 3, 2, 1, return_indices=True)
    _, expected_idx = ref_maxpool1d(x1, 3, 2, 1)
    assert idx.type == "u32"
    assert idx.shape == [2, 2, 5]
    assert_close(to_np(idx), expected_idx, msg="avgpool1d indices")

    x2 = rng(31).standard_normal((2, 2, 6, 7)).astype(np.float32)
    out, idx = F.avgpool2d(from_np(x2), (3, 2), (2, 2), (1, 1), return_indices=True)
    _, expected_idx = ref_maxpool2d(x2, (3, 2), (2, 2), (1, 1))
    assert idx.type == "u32"
    assert idx.shape == [2, 2, 3, 4]
    assert_close(to_np(idx), expected_idx, msg="avgpool2d indices")

    # Without the flag the output is a plain tensor.
    assert isinstance(F.avgpool1d(from_np(x1), 3, 2, 1), type(out))


def test_maxpool_return_indices():
    x1 = rng(32).standard_normal((2, 2, 9)).astype(np.float32)
    out, idx = F.maxpool1d(from_np(x1), 3, 2, 1, return_indices=True)
    expected, expected_idx = ref_maxpool1d(x1, 3, 2, 1)
    assert idx.type == "u32"
    assert_close(to_np(out), expected)
    assert_close(to_np(idx), expected_idx, msg="maxpool1d indices")

    x2 = rng(33).standard_normal((2, 2, 6, 7)).astype(np.float32)
    out, idx = F.maxpool2d(from_np(x2), (3, 2), (2, 2), (1, 1), return_indices=True)
    expected, expected_idx = ref_maxpool2d(x2, (3, 2), (2, 2), (1, 1))
    assert idx.type == "u32"
    assert_close(to_np(out), expected)
    assert_close(to_np(idx), expected_idx, msg="maxpool2d indices")


def test_pool_modules_return_indices_and_backward():
    x = rng(34).standard_normal((2, 2, 10)).astype(np.float32)

    for module, ref_fwd, ref_bwd in (
        (
            AvgPool1d(3, stride=2, padding=1, return_indices=True),
            ref_avgpool1d,
            lambda grad, shape: ref_avgpool1ddx(grad, shape, 3, 2, 1),
        ),
        (MaxPool1d(3, stride=2, padding=1, return_indices=True), ref_maxpool1d, None),
    ):
        tensor = from_np(x)
        tensor.requires_grad = True
        out, idx = module(tensor)
        if ref_bwd is None:
            expected, expected_idx = ref_fwd(x, 3, 2, 1)
            assert_close(to_np(idx), expected_idx)
            ref_bwd = lambda grad, shape, idx=expected_idx: ref_maxpool1ddx(grad, shape, idx)
        else:
            expected = ref_fwd(x, 3, 2, 1)
        assert_close(to_np(out), expected)
        out.sum().backward()
        assert_close(
            to_np(tensor.grad),
            ref_bwd(np.ones(tuple(out.shape), dtype=np.float32), x.shape),
        )

    x2 = rng(35).standard_normal((2, 2, 6, 6)).astype(np.float32)
    for module, ref_fwd, ref_bwd in (
        (
            AvgPool2d(2, stride=2, return_indices=True),
            ref_avgpool2d,
            lambda grad, shape: ref_avgpool2ddx(
                grad, shape, (2, 2), (2, 2), (0, 0)
            ),
        ),
        (MaxPool2d(2, stride=2, return_indices=True), ref_maxpool2d, None),
    ):
        result = ref_fwd(x2, (2, 2), (2, 2), (0, 0))
        expected, expected_idx = result if isinstance(result, tuple) else (result, None)
        tensor = from_np(x2)
        tensor.requires_grad = True
        out, idx = module(tensor)
        assert_close(to_np(out), expected)
        if expected_idx is not None:
            assert_close(to_np(idx), expected_idx)
            ref_bwd = lambda grad, shape, idx=expected_idx: ref_maxpool2ddx(
                grad, shape, idx
            )
        out.sum().backward()
        assert_close(
            to_np(tensor.grad),
            ref_bwd(np.ones(tuple(out.shape), dtype=np.float32), x2.shape),
        )


def test_pool_rejects_bad_arguments():
    x1 = from_np(np.zeros((2, 2, 8), dtype=np.float32))
    x2 = from_np(np.zeros((2, 2, 4, 4), dtype=np.float32))

    for fn, x in ((F.avgpool1d, x2), (F.maxpool1d, x2), (F.avgpool2d, x1), (F.maxpool2d, x1)):
        try:
            fn(x, 2, 2, 0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{fn.__name__} accepted the wrong rank")

    for fn in (F.avgpool1d, F.maxpool1d):
        for bad in (0, -1):
            try:
                fn(x1, bad, 1, 0)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{fn.__name__} accepted kernel={bad}")

    for fn, x in ((F.avgpool2d, x2), (F.maxpool2d, x2), (F.avgpool1d, x1), (F.maxpool1d, x1)):
        try:
            fn(from_np_u32(np.zeros(x.shape, dtype=np.uint32)), 2, 2, 0)
        except TypeError:
            pass
        else:
            raise AssertionError(f"{fn.__name__} accepted a u32 input")


TESTS = [
    ("adaptive_avgpool2d_forward", test_adaptive_avgpool2d_forward),
    ("adaptive_avgpool2d_backward", test_adaptive_avgpool2d_backward),
    (
        "adaptive_avgpool2d_module_backward",
        test_adaptive_avgpool2d_module_backward,
    ),
    ("avgpool2d_forward", test_avgpool2d_forward),
    ("avgpool2d_backward", test_avgpool2d_backward),
    ("avgpool2d_module_backward", test_avgpool2d_module_backward),
    ("avgpool1d_forward", test_avgpool1d_forward),
    ("avgpool1d_backward", test_avgpool1d_backward),
    ("avgpool1d_module_backward", test_avgpool1d_module_backward),
    ("maxpool2d_forward", test_maxpool2d_forward),
    ("maxpool2d_backward", test_maxpool2d_backward),
    ("maxpool1d_forward", test_maxpool1d_forward),
    ("maxpool1d_backward", test_maxpool1d_backward),
    ("avgpool_return_indices", test_avgpool_return_indices),
    ("maxpool_return_indices", test_maxpool_return_indices),
    (
        "pool_modules_return_indices_and_backward",
        test_pool_modules_return_indices_and_backward,
    ),
    ("pool_rejects_bad_arguments", test_pool_rejects_bad_arguments),
]

if __name__ == "__main__":
    from common import run_all
    run_all(TESTS)
