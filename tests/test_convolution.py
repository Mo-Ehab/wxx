import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from common import from_np, to_np, assert_close, rng

from wxx import functional
from wxx.nn.module import Conv2D, Conv2DTranspose


def test_conv2d_forward():
    c = Conv2D(2, 3, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
    # init weights deterministically
    from wxx import functional as F
    F.assign(c.W, from_np(rng(0).standard_normal(c.W.shape).astype(np.float32) * 0.1))
    if c.bias is not None:
        F.assign(c.bias, from_np(np.zeros(c.bias.shape).astype(np.float32)))
    x = from_np(rng(1).standard_normal((2, 2, 8, 8)).astype(np.float32))
    out = c(x)
    assert out.shape == [2, 3, 8, 8]
    # just sanity: finite
    assert np.all(np.isfinite(to_np(out)))


def test_conv2d_bias_gradient():
    """``bias`` grads reduce over every axis except the channel axis.

    ``sum(axis=(0, 2, 3))`` used to raise: the reducer only supports a
    contiguous run of axes, and the bias path was the only 4-D user.
    """
    c = Conv2D(2, 3, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
    w = rng(30).standard_normal(tuple(c.W.shape)).astype(np.float32)
    b = rng(31).standard_normal(tuple(c.bias.shape)).astype(np.float32)
    functional.assign(c.W, from_np(w))
    functional.assign(c.bias, from_np(b))

    x = rng(32).standard_normal((2, 2, 5, 5)).astype(np.float32)
    xt = from_np(x)
    xt.requires_grad = True
    out = c(xt)
    g = rng(33).standard_normal(tuple(out.shape)).astype(np.float32)
    (out * from_np(g)).sum().backward()
    assert_close(to_np(c.bias.grad), g.sum(axis=(0, 2, 3)))


def ref_causal_conv1d(x, w, b):
    B, D, L = x.shape
    K = w.shape[1]
    out = np.zeros_like(x)
    for k in range(K):
        out[:, :, k:] += w[:, k][None, :, None] * x[:, :, : L - k]
    return out + b[None, :, None]


def test_causal_conv1d_forward_backward():
    from wxx.functional.convolution import causal_conv1d, causal_conv1d_backward

    B, D, L, K = 3, 5, 9, 3
    x = rng(2).standard_normal((B, D, L)).astype(np.float32)
    w = rng(3).standard_normal((D, K)).astype(np.float32)
    b = rng(4).standard_normal((D,)).astype(np.float32)
    dy = rng(5).standard_normal((B, D, L)).astype(np.float32)

    out = causal_conv1d(from_np(x), from_np(w), from_np(b))
    dx, dw, db = causal_conv1d_backward(
        from_np(dy), from_np(x), from_np(w), from_np(b)
    )
    assert_close(to_np(out), ref_causal_conv1d(x, w, b))

    ref_dx = np.zeros_like(x)
    ref_dw = np.zeros_like(w)
    for k in range(K):
        ref_dx[:, :, : L - k] += w[:, k][None, :, None] * dy[:, :, k:]
        ref_dw[:, k] = np.einsum("bdl,bdl->d", dy[:, :, k:], x[:, :, : L - k])
    assert_close(to_np(dx), ref_dx)
    assert_close(to_np(dw), ref_dw)
    assert_close(to_np(db), dy.sum(axis=(0, 2)))


def test_causal_conv1d_strided_input():
    """A transposed (B, L, D) view is consumed without a contiguous copy."""
    from wxx.functional.convolution import causal_conv1d

    B, D, L, K = 2, 4, 7, 3
    x = rng(6).standard_normal((B, L, D)).astype(np.float32)
    w = rng(7).standard_normal((D, K)).astype(np.float32)
    b = rng(8).standard_normal((D,)).astype(np.float32)

    view = from_np(x).permute(0, 2, 1)
    assert not view.is_contiguous()
    out = causal_conv1d(view, from_np(w), from_np(b))
    assert_close(to_np(out), ref_causal_conv1d(np.transpose(x, (0, 2, 1)), w, b))


# ----------------------------------------------------------------------
# Transposed 2-D convolution
# ----------------------------------------------------------------------
def ref_conv_transpose(x, w, b, stride, padding, dilation, output_padding):
    """Naive scatter: out[oh] += x[ih] * w[k] for oh = ih * stride - pad + k * dil."""
    B, C_in, H_in, W_in = x.shape
    _, C_out, KH, KW = w.shape
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    H_out = (H_in - 1) * sh - 2 * ph + dh * (KH - 1) + output_padding[0] + 1
    W_out = (W_in - 1) * sw - 2 * pw + dw * (KW - 1) + output_padding[1] + 1

    out = np.zeros((B, C_out, H_out, W_out), dtype=np.float32)
    for batch in range(B):
        for ci in range(C_in):
            for co in range(C_out):
                for kh in range(KH):
                    for kw in range(KW):
                        weight = w[ci, co, kh, kw]
                        for ih in range(H_in):
                            oh = ih * sh - ph + kh * dh
                            if not 0 <= oh < H_out:
                                continue
                            for iw in range(W_in):
                                ow = iw * sw - pw + kw * dw
                                if 0 <= ow < W_out:
                                    out[batch, co, oh, ow] += (
                                        x[batch, ci, ih, iw] * weight
                                    )
    if b is not None:
        out += b[None, :, None, None]
    return out


def ref_conv_transpose_backward(g, x, w, stride, padding, dilation, output_padding):
    B, C_in, H_in, W_in = x.shape
    _, C_out, KH, KW = w.shape
    sh, sw = stride
    ph, pw = padding
    dil_h, dil_w = dilation
    H_out = (H_in - 1) * sh - 2 * ph + dil_h * (KH - 1) + output_padding[0] + 1
    W_out = (W_in - 1) * sw - 2 * pw + dil_w * (KW - 1) + output_padding[1] + 1

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    for batch in range(B):
        for ci in range(C_in):
            for co in range(C_out):
                for kh in range(KH):
                    for kw in range(KW):
                        for ih in range(H_in):
                            oh = ih * sh - ph + kh * dil_h
                            if not 0 <= oh < H_out:
                                continue
                            for iw in range(W_in):
                                ow = iw * sw - pw + kw * dil_w
                                if not 0 <= ow < W_out:
                                    continue
                                gv = g[batch, co, oh, ow]
                                xv = x[batch, ci, ih, iw]
                                dx[batch, ci, ih, iw] += gv * w[ci, co, kh, kw]
                                dw[ci, co, kh, kw] += xv * gv
    return dx, dw, g.sum(axis=(0, 2, 3))


def conv_transpose_out_size(shape, wshape, stride, padding, dilation, output_padding):
    H_out = (
        (shape[2] - 1) * stride[0]
        - 2 * padding[0]
        + dilation[0] * (wshape[2] - 1)
        + output_padding[0]
        + 1
    )
    W_out = (
        (shape[3] - 1) * stride[1]
        - 2 * padding[1]
        + dilation[1] * (wshape[3] - 1)
        + output_padding[1]
        + 1
    )
    return H_out, W_out


def conv_transpose_setup(
    shape, wshape, stride, padding, dilation, output_padding, seed, bias=True
):
    x = rng(seed).standard_normal(shape).astype(np.float32)
    w = rng(seed + 1).standard_normal(wshape).astype(np.float32)
    b = (
        rng(seed + 2).standard_normal((wshape[1],)).astype(np.float32)
        if bias
        else None
    )

    module = Conv2DTranspose(
        shape[1],
        wshape[1],
        kernel=(wshape[2], wshape[3]),
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        bias=bias,
    )
    functional.assign(module.W, from_np(w))
    if bias:
        functional.assign(module.bias, from_np(b))
    return module, x, w, b


# (name, input shape, weight shape, stride, padding, dilation, output_padding, bias)
FORWARD_CASES = [
    ("stride2", (2, 2, 5, 6), (2, 3, 3, 3), (2, 2), (1, 1), (1, 1), (0, 0), True),
    ("dilation2", (2, 2, 4, 4), (2, 2, 3, 3), (1, 1), (1, 1), (2, 2), (0, 0), True),
    ("output_padding", (2, 2, 4, 4), (2, 3, 2, 2), (2, 2), (0, 0), (1, 1), (1, 1), True),
    (
        "dilation_output_padding",
        (1, 2, 4, 4),
        (2, 2, 3, 3),
        (1, 1),
        (1, 1),
        (2, 2),
        (1, 1),
        True,
    ),
    ("ragged_stride", (2, 2, 4, 5), (2, 2, 2, 3), (3, 2), (1, 0), (1, 1), (1, 0), True),
    ("no_bias", (2, 2, 5, 6), (2, 3, 3, 3), (2, 2), (1, 1), (1, 1), (0, 0), False),
]


def check_conv_transpose_forward(case):
    name, shape, wshape, stride, padding, dilation, output_padding, bias = case
    module, x, w, b = conv_transpose_setup(
        shape, wshape, stride, padding, dilation, output_padding, seed=40, bias=bias
    )
    out = module(from_np(x))
    expected_shape = [
        shape[0],
        wshape[1],
        *conv_transpose_out_size(shape, wshape, stride, padding, dilation, output_padding),
    ]
    assert out.shape == expected_shape, f"{name}: {out.shape} != {expected_shape}"

    ref = ref_conv_transpose(x, w, b, stride, padding, dilation, output_padding)
    assert_close(to_np(out), ref, msg=name)
    assert np.all(np.isfinite(to_np(out)))


def test_conv2dtranspose_forward():
    for case in FORWARD_CASES:
        check_conv_transpose_forward(case)


def check_conv_transpose_backward(case, seed=50):
    name, shape, wshape, stride, padding, dilation, output_padding, bias = case
    module, x, w, b = conv_transpose_setup(
        shape, wshape, stride, padding, dilation, output_padding, seed=seed, bias=bias
    )
    H_out, W_out = conv_transpose_out_size(
        shape, wshape, stride, padding, dilation, output_padding
    )
    g = rng(seed + 10).standard_normal(
        (shape[0], wshape[1], H_out, W_out)
    ).astype(np.float32)

    xt = from_np(x)
    xt.requires_grad = True
    out = module(xt)
    (out * from_np(g)).sum().backward()

    dx_ref, dw_ref, db_ref = ref_conv_transpose_backward(
        g, x, w, stride, padding, dilation, output_padding
    )
    assert_close(to_np(xt.grad), dx_ref, msg=f"{name} dx")
    assert_close(to_np(module.W.grad), dw_ref, msg=f"{name} dw")
    if bias:
        assert_close(to_np(module.bias.grad), db_ref, msg=f"{name} db")
    else:
        assert module.bias is None


def test_conv2dtranspose_backward():
    for case in FORWARD_CASES:
        check_conv_transpose_backward(case)


def test_conv2dtranspose_output_padding_is_zero_padding():
    """Cells only reachable through ``output_padding`` get no input contribution.

    PyTorch still adds ``bias`` there, so with a bias they equal the bias and
    with ``bias=False`` they are exactly zero.
    """
    shape = (2, 2, 5, 5)
    wshape = (2, 3, 3, 3)
    module, x, w, b = conv_transpose_setup(
        shape, wshape, (2, 2), (0, 0), (1, 1), (1, 1), seed=60
    )
    out = to_np(module(from_np(x)))
    assert out.shape[2:] == (12, 12)

    plain, _, _, _ = conv_transpose_setup(
        shape, wshape, (2, 2), (0, 0), (1, 1), (0, 0), seed=60
    )
    plain_out = to_np(plain(from_np(x)))
    assert np.allclose(plain_out, out[:, :, :11, :11], atol=1e-4, rtol=1e-4)
    assert np.allclose(out[:, :, 11, :], b[None, :, None], atol=1e-6)
    assert np.allclose(out[:, :, :, 11], b[None, :, None], atol=1e-6)

    no_bias, x_nb, _, _ = conv_transpose_setup(
        shape, wshape, (2, 2), (0, 0), (1, 1), (1, 1), seed=60, bias=False
    )
    out_nb = to_np(no_bias(from_np(x_nb)))
    assert np.all(out_nb[:, :, 11, :] == 0.0)
    assert np.all(out_nb[:, :, :, 11] == 0.0)


def test_conv2dtranspose_dw_split_reduction():
    """A spatial volume above the 2048 split size uses the two-stage reducer."""
    shape = (3, 2, 16, 16)
    wshape = (2, 2, 3, 3)
    stride, padding, dilation, output_padding = (2, 2), (1, 1), (1, 1), (0, 0)
    total_spatial = shape[0] * 31 * 31
    assert total_spatial > 2048

    check_conv_transpose_backward(
        ("split", shape, wshape, stride, padding, dilation, output_padding, True),
        seed=70,
    )


def test_conv2dtranspose_1x1():
    """A 1x1 kernel is a per-pixel channel mix, independent of stride/padding."""
    module, x, w, b = conv_transpose_setup(
        (2, 3, 4, 5), (3, 2, 1, 1), (1, 1), (0, 0), (1, 1), (0, 0), seed=80
    )
    out = module(from_np(x))
    assert out.shape == [2, 2, 4, 5]
    ref = np.einsum("bchw,cd->bdhw", x, w[:, :, 0, 0]) + b[None, :, None, None]
    assert_close(to_np(out), ref)


def test_conv2dtranspose_rejects_bad_output_padding():
    for kwargs in (
        {"stride": 2, "output_padding": 2},
        {"stride": 1, "dilation": 1, "output_padding": 1},
        {"stride": (2, 3), "output_padding": (1, 3)},
    ):
        try:
            Conv2DTranspose(2, 3, kernel=3, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"Conv2DTranspose accepted {kwargs}")


def test_conv2dtranspose_rejects_channel_mismatch():
    try:
        functional.conv2dtranspose(
            from_np(np.zeros((1, 3, 4, 4), dtype=np.float32)),
            from_np(np.zeros((2, 2, 3, 3), dtype=np.float32)),
            (1, 1),
            (0, 0),
            (1, 1),
            (0, 0),
        )
    except ValueError:
        return
    raise AssertionError("conv2dtranspose accepted mismatched input channels")


TESTS = [
    ("conv2d_forward", test_conv2d_forward),
    ("conv2d_bias_gradient", test_conv2d_bias_gradient),
    ("causal_conv1d_forward_backward", test_causal_conv1d_forward_backward),
    ("causal_conv1d_strided_input", test_causal_conv1d_strided_input),
    ("conv2dtranspose_forward", test_conv2dtranspose_forward),
    ("conv2dtranspose_backward", test_conv2dtranspose_backward),
    (
        "conv2dtranspose_output_padding_is_zero_padding",
        test_conv2dtranspose_output_padding_is_zero_padding,
    ),
    (
        "conv2dtranspose_dw_split_reduction",
        test_conv2dtranspose_dw_split_reduction,
    ),
    ("conv2dtranspose_1x1", test_conv2dtranspose_1x1),
    (
        "conv2dtranspose_rejects_bad_output_padding",
        test_conv2dtranspose_rejects_bad_output_padding,
    ),
    (
        "conv2dtranspose_rejects_channel_mismatch",
        test_conv2dtranspose_rejects_channel_mismatch,
    ),
]


if __name__ == "__main__":
    from common import run_all
    run_all(TESTS)
