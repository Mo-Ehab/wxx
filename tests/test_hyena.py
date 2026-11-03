import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from common import from_np, from_np_u32, to_np, assert_close, rng

from wxx.nn.hyena import Hyena, HyenaOperator, TiedLinear


def test_hyena_batch_independence():
    """Distinct batch members must produce distinct logits.

    Regression for the embedding dispatch bug: at these shapes the old kernel
    only embedded the first few tokens, so every batch after the first (and
    most positions of the first) produced identical zero logits.
    """
    V, D, B, L = 16, 32, 3, 32
    ids = np.zeros((B, L), dtype=np.uint32)
    for b in range(B):
        ids[b] = (np.arange(L) + b * 5) % V

    model = Hyena(V, d_model=D, n_layers=1)
    out = to_np(model(from_np_u32(ids)))

    assert out.shape == (B, L, V)
    for i in range(B):
        for j in range(i + 1, B):
            assert not np.allclose(out[i], out[j]), f"batch {i} and {j} logits are identical"
    # every position should have received real (non-zero) signal
    assert np.count_nonzero(out) == out.size


def test_hyena_causal():
    """Changing a future token must not change earlier logits."""
    V, D, B, L = 16, 16, 1, 24
    ids = rng(0).integers(0, V, size=(B, L)).astype(np.uint32)
    ids2 = ids.copy()
    ids2[0, 16:] = (ids2[0, 16:] + 3) % V

    model = Hyena(V, d_model=D, n_layers=1)
    a = to_np(model(from_np_u32(ids)))
    b = to_np(model(from_np_u32(ids2)))

    assert_close(a[:, :16], b[:, :16], atol=1e-3, msg="prefix changed (not causal)")
    assert not np.allclose(a[:, 16:], b[:, 16:]), "future tokens had no effect"


def test_hyena_backward():
    V, D, B, L = 16, 16, 2, 16
    ids = rng(1).integers(0, V, size=(B, L)).astype(np.uint32)

    model = Hyena(V, d_model=D, n_layers=1)
    logits = model(from_np_u32(ids))
    logits.mean().backward()

    grads = [p.grad for _, p in model.parameters()]
    assert all(g is not None for g in grads)
    assert any(np.any(to_np(g) != 0) for g in grads)


def test_hyena_weight_tying():
    """The LM head must reuse the embedding matrix, matching PyTorch.

    PyTorch's reference model ties ``lm_head.weight`` to ``embedding.weight``
    and uses ``bias=False``.  wxx stores the embedding as ``(vocab, d_model)``
    and the linear as ``(in, out)``, so tying is done with ``transpose_B``.
    """
    V, D, N = 32, 16, 2
    model = Hyena(V, d_model=D, n_layers=N)

    assert isinstance(model.head, TiedLinear)
    assert model.head._embedding.weight is model.embed.weight

    names = [n for n, _ in model.named_parameters()]
    assert names.count("embed.weight") == 1, names
    assert not any(n.startswith("head.") for n in names), names

    # The tied head must not add parameters of its own.
    total = sum(int(np.prod(p.shape)) for _, p in model.parameters())
    assert total == _expected_params(V, D, N, order=2), total


def _expected_params(V, D, N, order, k=3, filter_hidden=64,
                     mlp_ratio=4, pe_features=8):
    """Structural parameter count for ``Hyena``.

    Per block (RMSNorm + Hyena + RMSNorm + SwiGLU, all linears bias-free
    except the depthwise short conv)::

        D                                    norm1
        + (2 + order) * D**2                 proj_v, proj_x, out_proj
        + (k * D + D)                        shared short conv
        + order * (2*h + h*D + D)            filter MLPs
                                             Linear(2K+1, h), Linear(h, D)
        + order * D                          decay
        + D                                  norm2
        + 3 * D * ffn                        SwiGLU w1/w2/w3, ffn = 8D/3

    The model adds ``V*D`` for the (tied) embedding and ``D`` for the final
    RMSNorm.
    """
    ffn = int(2 * (mlp_ratio * D) / 3)
    filter_in = 2 * pe_features + 1
    block = (
        D
        + (2 + order) * D * D
        + (k * D + D)
        + order * (filter_in * filter_hidden + filter_hidden + filter_hidden * D + D)
        + order * D
        + D
        + 3 * D * ffn
    )
    return N * block + V * D + D


def test_hyena_order_param_counts():
    """Each extra order step adds one gate column, filter MLP and decay row."""
    V, D, N = 32, 16, 2
    for order in (1, 2, 3, 4):
        model = Hyena(V, d_model=D, n_layers=N, order=order)
        total = sum(int(np.prod(p.shape)) for _, p in model.parameters())
        expected = _expected_params(V, D, N, order)
        assert total == expected, (order, total, expected)

    # The reference HyenaLLM config (vocab=2048, dim=256, depth=6, order=4).
    # Without the filter positional encoding it has 6,444,288 parameters; the
    # paper's PE adds N * order * 2K * filter_hidden = 24,576 at K=8.
    V, D, N, order = 2048, 256, 6, 4
    model = Hyena(V, d_model=D, n_layers=N, order=order)
    total = sum(int(np.prod(p.shape)) for _, p in model.parameters())
    assert total == _expected_params(V, D, N, order), total
    assert total == 6_444_288 + 6 * 4 * 2 * 8 * 64, total


def test_hyena_order_forward_backward():
    """Every order must produce finite logits and non-zero gradients."""
    V, D, B, L = 16, 16, 2, 16
    ids = rng(2).integers(0, V, size=(B, L)).astype(np.uint32)

    for order in (1, 3):
        model = Hyena(V, d_model=D, n_layers=1, order=order)
        logits = model(from_np_u32(ids))
        out = to_np(logits)
        assert out.shape == (B, L, V)
        assert np.isfinite(out).all(), f"order={order} produced non-finite logits"

        logits.mean().backward()
        grads = [p.grad for _, p in model.parameters()]
        assert all(g is not None for g in grads), f"order={order} missing gradients"
        assert any(np.any(to_np(g) != 0) for g in grads), f"order={order} zero grads"


def test_hyena_order_causality():
    """Higher orders must stay causal: future tokens cannot affect the prefix."""
    V, D, B, L = 16, 16, 1, 24
    ids = rng(3).integers(0, V, size=(B, L)).astype(np.uint32)
    ids2 = ids.copy()
    ids2[0, 16:] = (ids2[0, 16:] + 3) % V

    model = Hyena(V, d_model=D, n_layers=1, order=3)
    a = to_np(model(from_np_u32(ids)))
    b = to_np(model(from_np_u32(ids2)))

    assert_close(a[:, :16], b[:, :16], atol=1e-3, msg="order=3 prefix changed")
    assert not np.allclose(a[:, 16:], b[:, 16:]), "future tokens had no effect"


def test_hyena_short_conv_kernel():
    """Only the shared causal short conv depends on the kernel size."""
    V, D, B, L = 16, 16, 2, 16
    ids = rng(4).integers(0, V, size=(B, L)).astype(np.uint32)

    narrow = Hyena(V, d_model=D, n_layers=1, order=2)
    wide = Hyena(V, d_model=D, n_layers=1, order=2, short_conv_kernel=5)

    n_narrow = sum(int(np.prod(p.shape)) for _, p in narrow.parameters())
    n_wide = sum(int(np.prod(p.shape)) for _, p in wide.parameters())
    assert n_narrow == _expected_params(V, D, 1, 2, k=3)
    assert n_wide == _expected_params(V, D, 1, 2, k=5)
    assert n_wide - n_narrow == 2 * D, (n_wide - n_narrow)

    out = to_np(wide(from_np_u32(ids)))
    assert out.shape == (B, L, V)
    assert np.isfinite(out).all()


def test_hyena_reference_structure():
    """The model must match ``HyenaLLM``: RMSNorm/SwiGLU blocks, tied head."""
    from wxx.nn.hyena import HyenaBlock, SwiGLU
    from wxx.nn.module import RMSNorm

    V, D, N = 64, 32, 2
    model = Hyena(V, d_model=D, n_layers=N, order=3)

    assert isinstance(model.norm, RMSNorm)
    assert isinstance(model.head, TiedLinear)
    for block in model.layers:
        assert isinstance(block, HyenaBlock)
        assert isinstance(block.norm1, RMSNorm)
        assert isinstance(block.norm2, RMSNorm)
        assert isinstance(block.feed_forward, SwiGLU)
        # Reference SwiGLU width: int(2 * (4 * dim) / 3).
        assert block.feed_forward.hidden_dim == int(2 * (4 * D) / 3)

    # One shared short conv, ``order`` filters, ``order * D`` decay rates.
    mixer = model.layers[0].mixer
    assert tuple(mixer.proj_x.weight.shape) == (D, 3 * D)
    assert tuple(mixer.decay.shape) == (3, D)
    assert len(mixer.filter_mlp) == 3
    assert mixer.proj_v.bias is None and mixer.out_proj.bias is None


def test_hyena_decay_window():
    """``exp(-|decay| * t)`` must be causal, monotone and start at 1."""
    V, D = 32, 8
    model = Hyena(V, d_model=D, n_layers=1, order=2)
    mixer = model.layers[0].mixer
    win = to_np(mixer._decay_window(0, 16))

    assert win.shape == (D, 16)
    assert_close(win[:, 0], np.ones(D, np.float32), atol=1e-6, msg="window[0] != 1")
    assert (win > 0).all()
    assert (np.diff(win, axis=-1) <= 1e-6).all(), "window is not monotone"


def test_hyena_state_dict_roundtrip():
    """``state_dict`` must capture every parameter so a reload is identical."""
    V, D, B, L = 32, 16, 2, 12
    ids = rng(5).integers(0, V, size=(B, L)).astype(np.uint32)

    a = Hyena(V, d_model=D, n_layers=2, order=3)
    sd = a.state_dict()

    b = Hyena(V, d_model=D, n_layers=2, order=3)
    b.load_state_dict(sd)

    out_a = to_np(a(from_np_u32(ids)))
    out_b = to_np(b(from_np_u32(ids)))
    assert_close(out_a, out_b, atol=1e-6, msg="reloaded model differs")

    # The tied embedding must round-trip too (it is not listed under head.*).
    assert "embed.weight" in sd
    assert not any(k.startswith("head.") for k in sd)


def test_hyena_positional_encoding_values():
    """PE must match ``[t, cos(2*pi*k*t/L), sin(2*pi*k*t/L)]`` from the paper."""
    from wxx.nn.hyena import PositionalEncoding

    L, K = 8, 4
    pe = PositionalEncoding(K)
    got = to_np(pe(L))
    assert got.shape == (L, 2 * K + 1), got.shape

    t = np.linspace(0.0, 1.0, L)
    reference = np.concatenate(
        [t[:, None]]
        + [np.cos(2 * np.pi * k * t)[:, None] for k in range(K)]
        + [np.sin(2 * np.pi * k * t)[:, None] for k in range(K)],
        axis=1,
    )
    assert_close(got, reference, atol=1e-6, msg="positional encoding values differ")

    # Deterministic: repeated calls must return the same (cached) tensor.
    assert pe(L) is pe(L)


def test_hyena_filter_pe_param_counts():
    """Each filter's first linear must have ``2K + 1`` inputs (appendix D.3)."""
    V, D, N, order = 32, 16, 2, 3

    for K in (4, 8, 16):
        model = Hyena(V, d_model=D, n_layers=N, order=order, pe_features=K)
        total = sum(int(np.prod(p.shape)) for _, p in model.parameters())
        assert total == _expected_params(V, D, N, order, pe_features=K), (K, total)
        mlp = model.layers[0].mixer.filter_mlp[0].mlp
        assert tuple(mlp._modules["0"].weight.shape) == (2 * K + 1, 64)

    # The paper's defaults: K = 8, sinusoidal activation with omega = 10.
    from wxx.nn.hyena import Sine

    default = Hyena(V, d_model=D, n_layers=N, order=order)
    filt = default.layers[0].mixer.filter_mlp[0]
    assert filt.pe_features == 8
    assert isinstance(filt.mlp._modules["1"], Sine)
    assert filt.mlp._modules["1"].omega == 10.0
    n_default = sum(int(np.prod(p.shape)) for _, p in default.parameters())
    assert n_default == _expected_params(V, D, N, order)


def test_hyena_filter_pe_forward_backward():
    """The paper filters must keep the model finite, causal and differentiable."""
    V, D, B, L = 16, 16, 2, 24
    ids = rng(6).integers(0, V, size=(B, L)).astype(np.uint32)
    ids2 = ids.copy()
    ids2[0, 16:] = (ids2[0, 16:] + 3) % V

    model = Hyena(V, d_model=D, n_layers=1, order=2, pe_features=8)
    out = to_np(model(from_np_u32(ids)))
    assert out.shape == (B, L, V)
    assert np.isfinite(out).all()

    # The PE is a property of the filter, so the mixer stays causal.
    other = to_np(model(from_np_u32(ids2)))
    assert_close(out[:, :16], other[:, :16], atol=1e-3, msg="PE broke causality")

    model(from_np_u32(ids)).mean().backward()
    grads = {n: p.grad for n, p in model.named_parameters()}
    filter_grad = grads["layers.0.mixer.filter_mlp.0.mlp.0.weight"]
    assert filter_grad is not None
    assert np.isfinite(to_np(filter_grad)).all()
    assert np.any(to_np(filter_grad) != 0), "filter input weights got no gradient"

    # The decay window is part of the architecture and must be differentiable.
    decay_grad = grads["layers.0.mixer.decay"]
    assert decay_grad is not None
    assert np.isfinite(to_np(decay_grad)).all()


def test_hyena_no_legacy_options():
    """The model exposes the paper architecture only, with no compatibility toggles."""
    import inspect

    import wxx.nn as nn

    params = inspect.signature(Hyena.__init__).parameters
    for removed in ("filter_activation", "use_decay", "filter_pe_features"):
        assert removed not in params, f"{removed} should not be an option"
    for kept in ("pe_features", "filter_omega", "short_conv_kernel", "order"):
        assert kept in params, f"{kept} missing from Hyena"
    for gone in ("SiLU", "FeedForward", "HyenaLM"):
        assert not hasattr(nn, gone), f"wxx.nn.{gone} should be removed"


TESTS = [
    ("hyena_batch_independence", test_hyena_batch_independence),
    ("hyena_causal", test_hyena_causal),
    ("hyena_backward", test_hyena_backward),
    ("hyena_weight_tying", test_hyena_weight_tying),
    ("hyena_order_param_counts", test_hyena_order_param_counts),
    ("hyena_order_forward_backward", test_hyena_order_forward_backward),
    ("hyena_order_causality", test_hyena_order_causality),
    ("hyena_short_conv_kernel", test_hyena_short_conv_kernel),
    ("hyena_reference_structure", test_hyena_reference_structure),
    ("hyena_decay_window", test_hyena_decay_window),
    ("hyena_state_dict_roundtrip", test_hyena_state_dict_roundtrip),
    ("hyena_positional_encoding_values", test_hyena_positional_encoding_values),
    ("hyena_filter_pe_param_counts", test_hyena_filter_pe_param_counts),
    ("hyena_filter_pe_forward_backward", test_hyena_filter_pe_forward_backward),
    ("hyena_no_legacy_options", test_hyena_no_legacy_options),
]


if __name__ == "__main__":
    from common import run_all
    run_all(TESTS)
