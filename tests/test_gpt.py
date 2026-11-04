import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from common import from_np, from_np_u32, to_np

from wxx.nn.gpt import GPT2


def test_gpt2_forward_backward():
    V = 32
    m = GPT2(V, d_model=16, n_heads=4, n_layers=2, max_len=8)
    ids = np.random.default_rng(0).integers(0, V, size=(2, 6)).astype(np.uint32)
    x = from_np_u32(ids)
    out = m(x)
    assert out.shape == [2, 6, V]
    assert np.isfinite(to_np(out)).all()
    out.mean().backward()
    grads = [p.grad for _, p in m.named_parameters() if p.grad is not None]
    assert len(grads) > 0


def test_gpt2_shape_error():
    V = 16
    m = GPT2(V, d_model=8, n_heads=2, n_layers=1, max_len=4)
    ids = np.random.default_rng(1).integers(0, V, size=(1, 8)).astype(np.uint32)
    x = from_np_u32(ids)
    try:
        m(x)
        raised = False
    except ValueError:
        raised = True
    assert raised, "expected ValueError for seq > max_len"


def test_gpt2_state_dict_roundtrip():
    V = 24
    cfg = dict(d_model=16, n_heads=4, n_layers=2, max_len=8)
    m1 = GPT2(V, **cfg)
    state = m1.state_dict()
    m2 = GPT2(V, **cfg)
    m2.load_state_dict(state)
    restored = m2.state_dict()
    assert set(restored) == set(state)
    for key, value in state.items():
        assert np.allclose(restored[key], value), key


def test_gpt2_is_causal():
    """Changing a token must not change logits at earlier positions.

    Regression test: ``forward`` used to pass ``mask=None`` through to
    attention, so every position attended to the future.  The model could
    still fit the training loss but could not be sampled autoregressively.
    """
    V = 32
    m = GPT2(V, d_model=16, n_heads=4, n_layers=2, max_len=8, dropout=0.0)
    ids = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.uint32)
    changed = ids.copy()
    changed[0, -1] = 7

    before = to_np(m(from_np_u32(ids)))
    after = to_np(m(from_np_u32(changed)))

    assert np.allclose(before[0, :-1], after[0, :-1], atol=1e-5), (
        "logits at earlier positions changed when only the last token changed"
    )
    assert not np.allclose(before[0, -1], after[0, -1]), (
        "the last position ignored its own token"
    )


def test_gpt2_causal_mask_shape():
    m = GPT2(16, d_model=8, n_heads=2, n_layers=1, max_len=6, dropout=0.0)
    mask = to_np(m.causal_mask(4))

    assert mask.shape == (4, 4)
    assert not np.any(np.tril(mask)), "allowed positions must add 0"
    assert np.all(mask[np.triu_indices(4, k=1)] < -1e8), (
        "future positions must be masked out"
    )
    assert np.array_equal(mask, to_np(m.causal_mask(4))), "mask is not cached"


def test_gpt2_generate():
    V, T = 24, 10
    m = GPT2(V, d_model=16, n_heads=4, n_layers=1, max_len=T, dropout=0.0)
    prompt = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint32)

    out = m.generate(prompt, max_new_tokens=4, temperature=0.0)
    assert out.shape == (2, prompt.shape[1] + 4)
    assert out.dtype == np.uint32
    assert np.array_equal(out[:, : prompt.shape[1]], prompt)

    # Greedy sampling is deterministic; the context stops at max_len.
    assert np.array_equal(out, m.generate(prompt, max_new_tokens=4, temperature=0.0))
    assert m.generate(prompt, max_new_tokens=100, temperature=1.0).shape == (2, T)

    # A single (seq,) prompt is one batch row.
    assert m.generate(prompt[0], max_new_tokens=2, temperature=1.0).shape == (1, 5)


def test_gpt2_generate_restores_mode():
    from wxx.backend import context

    m = GPT2(16, d_model=8, n_heads=2, n_layers=1, max_len=8, dropout=0.0)
    prompt = np.array([[1, 2, 3]], dtype=np.uint32)
    m.generate(prompt, max_new_tokens=1, temperature=0.0)
    assert context.MODE == 0


TESTS = [
    ("gpt2_forward_backward", test_gpt2_forward_backward),
    ("gpt2_shape_error", test_gpt2_shape_error),
    ("gpt2_state_dict_roundtrip", test_gpt2_state_dict_roundtrip),
    ("gpt2_is_causal", test_gpt2_is_causal),
    ("gpt2_causal_mask_shape", test_gpt2_causal_mask_shape),
    ("gpt2_generate", test_gpt2_generate),
    ("gpt2_generate_restores_mode", test_gpt2_generate_restores_mode),
]

if __name__ == "__main__":
    from common import run_all
    run_all(TESTS)
