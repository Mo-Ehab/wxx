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


TESTS = [
    ("gpt2_forward_backward", test_gpt2_forward_backward),
    ("gpt2_shape_error", test_gpt2_shape_error),
    ("gpt2_state_dict_roundtrip", test_gpt2_state_dict_roundtrip),
]

if __name__ == "__main__":
    from common import run_all
    run_all(TESTS)
