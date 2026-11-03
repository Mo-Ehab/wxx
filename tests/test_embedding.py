import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from common import from_np, from_np_u32, to_np, assert_close, rng

from wxx.nn.module import Embedding


def _embedding_ref(weight, ids):
    return weight[ids]


def test_embedding_forward_covers_all_elements():
    """Regression: the kernel must write every ``num_tokens * embed_dim`` element.

    The old dispatch launched one thread per *token* instead of one per output
    element, so with ``embed_dim > 1`` only the first ``num_tokens`` values were
    written and the rest of the output stayed zero.  At training scale this made
    all batches except the first collapse to identical (zero) embeddings.
    """
    V, D, B, L = 64, 32, 8, 300
    ids = rng(0).integers(0, V, size=(B, L)).astype(np.uint32)

    emb = Embedding(V, D)
    out = emb(from_np_u32(ids))

    assert out.shape == [B, L, D]
    assert np.count_nonzero(to_np(out)) == B * L * D, "embedding left zeros in the output"
    assert_close(to_np(out), _embedding_ref(to_np(emb.weight), ids), atol=1e-6)


def test_embedding_batch_independence():
    V, D, B, L = 32, 8, 4, 16
    ids = np.zeros((B, L), dtype=np.uint32)
    for b in range(B):
        ids[b] = (np.arange(L) + b * 5) % V

    emb = Embedding(V, D)
    out = to_np(emb(from_np_u32(ids)))

    for i in range(B):
        for j in range(i + 1, B):
            assert not np.allclose(out[i], out[j]), f"batch {i} and {j} are identical"
    assert_close(out, _embedding_ref(to_np(emb.weight), ids), atol=1e-6)


def test_embedding_backward():
    V, D, B, L = 7, 3, 2, 5
    ids = rng(1).integers(0, V, size=(B, L)).astype(np.uint32)
    g = rng(2).standard_normal((B, L, D)).astype(np.float32)

    emb = Embedding(V, D)
    out = emb(from_np_u32(ids))
    out.grad = from_np(g)
    out.backward()

    ref = np.zeros((V, D), dtype=np.float32)
    for b in range(B):
        for l in range(L):
            ref[ids[b, l]] += g[b, l]

    assert_close(to_np(emb.weight.grad), ref, atol=1e-6)


def test_embedding_backward_large():
    """Backward must accumulate over all tokens too (not just the first ones)."""
    V, D, B, L = 64, 32, 8, 300
    ids = rng(3).integers(0, V, size=(B, L)).astype(np.uint32)
    g = np.ones((B, L, D), dtype=np.float32)

    emb = Embedding(V, D)
    out = emb(from_np_u32(ids))
    out.grad = from_np(g)
    out.backward()

    ref = np.zeros((V, D), dtype=np.float32)
    for b in range(B):
        for l in range(L):
            ref[ids[b, l]] += 1.0

    assert_close(to_np(emb.weight.grad), ref, atol=1e-6)


TESTS = [
    ("embedding_forward_covers_all_elements", test_embedding_forward_covers_all_elements),
    ("embedding_batch_independence", test_embedding_batch_independence),
    ("embedding_backward", test_embedding_backward),
    ("embedding_backward_large", test_embedding_backward_large),
]


if __name__ == "__main__":
    from common import run_all
    run_all(TESTS)
