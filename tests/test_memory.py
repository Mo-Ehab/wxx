import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from common import from_np_u32, rng

from wxx.backend.device import get_buffer_pool
from wxx.nn.gpt import GPT2
from wxx.nn.loss import CCELoss
from wxx.nn.optimizer import AdamW


def _active(pool):
    snapshot = pool.get_memory_snapshot()
    return len(snapshot["active"]), snapshot["active_bytes"]


def test_training_step_reuses_memory():
    """A training loop must not accumulate GPU buffers across steps.

    Regression test: the dropout layer used to return a tensor with
    ``requires_grad=False``, which detached the attention sub-graph.  The
    abandoned tensors formed reference cycles that were only reclaimed by a
    generation-2 GC pass (never scheduled automatically on Python 3.14), so
    every step leaked activations until the GPU ran out of memory.
    """
    pool = get_buffer_pool()
    vocab, d_model, n_heads, n_layers, seq, batch = 64, 16, 2, 2, 8, 2

    model = GPT2(
        vocab,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_len=seq,
        tie_weights=True,
    )
    loss_fn = CCELoss()
    optim = AdamW(list(model.parameters()), lr=1e-4)

    gen = rng(7)
    x_np = gen.integers(0, vocab, (batch, seq)).astype(np.uint32)
    y_np = gen.integers(0, vocab, (batch, seq)).astype(np.uint32)

    def train_step():
        x = from_np_u32(x_np)
        y = from_np_u32(y_np)

        out = model(x)
        b, s, v = out.shape
        loss = loss_fn(out.reshape((b * s, v)), y.reshape((b * s,)))
        loss.backward()
        loss.item()
        optim.step()
        optim.zero_grad()

    # Warm up: the buffer pool caches one buffer per (size, usage, label).
    for _ in range(3):
        train_step()

    count_before, bytes_before = _active(pool)
    for _ in range(8):
        train_step()
    count_after, bytes_after = _active(pool)

    assert count_after <= count_before + 4, (
        f"active GPU buffers grew from {count_before} to {count_after} "
        "over 8 training steps"
    )
    assert bytes_after <= bytes_before + 64 * 1024, (
        f"active GPU bytes grew from {bytes_before} to {bytes_after} "
        "over 8 training steps"
    )


TESTS = [
    ("training_step_reuses_memory", test_training_step_reuses_memory),
]


if __name__ == "__main__":
    from common import run_all
    run_all(TESTS)
