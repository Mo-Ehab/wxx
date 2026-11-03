import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from common import from_np, to_np, assert_close, rng

from wxx import nn
from wxx.backend.context import inferenceMode


def test_dropout_keeps_graph():
    """Dropout must produce a differentiable output, not a detached tensor."""
    x = from_np(np.ones((4, 8), dtype=np.float32))
    x.requires_grad = True

    out = nn.Dropout(0.5)(x)
    assert out.requires_grad is True
    assert out.creator is not None

    out.sum().backward()
    grad = to_np(x.grad)
    # Training-mode dropout scales kept values by 1 / (1 - p).
    assert set(np.unique(grad).tolist()) <= {0.0, 2.0}
    assert np.count_nonzero(grad) > 0


def test_dropout_mask_matches_forward():
    """The backward mask must follow the mask used in the forward pass."""
    x = from_np(rng(0).standard_normal((6, 10)).astype(np.float32))
    x.requires_grad = True

    out = nn.Dropout(0.25)(x)
    out.sum().backward()

    scale = 1.0 / (1.0 - 0.25)
    assert_close(to_np(x.grad), (to_np(out) != 0).astype(np.float32) * scale)


def test_dropout_inference_is_identity():
    x = from_np(rng(1).standard_normal((5, 7)).astype(np.float32))
    drop = nn.Dropout(0.5)
    with inferenceMode():
        out = drop(x)
    assert out is x


TESTS = [
    ("dropout_keeps_graph", test_dropout_keeps_graph),
    ("dropout_mask_matches_forward", test_dropout_mask_matches_forward),
    ("dropout_inference_is_identity", test_dropout_inference_is_identity),
]


if __name__ == "__main__":
    from common import run_all
    run_all(TESTS)
