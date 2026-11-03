import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from common import from_np, to_np

from wxx.nn.resnet import ResNet18


def test_resnet_forward_backward():
    m = ResNet18(num_classes=5)
    x = from_np(np.random.default_rng(0).standard_normal((2, 3, 16, 16)).astype(np.float32))
    out = m(x)
    assert out.shape == [2, 5]
    assert np.isfinite(to_np(out)).all()
    out.mean().backward()
    grads = [p.grad for _, p in m.named_parameters() if p.grad is not None]
    assert len(grads) > 0


def test_resnet_state_dict_roundtrip():
    m1 = ResNet18(num_classes=5)
    state = m1.state_dict()
    m2 = ResNet18(num_classes=5)
    m2.load_state_dict(state)
    restored = m2.state_dict()
    assert set(restored) == set(state)
    for key, value in state.items():
        assert np.allclose(restored[key], value), key


TESTS = [
    ("resnet_forward_backward", test_resnet_forward_backward),
    ("resnet_state_dict_roundtrip", test_resnet_state_dict_roundtrip),
]

if __name__ == "__main__":
    from common import run_all
    run_all(TESTS)
