import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gc

import numpy as np
from common import assert_close, from_np, rng, to_np

from wxx import Parameter, Tensor, functional


def test_detach_returns_view_without_grad():
    a = rng(0).standard_normal((3, 4)).astype(np.float32)
    x = from_np(a)
    x.requires_grad = True

    d = x.detach()
    assert d.shape == [3, 4]
    assert d.type == "f32"
    assert d.requires_grad is False
    assert d.creator is None
    assert d.grad is None
    assert d.buffer is x.buffer, "detach must share storage, not copy it"
    assert_close(to_np(d), a)


def test_detach_blocks_gradient_flow():
    x = from_np(np.ones((2, 3), dtype=np.float32))
    x.requires_grad = True

    y = (x.detach() * 3.0).sum()
    assert y.requires_grad is False
    try:
        y.backward()
    except RuntimeError:
        pass
    else:
        raise AssertionError("backward through a detached tensor should fail")
    assert x.grad is None


def test_detach_stops_only_its_branch():
    """Mixing a detached and a live branch must send grad to the live one."""
    x = from_np(np.ones((2, 3), dtype=np.float32))
    x.requires_grad = True

    z = (x.detach() + x).sum()
    assert z.requires_grad is True
    z.backward()
    assert_close(to_np(x.grad), np.ones((2, 3), dtype=np.float32))


def test_detach_shares_storage():
    a = rng(1).standard_normal((2, 5)).astype(np.float32)
    x = from_np(a)
    d = x.detach()

    functional.assign(d, 0.0)
    assert_close(to_np(x), np.zeros_like(a), msg="detach should alias storage")
    assert_close(to_np(d), np.zeros_like(a))


def test_detach_of_non_contiguous_view():
    a = rng(2).standard_normal((4, 6)).astype(np.float32)
    x = from_np(a)
    view = x[:, 1:5]
    d = view.detach()

    assert d.shape == [4, 4]
    assert not d.is_contiguous()
    assert_close(to_np(d), a[:, 1:5])

    functional.assign(d, -1.0)
    expected = a.copy()
    expected[:, 1:5] = -1.0
    assert_close(to_np(x), expected)


def test_detach_keeps_storage_alive():
    a = rng(3).standard_normal((2, 3)).astype(np.float32)
    d = from_np(a).detach()
    gc.collect()
    assert_close(to_np(d), a)


def test_detach_inplace():
    x = from_np(np.ones((2, 2), dtype=np.float32))
    x.requires_grad = True

    y = x * 2.0
    assert y.creator is not None
    out = y.detach_()

    assert out is y
    assert y.requires_grad is False
    assert y.creator is None
    assert y.grad is None


def test_detach_parameter():
    p = Parameter(Tensor((3,), value=2.0))
    d = p.detach()

    assert isinstance(d, Tensor)
    assert not isinstance(d, Parameter)
    assert d.requires_grad is False
    assert d.buffer is p.buffer
    assert_close(to_np(d), np.full(3, 2.0, dtype=np.float32))


def test_detach_preserves_dtype():
    from common import from_np_u32
    from wxx.functional.fft import from_numpy_complex

    ids = from_np_u32(np.arange(6, dtype=np.uint32).reshape(2, 3))
    d_ids = ids.detach()
    assert d_ids.type == "u32"
    assert_close(to_np(d_ids), np.arange(6).reshape(2, 3))

    z = np.ones((2, 2), dtype=np.complex64) * (1 + 2j)
    zt = from_numpy_complex(z)
    d_z = zt.detach()
    assert d_z.type == "c64"
    assert_close(to_np(d_z), z, atol=0)


def test_detach_stop_gradient_through_module():
    """``module(x.detach())`` must train the module but not reach ``x``."""
    from wxx.nn.module import LayerNorm

    x = from_np(np.ones((2, 4), dtype=np.float32))
    x.requires_grad = True
    ln = LayerNorm(4)

    out = ln(x.detach()).sum()
    assert out.requires_grad is True
    out.backward()

    assert x.grad is None, "gradient crossed the detach boundary"
    assert ln.weight.grad is not None, "module parameters must still learn"


TESTS = [
    ("detach_returns_view_without_grad", test_detach_returns_view_without_grad),
    ("detach_blocks_gradient_flow", test_detach_blocks_gradient_flow),
    ("detach_stops_only_its_branch", test_detach_stops_only_its_branch),
    ("detach_shares_storage", test_detach_shares_storage),
    ("detach_of_non_contiguous_view", test_detach_of_non_contiguous_view),
    ("detach_keeps_storage_alive", test_detach_keeps_storage_alive),
    ("detach_inplace", test_detach_inplace),
    ("detach_parameter", test_detach_parameter),
    ("detach_preserves_dtype", test_detach_preserves_dtype),
    ("detach_stop_gradient_through_module", test_detach_stop_gradient_through_module),
]


if __name__ == "__main__":
    from common import run_all
    run_all(TESTS)
