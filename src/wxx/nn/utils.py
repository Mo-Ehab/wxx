"""Utility helpers for ``wxx.nn`` (gradient clipping, ...)."""


import numpy as np

from .. import functional
from ..functional.fft import from_numpy_complex
from ..functional.utils import from_numpy


def _iter_params(parameters):
    """Normalize ``parameters`` into a flat list of tensors."""
    if hasattr(parameters, "parameters") and not hasattr(parameters, "grad"):
        parameters = parameters.parameters()
    out = []
    for p in parameters:
        if isinstance(p, tuple):
            p = p[1]
        if p is None:
            continue
        out.append(p)
    return out


def _grad_pairs(parameters):
    """Return ``(param, grad)`` pairs for every parameter with a gradient."""
    return [(p, p.grad) for p in _iter_params(parameters) if p.grad is not None]


def _clip_grad_norm_cpu(grads, max_norm, norm_type):
    """Fallback path: one readback per gradient, then scale on the CPU."""
    arrays = []
    if norm_type == float("inf"):
        total_norm = 0.0
        for _, g in grads:
            arr = g.numpy()
            arrays.append(arr)
            total_norm = max(total_norm, float(np.max(np.abs(arr))))
    else:
        total_sq = 0.0
        for _, g in grads:
            arr = g.numpy()
            arrays.append(arr)
            total_sq += float(np.sum(np.abs(arr) ** norm_type))
        total_norm = total_sq ** (1.0 / norm_type)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for (p, g), arr in zip(grads, arrays):
            scaled = np.asarray(arr) * clip_coef
            if g.type == "c64":
                functional.assign(g, from_numpy_complex(np.asarray(scaled)))
            elif g.type == "u32":
                functional.assign(g, from_numpy(scaled.astype(np.float32)))
            else:
                functional.assign(
                    g, from_numpy(np.asarray(scaled, dtype=np.float32))
                )
    return float(total_norm)


def clip_grad_norm_(parameters, max_norm, norm_type=2.0):
    """Clip the gradients of ``parameters`` in place.

    Mirrors ``torch.nn.utils.clip_grad_norm_``.  ``max_norm`` is the maximum
    allowed norm of the (flattened) gradients.  ``norm_type`` may be a finite
    number (Lp norm) or ``float('inf')`` (max norm).

    For the common ``norm_type=2`` case with float32 gradients the whole
    computation stays on the GPU: the squared norm is reduced on-device, only
    a single scalar is read back, and the gradients are rescaled by an
    elementwise kernel.  This avoids the per-parameter CPU round-trips that
    made clipping a major training bottleneck.

    Returns the total norm of the gradients (as a Python float).
    """
    grads = _grad_pairs(parameters)
    if not grads:
        return 0.0

    # Fast path: L2 norm of float32, offset-free gradients.
    if norm_type == 2.0 and all(
        g.type == "f32" and g.offset == 0 for _, g in grads
    ):
        total_sq = None
        for _, g in grads:
            partial = functional.sumsq(g)  # (1,) on the GPU
            if total_sq is None:
                total_sq = partial
            else:
                # in-place ``total_sq += partial``
                functional._pointwise(total_sq, partial, 0)

        # Exactly one device -> host synchronization per clipping call.
        total_norm = float(total_sq.item()) ** 0.5
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for p, g in grads:
                # ``op=2`` is multiply by the scalar; result stays on the GPU.
                p.grad = functional.scalarpointwise(g, clip_coef, 2)
        return float(total_norm)

    return _clip_grad_norm_cpu(grads, max_norm, norm_type)
