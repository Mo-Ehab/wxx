"""Save/load helpers for wxx models and tensors.

Models are serialized as ``.npz`` archives containing one array per parameter
(keyed by the dotted parameter name from ``Module.state_dict``).
"""

import os

import numpy as np

from .nn.module import Module


def _to_arrays(obj):
    if isinstance(obj, Module):
        obj = obj.state_dict()
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if hasattr(v, "numpy") and not isinstance(v, np.ndarray):
                v = v.numpy()
            out[str(k)] = np.asarray(v)
        return out
    # Single tensor / array-like
    if hasattr(obj, "numpy") and not isinstance(obj, np.ndarray):
        obj = obj.numpy()
    return {"tensor": np.asarray(obj)}


def save(obj, path):
    """Save a model, state_dict, or tensor to ``path`` (``.npz``)."""
    path = str(path)
    if not path.endswith(".npz"):
        path = path + ".npz"
    arrays = _to_arrays(obj)
    np.savez(path, **arrays)
    return path


def load(path, model=None):
    """Load a ``.npz`` archive.

    If ``model`` is given, its state_dict is populated and the model is
    returned. Otherwise a dict of numpy arrays is returned.
    """
    path = str(path)
    if not path.endswith(".npz") and not os.path.exists(path):
        path = path + ".npz"
    with np.load(path) as data:
        state = {k: data[k] for k in data.files}
    if model is not None:
        model.load_state_dict(state)
        return model
    return state
