import math

import numpy as np

from wxx.functional.utils import from_numpy, normal

from .. import autograd
from ..backend import context



from .. import functional
from ..Tensor import Parameter, Tensor
from ..backend.device import get_cmd_buffer

cmd_encoder = get_cmd_buffer()

class Module:
    def __init__(self):
        super().__setattr__("_parameters", {})
        super().__setattr__("_modules", {})

    def __setattr__(self, name: str, value):

        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, (list, tuple)):
            # Register any modules nested in a plain list/tuple.
            for item in value:
                if isinstance(item, Module):
                    pass

        super().__setattr__(name, value)

    def __call__(self, *x):
        out = self.forward(*x)
        return out

    def parameters(self, recurse=True):
        for param in self._parameters.values():
            if param is not None:
                yield ("", param)

        if recurse:
            for module in self._modules.values():
                if module is not None:
                    yield from module.parameters(recurse=True)

    def named_parameters(self, prefix="", recurse=True):
        """Yield ``(name, parameter)`` pairs for all (nested) parameters."""
        for name, param in self._parameters.items():
            if param is not None:
                yield (prefix + name, param)
        if recurse:
            for name, module in self._modules.items():
                if module is not None:
                    yield from module.named_parameters(
                        prefix + name + ".", recurse=True
                    )

    def state_dict(self, prefix="", destination=None):
        """Return a dict mapping parameter names to CPU numpy arrays."""
        if destination is None:
            destination = {}
        for name, param in self._parameters.items():
            if param is not None:
                arr = param.numpy() if hasattr(param, "numpy") else param
                destination[prefix + name] = np.asarray(arr).copy()
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(prefix + name + ".", destination)
        return destination

    def load_state_dict(self, state_dict, strict=True):
        """Load parameters from a dict (or numpy archive) into this module."""
        own_keys = set(self.state_dict().keys())
        for name, param in self.named_parameters():
            if name in state_dict:
                val = state_dict[name]
                if hasattr(val, "numpy"):
                    val = (
                        val.numpy()
                        if hasattr(val, "numpy") and not isinstance(val, np.ndarray)
                        else val
                    )
                val = np.asarray(val)
                if param.type == "c64":
                    from wxx.functional.fft import from_numpy_complex

                    functional.assign(param, from_numpy_complex(val))
                elif param.type == "u32":
                    functional.assign(param, from_numpy(val, dtype=np.uint32))
                else:
                    functional.assign(param, from_numpy(val))
            elif strict:
                raise KeyError(f"Missing key in state_dict: {name}")
        if strict:
            missing = own_keys - set(state_dict.keys())
            unexpected = set(state_dict.keys()) - own_keys
            if missing:
                raise RuntimeError(f"Missing keys: {sorted(missing)}")
            if unexpected:
                raise RuntimeError(f"Unexpected keys: {sorted(unexpected)}")

    def zero_grad(self, wait=False):
        for name, param in self.named_parameters():
            if param.grad is not None:
                param.grad = None
        cmd_encoder.flush(wait=wait)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        lines = [self.__class__.__name__ + "("]
        for name, mod in self._modules.items():
            lines.append(f"  ({name}): {repr(mod)}")
        lines.append(")")
        return "\n".join(lines)


class ModuleList(Module):
    def __init__(self, modules=None):
        super().__init__()
        if modules is not None:
            for i, m in enumerate(modules):
                setattr(self, str(i), m)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self._modules[str(i)] for i in range(*idx.indices(len(self)))]
        return self._modules[str(idx)]

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

    def append(self, module):
        idx = len(self._modules)
        setattr(self, str(idx), module)

    def forward(self, x, *args, **kwargs):
        # Default: apply sequentially (mirrors Sequential).  Subclasses/usage
        # may override; ModuleList is mostly a container.
        for m in self._modules.values():
            x = m(x, *args, **kwargs)
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.weight = Parameter(Tensor((in_features, out_features)))
        functional._kaiming_normal_init(self.weight, in_features)

        if bias:
            self.bias = Parameter(Tensor((out_features,), value=0.0))
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is None:
            return autograd.matmul(x, self.weight).forward(x, self.weight)
        return autograd.matmuladd(x, self.weight, self.bias).forward(x, self.weight, self.bias)


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, init_std=0.02):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(Tensor((num_embeddings, embedding_dim)))
        functional.assign(self.weight, normal(0, init_std, (num_embeddings, embedding_dim)))

    def forward(self, x):
        n = x.shape[0] * x.shape[1] if len(x.shape) > 1 else x.shape[0]
        return autograd.Embedding(x, self.weight).forward(
            x, self.weight, n, self.embedding_dim, self.num_embeddings
        )


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if context.MODE != 0:
            return x
        return autograd.Dropout(x).forward(x, self.p)


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weightg = Parameter(Tensor(self.normalized_shape, value=1.0))
        self.weightb = Parameter(Tensor(self.normalized_shape, value=0.0))
        self.weight = self.weightg
        self.bias = self.weightb

    def forward(self, x):
        axis = tuple(range(len(x.shape) - len(self.normalized_shape), len(x.shape)))
        axis_arg = axis if len(axis) > 1 else axis[0]
        return autograd.meanvarnorm(x, self.weightg, self.weightb).forward(
            x, self.weightg, self.weightb, axis=axis_arg
        )


class RMSNorm(Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = Parameter(Tensor((d_model,), value=1.0))

    def forward(self, x):
        return autograd.rmsnorm(x, self.weight).forward(x, self.weight, -1)


class BatchNorm(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(Tensor((num_features,), value=1.0))
        self.bias = Parameter(Tensor((num_features,), value=0.0))
        self.running_mean = Tensor((num_features,), value=0.0)
        self.running_std = Tensor((num_features,), value=1.0)

    def forward(self, x):
        training = context.MODE == 0
        return autograd.batchnorm(x, self.weight, self.bias).forward(
            x, self.weight, self.bias, self.running_mean, self.running_std, training
        )


class Conv2D(Module):
    def __init__(
        self, in_channels, out_channels, kernel=3, stride=1,
        padding=0, dilation=1, bias=True
    ):
        super().__init__()
        if isinstance(kernel, int):
            kernel = (kernel, kernel)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.W = Parameter(Tensor((out_channels, in_channels, kernel[0], kernel[1])))
        functional._kaiming_normal_init(self.W, math.prod(kernel) * in_channels)
        if bias:
            self.bias = Parameter(Tensor((out_channels,), value=0.0))
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is None:
            return autograd.Conv2D(x, self.W).forward(
                x, self.W, None, self.stride, self.padding, self.dilation
            )
        return autograd.Conv2D(x, self.W, self.bias).forward(
            x, self.W, self.bias, self.stride, self.padding, self.dilation
        )


class Conv2DTranspose(Module):
    """Transposed 2-D convolution (``ConvTranspose2d``).

    ``W`` has shape ``(in_channels, out_channels, KH, KW)``: every input cell
    scatters into ``stride``-spaced output windows, so the spatial size grows to
    ``(H - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1``.
    """

    def __init__(
        self, in_channels, out_channels, kernel=3, stride=1,
        padding=0, output_padding=0, dilation=1, bias=True
    ):
        super().__init__()
        if isinstance(kernel, int):
            kernel = (kernel, kernel)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(output_padding, int):
            output_padding = (output_padding, output_padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        for axis, name in enumerate(("height", "width")):
            limit = max(stride[axis], dilation[axis])
            if not 0 <= output_padding[axis] < limit:
                raise ValueError(
                    f"output_padding[{axis}] must be in [0, {limit}) for "
                    f"{name}, got {output_padding[axis]}"
                )

        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation

        self.W = Parameter(Tensor((in_channels, out_channels, kernel[0], kernel[1])))
        functional._kaiming_normal_init(self.W, math.prod(kernel) * in_channels)
        if bias:
            self.bias = Parameter(Tensor((out_channels,), value=0.0))
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is None:
            return autograd.Conv2DTranspose(x, self.W).forward(
                x,
                self.W,
                None,
                self.stride,
                self.padding,
                self.output_padding,
                self.dilation,
            )
        return autograd.Conv2DTranspose(x, self.W, self.bias).forward(
            x,
            self.W,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
        )


class MaxPool2d(Module):
    def __init__(self, kernel, stride=None, padding=0):
        super().__init__()
        if isinstance(kernel, int):
            kernel = (kernel, kernel)
        if stride is None:
            stride = kernel
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return autograd.maxpooling2d(x).forward(x, self.kernel, self.stride, self.padding)


class AvgPool2d(Module):
    def __init__(self, kernel, stride=None, padding=0):
        super().__init__()
        if isinstance(kernel, int):
            kernel = (kernel, kernel)
        if stride is None:
            stride = kernel
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return autograd.avgpooling2d(x).forward(x, self.kernel, self.stride, self.padding)


class AdaptiveAvgPool2d(Module):
    """Adaptive average pooling with a fixed output spatial size."""

    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = tuple(int(v) for v in output_size)

    def forward(self, x):
        return autograd.adaptive_avgpooling2d(x).forward(x, self.output_size)


class MultiHeadAttention(Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = Linear(d_model, d_model, bias=False)
        self.k = Linear(d_model, d_model, bias=False)
        self.v = Linear(d_model, d_model, bias=False)
        self.out = Linear(d_model, d_model, bias=False)
        self.drop = Dropout(dropout)

    def forward(self, x, mask=None):
        b, seq, _ = x.shape
        q = (
            self.q(x)
            .reshape((b, seq, self.n_heads, self.head_dim))
            .permute(0, 2, 1, 3)
            .reshape((b * self.n_heads, seq, self.head_dim))
        )
        k = (
            self.k(x)
            .reshape((b, seq, self.n_heads, self.head_dim))
            .permute(0, 2, 1, 3)
            .reshape((b * self.n_heads, seq, self.head_dim))
        )
        v = (
            self.v(x)
            .reshape((b, seq, self.n_heads, self.head_dim))
            .permute(0, 2, 1, 3)
            .reshape((b * self.n_heads, seq, self.head_dim))
        )

        score = (
            autograd.matmul(q, k).forward(q, k, transpose_b=True)
            * (1.0 / math.sqrt(self.head_dim))
        )
        if mask is not None:
            score = score + mask
        attn = score.softmax(dim=-1)
        attn = self.drop(attn)
        out = autograd.matmul(attn, v).forward(attn, v)
        out = (
            out.reshape((b, self.n_heads, seq, self.head_dim))
            .permute(0, 2, 1, 3)
            .reshape((b, seq, self.d_model))
        )
        return self.out(out)


class Attention(Module):
    def __init__(self, d_model, n_heads=1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)

    def forward(self, x, mask=None):
        return self.mha(x, mask)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        for i, m in enumerate(modules):
            setattr(self, str(i), m)

    def forward(self, x, *args, **kwargs):
        for m in self._modules.values():
            x = m(x, *args, **kwargs)
        return x


class GELU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.gelu()


# ---- auxiliary classes (shared with hyena.py) ----

class Sine(Module):
    def __init__(self, omega=10.0):
        super().__init__()
        self.omega = float(omega)

    def forward(self, x):
        return (x * self.omega).sin()

    def extra_repr(self):
        return f"omega={self.omega}"


class SwiGLU(Module):
    def __init__(self, d_model, hidden_dim=None, mlp_ratio=4):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * (mlp_ratio * d_model) / 3)
        self.hidden_dim = hidden_dim
        self.w1 = Linear(d_model, hidden_dim, bias=False)
        self.w2 = Linear(d_model, hidden_dim, bias=False)
        self.w3 = Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.w3(self.w1(x).silu() * self.w2(x))


class TiedLinear(Module):
    def __init__(self, embedding):
        super().__init__()
        object.__setattr__(self, "_embedding", embedding)

    def forward(self, x):
        weight = self._embedding.weight
        return autograd.matmul(x, weight).forward(x, weight, False, True)


class PositionalEncoding(Module):
    def __init__(self, features=8):
        super().__init__()
        features = int(features)
        if features < 1:
            raise ValueError(f"positional features must be >= 1, got {features}")
        self.features = features
        self.out_features = 2 * features + 1
        self._cache = {}

    def _build(self, length):
        import numpy as np

        t = np.linspace(0.0, 1.0, int(length), dtype=np.float32)
        k = np.arange(self.features, dtype=np.float32)
        angles = (2.0 * np.pi) * np.outer(t, k)
        return np.concatenate(
            [t[:, None], np.cos(angles), np.sin(angles)], axis=1
        ).astype(np.float32)

    def forward(self, length):
        key = (int(length), self.features)
        pe = self._cache.get(key)
        if pe is None:
            pe = Tensor(
                (key[0], self.out_features),
                data=self._build(length).tobytes(),
            )
            self._cache[key] = pe
        return pe

    def extra_repr(self):
        return f"features={self.features}, out_features={self.out_features}"
