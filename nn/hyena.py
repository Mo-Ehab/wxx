"""Hyena sequence mixer (Poli et al., 2023) on ``wxx`` FFT primitives.

This is the original Hyena architecture, with no alternative code paths.

Long filter
-----------
The filter of stage ``i`` is parameterised implicitly, as in the paper
(eq. 7 and appendix D.3)::

    h_t       = Window(t) * (FFN o PositionalEncoding)(t)
    PE(t)     = [t, Re[r_0](t) .. Re[r_{K-1}](t), Im[r_0](t) .. Im[r_{K-1}](t)]
    r_k(t)    = exp(i * 2*pi*k*t / L)                      k = 0 .. K-1
    Window(t) = exp(-|decay| * t)

``PE`` has dimension ``2K + 1``; the FFN uses the sinusoidal activation
``sigma(x) = sin(omega * x)``.  The paper's defaults, ``K = 8`` and
``omega = 10``, give filters with full-spectrum content at initialization
(a scalar ``t`` input instead produces "excessively smooth" filters, which the
paper shows converge to worse solutions).

Positional information enters **only through the filters**.  The token stream
is never given an absolute position, so the operator remains
translation-equivariant, as in the paper.

Operator
--------
The order-``n`` operator is

.. code-block:: text

    v      = short_conv(W_v x)
    x_i    = short_conv(chunk(W_x x, n)[i])     for i = 1..n
    z_0    = v
    z_i    = x_i * (h_i * z_{i-1})              for i = 1..n
    y      = W_o z_n

with one shared depthwise causal short convolution and ``*`` the long
convolution, evaluated with an FFT.

Modules
-------
``Sine``               -- ``sin(omega * x)`` activation.
``PositionalEncoding`` -- truncated complex-exponential basis.
``CausalShortConv1d``  -- depthwise causal 1-D convolution.
``ImplicitFilter``     -- synthesizes a long-convolution filter.
``HyenaOperator``      -- order-``n`` gated long convolution.
``SwiGLU``             -- gated feed-forward network.
``HyenaBlock``         -- RMSNorm + Hyena + RMSNorm + SwiGLU.
``Hyena``              -- causal language model.
"""

import numpy as np

from .. import autograd, functional
from ..Tensor import Parameter, Tensor
from .module import (
    Embedding,
    Linear,
    Module,
    ModuleList,
    PositionalEncoding,
    RMSNorm,
    Sequential,
    Sine,
    SwiGLU,
    TiedLinear,
)


def _next_pow2(n):
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


class CausalShortConv1d(Module):
    """Depthwise causal 1-D convolution with left padding.

    Note: wxx evaluates ``out[n] = sum_k w[k] * x[n-k]``, which is torch's
    ``Conv1d`` with its kernel flipped.  When porting ``nn.Conv1d`` weights
    from PyTorch, reverse the last axis of the kernel.
    """

    def __init__(self, d_model, kernel_size=3):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size

        self.weight = Parameter(Tensor((d_model, kernel_size), value=0.0))
        self.bias = Parameter(Tensor((d_model,), value=0.0))

        functional.assign(
            self.weight, functional.normal(0, 0.02, (d_model, kernel_size))
        )
        functional.assign(self.bias, functional.from_numpy(np.zeros(d_model, np.float32)))

    def forward(self, x):
        # x: (B, D, L).  The fused kernel consumes strided views directly and
        # computes out[..., n] = sum_k w[:, k] * x[..., n-k] in one dispatch.
        return autograd.causal_conv1d(x, self.weight, self.bias).forward(
            x, self.weight, self.bias
        )


class ImplicitFilter(Module):
    """Synthesize a long-convolution filter from its positional encoding.

    ``h(t) = FFN(PositionalEncoding(t))`` with ``t`` evenly spaced over
    ``[0, 1]``.  The FFN is ``Linear(2K+1, hidden) -> Sine(omega) ->
    Linear(hidden, d_model)``.  Returns ``(d_model, L)`` so it can be
    multiplied directly with the FFT of a ``(B, d_model, L)`` signal.
    """

    def __init__(self, d_model, hidden_dim=64, pe_features=8, omega=10.0):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.pe_features = pe_features
        self.positional_encoding = PositionalEncoding(pe_features)
        self.mlp = Sequential(
            Linear(self.positional_encoding.out_features, hidden_dim),
            Sine(omega),
            Linear(hidden_dim, d_model),
        )

    def forward(self, L):
        pos = self.positional_encoding(L)  # (L, 2K + 1)
        filter_weights = self.mlp(pos)  # (L, d_model)
        return filter_weights.transpose(0, 1)  # (d_model, L)


class HyenaOperator(Module):
    """Order-``order`` causal Hyena operator.

    Parameters
    ----------
    d_model:
        Channel width.
    order:
        Number of gating stages ``n``.
    fft_size:
        Optional fixed FFT length.  Defaults to ``next_pow2(2 * L)`` so the
        linear convolution does not wrap around.
    short_conv_kernel:
        Kernel size of the shared depthwise causal short convolution.
    filter_hidden:
        Hidden width of each implicit-filter MLP.
    pe_features:
        ``K``, the number of complex-exponential positional features; each
        filter sees ``2K + 1`` inputs.
    filter_omega:
        Frequency of the filter's sinusoidal activation.
    """

    def __init__(self, d_model, order=2, fft_size=None, short_conv_kernel=3,
                 filter_hidden=64, pe_features=8, filter_omega=10.0):
        super().__init__()
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        if short_conv_kernel < 1:
            raise ValueError(
                f"short_conv_kernel must be >= 1, got {short_conv_kernel}"
            )

        self.d_model = d_model
        self.order = int(order)
        self.fft_size = fft_size

        # One matrix producing the value projection plus every gate.
        self.proj_v = Linear(d_model, d_model, bias=False)
        self.proj_x = Linear(d_model, self.order * d_model, bias=False)
        self.out_proj = Linear(d_model, d_model, bias=False)

        # A single depthwise short conv, reused for v and for every gate.
        self.short_conv = CausalShortConv1d(d_model, kernel_size=short_conv_kernel)

        self.filter_mlp = ModuleList(
            [
                ImplicitFilter(
                    d_model,
                    hidden_dim=filter_hidden,
                    pe_features=pe_features,
                    omega=filter_omega,
                )
                for _ in range(self.order)
            ]
        )

        # ``torch.rand(order, dim)`` -- a per-(stage, channel) decay rate.
        self.decay = Parameter(Tensor((self.order, d_model), value=0.0))
        functional.assign(
            self.decay,
            functional.from_numpy(
                np.random.rand(self.order, d_model).astype(np.float32)
            ),
        )

    def _fft_size(self, length):
        if self.fft_size is not None:
            size = int(self.fft_size)
        else:
            size = _next_pow2(2 * length)
        if size < length:
            size = _next_pow2(length)
        return size

    def fft_conv1d(self, x, h):
        """Linear convolution of ``(B, D, L)`` with ``(D, L)``."""
        length = x.shape[-1]
        fft_size = self._fft_size(length)
        spectrum = x.rfft(fft_size).complex_mul(h.rfft(fft_size))
        out = spectrum.irfft(fft_size)
        if fft_size != length:
            out = out[..., :length]
        return out

    def _decay_window(self, stage, length):
        """``exp(-|decay_stage| * t)`` as a ``(d_model, length)`` tensor."""
        return autograd.decay_window(self.decay).forward(
            self.decay, stage, length
        )

    def forward(self, x):
        _, length, _ = x.shape

        v = self.proj_v(x)
        gates = list(self.proj_x(x).chunk(self.order, dim=-1))

        v = v.transpose(1, 2)
        gates = [gate.transpose(1, 2) for gate in gates]

        v = self.short_conv(v)
        gates = [self.short_conv(gate) for gate in gates]

        z = v
        for i in range(self.order):
            h = self.filter_mlp[i](length)  # (D, L)
            h = h * self._decay_window(i, length)
            z = gates[i] * self.fft_conv1d(z, h)

        return self.out_proj(z.transpose(1, 2))


class HyenaBlock(Module):
    """Pre-norm RMSNorm block: ``x + Hyena(norm(x))``, ``x + SwiGLU(norm(x))``."""

    def __init__(self, d_model, mlp_ratio=4, hidden_dim=None, order=2,
                 short_conv_kernel=3, filter_hidden=64, pe_features=8,
                 filter_omega=10.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.mixer = HyenaOperator(
            d_model,
            order=order,
            short_conv_kernel=short_conv_kernel,
            filter_hidden=filter_hidden,
            pe_features=pe_features,
            filter_omega=filter_omega,
        )
        self.norm2 = RMSNorm(d_model)
        self.feed_forward = SwiGLU(
            d_model, hidden_dim=hidden_dim, mlp_ratio=mlp_ratio
        )

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class Hyena(Module):
    """Causal Hyena language model.

    ``order`` controls the Hyena operator's gating depth and ``n_layers`` the
    number of blocks.  ``pe_features`` (``K``) and ``filter_omega`` are the
    paper's filter parameters: ``K = 8`` features and a sinusoidal activation
    frequency of ``10``.
    """

    def __init__(self, vocab_size, d_model=256, n_layers=8, mlp_ratio=4,
                 hidden_dim=None, tie_weights=True, embed_init_std=0.02,
                 order=2, short_conv_kernel=3, filter_hidden=64,
                 pe_features=8, filter_omega=10.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.order = order

        # GPT-2-style embedding scale: with a tied head, ``N(0, 1)`` logits
        # start ~sqrt(d_model) wide and the initial cross-entropy is ~250.
        self.embed = Embedding(vocab_size, d_model, init_std=embed_init_std)
        self.layers = ModuleList(
            [
                HyenaBlock(
                    d_model,
                    mlp_ratio=mlp_ratio,
                    hidden_dim=hidden_dim,
                    order=order,
                    short_conv_kernel=short_conv_kernel,
                    filter_hidden=filter_hidden,
                    pe_features=pe_features,
                    filter_omega=filter_omega,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        # ``tie_weights=False`` keeps a separate biased LM head.
        self.head = TiedLinear(self.embed) if tie_weights else Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for block in self.layers:
            h = block(h)
        return self.head(self.norm(h))


__all__ = [
    "Sine",
    "PositionalEncoding",
    "CausalShortConv1d",
    "ImplicitFilter",
    "HyenaOperator",
    "SwiGLU",
    "HyenaBlock",
    "TiedLinear",
    "Hyena",
]
