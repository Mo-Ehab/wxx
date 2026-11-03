"""GPT-2 style decoder-only transformer on ``wxx``."""

import numpy as np

from .. import functional
from ..functional.utils import from_numpy
from .module import (
    Dropout,
    Embedding,
    GELU,
    LayerNorm,
    Linear,
    Module,
    ModuleList,
    MultiHeadAttention,
    Sequential,
    TiedLinear,
)


class GPT2Block(Module):
    """Pre-LN decoder block: causal self-attention + GELU MLP."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.mlp = Sequential(
            Linear(d_model, 4 * d_model),
            GELU(),
            Linear(4 * d_model, d_model),
        )
        self.drop = Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.drop(self.attn(self.ln1(x), mask))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class GPT2(Module):
    """GPT-2-style decoder-only language model."""

    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6,
                 max_len=256, dropout=0.1, tie_weights=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tok_emb = Embedding(vocab_size, d_model, init_std=0.02)
        self.pos_emb = Embedding(max_len, d_model, init_std=0.02)
        self.blocks = ModuleList(
            [GPT2Block(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f = LayerNorm(d_model)
        self.head = (
            TiedLinear(self.tok_emb)
            if tie_weights
            else Linear(d_model, vocab_size, bias=False)
        )
        self._gpt2_init()

    def _gpt2_init(self):
        for name, p in self.named_parameters():
            if name.endswith(("tok_emb.weight", "pos_emb.weight")):
                continue
            if len(p.shape) >= 2:
                functional.assign(p, functional.normal(0, 0.02, p.shape))
            elif name.endswith("bias"):
                functional.assign(p, 0.0)

    def forward(self, x, mask=None):
        b, seq = x.shape
        if seq > self.max_len:
            raise ValueError(f"sequence length {seq} > max_len {self.max_len}")
        pos_arr = np.arange(seq, dtype=np.uint32).reshape(1, seq)
        pos = from_numpy(np.ascontiguousarray(pos_arr, dtype=np.uint32))
        h = self.tok_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h, mask)
        return self.head(self.ln_f(h))


__all__ = ["GPT2", "GPT2Block"]
