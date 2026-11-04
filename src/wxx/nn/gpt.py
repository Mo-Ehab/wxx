"""GPT-2 style decoder-only transformer on ``wxx``."""

import numpy as np

from .. import functional
from ..backend import context
from ..functional.utils import from_numpy
from ..Tensor import Tensor
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
        self._causal_mask_cache = None

    def _gpt2_init(self):
        for name, p in self.named_parameters():
            if name.endswith(("tok_emb.weight", "pos_emb.weight")):
                continue
            if len(p.shape) >= 2:
                functional.assign(p, functional.normal(0, 0.02, p.shape))
            elif name.endswith("bias"):
                functional.assign(p, 0.0)

    def causal_mask(self, seq):
        """Additive ``(seq, seq)`` mask: ``0`` on/below the diagonal, ``-1e9`` above.

        The full ``(max_len, max_len)`` mask is built once and sliced for
        shorter sequences, so training and generation share a single buffer.
        """
        mask = self._causal_mask_cache
        if mask is None:
            rows = np.arange(self.max_len, dtype=np.int64)[:, None]
            cols = np.arange(self.max_len, dtype=np.int64)[None, :]
            data = np.where(cols <= rows, 0.0, -1e9).astype(np.float32)
            mask = from_numpy(np.ascontiguousarray(data))
            self._causal_mask_cache = mask
        if seq == self.max_len:
            return mask
        return mask.narrow(0, 0, seq).narrow(1, 0, seq)

    def forward(self, x, mask=None):
        b, seq = x.shape
        if seq > self.max_len:
            raise ValueError(f"sequence length {seq} > max_len {self.max_len}")
        # Without a causal mask every position attends to the future, which
        # turns the model into a bidirectional encoder that cannot be sampled
        # autoregressively.  An explicit ``mask`` overrides the causal one.
        if mask is None:
            mask = self.causal_mask(seq)
        pos_arr = np.arange(seq, dtype=np.uint32).reshape(1, seq)
        pos = from_numpy(np.ascontiguousarray(pos_arr, dtype=np.uint32))
        h = self.tok_emb(x) + self.pos_emb(pos)
        for block in self.blocks:
            h = block(h, mask)
        return self.head(self.ln_f(h))

    def generate(self, prompt, max_new_tokens=64, temperature=1.0, top_k=None,
                 stop_token=None, rng=None):
        """Sample a continuation from ``prompt`` one token at a time.

        ``prompt`` is a ``(batch, seq)`` array of token ids (a NumPy array or
        a ``u32`` tensor; a ``(seq,)`` array is treated as a batch of one).
        Returns the prompt plus the generated ids as a ``uint32`` array.
        Generation stops early once every row has emitted ``stop_token``, and
        never exceeds ``max_len`` context tokens.  ``temperature=0`` selects
        the argmax.
        """
        if isinstance(prompt, Tensor):
            ids = prompt.numpy()
        else:
            ids = np.asarray(prompt)
        ids = np.ascontiguousarray(ids, dtype=np.uint32)
        if ids.ndim == 1:
            ids = ids[None, :]
        if ids.ndim != 2:
            raise ValueError(f"prompt must be 1D or 2D, got shape {ids.shape}")
        batch, seq = ids.shape
        if seq == 0:
            raise ValueError("prompt must contain at least one token")
        if seq > self.max_len:
            raise ValueError(f"prompt length {seq} > max_len {self.max_len}")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if rng is None:
            rng = np.random.default_rng()

        steps = min(int(max_new_tokens), self.max_len - seq)

        # Dropout must be off; restore the caller's mode instead of assuming
        # training so nested use does not silently exit inference mode.
        previous_mode = context.MODE
        context.MODE = 1
        try:
            for _ in range(steps):
                logits = self.forward(from_numpy(ids, dtype=np.uint32)).numpy()
                step_logits = logits[:, -1, :].astype(np.float64)

                if temperature == 0.0:
                    next_ids = step_logits.argmax(axis=-1)
                else:
                    step_logits = step_logits / temperature
                    if top_k is not None:
                        k = min(int(top_k), step_logits.shape[-1])
                        if k < step_logits.shape[-1]:
                            cutoffs = np.partition(step_logits, -k, axis=-1)[:, -k]
                            step_logits[step_logits < cutoffs[:, None]] = -np.inf
                    step_logits -= step_logits.max(axis=-1, keepdims=True)
                    probs = np.exp(step_logits)
                    probs /= probs.sum(axis=-1, keepdims=True)
                    draws = rng.random((batch, 1))
                    next_ids = (np.cumsum(probs, axis=-1) > draws).argmax(axis=-1)

                next_ids = next_ids.astype(np.uint32)
                ids = np.concatenate([ids, next_ids[:, None]], axis=1)
                if stop_token is not None and np.all(next_ids == stop_token):
                    break
        finally:
            context.MODE = previous_mode
        return ids


__all__ = ["GPT2", "GPT2Block"]

