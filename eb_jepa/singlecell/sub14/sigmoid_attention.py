"""Sigmoid self-attention (faithful port of Subliminal 1.4).

Reference:
    Ramapuram, Danelljan, et al., "Theory, Analysis, and Best Practices
    for Sigmoid Self-Attention", arXiv:2409.04431 (2024).

Softmax attention forces every token's attention row to sum to one. For
a set of genes (no positional ordering, a sampled subset per view) that
constraint is artificial — a token may legitimately attend to many or
few others. Sigmoid attention drops it:

    attn = sigmoid( (Q Kᵀ) / sqrt(d_k) + b )

with ``b`` a learnable per-head scalar bias initialised to
``-log(seq_len)`` so the initial pattern mimics softmax-under-uniform.

Two activations are supported (``activation=``):
- ``"sigmoid"`` (default, 1.4's winning choice): pure-PyTorch path,
  per-head learnable bias, arbitrary key-padding mask, compiles cleanly.
- ``"softmax"``: standard SDPA baseline (picks up FlashAttention / mem-
  efficient kernels on capable GPUs); the per-head bias is unused.

The Apple FlashSigmoid CUDA kernel from the reference is intentionally
omitted (it needs an out-of-tree build); the pure-PyTorch sigmoid path
is what the tuned 1.4 runs used.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SigmoidAttention(nn.Module):
    """Multi-head attention with sigmoid- or softmax-gated activation."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        dropout: float = 0.0,
        expected_seq_len: int = 512,
        bias: bool = False,
        activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if activation not in ("sigmoid", "softmax"):
            raise ValueError(f"Unknown activation: {activation!r}; expected sigmoid | softmax")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = float(dropout)
        self.activation = activation

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

        # Per-head learnable scalar bias. Init to -log(L): sigmoid(b) ≈ 1/L
        # at init mimics softmax-under-uniform.
        init_bias = -math.log(float(max(expected_seq_len, 1)))
        self.attn_bias = nn.Parameter(torch.full((n_heads,), init_bias))

        self._scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: Tensor, *, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """``x``: (B, S, D). ``key_padding_mask``: (B, S) bool, True = pad."""
        if self.activation == "softmax":
            return self._forward_softmax(x, key_padding_mask)
        return self._forward_sigmoid(x, key_padding_mask)

    def _forward_sigmoid(self, x: Tensor, key_padding_mask: Optional[Tensor]) -> Tensor:
        b, s, d = x.shape
        h, dh = self.n_heads, self.head_dim

        q = self.q_proj(x).view(b, s, h, dh).transpose(1, 2)  # (B, H, S, Dh)
        k = self.k_proj(x).view(b, s, h, dh).transpose(1, 2)
        v = self.v_proj(x).view(b, s, h, dh).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self._scale  # (B, H, S, S)
        scores = scores + self.attn_bias.view(1, h, 1, 1)

        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        attn = torch.sigmoid(scores)
        if self.training and self.dropout_p > 0.0:
            attn = F.dropout(attn, p=self.dropout_p)

        out = torch.matmul(attn, v)  # (B, H, S, Dh)
        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return self.o_proj(out)

    def _forward_softmax(self, x: Tensor, key_padding_mask: Optional[Tensor]) -> Tensor:
        b, s, d = x.shape
        h, dh = self.n_heads, self.head_dim

        q = self.q_proj(x).view(b, s, h, dh).transpose(1, 2)
        k = self.k_proj(x).view(b, s, h, dh).transpose(1, 2)
        v = self.v_proj(x).view(b, s, h, dh).transpose(1, 2)

        attn_mask: Optional[Tensor] = None
        if key_padding_mask is not None:
            # SDPA bool attn_mask: True = keep. pad (True) -> False.
            attn_mask = (~key_padding_mask).view(b, 1, 1, s).expand(b, h, s, s)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(b, s, d)
        return self.o_proj(out)
