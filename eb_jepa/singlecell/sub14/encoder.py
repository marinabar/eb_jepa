"""Subliminal 1.4 transformer encoder (faithful port).

A pre-norm stack of RMSNorm + SigmoidAttention + SwiGLU blocks. Prepends
a learnable ``[CELL]`` token; the cell representation is that token's
final-layer output. Reuses eb_jepa's :class:`RMSNorm` (float32-stable,
bf16-friendly).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from eb_jepa.singlecell.layers import RMSNorm
from eb_jepa.singlecell.sub14.sigmoid_attention import SigmoidAttention
from eb_jepa.singlecell.sub14.swiglu import SwiGLUFFN


class PreNormSigmoidBlock(nn.Module):
    """Pre-norm block: RMSNorm + SigmoidAttention + RMSNorm + SwiGLU."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        *,
        expected_seq_len: int,
        attention_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.self_attn = SigmoidAttention(
            d_model,
            n_heads,
            dropout=dropout,
            expected_seq_len=expected_seq_len,
            activation=attention_activation,
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        normed = self.attn_norm(x)
        attn_out = self.self_attn(normed, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_out)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class EncoderSub14(nn.Module):
    """Pre-norm sigmoid-attention encoder with a learnable ``[CELL]`` token."""

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
        *,
        expected_seq_len: int = 512,
        attention_activation: str = "sigmoid",
        grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        self.cell_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cell_token, std=0.02)

        self.layers = nn.ModuleList(
            PreNormSigmoidBlock(
                d_model,
                n_heads,
                d_ff,
                dropout,
                expected_seq_len=expected_seq_len,
                attention_activation=attention_activation,
            )
            for _ in range(n_layers)
        )
        self.final_norm = RMSNorm(d_model)

    def forward(
        self,
        embeddings: torch.Tensor,
        *,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``embeddings``: (B, G, D) gene-token embeddings (CELL not prepended).

        ``padding_mask``: (B, 1+G) bool, True = ignore. Returns the
        ``[CELL]`` representation, (B, D).
        """
        from torch.utils.checkpoint import checkpoint

        batch_size = embeddings.size(0)
        cell = self.cell_token.expand(batch_size, -1, -1)  # (B, 1, D)
        x = torch.cat([cell, embeddings], dim=1)  # (B, 1+G, D)

        for layer in self.layers:
            if self.grad_checkpoint and self.training:
                x = checkpoint(
                    lambda inp, m=padding_mask, lyr=layer: lyr(inp, key_padding_mask=m),
                    x,
                    use_reentrant=False,
                )
            else:
                x = layer(x, key_padding_mask=padding_mask)
        x = self.final_norm(x)
        return x[:, 0, :]  # [CELL] token
