"""SwiGLU gated feed-forward network (Shazeer 2020; used in LLaMA).

Faithful port of Subliminal 1.4's ``SwiGLUFFN``. The effective hidden
dim is ``d_ff`` directly (no 2/3 rescale): both the gate and up
projections map to ``d_ff``, then the down projection maps back to
``d_model``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """out = w_down( SiLU(w_gate(x)) * w_up(x) ). No biases."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))
