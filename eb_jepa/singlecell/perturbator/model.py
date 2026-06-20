"""FiLM-conditioned perturbator network (CLAUDE.md "Architectures / objectives to test").

v1 operates on the encoder's **pooled** latents (one vector per cell) — the encoder
is frozen. The perturbator maps a *source* (control) latent to its *perturbed*
counterpart, conditioned on the drug action (SMILES features + dose) via **FiLM**:
an MLP embeds the action vector, and per-block ``(gamma, beta)`` modulate the
hidden state (``h <- gamma * h + beta``) after each RMSNorm.

The backbone is a stack of per-cell residual blocks (RMSNorm -> FiLM -> SwiGLU),
i.e. it treats the source cloud as a set and acts on each cell independently. This
keeps constant ``[N, d_model]`` shapes (compile-friendly) and is permutation-
equivariant over the source distribution, which is exactly what an OT map needs. A
final residual head predicts the *displacement* from the source latent, so at init
(zero head) the perturbator is the identity — a strong, stable starting point.

The design anticipates later consuming the **full token set** instead of the pooled
latent: swap the per-cell blocks for self-attention blocks over the token dimension
(the FiLM conditioning and the residual-displacement head carry over unchanged).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from eb_jepa.singlecell.layers import RMSNorm, SwiGLU


class ActionEmbedding(nn.Module):
    """MLP embedding of the raw action vector ``[*, action_dim] -> [*, d_cond]``."""

    def __init__(self, action_dim: int, d_cond: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or max(d_cond, 4 * 2)
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_cond),
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.net(action)


class FiLMBlock(nn.Module):
    """Per-cell residual block with FiLM conditioning: RMSNorm -> FiLM -> SwiGLU.

    The conditioning ``cond`` (``[*, d_cond]``) produces ``(gamma, beta)`` that
    modulate the normalized hidden state before the SwiGLU FFN. ``gamma`` is
    initialized to 1 and ``beta`` to 0 (FiLM linear zero-init) so the block starts
    as a vanilla pre-norm FFN.
    """

    def __init__(self, d_model: int, d_cond: int, residual_scale: float = 1.0):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.film = nn.Linear(d_cond, 2 * d_model)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)  # gamma=1 (via +1), beta=0 at init
        self.ffn = SwiGLU(d_model)
        self.residual_scale = residual_scale

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        x = self.norm(h)
        x = (1.0 + gamma) * x + beta
        return h + self.residual_scale * self.ffn(x)


class Perturbator(nn.Module):
    """Map source latents ``[N, d_model]`` (+ action) -> predicted perturbed latents.

    Args:
        d_model: latent dimension (must match the frozen encoder's output).
        action_dim: dimension of the raw action vector (``DrugFeaturizer.action_dim``).
        depth: number of FiLM residual blocks.
        d_cond: action-embedding (conditioning) width.
        cond_hidden: hidden width of the action-embedding MLP.

    Predicts a residual displacement on top of the source latent, so the output is
    ``source + head(blocks(source, cond))``; zero-init on the head makes the model
    the identity at initialization.
    """

    def __init__(
        self,
        d_model: int,
        action_dim: int,
        depth: int = 4,
        d_cond: int = 256,
        cond_hidden: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim
        self.action_embed = ActionEmbedding(action_dim, d_cond, cond_hidden)
        residual_scale = (2.0 * max(1, depth)) ** -0.5
        self.in_proj = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList(
            FiLMBlock(d_model, d_cond, residual_scale=residual_scale)
            for _ in range(depth)
        )
        self.out_norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)  # identity at init

    def forward(self, source: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source: ``[N, d_model]`` source (control) latents.
            action: ``[action_dim]`` (one action for the whole source cloud) or
                ``[N, action_dim]`` (per-cell action).
        Returns:
            ``[N, d_model]`` predicted perturbed latents.
        """
        if action.dim() == 1:
            cond = self.action_embed(action).unsqueeze(0)  # [1, d_cond], broadcasts
        else:
            cond = self.action_embed(action)  # [N, d_cond]
        h = self.in_proj(source)
        for block in self.blocks:
            h = block(h, cond)
        return source + self.head(self.out_norm(h))
