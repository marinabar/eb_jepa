"""FiLM-conditioned perturbator network (CLAUDE.md "Architectures / objectives to test").

v1 operates on the encoder's **pooled** latents (one vector per cell) — the encoder
is frozen. The perturbator maps a *source* (control) latent to its *perturbed*
counterpart, conditioned on the drug action (SMILES features + dose) via **FiLM**:
an MLP embeds the action vector, and per-block ``(gamma, beta)`` modulate the
hidden state (``h <- gamma * h + beta``) after each RMSNorm.

The backbone is a stack of per-cell residual blocks (RMSNorm -> FiLM -> SwiGLU),
i.e. it treats the source cloud as a set and acts on each cell independently. This
keeps constant ``[N, d_model]`` shapes (compile-friendly) and is permutation-
equivariant over the source distribution, which is exactly what an OT map needs.

The same FiLM trunk drives **two selectable objectives** (CLAUDE.md Part II):

- **direct**: a residual-displacement head predicts the perturbed latent directly
  (``out = source + head(trunk(source, cond))``); zero-init on the head makes the
  perturbator the identity at init — a strong, stable start trained by the sliced-
  Wasserstein OT loss.
- **flow_matching**: the trunk is also conditioned on a diffusion **time** ``t`` and
  becomes a velocity field ``v_theta(x_t, t, action)`` regressed (conditional /
  rectified flow matching) to the source->target displacement ``x1 - x0``. Inference
  integrates the ODE ``dx/dt = v_theta`` from a source latent to a predicted
  perturbed latent (see ``flow.py``). Zero-init on the head makes ``v == 0`` at init,
  so the ODE is the identity map at init.

The design anticipates later consuming the **full token set** instead of the pooled
latent: swap the per-cell blocks for self-attention blocks over the token dimension
(the FiLM conditioning and the residual head carry over unchanged).
"""

from __future__ import annotations

import math

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


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding ``t in [0, 1] -> [*, d_cond]`` (flow matching).

    Maps the scalar diffusion time to a fixed sinusoidal feature, then an MLP to
    ``d_cond`` so it can be summed with the action conditioning. ``t`` may be a
    scalar, ``[N]`` (per-cell time), or already ``[N, 1]``.
    """

    def __init__(self, d_cond: int, n_freqs: int = 64):
        super().__init__()
        self.n_freqs = int(n_freqs)
        self.net = nn.Sequential(
            nn.Linear(2 * self.n_freqs, d_cond),
            nn.GELU(),
            nn.Linear(d_cond, d_cond),
        )
        # geometric frequencies (standard diffusion-time embedding band)
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(1000.0), self.n_freqs)
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(t, dtype=self.freqs.dtype, device=self.freqs.device)
        if t.dim() == 0:
            t = t.reshape(1)
        t = t.reshape(-1, 1)  # [N, 1]
        ang = t * self.freqs.unsqueeze(0) * (2.0 * math.pi)  # [N, n_freqs]
        feat = torch.cat([ang.sin(), ang.cos()], dim=-1)  # [N, 2*n_freqs]
        return self.net(feat)  # [N, d_cond]


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
    """FiLM trunk over source latents ``[N, d_model]`` (+ action [+ time]).

    Args:
        d_model: latent dimension (must match the frozen encoder's output).
        action_dim: dimension of the raw action vector (``DrugFeaturizer.action_dim``).
        depth: number of FiLM residual blocks.
        d_cond: action-embedding (conditioning) width.
        cond_hidden: hidden width of the action-embedding MLP.
        time_conditioned: if True, add a sinusoidal time embedding to the
            conditioning so the trunk is a velocity field ``v(x_t, t, action)``
            (flow matching). If False the trunk ignores time (direct mode).
        n_time_freqs: number of sinusoidal frequencies for the time embedding.

    A zero-init residual head means the model is the **identity** at init (direct
    mode) and the velocity is **zero** at init (flow-matching mode), both stable
    starting points.
    """

    def __init__(
        self,
        d_model: int,
        action_dim: int,
        depth: int = 4,
        d_cond: int = 256,
        cond_hidden: int | None = None,
        time_conditioned: bool = False,
        n_time_freqs: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.action_dim = action_dim
        self.time_conditioned = bool(time_conditioned)
        self.action_embed = ActionEmbedding(action_dim, d_cond, cond_hidden)
        self.time_embed = (
            TimeEmbedding(d_cond, n_freqs=n_time_freqs) if self.time_conditioned else None
        )
        residual_scale = (2.0 * max(1, depth)) ** -0.5
        self.in_proj = nn.Linear(d_model, d_model)
        self.blocks = nn.ModuleList(
            FiLMBlock(d_model, d_cond, residual_scale=residual_scale)
            for _ in range(depth)
        )
        self.out_norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)  # identity (direct) / zero velocity (flow) at init

    def _cond(
        self, action: torch.Tensor, n: int, t: torch.Tensor | None
    ) -> torch.Tensor:
        """Build the per-cell conditioning ``[N, d_cond]`` from action [+ time].

        ``action`` is ``[action_dim]`` (one action for the cloud, broadcast) or
        ``[N, action_dim]`` (per-cell). ``t`` (flow mode) is a scalar or ``[N]``.
        """
        if action.dim() == 1:
            cond = self.action_embed(action).unsqueeze(0).expand(n, -1)  # [N, d_cond]
        else:
            cond = self.action_embed(action)  # [N, d_cond]
        if self.time_conditioned:
            if t is None:
                raise ValueError("time_conditioned trunk requires a time tensor t")
            assert self.time_embed is not None  # set iff time_conditioned
            t_emb = self.time_embed(t)  # [1 or N, d_cond]
            if t_emb.shape[0] == 1:
                t_emb = t_emb.expand(n, -1)
            cond = cond + t_emb
        return cond

    def trunk(
        self, x: torch.Tensor, action: torch.Tensor, t: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Run the FiLM trunk + head on ``x`` ``[N, d_model]`` -> displacement-space
        output ``[N, d_model]`` (the residual added to ``x`` in direct mode, or the
        raw velocity in flow mode)."""
        cond = self._cond(action, x.shape[0], t)
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, cond)
        return self.head(self.out_norm(h))

    def velocity(
        self, x: torch.Tensor, t: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Velocity field ``v_theta(x_t, t, action)`` ``[N, d_model]`` (flow matching).

        Requires ``time_conditioned=True``.
        """
        if not self.time_conditioned:
            raise RuntimeError("velocity() requires time_conditioned=True")
        return self.trunk(x, action, t)

    def forward(
        self,
        source: torch.Tensor,
        action: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Direct-mode map: predicted perturbed latents ``source + trunk(source)``.

        Args:
            source: ``[N, d_model]`` source (control) latents.
            action: ``[action_dim]`` (one action for the whole cloud) or
                ``[N, action_dim]`` (per-cell action).
            t: optional time (only used / required when ``time_conditioned``).
        Returns:
            ``[N, d_model]`` predicted perturbed latents.
        """
        return source + self.trunk(source, action, t)
