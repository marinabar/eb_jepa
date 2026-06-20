"""SIGReg — LeJEPA's sliced Epps–Pulley Gaussianity regulariser.

Faithful port of Subliminal 1.4's SIGReg (num_slices=256 default) with
one addition for multi-GPU: the empirical characteristic function means
(``cos``/``sin`` over the batch) are all-reduced (AVG) across DDP ranks
so the Gaussianity test sees the global batch, and the projection RNG is
kept lock-step across ranks. Single-GPU behaviour is byte-identical to
the reference.

Reference: https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


class SIGReg(nn.Module):
    def __init__(self, num_slices: int = 256, knots: int = 17, t_max: float = 3.0):
        super().__init__()
        self.num_slices = num_slices
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)
        # Lock-step projection RNG across ranks (advanced once per call).
        self.step = 0

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """``proj``: (N, d_proj). Returns the scalar SIGReg statistic."""
        # Deterministic, rank-identical random slices so DDP ranks test the
        # same projection directions on the same global batch.
        gen = torch.Generator(device=proj.device)
        gen.manual_seed(0x5152 + self.step)
        self.step += 1
        A = torch.randn(
            proj.size(-1), self.num_slices, device=proj.device, dtype=proj.dtype, generator=gen
        )
        A = A.div_(A.norm(p=2, dim=0))

        x_t = (proj @ A).unsqueeze(-1) * self.t  # (N, S, knots)
        cos_mean = x_t.cos().mean(-3)  # (S, knots) — mean over the per-rank batch
        sin_mean = x_t.sin().mean(-3)
        n = proj.size(-2)

        if dist.is_available() and dist.is_initialized():
            # AVG of per-rank means == global mean (equal per-rank batch sizes,
            # drop_last=True). Scale by global N below.
            dist.all_reduce(cos_mean, op=dist.ReduceOp.AVG)
            dist.all_reduce(sin_mean, op=dist.ReduceOp.AVG)
            n = n * dist.get_world_size()

        err = (cos_mean - self.phi).square() + sin_mean.square()
        statistic = (err @ self.weights) * n
        return statistic.mean()
