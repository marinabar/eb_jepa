"""Optimal-transport objective for the perturbator (CLAUDE.md "Optimal-transport objective").

We do not have paired control/perturbed cells, only the two distributions, so the
training signal is the distance between the *predicted* perturbed distribution and
the *target* perturbed distribution. We use the **sliced-Wasserstein** distance:
project both point clouds onto many random unit directions, and average the 1-D
Wasserstein distance (a sorted-quantile comparison) over directions. The number of
slices is configurable. When the two clouds have different cardinalities the
quantiles are matched on a common grid by linear interpolation, so it works for
``N != M``.
"""

from __future__ import annotations

import torch


def _sorted_quantiles_on_grid(
    projections: torch.Tensor, grid: torch.Tensor
) -> torch.Tensor:
    """Per-slice empirical quantiles of ``projections`` sampled at ``grid``.

    Args:
        projections: ``[K, S]`` (K samples projected on S slices).
        grid: ``[Q]`` quantile levels in [0, 1].
    Returns:
        ``[S, Q]`` interpolated order statistics.
    """
    k = projections.shape[0]
    srt = projections.sort(dim=0).values  # [K, S]
    # Positions of the sorted samples on the [0, 1] quantile axis.
    if k == 1:
        sample_q = torch.zeros(1, device=projections.device, dtype=projections.dtype)
    else:
        sample_q = torch.linspace(
            0.0, 1.0, k, device=projections.device, dtype=projections.dtype
        )
    out = torch.empty(
        srt.shape[1], grid.shape[0], device=projections.device, dtype=projections.dtype
    )
    # torch has no batched 1-D interp; loop over slices (S is modest, e.g. 256).
    for s in range(srt.shape[1]):
        out[s] = _interp(grid, sample_q, srt[:, s])
    return out


def _interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """1-D linear interpolation (``numpy.interp`` semantics) on sorted ``xp``."""
    idx = torch.searchsorted(xp, x).clamp(1, xp.numel() - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = fp[idx - 1], fp[idx]
    denom = (x1 - x0).clamp_min(torch.finfo(x.dtype).eps)
    w = (x - x0) / denom
    return y0 + w * (y1 - y0)


def sliced_wasserstein(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_slices: int = 256,
    p: int = 2,
    generator: torch.Generator | None = None,
    n_quantiles: int | None = None,
) -> torch.Tensor:
    """Sliced ``p``-Wasserstein distance between two latent point clouds.

    Args:
        pred: ``[N, d]`` predicted samples (carries gradient to the perturbator).
        target: ``[M, d]`` target samples (typically detached / no grad).
        n_slices: number of random unit directions to average over.
        p: Wasserstein order (2 -> squared distance in the 1-D comparison).
        generator: optional RNG for reproducible slice directions.
        n_quantiles: size of the shared quantile grid when ``N != M``; defaults to
            ``max(N, M)``. Ignored (fast path) when ``N == M``.
    Returns:
        Scalar distance. ~0 for identical clouds, > 0 for shifted ones.
    """
    assert pred.dim() == 2 and target.dim() == 2, "pred/target must be [*, d]"
    assert pred.shape[1] == target.shape[1], "pred/target dims differ"
    d = pred.shape[1]

    directions = torch.randn(
        d, n_slices, device=pred.device, dtype=pred.dtype, generator=generator
    )
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(
        torch.finfo(pred.dtype).eps
    )

    proj_p = pred @ directions  # [N, S]
    proj_t = target @ directions  # [M, S]

    if pred.shape[0] == target.shape[0]:
        # equal cardinality: plain sorted-sample matching, no interpolation
        a = proj_p.sort(dim=0).values
        b = proj_t.sort(dim=0).values
    else:
        q = n_quantiles or max(pred.shape[0], target.shape[0])
        grid = torch.linspace(0.0, 1.0, q, device=pred.device, dtype=pred.dtype)
        a = _sorted_quantiles_on_grid(proj_p, grid).t()  # [Q, S]
        b = _sorted_quantiles_on_grid(proj_t, grid).t()

    diff = (a - b).abs()
    if p == 1:
        cost = diff.mean(dim=0)
    else:
        cost = diff.pow(p).mean(dim=0).pow(1.0 / p)
    return cost.mean()
