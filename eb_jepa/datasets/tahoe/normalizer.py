"""Count normalization and value encoding for Tahoe-100M.

Fixed transform (CLAUDE.md "Normalization / Count embedding"): per-cell depth
normalization (CP10k) then log1p, on the non-zero genes. The normalized value is
**continuous and mode-agnostic** — it is what gets cached. The two benchmarked
value-encoding modes are applied at load/collate time, not at cache time:

- mode A (continuous): the log scalar is fed to an MLP in the encoder.
- mode B (quantile binning): per-gene global quantile boundaries (~50 bins) map
  the scalar to a ``bin_id``. Boundaries are global, so a given ``(gene, value)``
  maps to the SAME bin in every drop/mask view — this is what keeps SIGReg stable.

This module owns the math; the encoder owns the embedding tables.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def cp10k_log1p(counts: torch.Tensor, target_sum: float = 1e4) -> torch.Tensor:
    """CP10k + log1p on a single cell's raw counts.

    counts: 1D tensor of raw UMI counts for the cell's non-zero genes.
    Returns log1p(counts / counts.sum() * target_sum), same shape. An all-zero
    cell (sum == 0) maps to zeros.
    """
    counts = counts.to(torch.float32)
    total = counts.sum()
    if total <= 0:
        return torch.zeros_like(counts)
    return torch.log1p(counts / total * target_sum)


class QuantileBinner:
    """Per-gene global quantile bins for mode-B count encoding.

    ``boundaries`` has shape ``[n_genes, n_bins - 1]`` (the inner quantile edges
    for each gene, indexed by token_id). ``bin(token_ids, values)`` returns the
    bin index in ``[0, n_bins - 1]`` via ``searchsorted`` on the per-gene edges.

    Because the boundaries are global and fixed, the bin for a ``(gene, value)``
    pair is deterministic and therefore identical across all views of a cell.
    """

    def __init__(self, boundaries: torch.Tensor, n_bins: int):
        assert boundaries.ndim == 2, "boundaries must be [n_genes, n_bins-1]"
        assert boundaries.shape[1] == n_bins - 1, "boundaries width must be n_bins-1"
        self.boundaries = boundaries.to(torch.float32).contiguous()
        self.n_bins = n_bins
        self.n_genes = boundaries.shape[0]

    def bin(self, token_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Return bin ids in [0, n_bins-1] for aligned (token_ids, values)."""
        token_ids = token_ids.to(torch.long)
        edges = self.boundaries[token_ids]  # [K, n_bins-1]
        # per-row searchsorted: count how many edges each value exceeds
        return torch.searchsorted(
            edges, values.unsqueeze(-1).to(torch.float32)
        ).squeeze(-1)

    # ---- persistence ----------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.save(path.with_suffix(".npy"), self.boundaries.numpy())
        path.with_suffix(".json").write_text(json.dumps({"n_bins": self.n_bins}))

    @classmethod
    def load(cls, path: str | Path) -> "QuantileBinner":
        path = Path(path)
        boundaries = torch.from_numpy(np.load(path.with_suffix(".npy")))
        meta = json.loads(path.with_suffix(".json").read_text())
        return cls(boundaries, n_bins=meta["n_bins"])


def fit_quantile_boundaries(
    token_ids: torch.Tensor,
    values: torch.Tensor,
    n_genes: int,
    n_bins: int = 50,
) -> torch.Tensor:
    """Compute per-gene quantile boundaries from a pooled (token_id, value) sample.

    Computes, for each gene, the ``n_bins - 1`` interior quantile edges of its
    observed CP10k+log1p values. Genes with no observations get zero edges (all
    their values fall in bin 0). Intended to run once on a cached subsample
    (preprocess step), then stored and reused at load time.

    Args:
        token_ids: 1D int tensor of gene token ids (pooled over many cells).
        values: 1D float tensor of CP10k+log1p values aligned with token_ids.
        n_genes: vocabulary size (e.g. 62710).
        n_bins: number of bins.
    Returns:
        boundaries: [n_genes, n_bins - 1] float tensor.
    """
    qs = torch.linspace(0, 1, n_bins + 1)[1:-1]  # interior quantiles
    boundaries = torch.zeros(n_genes, n_bins - 1, dtype=torch.float32)
    token_ids = token_ids.to(torch.long)
    order = torch.argsort(token_ids)
    sorted_ids = token_ids[order]
    sorted_vals = values[order].to(torch.float32)
    # contiguous runs of equal token id
    uniq, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
    starts = torch.cumsum(
        torch.cat([torch.zeros(1, dtype=counts.dtype), counts[:-1]]), 0
    )
    for gid, start, cnt in zip(uniq.tolist(), starts.tolist(), counts.tolist()):
        if cnt == 0:
            continue
        gene_vals = sorted_vals[start : start + cnt]
        boundaries[gid] = torch.quantile(gene_vals, qs)
    return boundaries
