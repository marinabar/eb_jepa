"""Count normalization and value encoding for Tahoe-100M.

Fixed transform (CLAUDE.md "Normalization / Count embedding"): per-cell depth
normalization (CP10k) then log1p, on the non-zero genes. The normalized value is
**continuous and mode-agnostic** — it is what gets cached. The two benchmarked
value-encoding modes are applied at load/collate time, not at cache time:

- mode A (continuous): the log scalar is fed to an MLP in the encoder.
- mode B (quantile binning): per-gene global quantile boundaries (~64 bins) map
  the scalar to a ``bin_id``. Boundaries are global, so a given ``(gene, value)``
  maps to the SAME bin in every drop/mask view — this is what keeps SIGReg stable.

Boundaries are fit over a (configurable) large sample of the dataset. Pooling that
many raw values in memory is infeasible, so ``GeneHistogram`` accumulates a fixed
per-gene grid in one streaming pass (memory independent of #cells) and derives the
quantile edges from it. The bin embedding table is indexed by ``bin_id`` only, so a
given quantile bin shares one learned embedding across all genes while each gene's
edges (and thus the value span of that bin) are its own.

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
    n_bins: int = 64,
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


class GeneHistogram:
    """Streaming per-gene histogram of CP10k+log1p values for quantile fitting.

    The exact fitter (``fit_quantile_boundaries``) holds every observed value in
    memory, which is fine for a small sample but explodes for the large samples
    (millions of cells) we want the quantiles fit on. This accumulator instead
    bins values into a fixed ``[n_genes, n_hist_bins]`` grid over ``[v_min, v_max]``
    in a single streaming pass: memory is ``n_genes * n_hist_bins`` regardless of
    how many cells are scanned, and ``update`` is fully vectorized.

    CP10k+log1p values are bounded by ``log1p(target_sum) ~= 9.21`` (one gene
    carrying all of a cell's depth), so the default range ``[0, 10]`` with a fine
    grid resolves quantiles to ~``(v_max - v_min) / n_hist_bins`` (~0.0024 with the
    defaults) — far below the spacing of the ~64 quantile bins.

    The histogram is itself saved (``save``/``load``) so boundaries for a different
    ``n_bins`` can be re-derived without re-scanning the dataset.
    """

    def __init__(
        self,
        n_genes: int,
        n_hist_bins: int = 4096,
        v_min: float = 0.0,
        v_max: float = 10.0,
    ):
        assert v_max > v_min and n_hist_bins > 1
        self.n_genes = n_genes
        self.n_hist_bins = n_hist_bins
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.hist = torch.zeros(n_genes, n_hist_bins, dtype=torch.int64)

    @property
    def n_observed(self) -> int:
        """Total number of (gene, value) observations accumulated."""
        return int(self.hist.sum().item())

    def update(self, token_ids: torch.Tensor, values: torch.Tensor) -> None:
        """Accumulate aligned (token_ids, CP10k+log1p values) into the histogram."""
        token_ids = token_ids.to(torch.long)
        v = values.to(torch.float32)
        scaled = (v - self.v_min) / (self.v_max - self.v_min) * self.n_hist_bins
        b = scaled.floor().to(torch.long).clamp_(0, self.n_hist_bins - 1)
        flat = token_ids * self.n_hist_bins + b
        self.hist.view(-1).index_add_(0, flat, torch.ones_like(flat))

    def quantile_boundaries(self, n_bins: int) -> torch.Tensor:
        """Derive per-gene quantile edges ``[n_genes, n_bins-1]`` from the grid.

        Same shape/semantics as ``fit_quantile_boundaries``: a gene with no
        observations gets all-zero edges (every value falls in bin 0).
        """
        qs = torch.linspace(0, 1, n_bins + 1, dtype=torch.float64)[1:-1]
        cdf = self.hist.to(torch.float64).cumsum(dim=1)  # [n_genes, n_hist_bins]
        total = cdf[:, -1].clone()  # [n_genes]
        right_edges = torch.linspace(
            self.v_min, self.v_max, self.n_hist_bins + 1, dtype=torch.float64
        )[1:]
        targets = qs[None, :] * total[:, None]  # [n_genes, n_bins-1]
        idx = torch.searchsorted(cdf.contiguous(), targets.contiguous())
        idx = idx.clamp_(0, self.n_hist_bins - 1)
        boundaries = right_edges[idx].to(torch.float32)
        boundaries[total == 0] = 0.0  # unseen genes -> bin 0
        return boundaries

    def binner(self, n_bins: int) -> "QuantileBinner":
        return QuantileBinner(self.quantile_boundaries(n_bins), n_bins=n_bins)

    # ---- persistence ----------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.save(path.with_suffix(".npy"), self.hist.numpy())
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "n_hist_bins": self.n_hist_bins,
                    "v_min": self.v_min,
                    "v_max": self.v_max,
                    "n_observed": self.n_observed,
                }
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> "GeneHistogram":
        path = Path(path)
        hist = np.load(path.with_suffix(".npy"))
        meta = json.loads(path.with_suffix(".json").read_text())
        obj = cls(
            hist.shape[0],
            n_hist_bins=meta["n_hist_bins"],
            v_min=meta["v_min"],
            v_max=meta["v_max"],
        )
        obj.hist = torch.from_numpy(hist)
        return obj
