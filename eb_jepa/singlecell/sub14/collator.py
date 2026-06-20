"""Subliminal 1.4 view collator over eb_jepa's sparse Tahoe cells.

Each cell arrives as the eb_jepa whole-cell item (sparse raw Tahoe
``gene_token_ids`` + ``raw_counts``/``values`` + probing metadata). For
every view this collator:

1. slices the cell to the protein-coding vocabulary (``token_to_pc_local``),
2. binomial-subsamples the integer counts (multi-view augmentation,
   p ~ U[p_min, p_max] drawn fresh per view),
3. per-cell equal-frequency quantile-bins the surviving genes and samples
   ``genes_per_bin`` per bin, padding to ``num_bins * genes_per_bin``.

Output tensors are ``(V, N, G)`` (gene_ids PC-local with pad sentinel
``n_pc_genes``; bin_ids with pad sentinel ``num_bins``; padding_mask
True=pad), plus per-cell metadata lists for the eval probe suite.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from eb_jepa.singlecell.sub14.quantile_binning import quantile_bin_and_sample_sparse

_META_KEYS = ("drug", "sample", "cell_line_id", "organ", "moa_fine", "plate", "canonical_smiles")


class Sub14Collator:
    def __init__(
        self,
        *,
        token_to_pc_local: np.ndarray,
        n_pc_genes: int,
        num_bins: int = 16,
        genes_per_bin: int = 32,
        num_views: int = 4,
        binomial_subsample: Optional[dict] = None,
        seed: int = 0,
    ) -> None:
        self.token_to_pc_local = np.asarray(token_to_pc_local, dtype=np.int64)
        self.n_pc_genes = int(n_pc_genes)
        self.num_bins = int(num_bins)
        self.genes_per_bin = int(genes_per_bin)
        self.max_genes_per_cell = self.num_bins * self.genes_per_bin
        self.num_views = int(num_views)
        self.binomial_subsample = self._normalize_subsample(binomial_subsample)
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def _normalize_subsample(spec: Optional[dict]) -> Optional[dict]:
        if not spec or not spec.get("enabled", True):
            return None
        p_min = float(spec.get("p_min", spec.get("p", 1.0)))
        p_max = float(spec.get("p_max", p_min))
        p_min, p_max = max(0.0, min(1.0, p_min)), max(0.0, min(1.0, p_max))
        if p_min > p_max:
            p_min, p_max = p_max, p_min
        return {"p_min": p_min, "p_max": p_max}

    def _subsample(self, counts: np.ndarray) -> np.ndarray:
        if self.binomial_subsample is None:
            return counts
        p_min, p_max = self.binomial_subsample["p_min"], self.binomial_subsample["p_max"]
        p = p_min if p_min == p_max else float(self._rng.uniform(p_min, p_max))
        if p >= 1.0:
            return counts
        counts_int = np.rint(counts).astype(np.int64, copy=False)
        return self._rng.binomial(counts_int, p).astype(np.float32, copy=False)

    def __call__(self, batch: list[dict]) -> dict:
        v, n = self.num_views, len(batch)
        g = self.max_genes_per_cell

        gene_ids = np.full((v, n, g), self.n_pc_genes, dtype=np.int64)
        bin_ids = np.full((v, n, g), self.num_bins, dtype=np.int64)
        valid = np.zeros((v, n, g), dtype=bool)

        for j, cell in enumerate(batch):
            tok = cell["gene_token_ids"].numpy()
            raw = cell.get("raw_counts")
            raw = (raw.numpy() if raw is not None else cell["values"].numpy()).astype(np.float32)
            pc_local = self.token_to_pc_local[tok]
            keep = pc_local >= 0
            gid_local = pc_local[keep]
            cnt = raw[keep]
            if gid_local.size == 0:
                continue
            for view in range(v):
                cnt_v = self._subsample(cnt)
                gi, bi, va = quantile_bin_and_sample_sparse(
                    gid_local,
                    cnt_v,
                    num_bins=self.num_bins,
                    genes_per_bin=self.genes_per_bin,
                    rng=self._rng,
                    pad_gene_index=self.n_pc_genes,
                )
                gene_ids[view, j] = gi
                bin_ids[view, j] = bi
                valid[view, j] = va

        out: dict = {
            "gene_ids": torch.from_numpy(gene_ids),
            "bin_ids": torch.from_numpy(bin_ids),
            "padding_mask": torch.from_numpy(~valid),  # True = pad
            "n_views": v,
            "batch_size": n,
        }
        for key in _META_KEYS:
            out[key] = [c.get(key) for c in batch]
        out["log_conc"] = torch.tensor(
            [float(c.get("log_conc", float("nan"))) for c in batch], dtype=torch.float32
        )
        return out
