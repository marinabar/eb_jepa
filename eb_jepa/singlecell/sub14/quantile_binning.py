"""Per-cell equal-frequency quantile binning + balanced sampling.

Faithful port of Subliminal 1.4's quantile binning, adapted to eb_jepa's
*sparse* cell representation: instead of a dense ``(n_pc,)`` row, each
cell arrives as aligned arrays of (PC-local gene id, count) for its
expressed genes only. The helper:

1. equal-frequency-bins the expressed genes into ``num_bins`` quantiles
   by rank,
2. samples up to ``genes_per_bin`` from each bin (without replacement),
3. pads gene ids + bin labels out to ``num_bins * genes_per_bin`` so the
   batch shape is static (torch.compile-friendly).

Binning is rank-based, so it is invariant to the CP10k+log1p transform
(a monotone map) — identical bins whether counts are raw or normalized.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def equal_frequency_bin_assignment(counts: np.ndarray, num_bins: int) -> np.ndarray:
    """Assign genes to equal-frequency quantile bins by rank.

    Lowest-expressing 1/num_bins → bin 0, etc. Ties broken by stable
    argsort. Returns ``(n,)`` int64 bin indices in ``[0, num_bins-1]``.
    """
    n = counts.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(counts, kind="stable")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.floor(np.arange(n, dtype=np.int64) * num_bins / n).astype(np.int64)
    np.clip(ranks, 0, num_bins - 1, out=ranks)
    return ranks


def quantile_bin_and_sample_sparse(
    gene_ids_local: np.ndarray,
    counts: np.ndarray,
    *,
    num_bins: int,
    genes_per_bin: int,
    rng: np.random.Generator,
    pad_gene_index: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin + sample one cell's expressed (PC-local gene id, count) arrays.

    Args:
        gene_ids_local: ``(n_expressed,)`` PC-local gene ids (row indices
            into the gene-feature tables).
        counts: ``(n_expressed,)`` aligned counts (> 0).
        num_bins / genes_per_bin: binning config; output length is always
            ``num_bins * genes_per_bin``.
        rng: numpy Generator (per-worker reproducibility).
        pad_gene_index: sentinel gene id for padded slots (= n_pc_genes).

    Returns ``(gene_ids, bin_ids, valid)`` each of length
    ``num_bins * genes_per_bin``:
        - gene_ids: int64 PC-local ids (pad_gene_index for pad),
        - bin_ids: int64 bin labels [0, num_bins-1], num_bins for pad,
        - valid: bool, True for real genes.
    """
    max_genes = num_bins * genes_per_bin
    gene_ids = np.full(max_genes, pad_gene_index, dtype=np.int64)
    bin_ids = np.full(max_genes, num_bins, dtype=np.int64)
    valid = np.zeros(max_genes, dtype=bool)

    # Drop any non-positive counts (binomial subsample can zero some genes).
    pos = counts > 0
    if not np.any(pos):
        return gene_ids, bin_ids, valid
    gene_ids_local = gene_ids_local[pos]
    counts = counts[pos]

    bin_assignment = equal_frequency_bin_assignment(counts, num_bins)

    sampled_gene_ids: list[int] = []
    sampled_bin_ids: list[int] = []
    for bin_idx in range(num_bins):
        candidates = gene_ids_local[bin_assignment == bin_idx]
        if candidates.size == 0:
            continue
        if candidates.size <= genes_per_bin:
            chosen = candidates
        else:
            chosen = rng.choice(candidates, size=genes_per_bin, replace=False)
        sampled_gene_ids.extend(chosen.tolist())
        sampled_bin_ids.extend([bin_idx] * chosen.shape[0])

    n_sampled = len(sampled_gene_ids)
    if n_sampled > 0:
        gene_ids[:n_sampled] = np.asarray(sampled_gene_ids, dtype=np.int64)
        bin_ids[:n_sampled] = np.asarray(sampled_bin_ids, dtype=np.int64)
        valid[:n_sampled] = True
    return gene_ids, bin_ids, valid
