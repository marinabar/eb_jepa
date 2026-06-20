"""Select a fixed panel of top-N highly-variable genes (HVGs) from Tahoe-100M.

The MATCHED baselines (mae_matched / vae_matched / pca_matched) restrict the dense
input to a fixed N-gene panel so they share Subliminal-14-small's context window
(num_bins * genes_per_bin = 16 * 32 = 512 tokens/cell). HVG selection is the
standard single-cell way to pick that panel: stream a sample of cells, accumulate
sparse per-gene sufficient statistics of the CP10k+log1p ``values`` (sum, sumsq,
count over the 62,713 token_id index space — no 62k densify), then take the top-N
genes by variance of the log-normalized expression.

Outputs to ``--out_dir`` (default the shared tahoe-cache):
  - ``hvg_{N}.npy``  : sorted int64 token_ids of the panel (the order the dense
                       HVG vector columns follow).
  - ``hvg_{N}.json`` : {method, N, n_cells, n_genes_index, token_ids, variances}.

Usage (Dalia, single GPU or CPU — streaming reader, no GPU needed):
    /lustre/work/vivatech-unaite/ljung/venv-arm/bin/python -m examples.tahoe_baselines.select_hvg \
        run --data_dir /lustre/work/vivatech-unaite/ljung/tahoe-subset \
            --n_cells 100000 --n_hvg 512 \
            --out_dir /lustre/work/vivatech-unaite/shared/tahoe-cache
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from eb_jepa.datasets.tahoe.dataset import TahoeConfig, TahoeIterableDataset
from eb_jepa.logging import get_logger

logger = get_logger(__name__)

# Index space = max Tahoe token_id (62712) + 1, so we can scatter raw token_ids.
N_GENES_INDEX = 62713


def accumulate_stats(
    dataset: TahoeIterableDataset, n_cells: int, n_index: int = N_GENES_INDEX
):
    """Stream up to ``n_cells`` cells; accumulate per-gene sum, sumsq, count.

    Sufficient statistics are kept sparsely over the full ``n_index`` token-id space
    (one float64 entry per id), never densifying a 62k vector per cell. ``values`` are
    the CP10k+log1p normalized expressions already produced by the dataset.
    """
    gsum = np.zeros(n_index, dtype=np.float64)
    gsumsq = np.zeros(n_index, dtype=np.float64)
    gcount = np.zeros(n_index, dtype=np.float64)
    seen = 0
    for item in dataset:
        tok = item["gene_token_ids"].numpy().astype(np.int64)
        val = item["values"].numpy().astype(np.float64)
        # Sparse scatter-add of the non-zero genes; zero genes contribute 0 to
        # sum/sumsq but DO count toward the per-gene n (handled below via n_cells).
        np.add.at(gsum, tok, val)
        np.add.at(gsumsq, tok, val * val)
        np.add.at(gcount, tok, 1.0)
        seen += 1
        if seen % 10000 == 0:
            logger.info(f"  accumulated {seen}/{n_cells} cells")
        if seen >= n_cells:
            break
    logger.info(f"accumulated stats over {seen} cells")
    return gsum, gsumsq, gcount, seen


def gene_variance(gsum, gsumsq, n_cells: int) -> np.ndarray:
    """Variance of log-normalized expression per gene over ALL ``n_cells`` cells.

    Cells where a gene is zero contribute 0 to sum/sumsq but still count toward the
    denominator (a gene off in most cells has low variance). Using the full ``n_cells``
    as the denominator is the standard HVG variance (population variance of the dense
    log-normalized vector). E[x] = sum/n, E[x^2] = sumsq/n, var = E[x^2] - E[x]^2.
    """
    n = max(int(n_cells), 1)
    mean = gsum / n
    var = gsumsq / n - mean * mean
    return np.clip(var, 0.0, None)


def select_hvg(var: np.ndarray, n_hvg: int) -> np.ndarray:
    """Return the sorted int64 token_ids of the top-``n_hvg`` genes by variance."""
    order = np.argsort(-var, kind="stable")[:n_hvg]
    return np.sort(order.astype(np.int64))


def run(
    data_dir: str = "/lustre/work/vivatech-unaite/ljung/tahoe-subset",
    n_cells: int = 100000,
    n_hvg: int = 512,
    out_dir: str = "/lustre/work/vivatech-unaite/shared/tahoe-cache",
    metric: str = "variance",
    shuffle_buffer: int = 4000,
    seed: int = 0,
):
    """Stream a sample, score genes by ``metric``, save the top-N panel.

    ``metric``: "variance" (HVG, default), "prevalence" (genes expressed in the most
    cells -> a dense per-cell input like sub14's expressed-gene sampling, no
    discriminative cherry-pick), or "mean" (highest mean log-normalized expression).
    Non-variance metrics save to ``hvg_{N}_{metric}.npy`` so the HVG panel is kept.
    """
    cfg = TahoeConfig(
        data_dir=data_dir,
        streaming=True,
        shuffle_buffer=int(shuffle_buffer),
        seed=int(seed),
    )
    dataset = TahoeIterableDataset(cfg, shuffle=True)
    gsum, gsumsq, gcount, seen = accumulate_stats(dataset, int(n_cells))
    var = gene_variance(gsum, gsumsq, seen)
    mean = gsum / max(int(seen), 1)
    prevalence = gcount.astype(np.float64)  # cells expressing each gene
    scores = {"variance": var, "mean": mean, "prevalence": prevalence}
    if metric not in scores:
        raise ValueError(f"metric must be one of {list(scores)}, got {metric!r}")
    score = scores[metric]

    order = np.argsort(-score, kind="stable")[: int(n_hvg)]
    hvg = np.sort(order.astype(np.int64))
    n_expressed = int((gcount > 0).sum())
    logger.info(
        f"selected {hvg.size} genes by {metric} out of {n_expressed} ever-expressed "
        f"({N_GENES_INDEX} index space)"
    )

    os.makedirs(out_dir, exist_ok=True)
    suffix = "" if metric == "variance" else f"_{metric}"
    npy_path = os.path.join(out_dir, f"hvg_{n_hvg}{suffix}.npy")
    json_path = os.path.join(out_dir, f"hvg_{n_hvg}{suffix}.json")
    np.save(npy_path, hvg)
    with open(json_path, "w") as f:
        json.dump(
            {
                "method": metric,
                "n_hvg": int(n_hvg),
                "n_cells": int(seen),
                "n_genes_index": N_GENES_INDEX,
                "n_expressed_genes": n_expressed,
                "data_dir": data_dir,
                "token_ids": hvg.tolist(),
                "score": score[hvg].tolist(),
            },
            f,
            indent=2,
        )
    logger.info(f"  -> saved {npy_path} and {json_path}")
    return hvg


if __name__ == "__main__":
    import fire

    fire.Fire({"run": run})
