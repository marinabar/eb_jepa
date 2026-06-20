"""Periodic t-SNE snapshots of the encoder representation during LeJEPA training.

Builds a FIXED eval set once (a random, seeded subset of cells, each as a single
clean full-gene view — no drop, no mask), then at intervals encodes it with the
current encoder and writes an elegant multi-panel t-SNE coloured by biological
class (organ, cell line, drug, MoA). The representation is the **pre-projection**
pooled latent (CLAUDE.md "Views and the LeJEPA objective").
"""

from __future__ import annotations

import copy
import os

import torch

from eb_jepa.datasets.tahoe.dataset import TahoeCollator
from eb_jepa.singlecell.visualize import plot_tsne_grid, tsne_embed

_CLASSES = ("organ", "cell_line_id", "drug", "moa_fine")


@torch.no_grad()
def build_eval_set(dataset, data_cfg, n_cells: int, seed: int = 0):
    """Return (eval_batch, labels) for a fixed seeded subset of cells.

    ``eval_batch`` is one clean view per cell ([1, N, L]) from a TahoeCollator
    configured with n_views=1, all genes kept, no masking. ``labels`` maps each
    probe class to its per-cell values.
    """
    g = torch.Generator().manual_seed(seed)
    n = len(dataset)
    idx = torch.randperm(n, generator=g)[: min(n_cells, n)].tolist()
    items = [dataset[i] for i in idx]

    ecfg = copy.deepcopy(data_cfg)
    ecfg.n_views = 1
    ecfg.view_mode = "drop"
    ecfg.gene_keep_frac = 1.0  # keep every gene (capped at L)
    ecfg.gene_mask_frac = 0.0
    eval_batch = TahoeCollator(ecfg)(items)
    labels = {c: [it.get(c) for it in items] for c in _CLASSES}
    return eval_batch, labels


@torch.no_grad()
def tsne_snapshot(
    encoder,
    eval_batch: dict,
    labels: dict,
    out_dir: str,
    step: int,
    device,
    classes=_CLASSES,
    chunk: int = 64,
    perplexity: float = 30.0,
    seed: int = 0,
    amp: bool = True,
):
    """Encode the eval set (chunked) and save a t-SNE panel figure. Returns path."""
    was_training = encoder.training
    encoder.eval()

    ids = eval_batch["gene_token_ids"][0]  # [N, L]  (single view)
    pad = eval_batch["pad_mask"][0]
    cv = eval_batch.get("count_value")
    cb = eval_batch.get("count_bin")
    cv = cv[0] if cv is not None else None
    cb = cb[0] if cb is not None else None

    reps = []
    n = ids.shape[0]
    for s in range(0, n, chunk):
        sl = slice(s, s + chunk)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            z = encoder(
                ids[sl].to(device),
                pad[sl].to(device),
                count_value=cv[sl].to(device) if cv is not None else None,
                count_bin=cb[sl].to(device) if cb is not None else None,
            )
        reps.append(z.float().cpu())
    reps = torch.cat(reps)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"tsne_step{step:06d}.png")
    emb = tsne_embed(reps, seed=seed, perplexity=perplexity)
    plot_tsne_grid(emb, {c: labels[c] for c in classes}, path, step=step)

    if was_training:
        encoder.train()
    return path
