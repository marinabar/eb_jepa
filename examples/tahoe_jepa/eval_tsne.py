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
from eb_jepa.singlecell.visualize import plot_tsne_grid, plot_tsne_single, tsne_embed

_CLASSES = ("organ", "cell_line_id", "drug", "moa_fine")


@torch.no_grad()
def build_eval_set(dataset, data_cfg, n_cells: int = 0, seed: int = 0, idx=None):
    """Return (eval_batch, labels) for a FIXED set of cells (one clean full-gene
    view each, no drop/mask).

    If ``idx`` is given, use exactly those cell indices (the held-out probe set,
    excluded from SSL training); otherwise sample ``n_cells`` (seeded). ``labels``
    maps each probe class (+ ``sample``) to its per-cell values.
    """
    if idx is not None:
        items = [dataset[i] for i in idx]
    elif hasattr(dataset, "sample_items"):  # streaming IterableDataset (no index space)
        items = dataset.sample_items(n_cells)
    else:
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
    labels = {c: [it.get(c) for it in items] for c in _CLASSES + ("sample",)}
    return eval_batch, labels


@torch.no_grad()
def encode_eval(encoder, eval_batch: dict, device, chunk: int = 128, amp: bool = True):
    """Encode the single-view eval batch in chunks -> [N, d_model] (pre-projection)."""
    was_training = encoder.training
    encoder.eval()
    ids = eval_batch["gene_token_ids"][0]  # [N, L] (single view)
    pad = eval_batch["pad_mask"][0]
    cv = eval_batch.get("count_value")
    cb = eval_batch.get("count_bin")
    cv = cv[0] if cv is not None else None
    cb = cb[0] if cb is not None else None
    reps = []
    for s in range(0, ids.shape[0], chunk):
        sl = slice(s, s + chunk)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            z = encoder(
                ids[sl].to(device),
                pad[sl].to(device),
                count_value=cv[sl].to(device) if cv is not None else None,
                count_bin=cb[sl].to(device) if cb is not None else None,
            )
        reps.append(z.float().cpu())
    if was_training:
        encoder.train()
    return torch.cat(reps)


def periodic_eval(
    encoder,
    eval_batch: dict,
    labels: dict,
    out_dir: str,
    step: int,
    device,
    run=None,
    classes=_CLASSES,
    chunk: int = 128,
    perplexity: float = 30.0,
    seed: int = 0,
    amp: bool = True,
):
    """Encode the held-out eval set, run detached probes + per-class t-SNE panels, and
    log to wandb. Returns (metrics, paths) where ``paths`` maps each class to its
    t-SNE image. Probes are imbalance-aware (balanced acc / macro-F1 for classes; R2
    for gene-count); ``repr/effective_rank`` tracks collapse.
    """
    from eb_jepa.singlecell.probes import run_probe_suite
    from eb_jepa.singlecell.visualize import effective_rank

    reps = encode_eval(encoder, eval_batch, device, chunk, amp)
    meta = dict(labels)
    meta["gene_count"] = eval_batch["pad_mask"][0].sum(-1).tolist()
    suite = run_probe_suite(reps, meta)

    metrics = {}
    for key, m in suite.items():  # key e.g. "clf/organ" or "reg/gene_count"
        for mk, mv in m.items():
            metrics[f"probe/{key}/{mk}"] = float(mv)
    metrics["repr/effective_rank"] = float(effective_rank(reps))

    os.makedirs(out_dir, exist_ok=True)
    emb = tsne_embed(reps, seed=seed, perplexity=perplexity)
    paths = {}
    for c in classes:
        p = os.path.join(out_dir, f"tsne_{c}_step{step:06d}.png")
        plot_tsne_single(emb, labels[c], p, name=c, step=step)
        paths[c] = p

    if run is not None:
        log = dict(metrics)
        try:
            import wandb

            for c, p in paths.items():
                log[f"tsne/{c}"] = wandb.Image(p, caption=f"step {step}")
        except Exception:
            pass
        run.log(log, step=step)
    return metrics, paths


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
    emb = tsne_embed(reps, seed=seed, perplexity=perplexity)
    paths = {}
    for c in classes:
        p = os.path.join(out_dir, f"tsne_{c}_step{step:06d}.png")
        plot_tsne_single(emb, labels[c], p, name=c, step=step)
        paths[c] = p

    if was_training:
        encoder.train()
    return paths  # {class_name -> image path}
