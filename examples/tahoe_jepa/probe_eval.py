"""Periodic detached-probe + representation diagnostics during LeJEPA training.

Companion to ``eval_tsne.py``: on the same FIXED eval set (a seeded subset of
cells, each a single clean full-gene view), encode the **pre-projection** pooled
latent, then run the detached linear-probe suite (organ / cell line / drug / ...
classification + gene-count regression, all imbalance-aware) and the collapse
diagnostics (covariance spectrum, effective rank). Everything is returned as a
flat dict of interpretable scalars + an elegant spectrum PNG; the caller logs to
wandb (this module stays wandb-free). The chunked-encode mirrors
``eval_tsne.tsne_snapshot`` exactly.
"""

from __future__ import annotations

import os

import torch

from eb_jepa.singlecell.probes import run_probe_suite
from eb_jepa.singlecell.visualize import (
    covariance_spectrum,
    effective_rank,
    plot_spectrum,
)

_CLASSES = ("organ", "cell_line_id", "drug", "sample", "moa_fine", "plate")


@torch.no_grad()
def _encode_eval(encoder, eval_batch: dict, device, chunk: int, amp: bool):
    """Chunked pre-projection encode of the single-view eval batch -> feats [N, d]."""
    was_training = encoder.training
    encoder.eval()
    try:
        ids = eval_batch["gene_token_ids"][0]  # [N, L] (single view)
        pad = eval_batch["pad_mask"][0]
        cv = eval_batch.get("count_value")
        cb = eval_batch.get("count_bin")
        cv = cv[0] if cv is not None else None
        cb = cb[0] if cb is not None else None

        reps = []
        n = ids.shape[0]
        for s in range(0, n, chunk):
            sl = slice(s, s + chunk)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=amp
            ):
                z = encoder(
                    ids[sl].to(device),
                    pad[sl].to(device),
                    count_value=cv[sl].to(device) if cv is not None else None,
                    count_bin=cb[sl].to(device) if cb is not None else None,
                )
            reps.append(z.float().cpu())
        return torch.cat(reps)
    finally:
        if was_training:
            encoder.train()


def _build_meta(eval_batch: dict) -> dict:
    """Probe metadata straight from the eval batch's own per-cell list fields."""
    meta = {}
    for key in _CLASSES:
        if key in eval_batch:
            meta[key] = list(eval_batch[key])
    meta["gene_count"] = eval_batch["pad_mask"][0].sum(-1).cpu().tolist()
    if eval_batch.get("log_conc") is not None:
        meta["log_conc"] = eval_batch["log_conc"].cpu().tolist()
    return meta


def _flatten_probe_results(results: dict) -> dict[str, float]:
    """Flatten run_probe_suite output into interpretable ``probe/...`` scalars."""
    scalars: dict[str, float] = {}
    for full_key, m in results.items():
        if full_key.startswith("clf/"):
            name = full_key[len("clf/") :]
            ba = float(m.get("balanced_accuracy", float("nan")))
            chance = float(m.get("chance", float("nan")))
            scalars[f"probe/clf/{name}/balanced_accuracy"] = ba
            scalars[f"probe/clf/{name}/macro_f1"] = float(
                m.get("macro_f1", float("nan"))
            )
            scalars[f"probe/clf/{name}/above_chance"] = ba - chance
            scalars[f"probe/clf/{name}/n_classes"] = float(m.get("n_classes", 0))
        elif full_key.startswith("reg/"):
            name = full_key[len("reg/") :]
            scalars[f"probe/reg/{name}/loss"] = float(m.get("loss", float("nan")))
            scalars[f"probe/reg/{name}/r2"] = float(m.get("r2", float("nan")))
    return scalars


def probe_report(
    encoder,
    eval_batch: dict,
    device,
    out_dir: str,
    step: int,
    *,
    probe_epochs: int = 150,
    chunk: int = 64,
    amp: bool = True,
):
    """Encode the fixed eval set, run detached probes + collapse diagnostics.

    Returns ``(scalars, spectrum_path)``: a flat dict of interpretable wandb
    scalars (``probe/clf/<key>/...``, ``probe/reg/<key>/...``, ``repr/...``) and
    the path to a house-style spectrum PNG. The caller owns wandb logging.

    The encode is no-grad (frozen encoder, detached features) but probe training
    needs autograd on the probe params, so this function is *not* globally
    no-grad — ``_encode_eval`` owns the no-grad context.
    """
    feats = _encode_eval(encoder, eval_batch, device, chunk=chunk, amp=amp)
    meta = _build_meta(eval_batch)

    results = run_probe_suite(feats, meta, epochs=probe_epochs)
    scalars = _flatten_probe_results(results)

    # collapse diagnostics on the pre-projection representation
    eig = covariance_spectrum(feats)
    d = feats.shape[1]
    total = float(eig.sum())
    scalars["repr/effective_rank"] = effective_rank(feats)
    scalars["repr/effrank_ratio"] = scalars["repr/effective_rank"] / max(1, d)
    scalars["repr/top1_eig_frac"] = float(eig[0]) / total if total > 0 else float("nan")

    os.makedirs(out_dir, exist_ok=True)
    spectrum_path = os.path.join(out_dir, f"spectrum_step{step:06d}.png")
    plot_spectrum(
        eig,
        spectrum_path,
        title=f"covariance spectrum · step {step}",
    )
    return scalars, spectrum_path
