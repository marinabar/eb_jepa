"""Train the MAE / VAE / PCA baselines on the densified Tahoe-100M stream.

These are the *well-tuned* baselines JEPA must beat (CLAUDE.md "Success criteria").
A single YAML config drives everything (cfgs/{mae,vae,pca}.yaml), mirroring the
sub14 config style. The baselines operate on the densified 62,713-gene CP10k+log1p
vector (``DensifyCollator``), so they share the fixed shared validation set and the
detached probe suite with the JEPA encoder.

  - MAE / VAE: stream cells, densify, optimise the reconstruction(+KL) loss with
    AdamW (no scheduler — same convention as sub14), bf16, single GPU. A periodic
    detached-probe snapshot on the FIXED shared eval set tracks representation
    quality during training.
  - PCA: draw ``optim.fit_cells`` cells from the same stream, densify, fit sklearn
    PCA, pickle the fitted model.

Usage:
    python -m examples.tahoe_baselines.train run --config examples/tahoe_baselines/cfgs/mae.yaml
"""
from __future__ import annotations

import os
import pickle
import time
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

from eb_jepa.logging import get_logger
from eb_jepa.singlecell.baselines import (
    DensifyCollator,
    PCABaseline,
    build_baseline,
)
from eb_jepa.singlecell.probes import run_probe_suite
from eb_jepa.singlecell.visualize import effective_rank
from eb_jepa.training_utils import load_config, setup_seed, setup_wandb

from examples.tahoe_baselines.common import (
    build_stream,
    densify_items,
    eval_meta,
    fixed_eval_items,
)

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Build                                                                       #
# --------------------------------------------------------------------------- #
def build_loader(cfg):
    dataset = build_stream(cfg, shuffle=True)
    collator = DensifyCollator(int(cfg.data.n_genes))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.get("num_workers", 0)),
        pin_memory=bool(cfg.data.get("pin_mem", False)),
        drop_last=True,
        collate_fn=collator,
        persistent_workers=int(cfg.data.get("num_workers", 0)) > 0,
    )
    return loader, dataset


def build_eval(cfg, dataset):
    """Fixed shared eval set: dense matrix + probe metadata (rank-0 / single GPU)."""
    n_eval = int(cfg.eval.get("eval_cells", 0))
    if not (cfg.eval.get("enabled", False) and n_eval > 0):
        return None, None
    items = fixed_eval_items(dataset, n_eval)
    dense = densify_items(items, int(cfg.data.n_genes))
    return dense, eval_meta(items)


# --------------------------------------------------------------------------- #
# Probe snapshot (shared with benchmark via run_probe_suite)                  #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def probe_snapshot(model, dense, meta, device, chunk):
    from examples.tahoe_baselines.common import encode_baseline

    reps = encode_baseline(model, dense, device, chunk=chunk)
    metrics: dict = {"repr/effective_rank": float(effective_rank(reps))}
    try:
        suite = run_probe_suite(reps, dict(meta))
        for key, m in suite.items():
            for mk, mv in m.items():
                metrics[f"probe/{key}/{mk}"] = float(mv)
    except Exception:
        logger.warning("probe suite failed", exc_info=True)
    return metrics


def _log_probes(run, metrics, step):
    if run is not None:
        run.log(metrics, step=step)
    line = " | ".join(
        f"{k.split('/', 1)[-1]}={v:.3f}"
        for k, v in metrics.items()
        if k.endswith("balanced_accuracy")
    )
    logger.info(f"[probe @ {step}] eff_rank={metrics['repr/effective_rank']:.2f} | {line}")


# --------------------------------------------------------------------------- #
# PCA (no iterative training)                                                 #
# --------------------------------------------------------------------------- #
def fit_pca(cfg, device):
    loader, dataset = build_loader(cfg)
    n_fit = int(cfg.optim.get("fit_cells", 50000))
    rows, seen = [], 0
    for batch in loader:
        rows.append(batch["dense"])
        seen += batch["dense"].shape[0]
        if seen >= n_fit:
            break
    X = torch.cat(rows, dim=0)[:n_fit]
    logger.info(f"fitting PCA on {X.shape[0]} cells x {X.shape[1]} genes ...")
    model = PCABaseline(n_components=int(cfg.model.n_components))
    model.fit(X)

    os.makedirs(cfg.meta.run_dir, exist_ok=True)
    out = os.path.join(cfg.meta.run_dir, "pca.pkl")
    with open(out, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"  -> saved {out}")

    run = _maybe_wandb(cfg)
    dense, meta = build_eval(cfg, dataset)
    if dense is not None and model._pca is not None:
        ev = float(model._pca.explained_variance_ratio_.sum())
        metrics = probe_snapshot(model, dense, meta, device, chunk=512)
        metrics["pca/cum_explained_var"] = ev
        _log_probes(run, metrics, step=0)
    if run is not None:
        run.finish()
    return model


# --------------------------------------------------------------------------- #
# MAE / VAE training                                                          #
# --------------------------------------------------------------------------- #
def _maybe_wandb(cfg):
    if not cfg.wandb.get("enabled", False):
        return None
    if cfg.wandb.get("entity"):
        os.environ["WANDB_ENTITY"] = cfg.wandb.entity
    return setup_wandb(cfg.wandb.project, cfg, cfg.meta.run_dir, enabled=True)


def train_neural(cfg, device):
    setup_seed(int(cfg.meta.seed))
    native_bf16 = bool(cfg.training.get("native_bf16", False)) and device.type == "cuda"
    train_dtype = torch.bfloat16 if native_bf16 else torch.float32

    loader, dataset = build_loader(cfg)
    mtype = str(cfg.model.type)
    kw: dict = dict(hidden=int(cfg.model.hidden), latent=int(cfg.model.latent))
    if mtype == "mae":
        kw["mask_frac"] = float(cfg.model.get("mask_frac", 0.5))
    elif mtype == "vae":
        kw["kl_coeff"] = float(cfg.model.get("kl_coeff", 1e-3))
    model = build_baseline(mtype, int(cfg.data.n_genes), **kw)
    assert isinstance(model, torch.nn.Module)  # mae/vae only (pca uses fit_pca)
    model = model.to(device=device, dtype=train_dtype)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.optim.lr),
        betas=tuple(cfg.optim.get("betas", (0.9, 0.999))),
        weight_decay=float(cfg.optim.get("weight_decay", 0.0)),
    )
    max_grad_norm = float(cfg.optim.get("max_grad_norm", 1.0))
    max_steps = int(cfg.optim.get("max_steps", 0))
    max_minutes = float(cfg.optim.get("max_minutes", cfg.training.get("max_minutes", 0)))
    log_every = int(cfg.training.get("log_every", 25))
    ckpt_every = int(cfg.training.get("ckpt_every_steps", 1000))
    eval_every = int(cfg.eval.get("eval_every", 0)) if cfg.eval.get("enabled", False) else 0
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{mtype.upper()} trainable params: {n_params:,} | dtype={train_dtype}")

    run = _maybe_wandb(cfg)
    dense, meta = build_eval(cfg, dataset)
    encode_chunk = int(cfg.eval.get("encode_chunk", 256))

    def _save(step, tag):
        os.makedirs(cfg.meta.run_dir, exist_ok=True)
        path = os.path.join(cfg.meta.run_dir, f"{tag}.pt")
        torch.save(
            {"model": model.state_dict(), "type": mtype, "n_genes": int(cfg.data.n_genes),
             "kwargs": kw, "step": step},
            path,
        )
        logger.info(f"  -> saved {tag}.pt @ step {step}")

    def _eval(step):
        if dense is not None:
            _log_probes(run, probe_snapshot(model, dense, meta, device, encode_chunk), step)

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (device.type == "cuda" and not native_bf16 and cfg.training.get("amp", False))
        else nullcontext()
    )

    _eval(0)  # random-init baseline
    t0, step, stop = time.time(), 0, False
    while not stop:
        for batch in loader:
            dense_b = batch["dense"].to(device=device, dtype=train_dtype, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast:
                out = model(dense_b)
                loss = out["loss"]
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            step += 1

            if step % log_every == 0:
                cells = step * int(cfg.data.batch_size)
                m = {"loss": float(loss.detach()), "data/cells_seen": cells,
                     "throughput/cells_per_s": cells / max(time.time() - t0, 1e-9)}
                for k in ("recon_loss", "kl"):
                    if k in out:
                        m[k] = float(out[k].detach())
                logger.info(f"step {step} | " + " ".join(f"{k.split('/')[-1]}={v:.4f}"
                            for k, v in m.items() if "loss" in k or k == "kl"))
                if run is not None:
                    run.log(m, step=step)
            if eval_every and step % eval_every == 0:
                _eval(step)
            if ckpt_every and step % ckpt_every == 0:
                _save(step, "encoder")
            if max_steps and step >= max_steps:
                stop = True
                break
            if max_minutes and (time.time() - t0) / 60.0 >= max_minutes:
                stop = True
                break

    _eval(step)
    _save(step, "encoder_final")
    if run is not None:
        run.finish()
    return model


# --------------------------------------------------------------------------- #
# Entry                                                                       #
# --------------------------------------------------------------------------- #
def run(config: str, **overrides):
    cfg = load_config(config, cli_overrides=overrides or None)
    os.makedirs(cfg.meta.run_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    if str(cfg.model.type) == "pca":
        fit_pca(cfg, device)
    else:
        train_neural(cfg, device)
    logger.info(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    import fire

    fire.Fire({"run": run})
