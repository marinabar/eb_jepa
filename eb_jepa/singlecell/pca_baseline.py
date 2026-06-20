#!/usr/bin/env python3
"""PCA baseline on Tahoe-100M — SAME eval as the sub14 JEPA, PCA instead of JEPA.

This is the headline baseline (CLAUDE.md "Success criteria"): the JEPA must beat a
well-tuned PCA. It is a one-for-one mirror of ``examples/tahoe_jepa/sub14_main``'s
evaluation — same full multi-organ Tahoe stream, same ``maps.pt`` organ/dose
labels, same ``sample_items`` fixed eval set, same detached probe suite, same
per-class t-SNE, and the SAME wandb keys (``probe/<key>/<metric>``,
``repr/effective_rank``, ``tsne/<class>``) — with the JEPA encoder replaced by a
PCA on the densified fixed-gene vocabulary. No training loop: PCA is fit once and
the result is logged to wandb ONCE.

Usage:
    python -m eb_jepa.singlecell.pca_baseline run \
        --config eb_jepa/singlecell/configs/pca_baseline.yaml
    # override anything, e.g. --pca.n_components 256 --pca.n_fit_cells 12000
"""

from __future__ import annotations

import os
import time

import torch

from eb_jepa.datasets.tahoe.dataset import TahoeConfig, TahoeIterableDataset
from eb_jepa.logging import get_logger
from eb_jepa.singlecell.baselines import DensifyCollator, PCABaseline
from eb_jepa.singlecell.probes import run_probe_suite
from eb_jepa.singlecell.visualize import (
    covariance_spectrum,
    effective_rank,
    plot_spectrum,
    plot_tsne_single,
    tsne_embed,
)
from eb_jepa.training_utils import load_config, setup_seed, setup_wandb

logger = get_logger(__name__)

# Probe metadata carried per cell (same fields the sub14 eval probes).
_META_KEYS = ("drug", "sample", "cell_line_id", "organ", "moa_fine", "plate")


# --------------------------------------------------------------------------- #
# Build (mirrors sub14_main.build_loader / build_eval_set)                    #
# --------------------------------------------------------------------------- #
def build_dataset(cfg) -> TahoeIterableDataset:
    """Full multi-organ streaming reader with the organ/dose maps attached."""
    data_cfg = TahoeConfig(
        **{k: cfg.data[k] for k in cfg.data if k in TahoeConfig.__dataclass_fields__}
    )
    maps = {}
    if cfg.data.get("maps_path") and os.path.exists(cfg.data.maps_path):
        maps = torch.load(cfg.data.maps_path)
        logger.info(f"loaded organ/dose maps from {cfg.data.maps_path}")
    else:
        logger.warning("no maps_path -> organ labels will be missing from the probe")
    return TahoeIterableDataset(
        data_cfg,
        binner=None,
        cell_line_to_organ=maps.get("cell_line_to_organ"),
        sample_to_logconc=maps.get("sample_to_logconc"),
        rank=0,
        world_size=1,
        shuffle=False,
    )


def build_eval_set(dataset, cfg):
    """Rank-0 fixed eval + fit sets via ``sample_items`` (deterministic, diverse).

    One ``sample_items`` draw of ``n_fit + n_eval`` cells across evenly-spaced
    shards (so organs/cell lines/drugs are mixed), split disjointly: the eval part
    is the SAME held-out set the JEPA is probed on; the fit part trains the PCA.
    Each cell is densified to the fixed [n_genes] vocabulary for the baseline.
    """
    n_genes = int(cfg.data.get("n_genes", TahoeConfig.n_genes))
    n_eval = int(cfg.eval.get("eval_cells", 3000))
    n_fit = int(cfg.pca.get("n_fit_cells", 8000))
    items = dataset.sample_items(n_fit + n_eval)
    if len(items) < n_fit + n_eval:
        logger.warning(f"requested {n_fit + n_eval} cells, got {len(items)}")
        n_eval = min(n_eval, max(1, len(items) // 3))
    eval_items, fit_items = items[:n_eval], items[n_eval:]

    collate = DensifyCollator(n_genes)
    eval_out = collate(eval_items)
    fit_dense = collate(fit_items)["dense"]
    eval_meta = {k: list(eval_out[k]) for k in _META_KEYS if k in eval_out}
    logger.info(
        f"eval set: {eval_out['dense'].shape[0]} cells | fit set: "
        f"{fit_dense.shape[0]} cells | n_genes={n_genes}"
    )
    return eval_out["dense"], eval_meta, fit_dense


# --------------------------------------------------------------------------- #
# Eval (mirrors sub14_main.run_eval — same wandb keys, PCA reps)              #
# --------------------------------------------------------------------------- #
def run_eval(pca, eval_dense, eval_meta, eval_dir, step, run, cfg):
    """Detached probes + per-class t-SNE on the PCA latent, logged with the SAME
    wandb keys as the sub14 JEPA run (probe/<key>/<metric>, repr/effective_rank,
    tsne/<class>) so the baseline lands in the shared dashboards."""
    reps = pca.encode(eval_dense)  # [N, n_components]
    metrics: dict = {}
    try:
        suite = run_probe_suite(reps, dict(eval_meta))  # {"clf/organ": {...}, ...}
        for key, m in suite.items():
            for mk, mv in m.items():
                metrics[f"probe/{key}/{mk}"] = float(mv)
    except Exception:
        logger.warning("probe suite failed at step %d", step, exc_info=True)
    metrics["repr/effective_rank"] = float(effective_rank(reps))

    # per-class t-SNE panels (same figures + keys as sub14 periodic eval)
    paths: dict = {}
    spectrum_path = None
    try:
        os.makedirs(eval_dir, exist_ok=True)
        emb = tsne_embed(
            reps, seed=int(cfg.meta.seed),
            perplexity=float(cfg.eval.get("perplexity", 30.0)),
        )
        for c in list(cfg.eval.get("classes", ["organ", "cell_line_id", "drug", "moa_fine"])):
            if c in eval_meta:
                p = os.path.join(eval_dir, f"tsne_{c}_step{step:06d}.png")
                plot_tsne_single(emb, eval_meta[c], p, name=c, step=step)
                paths[c] = p
        spectrum_path = os.path.join(eval_dir, f"spectrum_step{step:06d}.png")
        plot_spectrum(
            covariance_spectrum(reps), spectrum_path,
            title=f"PCA covariance spectrum · step {step}",
        )
    except Exception:
        logger.warning("t-SNE/spectrum snapshot failed at step %d", step, exc_info=True)

    if run is not None:
        log = dict(metrics)
        try:
            import wandb

            for c, p in paths.items():
                log[f"tsne/{c}"] = wandb.Image(p, caption=f"step {step}")
            if spectrum_path is not None:
                log["repr/spectrum"] = wandb.Image(spectrum_path, caption=f"step {step}")
        except Exception:
            pass
        run.log(log, step=step)
        run.summary.update(metrics)

    logger.info(
        f"[eval @ {step}] effective_rank={metrics['repr/effective_rank']:.2f}"
        + "".join(
            f" | {k.split('/', 1)[-1]}={v:.3f}"
            for k, v in metrics.items()
            if k.startswith("probe/") and k.endswith("balanced_accuracy")
        )
    )
    return metrics


# --------------------------------------------------------------------------- #
# Fit + eval (mirrors sub14_main.train, but PCA: one fit, one log)            #
# --------------------------------------------------------------------------- #
def fit(cfg):
    setup_seed(int(cfg.meta.seed))
    os.makedirs(cfg.meta.run_dir, exist_ok=True)
    eval_dir = os.path.join(cfg.meta.run_dir, "eval")

    dataset = build_dataset(cfg)
    eval_dense, eval_meta, fit_dense = build_eval_set(dataset, cfg)

    # wandb (rank 0) — same setup path as sub14_main
    run = None
    if cfg.wandb.get("enabled", False):
        if cfg.wandb.get("entity"):
            os.environ["WANDB_ENTITY"] = cfg.wandb.entity
        run = setup_wandb(
            cfg.wandb.project, cfg, cfg.meta.run_dir,
            run_name=cfg.wandb.get("run_name", "pca_baseline"),
            group=cfg.wandb.get("group"),
            tags=["baseline", "pca"],
            resume=False,  # one-shot run, never reattach
            enabled=True,
        )

    # fit the PCA "encoder" on the densified fit cells
    n_comp = int(cfg.pca.get("n_components", 128))
    n_comp = min(n_comp, fit_dense.shape[0], fit_dense.shape[1])
    logger.info(f"fitting PCA(n_components={n_comp}) on {fit_dense.shape[0]} cells")
    pca = PCABaseline(n_components=n_comp).fit(fit_dense)

    run_eval(pca, eval_dense, eval_meta, eval_dir, 0, run, cfg)

    if run is not None:
        run.finish()
    return pca


def run(config: str = "eb_jepa/singlecell/configs/pca_baseline.yaml", **overrides):
    cfg = load_config(config, cli_overrides=overrides or None)
    os.makedirs(cfg.meta.run_dir, exist_ok=True)
    t0 = time.time()
    fit(cfg)
    logger.info(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    import fire

    fire.Fire({"run": run})
