"""Headline benchmark: Subliminal-14 vs MAE / VAE / PCA on a FIXED shared eval set.

Loads each trained representation model, encodes the *same* fixed validation cells
(drawn deterministically from the Tahoe stream), and computes one common metric
table per model:

  - detached probe metrics via ``run_probe_suite`` for [organ, cell_line_id, drug,
    moa_fine] — imbalance-aware (balanced accuracy + macro-F1, with chance),
  - ``effective_rank`` of the representation (collapse diagnostic),
  - optional scib metrics if the ``scib`` package is importable (guarded; skipped
    cleanly otherwise).

Emits a tidy table to CSV + JSON, logs the scalars (and the comparison figures) to
wandb, and writes the figures to ``visualizations/benchmarks/``.

Usage (Dalia, single GPU):
    python -m examples.tahoe_baselines.benchmark run \
        --sub14_config examples/tahoe_jepa/cfgs/sub14_small.yaml \
        --sub14_ckpt   /lustre/work/vivatech-unaite/ljung/runs/sub14/sub14_small/encoder.pt \
        --mae_ckpt /.../runs/baselines/mae/encoder_final.pt \
        --vae_ckpt /.../runs/baselines/vae/encoder_final.pt \
        --pca_ckpt /.../runs/baselines/pca/pca.pkl \
        --eval_cells 3000 --out_dir visualizations/benchmarks
"""
from __future__ import annotations

import csv
import json
import os
import time

import torch

from eb_jepa.logging import get_logger
from eb_jepa.singlecell.probes import run_probe_suite
from eb_jepa.singlecell.visualize import effective_rank
from eb_jepa.training_utils import load_config, setup_wandb

from examples.tahoe_baselines.common import (
    build_stream,
    densify_items,
    encode_baseline,
    encode_sub14,
    eval_meta,
    fixed_eval_items,
    load_baseline_checkpoint,
    load_sub14_checkpoint,
)

logger = get_logger(__name__)

_CLF_CLASSES = ("organ", "cell_line_id", "drug", "moa_fine")


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #
def representation_metrics(reps: torch.Tensor, meta: dict) -> dict:
    """Probe suite (imbalance-aware) + effective rank for one representation."""
    row: dict = {
        "latent_dim": int(reps.shape[1]),
        "effective_rank": float(effective_rank(reps)),
    }
    suite = run_probe_suite(reps, dict(meta))
    for key in _CLF_CLASSES:
        m = suite.get(f"clf/{key}")
        if m is None:
            continue
        row[f"{key}/balanced_accuracy"] = float(m["balanced_accuracy"])
        row[f"{key}/macro_f1"] = float(m["macro_f1"])
        row[f"{key}/chance"] = float(m["chance"])
        row[f"{key}/above_chance"] = float(m["balanced_accuracy"] - m["chance"])
        row[f"{key}/n_classes"] = float(m["n_classes"])
    return row


def scib_metrics(reps: torch.Tensor, meta: dict) -> dict:
    """Optional scib batch/bio metrics. Returns {} (and logs) if scib is absent.

    Treats ``cell_line_id`` as the biological label and ``plate`` as the batch
    (technical) covariate — the standard atlas-integration framing.
    """
    try:
        import anndata as ad
        import numpy as np
        import scib
    except Exception:
        logger.info("scib not importable — skipping scib metrics.")
        return {}
    labels = meta.get("cell_line_id")
    batch = meta.get("plate")
    if not labels or not batch:
        return {}
    obs = {
        "cell_line": np.asarray([x if x is not None else "NA" for x in labels]),
        "batch": np.asarray([x if x is not None else "NA" for x in batch]),
    }
    adata = ad.AnnData(X=reps.numpy().astype("float32"))
    adata.obs["cell_line"] = obs["cell_line"]
    adata.obs["batch"] = obs["batch"]
    adata.obsm["X_emb"] = reps.numpy().astype("float32")
    out: dict = {}
    try:
        out["scib/silhouette_bio"] = float(
            scib.metrics.silhouette(adata, label_key="cell_line", embed="X_emb")
        )
        out["scib/silhouette_batch"] = float(
            scib.metrics.silhouette_batch(
                adata, batch_key="batch", label_key="cell_line", embed="X_emb"
            )
        )
    except Exception:
        logger.warning("scib metric computation failed", exc_info=True)
    return out


# --------------------------------------------------------------------------- #
# Encode each model on the SAME fixed cells                                   #
# --------------------------------------------------------------------------- #
def encode_all(items, *, sub14_config, sub14_ckpt, mae_ckpt, vae_ckpt, pca_ckpt,
               n_genes, device, encode_chunk):
    """Return {model_name -> features [N, d]} encoded on the identical eval cells."""
    feats: dict = {}

    # Baselines (densified vector)
    dense = densify_items(items, n_genes)
    for name, ckpt in (("MAE", mae_ckpt), ("VAE", vae_ckpt), ("PCA", pca_ckpt)):
        if ckpt and os.path.exists(ckpt):
            model = load_baseline_checkpoint(ckpt, device)
            feats[name] = encode_baseline(model, dense, device, chunk=encode_chunk)
            logger.info(f"encoded {name}: {tuple(feats[name].shape)}")
        else:
            logger.warning(f"{name} checkpoint missing ({ckpt}) — skipping.")

    # Subliminal-14 (PC quantile-thermometer views)
    if sub14_ckpt and os.path.exists(sub14_ckpt):
        from eb_jepa.singlecell.sub14.features import load_pc_features

        cfg = load_config(sub14_config, quiet=True)
        cache = cfg.model.get("gene_emb_cache", "")
        pc = load_pc_features(cache)
        model = load_sub14_checkpoint(sub14_ckpt, cfg, pc, device)
        feats["Subliminal14"] = encode_sub14(
            model, items, pc,
            num_bins=int(cfg.data.get("num_bins", 16)),
            genes_per_bin=int(cfg.data.get("genes_per_bin", 32)),
            device=device, chunk=encode_chunk,
        )
        logger.info(f"encoded Subliminal14: {tuple(feats['Subliminal14'].shape)}")
    else:
        logger.warning(f"sub14 checkpoint missing ({sub14_ckpt}) — skipping.")
    return feats


# --------------------------------------------------------------------------- #
# Emit                                                                        #
# --------------------------------------------------------------------------- #
def write_table(table: dict, out_dir: str):
    """Write the {model -> metrics} table to CSV + JSON. Returns (csv, json) paths."""
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "benchmark.json")
    with open(json_path, "w") as f:
        json.dump(table, f, indent=2)
    cols = sorted({k for row in table.values() for k in row})
    csv_path = os.path.join(out_dir, "benchmark.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + cols)
        for model, row in table.items():
            w.writerow([model] + [row.get(c, "") for c in cols])
    logger.info(f"wrote {csv_path} and {json_path}")
    return csv_path, json_path


# --------------------------------------------------------------------------- #
# Entry                                                                       #
# --------------------------------------------------------------------------- #
def run(
    sub14_config: str = "examples/tahoe_jepa/cfgs/sub14_small.yaml",
    sub14_ckpt: str = "",
    mae_ckpt: str = "",
    vae_ckpt: str = "",
    pca_ckpt: str = "",
    data_config: str = "examples/tahoe_baselines/cfgs/mae.yaml",
    eval_cells: int = 3000,
    n_genes: int = 62713,
    encode_chunk: int = 256,
    out_dir: str = "visualizations/benchmarks",
    wandb_enabled: bool = False,
    make_plots: bool = True,
):
    """Benchmark all available models on the fixed shared eval set."""
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fixed shared eval cells (same stream / draw used to train the baselines).
    data_cfg = load_config(data_config, quiet=True)
    dataset = build_stream(data_cfg, shuffle=False)
    items = fixed_eval_items(dataset, eval_cells)
    meta = eval_meta(items)
    logger.info(f"fixed eval set: {len(items)} cells")

    feats = encode_all(
        items, sub14_config=sub14_config, sub14_ckpt=sub14_ckpt,
        mae_ckpt=mae_ckpt, vae_ckpt=vae_ckpt, pca_ckpt=pca_ckpt,
        n_genes=n_genes, device=device, encode_chunk=encode_chunk,
    )
    if not feats:
        raise RuntimeError("No models could be loaded — pass at least one checkpoint.")

    table: dict = {}
    for name, reps in feats.items():
        row = representation_metrics(reps, meta)
        row.update(scib_metrics(reps, meta))
        table[name] = row
        logger.info(f"[{name}] eff_rank={row['effective_rank']:.2f} | " + " ".join(
            f"{c.split('/')[0]}={row[c]:.3f}" for c in row if c.endswith("balanced_accuracy")
        ))

    csv_path, json_path = write_table(table, out_dir)

    fig_paths: dict = {}
    if make_plots:
        from examples.tahoe_baselines.plots import make_all_plots

        fig_paths = make_all_plots(table, feats, meta, out_dir)

    if wandb_enabled:
        if data_cfg.wandb.get("entity"):
            os.environ["WANDB_ENTITY"] = data_cfg.wandb.entity
        run_wb = setup_wandb(data_cfg.wandb.project, {"benchmark": True}, out_dir, enabled=True)
        if run_wb is not None:
            import wandb

            flat = {f"benchmark/{m}/{k}": v for m, row in table.items() for k, v in row.items()}
            for tag, p in fig_paths.items():
                if p and os.path.exists(p) and p.endswith(".png"):
                    flat[f"benchmark/fig/{tag}"] = wandb.Image(p)
            run_wb.log(flat)
            run_wb.finish()

    logger.info(f"Benchmark done in {time.time() - t0:.1f}s")
    return table


if __name__ == "__main__":
    import fire

    fire.Fire({"run": run})
