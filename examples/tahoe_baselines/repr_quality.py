"""Representation-QUALITY benchmark: where a LeJEPA encoder is EXPECTED to win.

The headline probe table (``benchmark.py``) scores cell_line / organ classification
accuracy — axes where a high-capacity reconstruction baseline (MAE / VAE) or even
PCA can match or beat a JEPA, because they pack the full transcriptome into the
latent. This module instead measures the properties LeJEPA is *trained* to have and
the others are *not*, all on the IDENTICAL fixed eval cells / encoded matrix
``Z [N, 256]``:

  1. View-invariance to raw-cell gene dropout (the PRIMARY expected JEPA win):
     alignment (cosine of two dropped views) + self-retrieval (top-1 / mean rank).
  2. Isotropic-Gaussianity (the SIGReg target): SIGReg statistic on standardized Z,
     an isotropy score exp(mean log eig)/mean eig, condition number, top-eig fraction.
  3. Batch-invariance vs bio-conservation: plate (technical, lower better) vs
     cell_line (bio, higher better) probe accuracy, their gap and ratio; optional
     scib silhouette mixing when scib is importable.
  4. kNN vs linear gap: k=15 balanced kNN accuracy on cell_line / organ beside the
     linear-probe number (local-structure quality).
  5. Uniformity & alignment (Wang-Isola): uniformity = log E[exp(-2||zi-zj||^2)] on
     L2-normalized Z, alignment from metric 1.

FAIRNESS: the dropout perturbation is defined at the RAW cell level and applied
identically to every model; each model then encodes the two views through its OWN
native pipeline (sub14: Sub14Collator -> encode; baselines: densify over the SAME
panel passed via --hvg_path -> encode_baseline). All reps are pre-projection /
bottleneck; everything is detached, nothing trains.

Usage (Dalia, single GPU) on the EXPRESSED-prevalence panel:
    python -m examples.tahoe_baselines.repr_quality run \
        --sub14_config examples/tahoe_jepa/cfgs/sub14_small.yaml \
        --sub14_ckpt   /.../runs/sub14/sub14_small/encoder_frozen_pert.pt \
        --hvg_path     /.../tahoe-cache/hvg_512_prevalence.npy \
        --mae_matched_ckpt /.../runs/baselines_matched_expressed/mae/encoder_final.pt \
        --vae_matched_ckpt /.../runs/baselines_matched_expressed/vae/encoder_final.pt \
        --pca_matched_ckpt /.../runs/baselines_matched_expressed/pca/pca.pkl \
        --eval_cells 3000 --out_dir visualizations/repr_quality
"""
from __future__ import annotations

import csv
import json
import os
import time

import numpy as np
import torch

from eb_jepa.logging import get_logger
from eb_jepa.singlecell.probes import train_classification_probe
from eb_jepa.training_utils import load_config, setup_wandb

from examples.tahoe_baselines.common import (
    build_hvg_local_map,
    build_stream,
    densify_hvg,
    encode_baseline,
    eval_meta,
    fixed_eval_items,
    load_baseline_checkpoint,
    load_hvg_panel,
    load_sub14_checkpoint,
)

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Metric direction registry: orient every figure so "up = better".            #
# --------------------------------------------------------------------------- #
# value: +1 means higher is better, -1 means lower is better.
METRIC_DIRECTION = {
    "invariance/alignment_cos": +1,
    "invariance/self_retrieval_top1": +1,
    "invariance/self_retrieval_mean_rank": -1,
    "gaussianity/sigreg": -1,
    "gaussianity/isotropy_score": +1,
    "gaussianity/condition_number": -1,
    "gaussianity/top_eig_fraction": -1,
    "batch/plate_acc": -1,
    "batch/cell_line_acc": +1,
    "batch/bio_minus_batch": +1,
    "batch/bio_over_batch": +1,
    "batch/scib_silhouette_batch": +1,
    "knn/cell_line_acc": +1,
    "knn/organ_acc": +1,
    # Wang-Isola uniformity = log E[exp(-2||zi-zj||^2)]: more NEGATIVE = more
    # uniformly spread on the sphere = better (collapsed reps sit near 0).
    "uniformity/uniformity": -1,
    "uniformity/alignment_cos": +1,
}

# The subset surfaced as "JEPA-expected-win" headline metrics.
HEADLINE_METRICS = (
    "invariance/alignment_cos",
    "invariance/self_retrieval_top1",
    "gaussianity/sigreg",
    "gaussianity/isotropy_score",
    "batch/bio_minus_batch",
    "uniformity/uniformity",
)


# --------------------------------------------------------------------------- #
# Raw-cell dropout perturbation (fair across architectures)                   #
# --------------------------------------------------------------------------- #
def drop_view_item(item: dict, p: float, rng: np.random.Generator) -> dict:
    """Return a copy of ``item`` keeping a random (1-p) fraction of expressed genes.

    The perturbation is defined on the RAW sparse cell (gene_token_ids + aligned
    values / raw_counts), so it is identical regardless of which model later encodes
    it. At least one gene is always kept.
    """
    tok = item["gene_token_ids"]
    g = int(tok.numel())
    if g == 0:
        return item
    keep_n = max(1, int(round(g * (1.0 - p))))
    sel = np.sort(rng.choice(g, size=keep_n, replace=False))
    sel_t = torch.from_numpy(sel).long()
    out = dict(item)
    out["gene_token_ids"] = tok[sel_t]
    out["values"] = item["values"][sel_t]
    rc = item.get("raw_counts")
    if rc is not None:
        out["raw_counts"] = rc[sel_t]
    return out


def make_two_views(items: list[dict], p: float, seed: int):
    """Two independent dropout views of every cell (lists aligned to ``items``)."""
    rng = np.random.default_rng(seed)
    view_a = [drop_view_item(c, p, rng) for c in items]
    view_b = [drop_view_item(c, p, rng) for c in items]
    return view_a, view_b


# --------------------------------------------------------------------------- #
# Per-model encoders of an arbitrary item list -> Z [N, d]                     #
# --------------------------------------------------------------------------- #
def _encode_sub14_items(model, items, pc, *, num_bins, genes_per_bin, device,
                        seed, chunk):
    """Encode a list of (possibly dropped) cells with sub14's native pipeline."""
    from eb_jepa.singlecell.sub14.collator import Sub14Collator

    coll = Sub14Collator(
        token_to_pc_local=pc.token_to_pc_local,
        n_pc_genes=pc.n_pc_genes,
        num_bins=num_bins,
        genes_per_bin=genes_per_bin,
        num_views=1,
        binomial_subsample=None,
        seed=seed,
    )
    view = coll(items)
    gene_ids, bin_ids, pad = view["gene_ids"][0], view["bin_ids"][0], view["padding_mask"][0]
    model.eval()
    reps = []
    with torch.no_grad():
        for s in range(0, gene_ids.size(0), chunk):
            sl = slice(s, s + chunk)
            r = model.encode(gene_ids[sl].to(device), bin_ids[sl].to(device),
                             pad[sl].to(device))
            reps.append(r.float().cpu())
    return torch.cat(reps, dim=0)


def _encode_baseline_items(model, items, hvg_local_map, n_hvg, device, chunk):
    """Densify a list of (possibly dropped) cells over the panel, then encode."""
    dense = densify_hvg(items, hvg_local_map, n_hvg)
    return encode_baseline(model, dense, device, chunk=chunk)


# --------------------------------------------------------------------------- #
# Metric 1 — view invariance (alignment + self-retrieval)                     #
# --------------------------------------------------------------------------- #
def _cosine_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    an = torch.nn.functional.normalize(a.float(), dim=1)
    bn = torch.nn.functional.normalize(b.float(), dim=1)
    return (an * bn).sum(1)


def invariance_metrics(za: torch.Tensor, zb: torch.Tensor) -> dict:
    """Alignment (mean row cosine) + self-retrieval (top-1 + mean rank) A->B."""
    align = float(_cosine_rows(za, zb).mean())
    an = torch.nn.functional.normalize(za.float(), dim=1)
    bn = torch.nn.functional.normalize(zb.float(), dim=1)
    sim = an @ bn.t()  # [N, N], rows = view-A query, cols = view-B keys
    n = sim.shape[0]
    self_sim = sim.diagonal()
    # rank of the true match = #keys strictly more similar than the self match (+1)
    rank = (sim > self_sim.unsqueeze(1)).sum(1) + 1
    top1 = float((rank == 1).float().mean())
    return {
        "alignment_cos": align,
        "self_retrieval_top1": top1,
        "self_retrieval_mean_rank": float(rank.float().mean()),
    }


# --------------------------------------------------------------------------- #
# Metric 2 — isotropic-Gaussianity                                            #
# --------------------------------------------------------------------------- #
def _standardize_Z(Z: torch.Tensor) -> torch.Tensor:
    """Center + scale by a single global scalar (matches probes._standardize)."""
    Z = Z.float()
    mu = Z.mean(0, keepdim=True)
    s = (Z - mu).std().clamp(min=1e-6)
    return (Z - mu) / s


def isotropy_score(Z: torch.Tensor) -> dict:
    """exp(mean log eig)/mean eig (1=isotropic), condition number, top-eig fraction."""
    from eb_jepa.singlecell.visualize import covariance_spectrum

    eig = covariance_spectrum(Z).clamp(min=0)
    eig_pos = eig[eig > 1e-12]
    if eig_pos.numel() == 0:
        return {"isotropy_score": 0.0, "condition_number": float("inf"),
                "top_eig_fraction": 1.0}
    geo = torch.exp(eig_pos.log().mean())
    arith = eig.mean().clamp(min=1e-12)
    return {
        "isotropy_score": float(geo / arith),
        "condition_number": float(eig.max() / eig_pos.min()),
        "top_eig_fraction": float(eig.max() / eig.sum().clamp(min=1e-12)),
    }


def gaussianity_metrics(Z: torch.Tensor, *, num_slices: int, seed: int) -> dict:
    """SIGReg statistic on standardized Z + isotropy diagnostics."""
    from eb_jepa.singlecell.sub14.sigreg import SIGReg

    Zs = _standardize_Z(Z)
    torch.manual_seed(seed)
    sig = SIGReg(num_slices=num_slices)
    stat = float(sig(Zs).item())
    out = {"sigreg": stat}
    out.update(isotropy_score(Zs))
    return out


# --------------------------------------------------------------------------- #
# Metric 4 — kNN balanced accuracy (local structure)                          #
# --------------------------------------------------------------------------- #
def knn_balanced_accuracy(Z: torch.Tensor, labels: list, k: int = 15,
                          seed: int = 0) -> float:
    """k-NN (cosine) balanced accuracy on a 80/20 split. NaN if <2 usable classes."""
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.neighbors import KNeighborsClassifier

    classes = sorted({x for x in labels if x is not None})
    if len(classes) < 2:
        return float("nan")
    idx = {c: i for i, c in enumerate(classes)}
    y = np.array([idx.get(x, -1) for x in labels])
    keep = y >= 0
    X = _standardize_Z(Z).numpy()[keep]
    y = y[keep]
    if X.shape[0] < 8:
        return float("nan")
    g = np.random.default_rng(seed)
    perm = g.permutation(X.shape[0])
    n_val = max(1, int(0.2 * X.shape[0]))
    va, tr = perm[:n_val], perm[n_val:]
    if len(set(y[tr].tolist())) < 2:
        return float("nan")
    kk = min(k, len(tr))
    clf = KNeighborsClassifier(n_neighbors=kk, metric="cosine", weights="distance")
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[va])
    return float(balanced_accuracy_score(y[va], pred))


# --------------------------------------------------------------------------- #
# Metric 5 — Wang-Isola uniformity                                            #
# --------------------------------------------------------------------------- #
def uniformity(Z: torch.Tensor, t: float = 2.0, max_n: int = 4096,
               seed: int = 0) -> float:
    """log E[exp(-t ||zi - zj||^2)] on L2-normalized Z (less negative = more spread)."""
    Zn = torch.nn.functional.normalize(Z.float(), dim=1)
    n = Zn.shape[0]
    if n > max_n:
        g = torch.Generator().manual_seed(seed)
        sub = torch.randperm(n, generator=g)[:max_n]
        Zn = Zn[sub]
    sq = torch.pdist(Zn, p=2).pow(2)
    if sq.numel() == 0:
        return float("nan")
    return float((-t * sq).exp().mean().clamp(min=1e-30).log())


# --------------------------------------------------------------------------- #
# Optional scib mixing                                                        #
# --------------------------------------------------------------------------- #
def scib_batch_mixing(Z: torch.Tensor, meta: dict) -> dict:
    """scib silhouette_batch (batch-mixing, higher=better). {} if scib absent."""
    try:
        import anndata as ad
        import scib
    except Exception:
        return {}
    labels = meta.get("cell_line_id")
    batch = meta.get("plate")
    if not labels or not batch:
        return {}
    adata = ad.AnnData(X=Z.numpy().astype("float32"))
    adata.obs["cell_line"] = np.asarray([x or "NA" for x in labels])
    adata.obs["batch"] = np.asarray([x or "NA" for x in batch])
    adata.obsm["X_emb"] = Z.numpy().astype("float32")
    try:
        return {"scib_silhouette_batch": float(
            scib.metrics.silhouette_batch(
                adata, batch_key="batch", label_key="cell_line", embed="X_emb"))}
    except Exception:
        logger.warning("scib silhouette_batch failed", exc_info=True)
        return {}


# --------------------------------------------------------------------------- #
# Encode all models on (clean, view-A, view-B) of the SAME fixed cells         #
# --------------------------------------------------------------------------- #
def _resolve_baselines(mae_ckpt, vae_ckpt, pca_ckpt, n_hvg):
    return [(f"MAE-{n_hvg}", mae_ckpt), (f"VAE-{n_hvg}", vae_ckpt),
            (f"PCA-{n_hvg}", pca_ckpt)]


def encode_all_views(items, view_a, view_b, *, sub14_config, sub14_ckpt,
                     mae_ckpt, vae_ckpt, pca_ckpt, hvg_path, device,
                     encode_chunk, view_seed):
    """Return {model -> {"clean","a","b": Z[N,d]}} on the identical eval cells.

    Baselines densify over the panel from ``hvg_path``; sub14 uses its own pipeline.
    Clean, view-A and view-B are encoded with each model's NATIVE forward so the
    invariance comparison is fair (same raw dropout, model-native encoding).
    """
    if not hvg_path:
        raise ValueError("repr_quality needs --hvg_path (the shared dense panel).")
    panel = load_hvg_panel(hvg_path)
    n_hvg = int(panel.numel())
    hvg_local = build_hvg_local_map(panel)
    feats: dict = {}

    for name, ckpt in _resolve_baselines(mae_ckpt, vae_ckpt, pca_ckpt, n_hvg):
        if not (ckpt and os.path.exists(ckpt)):
            logger.warning(f"{name} checkpoint missing ({ckpt}) — skipping.")
            continue
        model = load_baseline_checkpoint(ckpt, device)
        feats[name] = {
            "clean": _encode_baseline_items(model, items, hvg_local, n_hvg, device, encode_chunk),
            "a": _encode_baseline_items(model, view_a, hvg_local, n_hvg, device, encode_chunk),
            "b": _encode_baseline_items(model, view_b, hvg_local, n_hvg, device, encode_chunk),
        }
        logger.info(f"encoded {name}: clean={tuple(feats[name]['clean'].shape)}")

    if sub14_ckpt and os.path.exists(sub14_ckpt):
        from eb_jepa.singlecell.sub14.features import load_pc_features

        cfg = load_config(sub14_config, quiet=True)
        pc = load_pc_features(cfg.model.get("gene_emb_cache", ""))
        model = load_sub14_checkpoint(sub14_ckpt, cfg, pc, device)
        nb = int(cfg.data.get("num_bins", 16))
        gpb = int(cfg.data.get("genes_per_bin", 32))
        kw = dict(model=model, pc=pc, num_bins=nb, genes_per_bin=gpb,
                  device=device, chunk=encode_chunk)
        feats["Subliminal14"] = {
            "clean": _encode_sub14_items(items=items, seed=view_seed, **kw),
            "a": _encode_sub14_items(items=view_a, seed=view_seed + 1, **kw),
            "b": _encode_sub14_items(items=view_b, seed=view_seed + 2, **kw),
        }
        logger.info(f"encoded Subliminal14: clean={tuple(feats['Subliminal14']['clean'].shape)}")
    else:
        logger.warning(f"sub14 checkpoint missing ({sub14_ckpt}) — skipping.")
    return feats


# --------------------------------------------------------------------------- #
# Assemble the metric row for one model                                       #
# --------------------------------------------------------------------------- #
def model_metrics(z: dict, meta: dict, *, sigreg_slices: int, knn_k: int,
                  seed: int) -> dict:
    """All five metric families for one model from its {clean,a,b} encodings."""
    Z, za, zb = z["clean"], z["a"], z["b"]
    row: dict = {"latent_dim": int(Z.shape[1])}

    for k, v in invariance_metrics(za, zb).items():
        row[f"invariance/{k}"] = v
    row["uniformity/alignment_cos"] = row["invariance/alignment_cos"]

    for k, v in gaussianity_metrics(Z, num_slices=sigreg_slices, seed=seed).items():
        row[f"gaussianity/{k}"] = v

    plate_acc = train_classification_probe(Z, meta.get("plate", []), seed=seed)
    line_acc = train_classification_probe(Z, meta.get("cell_line_id", []), seed=seed)
    pa = float(plate_acc.get("balanced_accuracy", float("nan")))
    ca = float(line_acc.get("balanced_accuracy", float("nan")))
    row["batch/plate_acc"] = pa
    row["batch/cell_line_acc"] = ca
    row["batch/bio_minus_batch"] = ca - pa
    row["batch/bio_over_batch"] = ca / pa if pa > 1e-6 else float("nan")
    row.update({f"batch/{k}": v for k, v in scib_batch_mixing(Z, meta).items()})

    row["knn/cell_line_acc"] = knn_balanced_accuracy(Z, meta.get("cell_line_id", []), k=knn_k, seed=seed)
    row["knn/organ_acc"] = knn_balanced_accuracy(Z, meta.get("organ", []), k=knn_k, seed=seed)

    row["uniformity/uniformity"] = uniformity(Z, seed=seed)
    return row


# --------------------------------------------------------------------------- #
# Win summary                                                                 #
# --------------------------------------------------------------------------- #
def compute_wins(table: dict, target: str = "Subliminal14") -> dict:
    """For each metric, the best model (orientation-aware) + whether sub14 wins."""
    wins: dict = {}
    if target not in table:
        return wins
    for metric, direction in METRIC_DIRECTION.items():
        vals = {m: row[metric] for m, row in table.items()
                if metric in row and np.isfinite(row[metric])}
        if target not in vals or len(vals) < 2:
            continue
        best = (max if direction > 0 else min)(vals, key=vals.get)
        wins[metric] = {
            "winner": best,
            "sub14_wins": best == target,
            "sub14_value": vals[target],
            "best_value": vals[best],
            "direction": "higher" if direction > 0 else "lower",
        }
    return wins


# --------------------------------------------------------------------------- #
# Emit table                                                                  #
# --------------------------------------------------------------------------- #
def write_table(table: dict, wins: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "repr_quality.json")
    with open(json_path, "w") as f:
        json.dump({"table": table, "wins": wins}, f, indent=2)
    cols = sorted({k for row in table.values() for k in row})
    csv_path = os.path.join(out_dir, "repr_quality.csv")
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
    mae_matched_ckpt: str = "",
    vae_matched_ckpt: str = "",
    pca_matched_ckpt: str = "",
    hvg_path: str = "",
    data_config: str = "examples/tahoe_baselines/cfgs/mae.yaml",
    eval_cells: int = 3000,
    dropout_p: float = 0.5,
    sigreg_slices: int = 256,
    knn_k: int = 15,
    encode_chunk: int = 256,
    seed: int = 0,
    out_dir: str = "visualizations/repr_quality",
    wandb_enabled: bool = False,
    make_plots: bool = True,
):
    """Representation-quality benchmark on the fixed shared eval set."""
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = load_config(data_config, quiet=True)
    dataset = build_stream(data_cfg, shuffle=False)
    items = fixed_eval_items(dataset, eval_cells)
    meta = eval_meta(items)
    logger.info(f"fixed eval set: {len(items)} cells")

    # ONE raw-cell dropout, shared by every model (fairness).
    view_a, view_b = make_two_views(items, dropout_p, seed)

    feats = encode_all_views(
        items, view_a, view_b,
        sub14_config=sub14_config, sub14_ckpt=sub14_ckpt,
        mae_ckpt=mae_matched_ckpt, vae_ckpt=vae_matched_ckpt, pca_ckpt=pca_matched_ckpt,
        hvg_path=hvg_path, device=device, encode_chunk=encode_chunk, view_seed=seed,
    )
    if not feats:
        raise RuntimeError("No models could be loaded — pass at least one checkpoint.")

    table: dict = {}
    for name, z in feats.items():
        table[name] = model_metrics(z, meta, sigreg_slices=sigreg_slices,
                                    knn_k=knn_k, seed=seed)
        r = table[name]
        logger.info(
            f"[{name}] align={r['invariance/alignment_cos']:.3f} "
            f"sigreg={r['gaussianity/sigreg']:.2f} iso={r['gaussianity/isotropy_score']:.3f} "
            f"bio-batch={r['batch/bio_minus_batch']:.3f}"
        )

    wins = compute_wins(table)
    n_win = sum(1 for w in wins.values() if w["sub14_wins"])
    logger.info(f"Subliminal14 wins {n_win}/{len(wins)} oriented metrics")

    csv_path, json_path = write_table(table, wins, out_dir)

    fig_paths: dict = {}
    if make_plots:
        from examples.tahoe_baselines.repr_quality_plots import make_all_plots

        fig_paths = make_all_plots(table, wins, METRIC_DIRECTION, HEADLINE_METRICS, out_dir)

    if wandb_enabled:
        if data_cfg.wandb.get("entity"):
            os.environ["WANDB_ENTITY"] = data_cfg.wandb.entity
        run_wb = setup_wandb(data_cfg.wandb.project, {"repr_quality": True}, out_dir, enabled=True)
        if run_wb is not None:
            import wandb

            flat = {f"repr_quality/{m}/{k}": v for m, row in table.items()
                    for k, v in row.items() if isinstance(v, (int, float))}
            flat["repr_quality/sub14_wins"] = n_win
            for tag, p in fig_paths.items():
                if p and os.path.exists(p) and p.endswith(".png"):
                    flat[f"repr_quality/fig/{tag}"] = wandb.Image(p)
            run_wb.log(flat)
            run_wb.finish()

    logger.info(f"repr_quality done in {time.time() - t0:.1f}s")
    return table, wins


if __name__ == "__main__":
    import fire

    fire.Fire({"run": run})
