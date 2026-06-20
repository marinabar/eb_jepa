"""Hepatotoxicity-prediction validation suite for the trained perturbator (CLAUDE.md II).

Loads the LIVER-finetuned (frozen) encoder + the trained hepatotox perturbator,
streams liver cells, encodes them, builds per-``(cell_line, plate, drug, dose)`` OT
strata, predicts perturbed latents via the ODE (``predict_perturbed``), and computes
a battery of hepatotoxicity-relevant metrics + publication-grade figures. Every
metric/figure skips gracefully (and logs the skip) when its label/stratum support is
too sparse — the liver subset is only ~3 cell lines, so strata are few.

Metrics (logged to wandb + ``visualizations/hepatotox/dose_hepatotox_ranking.{csv,json}``):
  - Perturbator accuracy on liver — per-stratum sliced-Wasserstein + centroid cosine
    + gap-closed between PREDICTED and REAL treated distributions.
  - DILI classification from the predicted perturbation — ||predicted shift|| at the
    top dose (and its projection on a toxic-vs-safe axis) scored against a curated
    DILI label set (reused from evaluate.py; FDA DILIrank / LiverTox). ROC-AUC +
    balanced accuracy + class counts.
  - Dose-response / potency — predicted ||shift|| vs log10 dose per drug; slope (and
    an EC50-like midpoint when monotone); hepatotoxic vs safe separation.
  - Virtual-pathway attribution — Spearman of each hepatotox pathway feature
    (CYP/BSEP/NRF2/mito-tox) vs predicted shift magnitude across drugs.
  - MoA hierarchy — same ``moa_fine`` drugs -> similar predicted displacement
    direction; mean intra-MoA vs inter-MoA cosine.

Figures (house style, PNG + PDF, dpi>=200, ``visualizations/hepatotox/``): DILI ROC,
dose-response curves for top hepatotoxins, pathway-attribution bar, predicted-vs-real
agreement scatter, and cherry-picked latent dose-shift trajectories.

Run on Dalia (1 GPU):
    /lustre/work/vivatech-unaite/ljung/venv-arm/bin/python -m \
        examples.tahoe_hepatotox.validate_perturbator run \
        --config examples/tahoe_perturbator/cfgs/train_hepatotox.yaml \
        --perturbator_ckpt /lustre/work/vivatech-unaite/ljung/runs/perturbator/hepatotox_liver/perturbator_final.pt
"""
from __future__ import annotations

import collections
import csv
import json
import os
import time

import numpy as np
import torch

from eb_jepa.logging import get_logger
from eb_jepa.singlecell.perturbator.flow import predict_perturbed
from eb_jepa.singlecell.perturbator.hepatotox_features import HepatotoxPathwayFeaturizer
from eb_jepa.singlecell.perturbator.losses import sliced_wasserstein
from eb_jepa.singlecell.perturbator.matching import build_strata
from eb_jepa.singlecell.perturbator.model import Perturbator
from eb_jepa.training_utils import load_config, setup_seed

# Reuse the curated DILI label set from the encoder-level eval (single source).
from examples.tahoe_hepatotox.evaluate import (
    HEPATOTOX_DRUGS,
    LOW_DILI_DRUGS,
    _dili_label,
)

logger = get_logger(__name__)

# House palette (single source of truth: eb_jepa.singlecell.visualize — do not edit it).
from eb_jepa.singlecell.visualize import _INK, _SUB, _palette

_ACCENT = "#2a6f97"
_ACCENT2 = "#3f8bb5"
_GRID = "#e9edf2"
_AXIS = "#c7cfdb"


# =========================================================================== #
# Pure metric helpers (unit-tested on synthetic data)                         #
# =========================================================================== #
def _np(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float64)


def centroid_cosine(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Cosine similarity of the predicted vs real distribution centroids.

    1.0 = predicted cloud sits in the same latent direction as the real treated
    cloud (relative to the origin). Used per-stratum as a coarse direction match.
    """
    a = _np(pred).mean(0)
    b = _np(target).mean(0)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def gap_closed(pred_sw: float, base_sw: float) -> float:
    """Fraction of the source->target OT gap closed by the prediction.

    ``1 - d(pred, target) / d(source, target)``: 1 = perfect, 0 = no better than the
    untouched control, <0 = worse than control.
    """
    if base_sw <= 1e-8:
        return 0.0
    return float(1.0 - pred_sw / base_sw)


def roc_auc(scores, labels) -> float:
    """ROC-AUC of ``scores`` against binary ``labels`` (1 = positive class).

    Rank-based (Mann-Whitney U) so it needs no sklearn; ties get the average rank.
    Returns nan if either class is empty.
    """
    scores = _np(scores).astype(np.float64)
    labels = np.asarray(labels).astype(int)
    pos = labels == 1
    neg = labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average-rank tie correction
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    sum_pos = ranks[pos].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def balanced_accuracy_at_threshold(scores, labels, threshold=None) -> tuple[float, float]:
    """Balanced accuracy of ``scores >= threshold`` vs binary ``labels``.

    If ``threshold`` is None it sweeps candidate thresholds (the score midpoints) and
    returns the best balanced accuracy. Returns ``(balanced_acc, threshold)``.
    """
    scores = _np(scores).astype(np.float64)
    labels = np.asarray(labels).astype(int)
    pos = labels == 1
    neg = labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan"), float("nan")

    def bal_acc(thr):
        pred = scores >= thr
        tpr = (pred & pos).sum() / max(1, pos.sum())
        tnr = (~pred & neg).sum() / max(1, neg.sum())
        return 0.5 * (tpr + tnr)

    if threshold is not None:
        return float(bal_acc(threshold)), float(threshold)
    cands = np.unique(scores)
    mids = (cands[:-1] + cands[1:]) / 2.0 if len(cands) > 1 else cands
    thrs = np.concatenate([[cands.min() - 1.0], mids, [cands.max() + 1.0]])
    accs = [bal_acc(t) for t in thrs]
    best = int(np.argmax(accs))
    return float(accs[best]), float(thrs[best])


def dose_response_slope(log_doses, shifts) -> dict:
    """Least-squares slope of predicted ||shift|| vs log10 dose, with an EC50-like point.

    Returns dict with ``slope`` (shift per log10-molar), ``r`` (Pearson r), and
    ``ec50_log`` — the log10 dose at the half-max shift, linearly interpolated, only
    when the response is monotone non-decreasing (else nan).
    """
    x = _np(log_doses).astype(np.float64)
    y = _np(shifts).astype(np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    out = {"slope": float("nan"), "r": float("nan"), "ec50_log": float("nan"), "n": int(len(x))}
    if len(x) < 2 or np.allclose(x, x[0]):
        return out
    order = np.argsort(x)
    x, y = x[order], y[order]
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    out["slope"] = float(slope)
    if y.std() > 1e-12 and x.std() > 1e-12:
        out["r"] = float(np.corrcoef(x, y)[0, 1])
    # EC50-like midpoint (only meaningful when monotone increasing)
    if np.all(np.diff(y) >= -1e-9) and y[-1] > y[0]:
        half = 0.5 * (y[0] + y[-1])
        for i in range(len(y) - 1):
            if y[i] <= half <= y[i + 1] and y[i + 1] > y[i]:
                frac = (half - y[i]) / (y[i + 1] - y[i])
                out["ec50_log"] = float(x[i] + frac * (x[i + 1] - x[i]))
                break
    return out


def spearman(a, b) -> float:
    """Spearman rank correlation between two 1-D arrays (nan if degenerate)."""
    a = _np(a).astype(np.float64)
    b = _np(b).astype(np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 3:
        return float("nan")

    def rankdata(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=np.float64)
        r[order] = np.arange(1, len(v) + 1)
        # average ties
        sv = v[order]
        i = 0
        while i < len(sv):
            j = i
            while j + 1 < len(sv) and sv[j + 1] == sv[i]:
                j += 1
            if j > i:
                avg = (r[order[i]] + r[order[j]]) / 2.0
                for k in range(i, j + 1):
                    r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = rankdata(a), rankdata(b)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def moa_hierarchy_cosine(displacements, moa_labels) -> dict:
    """Mean intra-MoA vs inter-MoA cosine of per-drug predicted displacement vectors.

    Args:
        displacements: ``[D, d]`` per-drug predicted shift directions (centroid shift).
        moa_labels: length-``D`` MoA labels (None / "" -> excluded).
    Returns:
        dict with ``intra``, ``inter``, ``separation`` (intra - inter), ``n_pairs_*``.
        Drugs whose MoA appears only once contribute to inter pairs only.
    """
    X = _np(displacements)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    U = X / norms
    labs = [m if (m is not None and str(m) != "" and str(m).lower() != "nan") else None
            for m in moa_labels]
    intra, inter = [], []
    n = len(labs)
    for i in range(n):
        for j in range(i + 1, n):
            if labs[i] is None or labs[j] is None:
                continue
            cos = float(np.dot(U[i], U[j]))
            (intra if labs[i] == labs[j] else inter).append(cos)
    out = {
        "intra": float(np.mean(intra)) if intra else float("nan"),
        "inter": float(np.mean(inter)) if inter else float("nan"),
        "n_pairs_intra": len(intra),
        "n_pairs_inter": len(inter),
    }
    out["separation"] = (
        out["intra"] - out["inter"]
        if intra and inter else float("nan")
    )
    return out


# =========================================================================== #
# Encoding + perturbator loading                                              #
# =========================================================================== #
def _load_perturbator(ckpt_path: str, cfg, featurizer, device):
    """Rebuild the perturbator from its checkpoint (action_dim/d_model/objective)."""
    from examples.tahoe_perturbator.main import build_featurizer  # noqa: F401

    state = torch.load(ckpt_path, map_location="cpu")
    objective = str(state.get("objective", cfg.loss.get("objective", "flow_matching")))
    action_dim = int(state.get("action_dim", featurizer.action_dim))
    d_model = int(state.get("d_model", cfg.encoder.d_model))
    if action_dim != featurizer.action_dim:
        logger.warning(
            "checkpoint action_dim=%d != featurizer.action_dim=%d — config/featurizer "
            "mismatch with the trained perturbator.", action_dim, featurizer.action_dim
        )
    model = Perturbator(
        d_model=d_model,
        action_dim=action_dim,
        depth=int(cfg.model.get("depth", 4)),
        d_cond=int(cfg.model.get("d_cond", 256)),
        cond_hidden=cfg.model.get("cond_hidden"),
        time_conditioned=(objective == "flow_matching"),
        n_time_freqs=int(cfg.model.get("n_time_freqs", 64)),
    )
    sd = state.get("model_state_dict", state)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, objective


def _gather_liver_latents(cfg, pc, encoder, device, n_cells, amp):
    """Stream liver cells, encode them, return (latents [N,d], meta dict of lists)."""
    from examples.tahoe_perturbator.main import build_loader, encode_cells

    loader, _, _ = build_loader(cfg, pc)
    keys = ("cell_line_id", "plate", "drug", "canonical_smiles", "moa_fine")
    latents = []
    meta = collections.defaultdict(list)
    seen = 0
    for batch in loader:
        z = encode_cells(encoder, batch, device, amp).cpu()
        latents.append(z)
        for k in keys:
            meta[k].extend(batch[k])
        meta["log_conc"].extend(batch["log_conc"].tolist())
        seen += z.shape[0]
        if seen >= n_cells:
            break
    if not latents:
        return torch.empty(0), meta
    return torch.cat(latents, 0), meta


# =========================================================================== #
# Figures (house style)                                                       #
# =========================================================================== #
def _style_ax(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_AXIS)
        ax.spines[s].set_linewidth(0.9)
    ax.tick_params(colors=_SUB, labelsize=8)
    ax.grid(True, axis="x", color=_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _save(fig, out_prefix, formats=("png", "pdf")):
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    paths = []
    for fmt in formats:
        p = f"{out_prefix}.{fmt}"
        fig.savefig(p, dpi=220, bbox_inches="tight", facecolor="white")
        paths.append(p)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return paths


def plot_roc(scores, labels, out_prefix, auc, title="DILI prediction"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = _np(scores)
    labels = np.asarray(labels).astype(int)
    pos, neg = labels == 1, labels == 0
    thrs = np.concatenate([[np.inf], np.sort(np.unique(scores))[::-1], [-np.inf]])
    tpr = [(scores[pos] >= t).mean() if pos.sum() else 0.0 for t in thrs]
    fpr = [(scores[neg] >= t).mean() if neg.sum() else 0.0 for t in thrs]
    fig, ax = plt.subplots(figsize=(6.4, 6.0), facecolor="white")
    _style_ax(ax)
    ax.grid(True, axis="both", color=_GRID, linewidth=0.8, zorder=0)
    ax.plot([0, 1], [0, 1], color=_SUB, lw=1.0, ls="--", zorder=1)
    ax.plot(fpr, tpr, color=_ACCENT, lw=2.2, zorder=3)
    ax.fill_between(fpr, tpr, color=_ACCENT, alpha=0.12, zorder=2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=_INK, pad=22)
    ax.text(0.0, 1.015, f"hepatotoxicity from predicted perturbation  ·  AUC = {auc:.3f}",
            transform=ax.transAxes, fontsize=9, color=_SUB)
    ax.set_xlabel("false positive rate", color=_SUB, fontsize=9)
    ax.set_ylabel("true positive rate", color=_SUB, fontsize=9)
    return _save(fig, out_prefix)


def plot_dose_response(drug_curves, out_prefix, title="Predicted dose-response"):
    """drug_curves: list of (drug, log_doses[asc], shifts[asc], is_toxic)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 6.0), facecolor="white")
    _style_ax(ax)
    ax.grid(True, axis="both", color=_GRID, linewidth=0.8, zorder=0)
    cols = _palette(max(1, len(drug_curves)))
    for i, (drug, ld, sh, _tox) in enumerate(drug_curves):
        ax.plot(ld, sh, color=cols[i], lw=2.0, marker="o", ms=4, zorder=3, label=str(drug))
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=_INK, pad=22)
    ax.text(0.0, 1.015, f"predicted ‖control→perturbed‖ vs dose  ·  {len(drug_curves)} drug(s)",
            transform=ax.transAxes, fontsize=9, color=_SUB)
    ax.set_xlabel("log10 dose (molar)", color=_SUB, fontsize=9)
    ax.set_ylabel("predicted shift magnitude", color=_SUB, fontsize=9)
    leg = ax.legend(loc="best", frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(_INK)
    return _save(fig, out_prefix)


def plot_pathway_attribution(names, rhos, out_prefix, title="Pathway attribution"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(np.abs(_np(rhos)))[::-1]
    names = [names[i] for i in order]
    rhos = _np(rhos)[order]
    fig, ax = plt.subplots(figsize=(7.4, max(3.0, 0.32 * len(names) + 1.5)), facecolor="white")
    _style_ax(ax)
    y = np.arange(len(names))[::-1]
    cols = [_ACCENT if r >= 0 else _ACCENT2 for r in rhos]
    ax.barh(y, rhos, color=cols, zorder=3, height=0.7)
    ax.axvline(0, color=_AXIS, lw=0.9, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8, color=_INK)
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=_INK, pad=22)
    ax.text(0.0, 1.01, "Spearman(pathway score, predicted shift magnitude) across drugs",
            transform=ax.transAxes, fontsize=9, color=_SUB)
    ax.set_xlabel("Spearman ρ", color=_SUB, fontsize=9)
    return _save(fig, out_prefix)


def plot_agreement_scatter(base_sw, pred_sw, cos, out_prefix, title="Predicted vs real agreement"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_sw = _np(base_sw)
    pred_sw = _np(pred_sw)
    cos = _np(cos)
    fig, ax = plt.subplots(figsize=(6.8, 6.0), facecolor="white")
    _style_ax(ax)
    ax.grid(True, axis="both", color=_GRID, linewidth=0.8, zorder=0)
    lim = max(float(base_sw.max()) if base_sw.size else 1.0,
              float(pred_sw.max()) if pred_sw.size else 1.0) * 1.05
    ax.plot([0, lim], [0, lim], color=_SUB, lw=1.0, ls="--", zorder=1,
            label="no improvement (pred = control)")
    sc = ax.scatter(base_sw, pred_sw, c=cos, cmap="viridis", s=42, zorder=3,
                    edgecolors="white", linewidths=0.6, vmin=-1, vmax=1)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("centroid cosine", color=_SUB, fontsize=8)
    cb.ax.tick_params(colors=_SUB, labelsize=7)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=_INK, pad=22)
    ax.text(0.0, 1.015, "per-stratum sliced-Wasserstein: below the diagonal = perturbator closes the gap",
            transform=ax.transAxes, fontsize=9, color=_SUB)
    ax.set_xlabel("d(control, real treated)", color=_SUB, fontsize=9)
    ax.set_ylabel("d(predicted, real treated)", color=_SUB, fontsize=9)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(_INK)
    return _save(fig, out_prefix)


# =========================================================================== #
# Driver                                                                       #
# =========================================================================== #
def run(
    config: str = "examples/tahoe_perturbator/cfgs/train_hepatotox.yaml",
    perturbator_ckpt: str = "",
    out_dir: str = "visualizations/hepatotox",
    eval_cells: int = 0,
    holdout_frac: float = 0.5,
    top_drugs_fig: int = 6,
    **overrides,
):
    cfg = load_config(config, cli_overrides=overrides or None)
    setup_seed(int(cfg.meta.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    # Encoder + featurizer + perturbator -------------------------------------
    from examples.tahoe_perturbator.main import build_featurizer, build_frozen_encoder
    from eb_jepa.singlecell.sub14.features import load_pc_features, random_pc_features

    cache = cfg.encoder.get("gene_emb_cache", "random")
    pc = load_pc_features(cache) if cache and cache != "random" else random_pc_features(
        n_pc=int(cfg.encoder.get("smoke_n_pc", 2000))
    )
    encoder = build_frozen_encoder(cfg, pc, device)
    featurizer = build_featurizer(cfg)
    # virtual-pathway featurizer for attribution (raw scores, no dose channels)
    path_feat = HepatotoxPathwayFeaturizer()

    if not perturbator_ckpt or not os.path.exists(perturbator_ckpt):
        raise FileNotFoundError(
            f"perturbator_ckpt={perturbator_ckpt!r} not found — train it first with "
            "examples/tahoe_perturbator/cfgs/train_hepatotox.yaml."
        )
    perturbator, objective = _load_perturbator(perturbator_ckpt, cfg, featurizer, device)
    ode_steps = int(cfg.loss.get("ode_steps", 20))
    ode_method = str(cfg.loss.get("ode_method", "heun"))
    sw_slices = int(cfg.loss.get("sw_slices", 256))
    amp = bool(cfg.training.get("amp", True))

    # Liver latents -----------------------------------------------------------
    n_cells = int(eval_cells or cfg.eval.get("eval_cells", 8000))
    latents, meta = _gather_liver_latents(cfg, pc, encoder, device, n_cells, amp)
    if latents.shape[0] == 0:
        raise RuntimeError("No liver cells streamed — check data.liver_only / maps_path.")
    logger.info("encoded %d liver cells -> latents %s", latents.shape[0], tuple(latents.shape))

    # Strata (cell_line, plate, drug, dose) with a held-out target split ------
    strata = build_strata(
        latents, meta["cell_line_id"], meta["plate"], meta["drug"],
        meta["canonical_smiles"], meta["log_conc"],
    )
    min_src = int(cfg.loss.get("min_source", 4))
    min_tgt = int(cfg.loss.get("min_target", 4))
    strata = [s for s in strata if s.source.shape[0] >= min_src and s.target.shape[0] >= min_tgt]
    logger.info("built %d valid liver strata (min_src=%d min_tgt=%d)", len(strata), min_src, min_tgt)
    skipped: dict[str, str] = {}
    if not strata:
        skipped["all"] = "no valid liver strata (too few cells per (line,plate,drug,dose))"

    # MoA per (drug, smiles) for the hierarchy metric
    drug_to_moa: dict[str, str] = {}
    for d, mo in zip(meta["drug"], meta["moa_fine"]):
        if d and d not in drug_to_moa and mo:
            drug_to_moa[d] = mo

    rng = np.random.default_rng(int(cfg.meta.seed))
    results: dict = {}
    per_stratum_rows = []  # for CSV
    drug_shift: dict[tuple, list] = collections.defaultdict(list)  # (drug,smiles)->(log_conc,shift)
    base_sw_all, pred_sw_all, cos_all = [], [], []

    # --- Perturbator accuracy + per-stratum predicted shift ------------------
    for s in strata:
        # held-out split of the REAL target so accuracy is measured out-of-sample
        nt = s.target.shape[0]
        perm = rng.permutation(nt)
        n_eval = max(1, int(round(holdout_frac * nt)))
        eval_idx = perm[:n_eval]
        target_eval = s.target[eval_idx].to(device)
        action = featurizer.featurize(s.smiles, s.log_conc).to(device)
        src = s.source.to(device)
        pred = predict_perturbed(perturbator, src, action, objective,
                                 n_steps=ode_steps, method=ode_method)
        psw = float(sliced_wasserstein(pred, target_eval, n_slices=sw_slices))
        bsw = float(sliced_wasserstein(src, target_eval, n_slices=sw_slices))
        cos = centroid_cosine(pred, target_eval)
        gc = gap_closed(psw, bsw)
        base_sw_all.append(bsw); pred_sw_all.append(psw); cos_all.append(cos)
        # predicted centroid shift (control -> predicted perturbed), full source cloud
        shift_vec = (pred.float().mean(0) - src.float().mean(0)).cpu().numpy()
        shift_mag = float(np.linalg.norm(shift_vec))
        drug_shift[(s.drug, s.smiles)].append(
            (float(s.log_conc), shift_mag, shift_vec)
        )
        per_stratum_rows.append({
            "cell_line": s.stratum[0], "plate": s.stratum[1], "drug": s.drug,
            "log_conc": float(s.log_conc), "n_source": int(src.shape[0]),
            "n_target_eval": int(n_eval), "sliced_w_pred": psw,
            "sliced_w_baseline": bsw, "gap_closed": gc, "centroid_cosine": cos,
            "pred_shift_mag": shift_mag,
        })

    if per_stratum_rows:
        gcs = np.array([r["gap_closed"] for r in per_stratum_rows])
        coss = np.array([r["centroid_cosine"] for r in per_stratum_rows])
        results["accuracy"] = {
            "n_strata": len(per_stratum_rows),
            "mean_sliced_w_pred": float(np.mean(pred_sw_all)),
            "mean_sliced_w_baseline": float(np.mean(base_sw_all)),
            "mean_gap_closed": float(np.mean(gcs)),
            "median_gap_closed": float(np.median(gcs)),
            "mean_centroid_cosine": float(np.nanmean(coss)),
        }
        plot_agreement_scatter(base_sw_all, pred_sw_all, cos_all,
                               os.path.join(out_dir, "perturbator_agreement_scatter"))
    else:
        skipped["accuracy"] = "no strata to score"

    # --- Per-drug summary: top-dose shift, dose slope, pathway scores --------
    drug_rows = []  # one per drug (for ranking CSV)
    drug_disp_dirs = []  # per-drug top-dose displacement direction (MoA hierarchy)
    drug_disp_labels = []
    drug_pathway = {}  # drug -> pathway score vector (raw)
    drug_topshift = {}  # drug -> top-dose shift magnitude
    for (drug, smiles), entries in drug_shift.items():
        entries = sorted(entries, key=lambda e: e[0])  # ascending dose
        log_doses = [e[0] for e in entries]
        mags = [e[1] for e in entries]
        top = entries[-1]  # highest dose
        dr = dose_response_slope(log_doses, mags)
        pv = path_feat.featurize(smiles).numpy() if smiles else None
        toxic = _dili_label(drug)
        row = {
            "drug": drug, "smiles": smiles, "moa_fine": drug_to_moa.get(drug),
            "n_doses": len(entries), "top_log_conc": float(top[0]),
            "top_shift_mag": float(top[1]), "dose_slope": dr["slope"],
            "dose_r": dr["r"], "ec50_log": dr["ec50_log"],
            "dili_label": toxic,
        }
        drug_rows.append(row)
        drug_topshift[drug] = float(top[1])
        drug_disp_dirs.append(top[2])
        drug_disp_labels.append(drug_to_moa.get(drug))
        if pv is not None:
            drug_pathway[drug] = pv

    # --- DILI classification from predicted perturbation ---------------------
    dili_drugs = [r for r in drug_rows if r["dili_label"] in ("hepatotoxic", "low_concern")]
    if len({r["dili_label"] for r in dili_drugs}) >= 2 and len(dili_drugs) >= 4:
        scores = np.array([r["top_shift_mag"] for r in dili_drugs])
        labels = np.array([1 if r["dili_label"] == "hepatotoxic" else 0 for r in dili_drugs])
        auc = roc_auc(scores, labels)
        bacc, thr = balanced_accuracy_at_threshold(scores, labels)
        results["dili"] = {
            "roc_auc": auc, "balanced_accuracy": bacc, "threshold": thr,
            "n_hepatotoxic": int(labels.sum()), "n_low_concern": int((labels == 0).sum()),
            "label_source": "FDA DILIrank / NIH LiverTox curated set (evaluate.py)",
        }
        plot_roc(scores, labels, os.path.join(out_dir, "dili_roc"), auc)
    else:
        skipped["dili"] = (
            f"too few labelled drugs (have {len(dili_drugs)}; need >=4 across 2 classes)"
        )

    # --- Dose-response: hepatotoxic vs safe separation -----------------------
    tox_slopes = [r["dose_slope"] for r in drug_rows
                  if r["dili_label"] == "hepatotoxic" and np.isfinite(r["dose_slope"])]
    safe_slopes = [r["dose_slope"] for r in drug_rows
                   if r["dili_label"] == "low_concern" and np.isfinite(r["dose_slope"])]
    if tox_slopes and safe_slopes:
        results["dose_response"] = {
            "mean_slope_hepatotoxic": float(np.mean(tox_slopes)),
            "mean_slope_low_concern": float(np.mean(safe_slopes)),
            "slope_separation": float(np.mean(tox_slopes) - np.mean(safe_slopes)),
            "n_hepatotoxic": len(tox_slopes), "n_low_concern": len(safe_slopes),
        }
    else:
        skipped["dose_response"] = "need dose slopes for both hepatotoxic and low-concern drugs"
    # dose-response figure for the top hepatotoxins (by top-dose shift)
    tox_curves = []
    for r in sorted([r for r in drug_rows if r["dili_label"] == "hepatotoxic"],
                    key=lambda r: r["top_shift_mag"], reverse=True)[:top_drugs_fig]:
        entries = sorted(drug_shift[(r["drug"], r["smiles"])], key=lambda e: e[0])
        if len(entries) >= 2:
            tox_curves.append((r["drug"], [e[0] for e in entries],
                               [e[1] for e in entries], True))
    if tox_curves:
        plot_dose_response(tox_curves, os.path.join(out_dir, "dose_response_top_hepatotoxins"))
    else:
        skipped["dose_response_fig"] = "no hepatotoxin with >=2 doses for the dose-response figure"

    # --- Virtual-pathway attribution (Spearman across drugs) -----------------
    if len(drug_pathway) >= 3:
        drugs_p = list(drug_pathway.keys())
        P = np.stack([drug_pathway[d] for d in drugs_p])          # [D, n_path]
        mags = np.array([drug_topshift[d] for d in drugs_p])      # [D]
        rhos = [spearman(P[:, j], mags) for j in range(P.shape[1])]
        attribution = dict(zip(path_feat.feature_names, [float(r) for r in rhos]))
        results["pathway_attribution"] = attribution
        plot_pathway_attribution(path_feat.feature_names, rhos,
                                 os.path.join(out_dir, "pathway_attribution"))
    else:
        skipped["pathway_attribution"] = f"too few drugs with SMILES ({len(drug_pathway)}; need >=3)"

    # --- MoA hierarchy (intra vs inter cosine) -------------------------------
    labelled = [(d, l) for d, l in zip(drug_disp_dirs, drug_disp_labels) if l]
    if len(labelled) >= 4 and len({l for _, l in labelled}) >= 2:
        moa = moa_hierarchy_cosine([d for d, _ in labelled], [l for _, l in labelled])
        results["moa_hierarchy"] = moa
    else:
        skipped["moa_hierarchy"] = "too few drugs with a known MoA across >=2 MoAs"

    # --- Cherry-picked dose-shift trajectory figures -------------------------
    try:
        _trajectory_figures(perturbator, featurizer, latents, meta, objective,
                            ode_steps, ode_method, drug_rows, out_dir, top_drugs_fig,
                            int(cfg.meta.seed), min_src)
    except Exception as e:  # display-only; never fail the metric run
        logger.warning("trajectory figures skipped: %s", e)
        skipped["trajectory_figures"] = str(e)

    # --- Persist CSV + JSON --------------------------------------------------
    csv_path = os.path.join(out_dir, "dose_hepatotox_ranking.csv")
    drug_rows_sorted = sorted(drug_rows, key=lambda r: r["top_shift_mag"], reverse=True)
    if drug_rows_sorted:
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(drug_rows_sorted[0].keys()))
            w.writeheader()
            for r in drug_rows_sorted:
                w.writerow(r)
        logger.info("wrote %s (%d drugs)", csv_path, len(drug_rows_sorted))
    results["skipped"] = skipped
    results["n_drugs"] = len(drug_rows)
    summary_path = os.path.join(out_dir, "dose_hepatotox_ranking.json")
    with open(summary_path, "w") as fh:
        json.dump({"summary": results, "drugs": drug_rows_sorted}, fh, indent=2, default=float)
    logger.info("wrote %s", summary_path)

    # --- wandb scalars -------------------------------------------------------
    _log_wandb(cfg, results)

    # --- Console report ------------------------------------------------------
    logger.info("=== Hepatotox perturbator validation (%.1fs) ===", time.time() - t0)
    for section in ("accuracy", "dili", "dose_response", "moa_hierarchy"):
        if section in results:
            logger.info("  %s: %s", section, results[section])
    if "pathway_attribution" in results:
        top = sorted(results["pathway_attribution"].items(),
                     key=lambda kv: abs(kv[1]) if np.isfinite(kv[1]) else -1, reverse=True)[:5]
        logger.info("  top pathway attributions (Spearman): %s", top)
    if skipped:
        logger.info("  SKIPPED: %s", skipped)
    return results


def _trajectory_figures(perturbator, featurizer, latents, meta, objective,
                        ode_steps, ode_method, drug_rows, out_dir, top_k, seed, min_src):
    """Cherry-pick the most hepatotoxic drugs and render latent dose-shift tracks.

    Reuses ``perturbator/visualize.py`` (build_dose_track + monotonicity ranking +
    plot_dose_shift). Control cells of the most-represented hepatic line are the
    shared backdrop; each drug's ascending doses (seen in the stream) form one track.
    """
    from eb_jepa.singlecell.perturbator.visualize import (
        build_dose_track, plot_dose_shift, rank_combos,
    )

    cl = np.array(meta["cell_line_id"], dtype=object)
    drug = np.array(meta["drug"], dtype=object)
    smiles = np.array(meta["canonical_smiles"], dtype=object)
    log_conc = np.array(meta["log_conc"], dtype=np.float64)

    # busiest hepatic line with controls
    ctrl_mask = drug == "DMSO_TF"
    lines = collections.Counter(cl[ctrl_mask].tolist())
    if not lines:
        raise RuntimeError("no control cells for trajectory figures")
    line = lines.most_common(1)[0][0]
    cmask = (cl == line) & ctrl_mask
    control = latents[torch.from_numpy(np.where(cmask)[0])]
    if control.shape[0] < min_src:
        raise RuntimeError(f"too few control cells on line {line}")

    # hepatotoxic drugs present on this line, with >=2 doses
    tox = {r["drug"] for r in drug_rows if r["dili_label"] == "hepatotoxic"}
    tracks = []
    for d in sorted(tox):
        m = (cl == line) & (drug == d)
        if not m.any():
            continue
        doses = sorted({round(float(x), 6) for x in log_conc[m] if np.isfinite(x)})
        if len(doses) < 2:
            continue
        sm = smiles[m][0]
        tracks.append(build_dose_track(
            perturbator, featurizer, control, d, sm, doses,
            objective=objective, ode_steps=ode_steps, ode_method=ode_method,
        ))
    if not tracks:
        raise RuntimeError("no hepatotoxic drug with >=2 doses on the busiest line")
    tracks = rank_combos(tracks)[:top_k]
    plot_dose_shift(control, tracks, os.path.join(out_dir, "dose_shift_hepatotoxins"),
                    cell_line=str(line), projector="pca", seed=seed)


def _log_wandb(cfg, results):
    if not cfg.wandb.get("enabled", False):
        return
    try:
        from eb_jepa.training_utils import setup_wandb

        if cfg.wandb.get("entity"):
            os.environ["WANDB_ENTITY"] = cfg.wandb.entity
        run = setup_wandb(cfg.wandb.project, cfg, cfg.meta.run_dir, enabled=True)
        flat = {}
        for sect, d in results.items():
            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(v, (int, float)):
                        flat[f"validate/{sect}/{k}"] = v
        if flat:
            run.log(flat)
    except Exception as e:
        logger.warning("wandb logging skipped: %s", e)


if __name__ == "__main__":
    import fire

    fire.Fire({"run": run})
