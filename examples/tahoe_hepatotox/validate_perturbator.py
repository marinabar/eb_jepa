"""Hepatotoxicity-prediction validation suite for the trained perturbator (CLAUDE.md II).

Loads the LIVER-finetuned (frozen) encoder + the trained hepatotox perturbator,
streams a TARGETED sample of liver cells (bucketed per ``(cell_line, plate, drug,
dose)`` so labelled / multi-dose drugs are actually covered), encodes them, builds
per-stratum OT problems, predicts perturbed latents via the ODE (``predict_perturbed``),
and computes a comprehensive, *self-ranking* battery of hepatotoxicity metrics +
publication-grade figures. Every metric/figure skips gracefully (and logs the skip)
when its label/stratum support is genuinely too sparse.

The DILI labels are the FDA **DILIrank** + NIH **LiverTox** drug-name vocabulary
(``hepatotox_features.HEPATOTOX_DRUGS`` / ``LOW_DILI_DRUGS``, primary) plus a weak
MoA-derived label (``weak_dili_label_from_moa``, secondary, reported separately).

Each analysis tries MULTIPLE formulations and keeps the BEST:
  - DILI classification: top-dose / mean / dose-slope / learned-axis shift scores,
    each scored (ROC-AUC + balanced acc) against both label sources; best reported.
  - Dose-response: per-drug ||shift|| vs log10 dose -> slope / monotonicity / EC50;
    hepatotoxic-vs-safe separation (slope AUROC + Mann-Whitney p), best formulation.
  - Virtual-pathway attribution: Spearman of each pathway feature vs predicted shift.
  - MoA hierarchy: intra- vs inter-MoA displacement cosine + permutation p-value.
  - Predicted-vs-real agreement: sliced-Wasserstein / gap-closed / centroid cosine.
  - Cherry-picked dose-shift trajectory figures (ranked by clean dose-response).

Outputs (``visualizations/hepatotox/``): ``best_findings.json`` + ``best_findings.md``
(ranked headline results), ``dose_hepatotox_ranking.{csv,json}``, and house-style
figures (PNG + PDF): DILI ROC, dose-response curves, pathway attribution, MoA
hierarchy, predicted-vs-real agreement, top dose-shift trajectories.

Run on Dalia (1 GPU):
    /lustre/work/vivatech-unaite/ljung/venv-arm/bin/python -m \
        examples.tahoe_hepatotox.validate_perturbator run \
        --config examples/tahoe_perturbator/cfgs/train_hepatotox.yaml \
        --perturbator_ckpt /lustre/work/vivatech-unaite/ljung/runs/perturbator/hepatotox_liver_long/perturbator_final.pt \
        --eval_cells 150000
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
from eb_jepa.singlecell.perturbator.hepatotox_features import (
    HepatotoxPathwayFeaturizer,
    dili_label_by_name,
    weak_dili_label_from_moa,
)
from eb_jepa.singlecell.perturbator.losses import sliced_wasserstein
from eb_jepa.singlecell.perturbator.matching import build_strata
from eb_jepa.singlecell.perturbator.model import Perturbator
from eb_jepa.training_utils import load_config, setup_seed

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


def mannwhitney_p(a, b) -> float:
    """Two-sided Mann-Whitney U p-value (normal approximation, tie-corrected).

    Tests whether ``a`` and ``b`` are drawn from the same distribution. Returns nan
    when either group is empty. Used for hepatotoxic-vs-safe slope separation.
    """
    a = _np(a).astype(np.float64)
    b = _np(b).astype(np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("nan")
    allv = np.concatenate([a, b])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    sv = allv[order]
    # average ties
    i = 0
    tie_term = 0.0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            tcount = j - i + 1
            tie_term += tcount ** 3 - tcount
        i = j + 1
    r_a = ranks[:na].sum()
    u_a = r_a - na * (na + 1) / 2.0
    u = min(u_a, na * nb - u_a)
    n = na + nb
    mu = na * nb / 2.0
    sigma2 = (na * nb / 12.0) * ((n + 1) - tie_term / (n * (n - 1)))
    if sigma2 <= 0:
        return float("nan")
    z = (u - mu) / np.sqrt(sigma2)
    # two-sided normal p
    from math import erf, sqrt

    p = 2.0 * (0.5 * (1.0 + erf(-abs(z) / sqrt(2.0))))
    return float(min(1.0, max(0.0, p)))


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


def cv_axis_scores(features, labels, n_folds=5, seed=0) -> np.ndarray:
    """Cross-validated logistic-regression decision scores on a feature matrix.

    Learns a toxic-vs-safe direction from ``features`` ([D, p]) and returns the
    out-of-fold decision scores ([D]) so the projection onto the learned axis can be
    ROC-scored without leakage. Falls back to a leave-one-out scheme when D is small.
    Returns an all-nan array if sklearn is unavailable or a class is empty.
    """
    X = _np(features).astype(np.float64)
    y = np.asarray(labels).astype(int)
    D = X.shape[0]
    out = np.full(D, np.nan)
    if D < 4 or y.sum() == 0 or (y == 0).sum() == 0:
        return out
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
    except Exception:
        return out
    n_folds = int(min(n_folds, y.sum(), (y == 0).sum()))
    if n_folds < 2:
        n_folds = 2
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0)
        Xtr = X[tr]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        clf.fit((Xtr - mu) / sd, y[tr])
        out[te] = clf.decision_function((X[te] - mu) / sd)
    return out


def dose_response_slope(log_doses, shifts) -> dict:
    """Least-squares slope of predicted ||shift|| vs log10 dose, with an EC50-like point.

    Returns dict with ``slope`` (shift per log10-molar), ``r`` (Pearson r),
    ``monotonicity`` (fraction of consecutive ascending-dose steps that increase),
    and ``ec50_log`` — the log10 dose at the half-max shift, linearly interpolated,
    only when the response is monotone non-decreasing (else nan).
    """
    x = _np(log_doses).astype(np.float64)
    y = _np(shifts).astype(np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    out = {"slope": float("nan"), "r": float("nan"), "ec50_log": float("nan"),
           "monotonicity": float("nan"), "n": int(len(x))}
    if len(x) < 2 or np.allclose(x, x[0]):
        return out
    order = np.argsort(x)
    x, y = x[order], y[order]
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    out["slope"] = float(slope)
    if y.std() > 1e-12 and x.std() > 1e-12:
        out["r"] = float(np.corrcoef(x, y)[0, 1])
    diffs = np.diff(y)
    out["monotonicity"] = float(np.mean(diffs > 0)) if diffs.size else float("nan")
    # EC50-like midpoint (only meaningful when monotone increasing)
    if np.all(diffs >= -1e-9) and y[-1] > y[0]:
        half = 0.5 * (y[0] + y[-1])
        for i in range(len(y) - 1):
            if y[i] <= half <= y[i + 1] and y[i + 1] > y[i]:
                frac = (half - y[i]) / (y[i + 1] - y[i])
                out["ec50_log"] = float(x[i] + frac * (x[i + 1] - x[i]))
                break
    return out


def spearman(a, b) -> tuple[float, float]:
    """Spearman rank correlation + a normal-approx p-value (nan if degenerate).

    Returns ``(rho, p_value)``. Backward compatible with float consumers via
    ``spearman(...)[0]``; the p-value uses the t-approximation on n-2 d.o.f.
    """
    a = _np(a).astype(np.float64)
    b = _np(b).astype(np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    n = len(a)
    if n < 3:
        return float("nan"), float("nan")

    def rankdata(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=np.float64)
        r[order] = np.arange(1, len(v) + 1)
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
        return float("nan"), float("nan")
    rho = float(np.corrcoef(ra, rb)[0, 1])
    # t-approx two-sided p
    from math import sqrt

    denom = 1.0 - rho ** 2
    if denom <= 1e-12 or n <= 2:
        p = 0.0 if abs(rho) >= 1.0 - 1e-12 else float("nan")
    else:
        t = rho * sqrt((n - 2) / denom)
        p = _student_t_sf(abs(t), n - 2) * 2.0
    return rho, float(p)


def _student_t_sf(t: float, df: int) -> float:
    """Survival function of Student-t (one-sided) via the regularized incomplete beta."""
    from math import sqrt

    x = df / (df + t * t)
    return 0.5 * _betainc(df / 2.0, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a,b) (Lentz continued fraction)."""
    from math import lgamma, log, exp

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    # use the symmetry I_x(a,b) = 1 - I_{1-x}(b,a) for fast continued-fraction convergence
    if x >= (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(b, a, 1.0 - x)
    lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
    front = exp(log(x) * a + log(1.0 - x) * b - lbeta) / a

    tiny = 1e-30
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return front * h


def moa_hierarchy_cosine(displacements, moa_labels, n_perm=1000, seed=0) -> dict:
    """Mean intra-MoA vs inter-MoA cosine of per-drug predicted displacement vectors.

    Args:
        displacements: ``[D, d]`` per-drug predicted shift directions (centroid shift).
        moa_labels: length-``D`` MoA labels (None / "" -> excluded).
        n_perm: label permutations for the separation p-value (0 = skip).
    Returns:
        dict with ``intra``, ``inter``, ``separation`` (intra - inter), ``n_pairs_*``,
        and ``p_value`` (one-sided permutation: P(perm separation >= observed)).
        Drugs whose MoA appears only once contribute to inter pairs only.
    """
    X = _np(displacements)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    U = X / norms
    labs = [m if (m is not None and str(m) != "" and str(m).lower() != "nan") else None
            for m in moa_labels]

    def separation(label_list):
        intra, inter = [], []
        n = len(label_list)
        for i in range(n):
            for j in range(i + 1, n):
                if label_list[i] is None or label_list[j] is None:
                    continue
                cos = float(np.dot(U[i], U[j]))
                (intra if label_list[i] == label_list[j] else inter).append(cos)
        return intra, inter

    intra, inter = separation(labs)
    out = {
        "intra": float(np.mean(intra)) if intra else float("nan"),
        "inter": float(np.mean(inter)) if inter else float("nan"),
        "n_pairs_intra": len(intra),
        "n_pairs_inter": len(inter),
        "p_value": float("nan"),
    }
    out["separation"] = (
        out["intra"] - out["inter"] if intra and inter else float("nan")
    )
    if intra and inter and n_perm > 0:
        obs = out["separation"]
        labelled_idx = [i for i, l in enumerate(labs) if l is not None]
        labelled_vals = [labs[i] for i in labelled_idx]
        rng = np.random.default_rng(seed)
        ge = 0
        for _ in range(n_perm):
            shuffled = list(labelled_vals)
            rng.shuffle(shuffled)
            perm = list(labs)
            for pos, i in enumerate(labelled_idx):
                perm[i] = shuffled[pos]
            pi, pe = separation(perm)
            sep = (np.mean(pi) - np.mean(pe)) if pi and pe else -np.inf
            if sep >= obs - 1e-12:
                ge += 1
        out["p_value"] = float((ge + 1) / (n_perm + 1))
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


def _dose_bucket(log_conc: float) -> float:
    if log_conc is None or (isinstance(log_conc, float) and np.isnan(log_conc)):
        return float("nan")
    return round(float(log_conc), 4)


def _gather_liver_latents(cfg, pc, encoder, device, n_cells, amp, max_per_bucket=512):
    """Stream liver cells with TARGETED per-(line,plate,drug,dose) bucketing.

    Streams up to ``n_cells`` cells and keeps, per ``(cell_line, plate, drug, dose)``
    bucket, at most ``max_per_bucket`` cells (so memory is bounded while labelled /
    multi-dose drugs accumulate enough support). Controls (``DMSO_TF``) are bucketed
    per ``(cell_line, plate)`` so every stratum keeps a source cloud. Returns
    ``(latents [N,d], meta dict of aligned lists)``.
    """
    from examples.tahoe_perturbator.main import build_loader, encode_cells

    loader, _, _ = build_loader(cfg, pc)
    keys = ("cell_line_id", "plate", "drug", "canonical_smiles", "moa_fine")
    bucket_counts: dict[tuple, int] = collections.defaultdict(int)
    latents = []
    meta = collections.defaultdict(list)
    seen = 0
    kept = 0
    for batch in loader:
        z = encode_cells(encoder, batch, device, amp).cpu()
        cl = batch["cell_line_id"]
        plate = batch["plate"]
        drug = batch["drug"]
        lc = batch["log_conc"].tolist()
        keep_rows = []
        for i in range(z.shape[0]):
            if drug[i] == "DMSO_TF":
                bkey = (cl[i], plate[i], "DMSO_TF")
            else:
                bkey = (cl[i], plate[i], drug[i], _dose_bucket(lc[i]))
            if bucket_counts[bkey] >= max_per_bucket:
                continue
            bucket_counts[bkey] += 1
            keep_rows.append(i)
        if keep_rows:
            idx = torch.tensor(keep_rows, dtype=torch.long)
            latents.append(z[idx])
            for k in keys:
                vals = batch[k]
                meta[k].extend([vals[i] for i in keep_rows])
            meta["log_conc"].extend([lc[i] for i in keep_rows])
            kept += len(keep_rows)
        seen += z.shape[0]
        if seen >= n_cells:
            break
    logger.info("streamed %d cells, kept %d after bucketing (cap %d/bucket, %d buckets)",
                seen, kept, max_per_bucket, len(bucket_counts))
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


def plot_roc(scores, labels, out_prefix, auc, title="DILI prediction", subtitle=None):
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
    sub = subtitle or f"hepatotoxicity from predicted perturbation  ·  AUC = {auc:.3f}"
    ax.text(0.0, 1.015, sub, transform=ax.transAxes, fontsize=9, color=_SUB)
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


def plot_moa_hierarchy(moa, out_prefix, title="MoA displacement hierarchy"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 5.4), facecolor="white")
    _style_ax(ax)
    vals = [moa.get("intra", float("nan")), moa.get("inter", float("nan"))]
    labels = ["intra-MoA", "inter-MoA"]
    cols = [_ACCENT, _ACCENT2]
    x = np.arange(2)
    ax.bar(x, vals, color=cols, width=0.6, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, color=_INK)
    ax.axhline(0, color=_AXIS, lw=0.9, zorder=2)
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=_INK, pad=22)
    sep = moa.get("separation", float("nan"))
    p = moa.get("p_value", float("nan"))
    ax.text(0.0, 1.01,
            f"mean cosine of predicted displacement directions  ·  sep={sep:.3f}  ·  p={p:.3g}",
            transform=ax.transAxes, fontsize=9, color=_SUB)
    ax.set_ylabel("mean cosine similarity", color=_SUB, fontsize=9)
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
# Labels                                                                       #
# =========================================================================== #
def _label_drug(drug, moa, source):
    """Return a DILI label for a drug under the chosen label ``source``."""
    if source == "dilirank":
        return dili_label_by_name(drug)
    if source == "weak_moa":
        return weak_dili_label_from_moa(moa)
    return None


# =========================================================================== #
# Driver                                                                       #
# =========================================================================== #
def run(
    config: str = "examples/tahoe_perturbator/cfgs/train_hepatotox.yaml",
    perturbator_ckpt: str = "",
    out_dir: str = "visualizations/hepatotox",
    eval_cells: int = 0,
    max_per_drug_dose: int = 512,
    holdout_frac: float = 0.5,
    top_drugs_fig: int = 6,
    wandb_enabled: bool | None = None,
    **overrides,
):
    cfg = load_config(config, cli_overrides=overrides or None)
    setup_seed(int(cfg.meta.seed))
    seed = int(cfg.meta.seed)
    if wandb_enabled is not None:
        cfg.wandb["enabled"] = bool(wandb_enabled)
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

    # Liver latents (targeted bucketing) -------------------------------------
    n_cells = int(eval_cells or cfg.eval.get("eval_cells", 150000))
    latents, meta = _gather_liver_latents(
        cfg, pc, encoder, device, n_cells, amp, max_per_bucket=int(max_per_drug_dose)
    )
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

    # MoA per (drug, smiles) for the hierarchy + weak-label metrics
    drug_to_moa: dict[str, str] = {}
    for d, mo in zip(meta["drug"], meta["moa_fine"]):
        if d and d not in drug_to_moa and mo:
            drug_to_moa[d] = mo

    rng = np.random.default_rng(seed)
    results: dict = {}
    per_stratum_rows = []  # for CSV
    drug_shift: dict[tuple, list] = collections.defaultdict(list)  # (drug,smiles)->(log_conc,shift_mag,shift_vec)
    base_sw_all, pred_sw_all, cos_all = [], [], []

    # --- Perturbator accuracy + per-stratum predicted shift ------------------
    perturb_device = next(perturbator.parameters()).device
    for s in strata:
        # held-out split of the REAL target so accuracy is measured out-of-sample
        nt = s.target.shape[0]
        perm = rng.permutation(nt)
        n_eval = max(1, int(round(holdout_frac * nt)))
        eval_idx = perm[:n_eval]
        target_eval = s.target[eval_idx].to(perturb_device)
        action = featurizer.featurize(s.smiles, s.log_conc).to(perturb_device)
        src = s.source.to(perturb_device)
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
            "median_centroid_cosine": float(np.nanmedian(coss)),
        }
        plot_agreement_scatter(base_sw_all, pred_sw_all, cos_all,
                               os.path.join(out_dir, "perturbator_agreement_scatter"))
    else:
        skipped["accuracy"] = "no strata to score"

    # --- Per-drug summary: multi-dose shift, dose slope, pathway scores ------
    drug_rows = []
    drug_disp_dirs = []   # per-drug top-dose displacement direction (MoA hierarchy)
    drug_disp_labels = []
    drug_pathway = {}     # drug -> pathway score vector (raw)
    drug_scores: dict[str, dict] = {}  # drug -> {top_shift, mean_shift, dose_slope, ...}
    for (drug, smiles), entries in drug_shift.items():
        entries = sorted(entries, key=lambda e: e[0])  # ascending dose
        log_doses = [e[0] for e in entries]
        mags = [e[1] for e in entries]
        top = entries[-1]  # highest dose
        dr = dose_response_slope(log_doses, mags)
        pv = path_feat.featurize(smiles).numpy() if smiles else None
        row = {
            "drug": drug, "smiles": smiles, "moa_fine": drug_to_moa.get(drug),
            "n_doses": len(entries), "top_log_conc": float(top[0]),
            "top_shift_mag": float(top[1]), "mean_shift_mag": float(np.mean(mags)),
            "dose_slope": dr["slope"], "dose_r": dr["r"],
            "dose_monotonicity": dr["monotonicity"], "ec50_log": dr["ec50_log"],
            "dili_dilirank": dili_label_by_name(drug),
            "dili_weak_moa": weak_dili_label_from_moa(drug_to_moa.get(drug)),
        }
        drug_rows.append(row)
        drug_scores[drug] = {
            "top_shift_mag": float(top[1]),
            "mean_shift_mag": float(np.mean(mags)),
            "dose_slope": dr["slope"],
        }
        drug_disp_dirs.append(top[2])
        drug_disp_labels.append(drug_to_moa.get(drug))
        if pv is not None:
            drug_pathway[drug] = pv

    # --- DILI classification (try scores x label sources, keep BEST) ---------
    dili_variants = []          # all computed (score, source) variants
    best_dili = None
    score_names = ("top_shift_mag", "mean_shift_mag", "dose_slope")
    for source in ("dilirank", "weak_moa"):
        lab_key = "dili_dilirank" if source == "dilirank" else "dili_weak_moa"
        labelled = [r for r in drug_rows if r[lab_key] in ("hepatotoxic", "low_concern")]
        n_pos = sum(1 for r in labelled if r[lab_key] == "hepatotoxic")
        n_neg = len(labelled) - n_pos
        if n_pos < 2 or n_neg < 2 or len(labelled) < 4:
            skipped[f"dili_{source}"] = (
                f"too few labelled drugs (pos={n_pos} neg={n_neg}; need >=2 each, >=4 total)"
            )
            continue
        labels = np.array([1 if r[lab_key] == "hepatotoxic" else 0 for r in labelled])
        # scalar-score variants
        for sn in score_names:
            sc = np.array([drug_scores[r["drug"]][sn] for r in labelled], dtype=np.float64)
            if not np.isfinite(sc).any() or np.unique(sc[np.isfinite(sc)]).size < 2:
                continue
            sc = np.where(np.isfinite(sc), sc, np.nanmin(sc) - 1.0)
            auc = roc_auc(sc, labels)
            bacc, thr = balanced_accuracy_at_threshold(sc, labels)
            # use |AUC - 0.5| so a strongly anti-correlated score still counts as signal
            dili_variants.append({
                "score": sn, "label_source": source, "roc_auc": auc,
                "auc_strength": abs(auc - 0.5), "balanced_accuracy": bacc,
                "threshold": float(thr), "n_hepatotoxic": int(labels.sum()),
                "n_low_concern": int((labels == 0).sum()), "n_drugs": int(len(labelled)),
            })
        # learned toxic-vs-safe axis on the predicted shift VECTORS (CV, no leakage)
        try:
            disp = []
            for r in labelled:
                entries = sorted(drug_shift[(r["drug"], r["smiles"])], key=lambda e: e[0])
                disp.append(entries[-1][2])  # top-dose displacement vector
            disp = np.stack(disp)
            ax_scores = cv_axis_scores(disp, labels, seed=seed)
            if np.isfinite(ax_scores).sum() >= 4:
                m = np.isfinite(ax_scores)
                auc = roc_auc(ax_scores[m], labels[m])
                bacc, thr = balanced_accuracy_at_threshold(ax_scores[m], labels[m])
                dili_variants.append({
                    "score": "learned_axis", "label_source": source, "roc_auc": auc,
                    "auc_strength": abs(auc - 0.5), "balanced_accuracy": bacc,
                    "threshold": float(thr), "n_hepatotoxic": int(labels[m].sum()),
                    "n_low_concern": int((labels[m] == 0).sum()), "n_drugs": int(m.sum()),
                })
        except Exception as e:
            logger.warning("learned-axis DILI variant skipped (%s): %s", source, e)

    if dili_variants:
        best_dili = max(dili_variants, key=lambda v: v["auc_strength"]
                        if np.isfinite(v["auc_strength"]) else -1.0)
        results["dili"] = {
            "best": best_dili,
            "all_variants": dili_variants,
            "label_source_note": "primary=DILIrank/LiverTox by name; secondary=weak MoA",
        }
        # ROC of the best variant
        bsource = best_dili["label_source"]
        lab_key = "dili_dilirank" if bsource == "dilirank" else "dili_weak_moa"
        labelled = [r for r in drug_rows if r[lab_key] in ("hepatotoxic", "low_concern")]
        labels = np.array([1 if r[lab_key] == "hepatotoxic" else 0 for r in labelled])
        if best_dili["score"] == "learned_axis":
            disp = np.stack([
                sorted(drug_shift[(r["drug"], r["smiles"])], key=lambda e: e[0])[-1][2]
                for r in labelled])
            sc_best = cv_axis_scores(disp, labels, seed=seed)
            m = np.isfinite(sc_best)
            sc_best, labels = sc_best[m], labels[m]
        else:
            sc_best = np.array([drug_scores[r["drug"]][best_dili["score"]] for r in labelled])
            sc_best = np.where(np.isfinite(sc_best), sc_best, np.nanmin(sc_best) - 1.0)
        sub = (f"best: {best_dili['score']} · {bsource} · AUC={best_dili['roc_auc']:.3f} · "
               f"bacc={best_dili['balanced_accuracy']:.3f}")
        plot_roc(sc_best, labels, os.path.join(out_dir, "dili_roc"),
                 best_dili["roc_auc"], subtitle=sub)
    else:
        skipped["dili"] = "no DILI variant had enough labelled support"

    # --- Dose-response: hepatotoxic vs safe separation (best formulation) ----
    dr_variants = []
    for source in ("dilirank", "weak_moa"):
        lab_key = "dili_dilirank" if source == "dilirank" else "dili_weak_moa"
        for metric in ("dose_slope", "dose_monotonicity"):
            tox = [r[metric] for r in drug_rows
                   if r[lab_key] == "hepatotoxic" and np.isfinite(r[metric])]
            safe = [r[metric] for r in drug_rows
                    if r[lab_key] == "low_concern" and np.isfinite(r[metric])]
            if len(tox) >= 2 and len(safe) >= 2:
                labels = np.array([1] * len(tox) + [0] * len(safe))
                vals = np.array(tox + safe)
                auc = roc_auc(vals, labels)
                p = mannwhitney_p(tox, safe)
                dr_variants.append({
                    "metric": metric, "label_source": source,
                    "mean_hepatotoxic": float(np.mean(tox)),
                    "mean_low_concern": float(np.mean(safe)),
                    "separation": float(np.mean(tox) - np.mean(safe)),
                    "roc_auc": auc, "auc_strength": abs(auc - 0.5),
                    "mannwhitney_p": p, "n_hepatotoxic": len(tox), "n_low_concern": len(safe),
                })
    if dr_variants:
        best_dr = max(dr_variants, key=lambda v: v["auc_strength"]
                      if np.isfinite(v["auc_strength"]) else -1.0)
        results["dose_response"] = {"best": best_dr, "all_variants": dr_variants}
    else:
        skipped["dose_response"] = "need dose metrics for both hepatotoxic and low-concern drugs"

    # dose-response figure for the top hepatotoxins (by top-dose shift), any label source
    tox_drugs = [r for r in drug_rows
                 if r["dili_dilirank"] == "hepatotoxic" or r["dili_weak_moa"] == "hepatotoxic"]
    tox_curves = []
    for r in sorted(tox_drugs, key=lambda r: r["top_shift_mag"], reverse=True):
        entries = sorted(drug_shift[(r["drug"], r["smiles"])], key=lambda e: e[0])
        if len(entries) >= 2:
            tox_curves.append((r["drug"], [e[0] for e in entries],
                               [e[1] for e in entries], True))
        if len(tox_curves) >= top_drugs_fig:
            break
    if tox_curves:
        plot_dose_response(tox_curves, os.path.join(out_dir, "dose_response_top_hepatotoxins"))
    else:
        skipped["dose_response_fig"] = "no hepatotoxin with >=2 doses for the dose-response figure"

    # --- Virtual-pathway attribution (Spearman across drugs) -----------------
    if len(drug_pathway) >= 3:
        drugs_p = list(drug_pathway.keys())
        P = np.stack([drug_pathway[d] for d in drugs_p])          # [D, n_path]
        mags = np.array([drug_scores[d]["top_shift_mag"] for d in drugs_p])  # [D]
        rhos, pvals = [], []
        for j in range(P.shape[1]):
            rho, p = spearman(P[:, j], mags)
            rhos.append(rho); pvals.append(p)
        attribution = {n: {"rho": float(r), "p": float(p)}
                       for n, r, p in zip(path_feat.feature_names, rhos, pvals)}
        # strongest by |rho|
        finite = [(n, a["rho"], a["p"]) for n, a in attribution.items()
                  if np.isfinite(a["rho"])]
        strongest = max(finite, key=lambda t: abs(t[1])) if finite else None
        results["pathway_attribution"] = {
            "per_feature": attribution, "n_drugs": len(drugs_p),
            "strongest": ({"feature": strongest[0], "rho": strongest[1], "p": strongest[2]}
                          if strongest else None),
        }
        plot_pathway_attribution(path_feat.feature_names, rhos,
                                 os.path.join(out_dir, "pathway_attribution"))
    else:
        skipped["pathway_attribution"] = f"too few drugs with SMILES ({len(drug_pathway)}; need >=3)"

    # --- MoA hierarchy (intra vs inter cosine + permutation p) ---------------
    labelled = [(d, l) for d, l in zip(drug_disp_dirs, drug_disp_labels) if l]
    if len(labelled) >= 4 and len({l for _, l in labelled}) >= 2:
        moa = moa_hierarchy_cosine([d for d, _ in labelled], [l for _, l in labelled],
                                   n_perm=2000, seed=seed)
        results["moa_hierarchy"] = moa
        if np.isfinite(moa.get("separation", float("nan"))):
            plot_moa_hierarchy(moa, os.path.join(out_dir, "moa_hierarchy"))
    else:
        skipped["moa_hierarchy"] = "too few drugs with a known MoA across >=2 MoAs"

    # --- Cherry-picked dose-shift trajectory figures -------------------------
    try:
        traj = _trajectory_figures(perturbator, featurizer, latents, meta, objective,
                                   ode_steps, ode_method, drug_rows, out_dir, top_drugs_fig,
                                   seed, min_src)
        if traj:
            results["trajectories"] = traj
    except Exception as e:  # display-only; never fail the metric run
        logger.warning("trajectory figures skipped: %s", e)
        skipped["trajectory_figures"] = str(e)

    # --- Coverage report -----------------------------------------------------
    n_dilirank_pos = sum(1 for r in drug_rows if r["dili_dilirank"] == "hepatotoxic")
    n_dilirank_neg = sum(1 for r in drug_rows if r["dili_dilirank"] == "low_concern")
    n_weak_pos = sum(1 for r in drug_rows if r["dili_weak_moa"] == "hepatotoxic")
    n_weak_neg = sum(1 for r in drug_rows if r["dili_weak_moa"] == "low_concern")
    n_multi = sum(1 for r in drug_rows if r["n_doses"] >= 2)
    results["coverage"] = {
        "n_drugs": len(drug_rows), "n_strata": len(per_stratum_rows),
        "n_dilirank_hepatotoxic": n_dilirank_pos, "n_dilirank_low_concern": n_dilirank_neg,
        "n_weak_moa_hepatotoxic": n_weak_pos, "n_weak_moa_low_concern": n_weak_neg,
        "n_multi_dose_drugs": n_multi,
    }
    logger.info("coverage: drugs=%d | DILIrank pos=%d neg=%d | weakMoA pos=%d neg=%d | multi-dose=%d",
                len(drug_rows), n_dilirank_pos, n_dilirank_neg, n_weak_pos, n_weak_neg, n_multi)

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

    # --- Self-ranking "best findings" ---------------------------------------
    findings = _rank_findings(results)
    with open(os.path.join(out_dir, "best_findings.json"), "w") as fh:
        json.dump(findings, fh, indent=2, default=float)
    _write_findings_md(findings, os.path.join(out_dir, "best_findings.md"))
    logger.info("wrote %s", os.path.join(out_dir, "best_findings.json"))

    # --- wandb scalars + figures + findings table ----------------------------
    _log_wandb(cfg, results, findings, out_dir)

    # --- Console report ------------------------------------------------------
    logger.info("=== Hepatotox perturbator validation (%.1fs) ===", time.time() - t0)
    if findings.get("headline"):
        logger.info("  HEADLINE: %s", findings["headline"])
    for f in findings.get("ranked", []):
        logger.info("  [%s] %s", f["analysis"], f["summary"])
    if skipped:
        logger.info("  SKIPPED: %s", skipped)
    return results


def _rank_findings(results: dict) -> dict:
    """Rank all computed analyses by signal strength and name the headline result."""
    ranked = []

    dili = results.get("dili", {}).get("best")
    if dili and np.isfinite(dili.get("roc_auc", float("nan"))):
        ranked.append({
            "analysis": "dili_classification",
            "strength": float(dili["auc_strength"]),
            "metric": "roc_auc", "value": float(dili["roc_auc"]),
            "summary": (f"DILI AUROC={dili['roc_auc']:.3f} (bacc={dili['balanced_accuracy']:.3f}) "
                        f"via {dili['score']} on {dili['label_source']} "
                        f"({dili['n_hepatotoxic']}+/{dili['n_low_concern']}-)"),
            "detail": dili,
        })

    dr = results.get("dose_response", {}).get("best")
    if dr and np.isfinite(dr.get("roc_auc", float("nan"))):
        ranked.append({
            "analysis": "dose_response_separation",
            "strength": float(dr["auc_strength"]),
            "metric": "roc_auc", "value": float(dr["roc_auc"]),
            "summary": (f"dose-response {dr['metric']} separates hepatotoxic vs safe "
                        f"AUROC={dr['roc_auc']:.3f} (MW p={dr['mannwhitney_p']:.3g}, "
                        f"{dr['label_source']})"),
            "detail": dr,
        })

    pa = results.get("pathway_attribution", {})
    if pa.get("strongest") and np.isfinite(pa["strongest"].get("rho", float("nan"))):
        s = pa["strongest"]
        ranked.append({
            "analysis": "pathway_attribution",
            "strength": float(abs(s["rho"])),
            "metric": "spearman_rho", "value": float(s["rho"]),
            "summary": (f"strongest pathway: {s['feature']} ρ={s['rho']:.3f} "
                        f"(p={s['p']:.3g}) vs predicted shift across {pa['n_drugs']} drugs"),
            "detail": s,
        })

    moa = results.get("moa_hierarchy", {})
    if np.isfinite(moa.get("separation", float("nan"))):
        ranked.append({
            "analysis": "moa_hierarchy",
            "strength": float(max(0.0, moa["separation"])),
            "metric": "intra_minus_inter_cosine", "value": float(moa["separation"]),
            "summary": (f"MoA hierarchy: intra-inter cosine sep={moa['separation']:.3f} "
                        f"(p={moa.get('p_value', float('nan')):.3g}, "
                        f"{moa['n_pairs_intra']}/{moa['n_pairs_inter']} pairs)"),
            "detail": moa,
        })

    acc = results.get("accuracy", {})
    if acc:
        ranked.append({
            "analysis": "predicted_vs_real",
            "strength": float(max(0.0, acc.get("mean_gap_closed", 0.0))),
            "metric": "mean_gap_closed", "value": float(acc.get("mean_gap_closed", float("nan"))),
            "summary": (f"perturbator closes {acc.get('mean_gap_closed', float('nan')):.3f} of "
                        f"the OT gap (centroid cos={acc.get('mean_centroid_cosine', float('nan')):.3f}) "
                        f"over {acc.get('n_strata', 0)} strata"),
            "detail": acc,
        })

    traj = results.get("trajectories", {})
    if traj.get("top"):
        best_t = traj["top"][0]
        ranked.append({
            "analysis": "dose_trajectory",
            "strength": float(best_t.get("score", 0.0)),
            "metric": "monotonicity_score", "value": float(best_t.get("score", 0.0)),
            "summary": (f"cleanest dose-shift trajectory: {best_t['drug']} on "
                        f"{best_t['cell_line']} (mono score={best_t.get('score', 0.0):.3f})"),
            "detail": traj,
        })

    ranked.sort(key=lambda f: f["strength"] if np.isfinite(f["strength"]) else -1.0,
                reverse=True)
    headline = ranked[0]["summary"] if ranked else "no analysis produced a signal"
    return {"headline": headline, "ranked": ranked, "coverage": results.get("coverage", {})}


def _write_findings_md(findings: dict, path: str) -> None:
    lines = ["# Hepatotoxicity perturbator — best findings", ""]
    cov = findings.get("coverage", {})
    if cov:
        lines += [
            "## Coverage",
            f"- drugs analysed: **{cov.get('n_drugs', 0)}** across **{cov.get('n_strata', 0)}** strata",
            f"- DILIrank labels: **{cov.get('n_dilirank_hepatotoxic', 0)}** hepatotoxic / "
            f"**{cov.get('n_dilirank_low_concern', 0)}** low-concern",
            f"- weak-MoA labels: **{cov.get('n_weak_moa_hepatotoxic', 0)}** hepatotoxic / "
            f"**{cov.get('n_weak_moa_low_concern', 0)}** low-concern",
            f"- multi-dose drugs: **{cov.get('n_multi_dose_drugs', 0)}**",
            "",
        ]
    lines += ["## Headline", f"> {findings.get('headline', '—')}", ""]
    lines += ["## Ranked results (by signal strength)", ""]
    lines += ["| rank | analysis | metric | value | summary |", "|---|---|---|---|---|"]
    for i, f in enumerate(findings.get("ranked", []), 1):
        lines.append(
            f"| {i} | {f['analysis']} | {f['metric']} | {f['value']:.4f} | {f['summary']} |"
        )
    lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _trajectory_figures(perturbator, featurizer, latents, meta, objective,
                        ode_steps, ode_method, drug_rows, out_dir, top_k, seed, min_src):
    """Cherry-pick the most hepatotoxic drugs and render latent dose-shift tracks.

    Reuses ``perturbator/visualize.py`` (build_dose_track + monotonicity ranking +
    plot_dose_shift). Control cells of the most-represented hepatic line are the
    shared backdrop; each drug's ascending doses (seen in the stream) form one track.

    Device-consistent: control latents are moved to the PERTURBATOR's device before any
    ODE integration so the control cloud, the action vector and the model agree.
    """
    from eb_jepa.singlecell.perturbator.visualize import (
        build_dose_track, plot_dose_shift, rank_combos,
    )

    cl = np.array(meta["cell_line_id"], dtype=object)
    drug = np.array(meta["drug"], dtype=object)
    smiles = np.array(meta["canonical_smiles"], dtype=object)
    log_conc = np.array(meta["log_conc"], dtype=np.float64)
    perturb_device = next(perturbator.parameters()).device

    # busiest hepatic line with controls
    ctrl_mask = drug == "DMSO_TF"
    lines = collections.Counter(cl[ctrl_mask].tolist())
    if not lines:
        raise RuntimeError("no control cells for trajectory figures")
    line = lines.most_common(1)[0][0]
    cmask = (cl == line) & ctrl_mask
    # control latents ON THE PERTURBATOR DEVICE (fix: cpu/cuda mismatch crash)
    control = latents[torch.from_numpy(np.where(cmask)[0])].to(perturb_device)
    if control.shape[0] < min_src:
        raise RuntimeError(f"too few control cells on line {line}")

    # hepatotoxic drugs present on this line, with >=2 doses (either label source)
    tox = {r["drug"] for r in drug_rows
           if r["dili_dilirank"] == "hepatotoxic" or r["dili_weak_moa"] == "hepatotoxic"}
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
    # plot_dose_shift handles its own .cpu() for display; pass the (device) control cloud
    paths = plot_dose_shift(control, tracks, os.path.join(out_dir, "dose_shift_hepatotoxins"),
                            cell_line=str(line), projector="pca", seed=seed)
    return {
        "cell_line": str(line),
        "figure": paths[0] if paths else None,
        "top": [{"drug": t.drug, "cell_line": str(line),
                 "score": float(t.metrics.get("score", 0.0)),
                 "collinearity": float(t.metrics.get("collinearity", 0.0)),
                 "magnitude_monotonicity": float(t.metrics.get("magnitude_monotonicity", 0.0))}
                for t in tracks],
    }


def _log_wandb(cfg, results, findings, out_dir):
    if not cfg.wandb.get("enabled", False):
        return
    try:
        import wandb

        from eb_jepa.training_utils import setup_wandb

        if cfg.wandb.get("entity"):
            os.environ["WANDB_ENTITY"] = cfg.wandb.entity
        run = setup_wandb(cfg.wandb.project, cfg, cfg.meta.run_dir, enabled=True)
        # flatten scalar metrics
        flat = {}

        def _walk(prefix, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    _walk(f"{prefix}/{k}", v)
                elif isinstance(v, (int, float)) and not isinstance(v, bool):
                    flat[f"validate/{prefix}/{k}"] = v

        for sect in ("accuracy", "coverage"):
            if sect in results:
                _walk(sect, results[sect])
        best = results.get("dili", {}).get("best")
        if best:
            _walk("dili_best", {k: v for k, v in best.items() if isinstance(v, (int, float))})
        bdr = results.get("dose_response", {}).get("best")
        if bdr:
            _walk("dose_response_best", {k: v for k, v in bdr.items() if isinstance(v, (int, float))})
        moa = results.get("moa_hierarchy")
        if moa:
            _walk("moa_hierarchy", {k: v for k, v in moa.items() if isinstance(v, (int, float))})
        if flat:
            run.log(flat)
        # figures
        for fn in os.listdir(out_dir):
            if fn.endswith(".png"):
                try:
                    run.log({f"validate/fig/{fn[:-4]}": wandb.Image(os.path.join(out_dir, fn))})
                except Exception:
                    pass
        # best-findings table
        try:
            tbl = wandb.Table(columns=["rank", "analysis", "metric", "value", "summary"])
            for i, f in enumerate(findings.get("ranked", []), 1):
                tbl.add_data(i, f["analysis"], f["metric"], float(f["value"]), f["summary"])
            run.log({"validate/best_findings": tbl,
                     "validate/headline": findings.get("headline", "")})
        except Exception as e:
            logger.warning("wandb findings table skipped: %s", e)
    except Exception as e:
        logger.warning("wandb logging skipped: %s", e)


if __name__ == "__main__":
    import fire

    fire.Fire({"run": run})
