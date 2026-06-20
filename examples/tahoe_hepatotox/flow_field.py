"""The latent flow of a drug on liver cells — flow-matching ODE transport figure.

Loads the frozen LIVER-finetuned encoder + the trained hepatotox perturbator + the
hepatotox action featurizer, picks a ``(cell_line, drug, dose)`` (auto-picks among a
candidate list of flagged hepatotoxins when not given), streams the liver cells of
that stratum, encodes them, and integrates the learned flow-matching ODE from the
DMSO control latents to the predicted perturbed state — recording the FULL integrated
path. The control cloud, the predicted cloud, the recorded flow paths, and the REAL
treated cloud (held-out validation target) are projected through ONE shared PCA(2)
fitted on (control ∪ real-treated), so the straight-line flow paths project honestly.

The figure ("Latent flow of {drug} on {cell_line} liver cells", house style) layers
back-to-front: control cloud (t=0), real treated cloud (validation target), per-cell
flow streamlines coloured by integration time, predicted perturbed cloud (t=1),
summary centroid arrows (control→predicted bold; predicted→real dashed residual), and
rigor annotations (sliced-Wasserstein, gap-closed, directional cosine, n_control /
n_treated).

Run on Dalia (1 GPU) on the CURRENT perturbator::

    /lustre/work/vivatech-unaite/ljung/venv-arm/bin/python -m \
        examples.tahoe_hepatotox.flow_field run \
        --config examples/tahoe_perturbator/cfgs/train_hepatotox.yaml \
        --perturbator_ckpt /lustre/work/vivatech-unaite/ljung/runs/perturbator/hepatotox_liver/perturbator_final.pt

It works identically on the longer run (``hepatotox_liver_long/perturbator_final.pt``).
"""
from __future__ import annotations

import collections
import json
import os
import time

import numpy as np
import torch

from eb_jepa.logging import get_logger
from eb_jepa.singlecell.perturbator.flow import ode_sample
from eb_jepa.singlecell.perturbator.losses import sliced_wasserstein
from eb_jepa.training_utils import load_config, setup_seed

logger = get_logger(__name__)

# House palette (single source of truth: eb_jepa.singlecell.visualize — do not edit it).
from eb_jepa.singlecell.visualize import _INK, _SUB

_ACCENT = "#2a6f97"
_ACCENT2 = "#3f8bb5"
_GRID = "#e9edf2"
_AXIS = "#c7cfdb"
_ROSE = "#b5566e"  # real-treated validation cloud

# Flagged hepatotoxins to auto-pick from (DILIrank / LiverTox concern in Tahoe).
CANDIDATE_HEPATOTOXINS = [
    "Dabrafenib", "Afatinib", "Encorafenib", "Berbamine", "Tucidinostat", "Selinexor",
]


# =========================================================================== #
# Small helpers                                                                #
# =========================================================================== #
def _np(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float64)


def _dose_bucket(log_conc) -> float:
    if log_conc is None or (isinstance(log_conc, float) and np.isnan(log_conc)):
        return float("nan")
    return round(float(log_conc), 4)


# =========================================================================== #
# Streaming + selection                                                        #
# =========================================================================== #
def _gather_flow_cells(
    cfg, pc, encoder, device, n_cells, amp,
    candidates, drug=None, cell_line=None, max_per_bucket=4096,
):
    """Stream liver cells and bucket per ``(cell_line, plate, drug, dose)``.

    Keeps every DMSO_TF control bucketed per ``(cell_line, plate)`` and every treated
    bucket of a *candidate* drug (or the requested ``drug``) up to ``max_per_bucket``.
    Returns ``(latents [N, d], meta dict of aligned lists)``.
    """
    from examples.tahoe_perturbator.main import build_loader, encode_cells

    loader, _, _ = build_loader(cfg, pc)
    keys = ("cell_line_id", "plate", "drug", "canonical_smiles")
    wanted = {d.lower() for d in candidates}
    if drug:
        wanted.add(drug.lower())
    bucket_counts: dict[tuple, int] = collections.defaultdict(int)
    latents, meta, seen, kept = [], collections.defaultdict(list), 0, 0
    for batch in loader:
        z = encode_cells(encoder, batch, device, amp).cpu()
        cl, plate, dr = batch["cell_line_id"], batch["plate"], batch["drug"]
        lc = batch["log_conc"].tolist()
        keep_rows = []
        for i in range(z.shape[0]):
            is_ctrl = dr[i] == "DMSO_TF"
            if not is_ctrl:
                if dr[i].lower() not in wanted:
                    continue
                if cell_line and cl[i] != cell_line:
                    continue
            bkey = (cl[i], plate[i], "DMSO_TF") if is_ctrl else (
                cl[i], plate[i], dr[i], _dose_bucket(lc[i]))
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
    logger.info("streamed %d cells, kept %d after bucketing (%d buckets)",
                seen, kept, len(bucket_counts))
    if not latents:
        return torch.empty(0), meta
    return torch.cat(latents, 0), meta


def _select_stratum(meta, candidates, drug=None, cell_line=None, dose=None, min_control=8):
    """Choose the ``(cell_line, plate, drug, dose)`` to visualize.

    If ``drug``/``cell_line`` are given they constrain the search; ``dose`` (a
    ``log_conc`` bucket) is honoured if present else the drug's TOP dose is used.
    AUTO-PICK (no drug given): among ``candidates``, pick the ``(cell_line, drug)`` with
    the most treated cells on a liver line that ALSO has >= ``min_control`` DMSO
    controls on a shared plate, then take that drug's TOP dose.
    Returns a dict describing the choice (with the row indices of source / target).
    """
    cl = meta["cell_line_id"]
    plate = meta["plate"]
    dr = meta["drug"]
    smiles = meta["canonical_smiles"]
    lc = meta["log_conc"]
    n = len(cl)

    # control rows per (cell_line, plate)
    ctrl_by_lp: dict[tuple, list] = collections.defaultdict(list)
    # treated rows per (cell_line, plate, drug, dose)
    treat: dict[tuple, list] = collections.defaultdict(list)
    for i in range(n):
        if dr[i] == "DMSO_TF":
            ctrl_by_lp[(cl[i], plate[i])].append(i)
        else:
            treat[(cl[i], plate[i], dr[i], _dose_bucket(lc[i]))].append(i)

    wanted = {d.lower() for d in candidates}
    if drug:
        wanted = {drug.lower()}

    # group treated by (cell_line, drug) to total support; require a control plate match
    best = None
    by_cd: dict[tuple, list] = collections.defaultdict(list)  # (cl, drug) -> [(plate,dose,rows)]
    for (lline, pl, drg, ds), rows in treat.items():
        if drg.lower() not in wanted:
            continue
        if cell_line and lline != cell_line:
            continue
        if len(ctrl_by_lp.get((lline, pl), [])) < min_control:
            continue
        by_cd[(lline, drg)].append((pl, ds, rows))

    if not by_cd:
        raise RuntimeError(
            "No (cell_line, drug) on a liver line with both treated cells and "
            f">= {min_control} DMSO controls on a shared plate "
            f"(candidates={sorted(wanted)}, cell_line={cell_line!r})."
        )

    # rank (cell_line, drug) by total treated support
    ranked = sorted(by_cd.items(), key=lambda kv: -sum(len(r) for _, _, r in kv[1]))
    (sel_line, sel_drug), entries = ranked[0]

    # choose the dose: requested bucket if present, else the TOP (highest log_conc)
    def pick_dose(entries):
        if dose is not None:
            db = _dose_bucket(dose)
            hits = [e for e in entries if e[1] == db]
            if hits:
                return db, hits
            logger.warning("requested dose %s not found; using top dose", dose)
        top = max((e for e in entries), key=lambda e: (e[1] if np.isfinite(e[1]) else -np.inf))
        db = top[1]
        return db, [e for e in entries if e[1] == db]

    sel_dose, dose_entries = pick_dose(entries)
    # pick the single plate with the most treated cells at that dose (shared-plate OT)
    dose_entries = sorted(dose_entries, key=lambda e: -len(e[2]))
    sel_plate, _, tgt_rows = dose_entries[0]
    src_rows = ctrl_by_lp[(sel_line, sel_plate)]
    rep = tgt_rows[0]
    return {
        "cell_line": sel_line, "plate": sel_plate, "drug": sel_drug,
        "dose": float(sel_dose), "smiles": smiles[rep],
        "src_rows": src_rows, "tgt_rows": tgt_rows,
        "n_control": len(src_rows), "n_treated": len(tgt_rows),
    }


# =========================================================================== #
# Figure                                                                        #
# =========================================================================== #
def _style_ax(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_AXIS)
        ax.spines[s].set_linewidth(0.9)
    ax.tick_params(colors=_SUB, labelsize=8)
    ax.set_axisbelow(True)


def _save(fig, out_prefix, formats=("png", "pdf", "svg")):
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    paths = []
    for fmt in formats:
        p = f"{out_prefix}.{fmt}"
        fig.savefig(p, dpi=220, bbox_inches="tight", facecolor="white")
        paths.append(p)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return paths


def plot_flow_field(
    ctrl2d, treat2d, pred2d, paths2d, choice, metrics, out_prefix,
    n_streamlines=80, seed=0,
):
    """Compose the latent-flow figure (house style). All inputs are 2-D (post-PCA).

    Args:
        ctrl2d:  ``[Nc, 2]`` projected control cloud (t=0).
        treat2d: ``[Nt, 2]`` projected real treated cloud (validation target).
        pred2d:  ``[Nc, 2]`` projected predicted perturbed cloud (t=1).
        paths2d: ``[T+1, Nc, 2]`` projected flow paths.
        choice:  dict from :func:`_select_stratum` (drug / cell_line / dose / counts).
        metrics: dict of rigor scalars (sliced_w / gap_closed / cosine).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(8.6, 7.8), facecolor="white")
    _style_ax(ax)

    # 1. control cloud (t=0)
    ax.scatter(ctrl2d[:, 0], ctrl2d[:, 1], s=9, color=_ACCENT, alpha=0.18,
               linewidths=0, zorder=2, rasterized=True)
    # 2. real treated cloud (validation target)
    ax.scatter(treat2d[:, 0], treat2d[:, 1], s=9, color=_ROSE, alpha=0.18,
               linewidths=0, zorder=2, rasterized=True)

    # 3. flow streamlines (subset), coloured by integration time t (white->accent->ink)
    flow_cmap = LinearSegmentedColormap.from_list(
        "flow", ["#ffffff", _ACCENT2, _ACCENT, _INK])
    Tp1 = paths2d.shape[0]
    nc = paths2d.shape[1]
    n_sl = int(min(n_streamlines, nc))
    sl_idx = rng.choice(nc, size=n_sl, replace=False) if nc > n_sl else np.arange(nc)
    tvals = np.linspace(0.0, 1.0, Tp1)
    segs, seg_t = [], []
    for j in sl_idx:
        pts = paths2d[:, j, :]  # [T+1, 2]
        for k in range(Tp1 - 1):
            segs.append([pts[k], pts[k + 1]])
            seg_t.append(0.5 * (tvals[k] + tvals[k + 1]))
    lc = LineCollection(segs, cmap=flow_cmap, linewidths=0.7, alpha=0.5, zorder=3)
    lc.set_array(np.asarray(seg_t))
    lc.set_clim(0.0, 1.0)
    ax.add_collection(lc)
    # a few arrowheads (direction) on a sub-subset of streamlines
    for j in sl_idx[:: max(1, n_sl // 12)]:
        pts = paths2d[:, j, :]
        a, b = pts[-2], pts[-1]
        if np.linalg.norm(b - a) > 1e-9:
            ax.annotate("", xy=b, xytext=a, zorder=4,
                        arrowprops=dict(arrowstyle="-|>", color=_ACCENT,
                                        lw=0.8, alpha=0.7, mutation_scale=8))

    # 4. predicted perturbed cloud (t=1)
    ax.scatter(pred2d[:, 0], pred2d[:, 1], s=16, color=_ACCENT,
               edgecolors="white", linewidths=0.4, alpha=0.85, zorder=5,
               rasterized=True)

    # 5. summary centroid arrows + stars
    c_ctrl = ctrl2d.mean(0)
    c_pred = pred2d.mean(0)
    c_real = treat2d.mean(0)
    ax.annotate("", xy=c_pred, xytext=c_ctrl, zorder=7,
                arrowprops=dict(arrowstyle="-|>", color=_ACCENT, lw=2.6,
                                mutation_scale=20))
    ax.annotate("", xy=c_real, xytext=c_pred, zorder=7,
                arrowprops=dict(arrowstyle="-|>", color=_SUB, lw=1.6, ls="--",
                                mutation_scale=14))
    for c, col in ((c_ctrl, _ACCENT), (c_pred, _INK), (c_real, _ROSE)):
        ax.scatter([c[0]], [c[1]], marker="*", s=240, color=col,
                   edgecolors="white", linewidths=1.0, zorder=8)

    # titles
    drug, line, dose = choice["drug"], choice["cell_line"], choice["dose"]
    ax.set_title(f"Latent flow of {drug} on {line} liver cells",
                 loc="left", fontsize=16, fontweight="bold", color=_INK, pad=26)
    dose_str = f"log10[M] = {dose:.2f}" if np.isfinite(dose) else "top dose"
    ax.text(0.0, 1.022,
            "flow-matching ODE transport (control → perturbed)  ·  "
            f"liver-finetuned JEPA encoder  ·  {dose_str}  ·  validated vs real treated",
            transform=ax.transAxes, fontsize=9, color=_SUB)
    ax.set_xlabel("PC 1  (control ∪ real-treated)", color=_SUB, fontsize=9)
    ax.set_ylabel("PC 2", color=_SUB, fontsize=9)

    # 6. rigor annotation box (small grey, bottom-left corner)
    box = (
        f"sliced-W(pred, real) = {metrics['sliced_w_pred']:.3f}"
        f"   (control baseline {metrics['sliced_w_base']:.3f})\n"
        f"gap closed = {metrics['gap_closed']:.2f}"
        f"   ·   shift cosine(pred, real) = {metrics['shift_cosine']:.3f}\n"
        f"n control = {choice['n_control']}   ·   n treated = {choice['n_treated']}"
    )
    ax.text(0.012, 0.012, box, transform=ax.transAxes, fontsize=8, color=_SUB,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor=_GRID, linewidth=0.9))

    # legend
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=_ACCENT,
               markersize=8, alpha=0.5, label="control (t=0)"),
        Line2D([0], [0], color=_ACCENT, lw=1.4, alpha=0.7, label="flow (control → perturbed)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=_ACCENT,
               markeredgecolor="white", markersize=8, label="predicted perturbed (t=1)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=_ROSE,
               markersize=8, alpha=0.6, label="real treated (validation)"),
    ]
    leg = ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(_INK)

    ax.margins(0.06)
    return _save(fig, out_prefix)


# =========================================================================== #
# Driver                                                                        #
# =========================================================================== #
def run(
    config: str = "examples/tahoe_perturbator/cfgs/train_hepatotox.yaml",
    perturbator_ckpt: str = "",
    out_dir: str = "visualizations/hepatotox/flow",
    drug: str | None = None,
    cell_line: str | None = None,
    dose: float | None = None,
    eval_cells: int = 0,
    ode_steps: int = 24,
    n_streamlines: int = 80,
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
    from examples.tahoe_hepatotox.validate_perturbator import _load_perturbator
    from eb_jepa.singlecell.sub14.features import load_pc_features, random_pc_features

    cache = cfg.encoder.get("gene_emb_cache", "random")
    pc = load_pc_features(cache) if cache and cache != "random" else random_pc_features(
        n_pc=int(cfg.encoder.get("smoke_n_pc", 2000)))
    encoder = build_frozen_encoder(cfg, pc, device)
    featurizer = build_featurizer(cfg)

    if not perturbator_ckpt or not os.path.exists(perturbator_ckpt):
        raise FileNotFoundError(
            f"perturbator_ckpt={perturbator_ckpt!r} not found — train it first with "
            "examples/tahoe_perturbator/cfgs/train_hepatotox.yaml.")
    perturbator, objective = _load_perturbator(perturbator_ckpt, cfg, featurizer, device)
    if objective != "flow_matching":
        raise ValueError(
            f"flow_field requires a flow_matching perturbator, got objective={objective!r}.")
    ode_method = str(cfg.loss.get("ode_method", "heun"))
    sw_slices = int(cfg.loss.get("sw_slices", 256))
    amp = bool(cfg.training.get("amp", True))

    # Liver latents ----------------------------------------------------------
    n_cells = int(eval_cells or cfg.eval.get("eval_cells", 150000))
    latents, meta = _gather_flow_cells(
        cfg, pc, encoder, device, n_cells, amp, CANDIDATE_HEPATOTOXINS,
        drug=drug, cell_line=cell_line)
    if latents.shape[0] == 0:
        raise RuntimeError("No liver cells streamed — check data.liver_only / maps_path.")

    # Pick the (cell_line, drug, dose) ---------------------------------------
    choice = _select_stratum(meta, CANDIDATE_HEPATOTOXINS, drug=drug,
                             cell_line=cell_line, dose=dose)
    logger.info(
        "chosen stratum: cell_line=%s plate=%s drug=%s dose(log10 M)=%.3f "
        "| n_control=%d n_treated=%d",
        choice["cell_line"], choice["plate"], choice["drug"], choice["dose"],
        choice["n_control"], choice["n_treated"])

    src = latents[torch.tensor(choice["src_rows"], dtype=torch.long)]
    real = latents[torch.tensor(choice["tgt_rows"], dtype=torch.long)]

    # Build the action + integrate the flow with the recorded path ------------
    pdev = next(perturbator.parameters()).device
    action = featurizer.featurize(choice["smiles"], choice["dose"]).to(pdev)
    src_d = src.to(pdev)
    pred, path = ode_sample(perturbator, src_d, action, n_steps=int(ode_steps),
                            method=ode_method, return_path=True)
    pred = pred.cpu()
    path = path.cpu()  # [T+1, Nc, d]
    logger.info("integrated flow: path %s (steps=%d, method=%s)",
                tuple(path.shape), int(ode_steps), ode_method)

    # Rigor metrics (full-cloud) ---------------------------------------------
    sw_pred = float(sliced_wasserstein(pred, real, n_slices=sw_slices))
    sw_base = float(sliced_wasserstein(src, real, n_slices=sw_slices))
    gap = float(1.0 - sw_pred / sw_base) if sw_base > 1e-8 else 0.0
    cc, cp, cr = _np(src).mean(0), _np(pred).mean(0), _np(real).mean(0)
    pred_shift = cp - cc
    real_shift = cr - cc
    np_, nr = np.linalg.norm(pred_shift), np.linalg.norm(real_shift)
    shift_cos = float(np.dot(pred_shift, real_shift) / (np_ * nr)) if np_ > 1e-9 and nr > 1e-9 else float("nan")
    gap_centroid = float(np_ / nr) if nr > 1e-9 else float("nan")
    metrics = {
        "sliced_w_pred": sw_pred, "sliced_w_base": sw_base, "gap_closed": gap,
        "shift_cosine": shift_cos, "gap_closed_centroid": gap_centroid,
        "n_control": choice["n_control"], "n_treated": choice["n_treated"],
        "ode_steps": int(ode_steps), "ode_method": ode_method,
    }
    logger.info("rigor: sliced_W(pred,real)=%.4f base=%.4f gap_closed=%.3f "
                "shift_cos=%.3f", sw_pred, sw_base, gap, shift_cos)

    # 2-D projection: PCA(2) on (control ∪ real-treated) ----------------------
    ctrl_np = _np(src)
    real_np = _np(real)
    fit = np.concatenate([ctrl_np, real_np], 0)
    mu = fit.mean(0, keepdims=True)
    # SVD-based PCA (faithful linear map; straight paths project as straight lines)
    _, _, Vt = np.linalg.svd(fit - mu, full_matrices=False)
    comp = Vt[:2].T  # [d, 2]

    def project(x):
        return (_np(x) - mu) @ comp

    ctrl2d = project(src)
    treat2d = project(real)
    pred2d = project(pred)
    T = path.shape[0]
    paths2d = project(path.reshape(-1, path.shape[-1])).reshape(T, path.shape[1], 2)

    # Figure ------------------------------------------------------------------
    safe = f"{choice['drug']}_{choice['cell_line']}".replace("/", "-").replace(" ", "")
    out_prefix = os.path.join(out_dir, f"latent_flow_{safe}")
    paths_saved = plot_flow_field(ctrl2d, treat2d, pred2d, paths2d, choice, metrics,
                                  out_prefix, n_streamlines=int(n_streamlines), seed=seed)
    logger.info("wrote %s", paths_saved)

    # JSON sidecar ------------------------------------------------------------
    summary = {
        "choice": {k: choice[k] for k in
                   ("cell_line", "plate", "drug", "dose", "smiles", "n_control", "n_treated")},
        "metrics": metrics,
        "figure": paths_saved,
        "perturbator_ckpt": perturbator_ckpt,
    }
    with open(out_prefix + ".json", "w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    # wandb (optional) --------------------------------------------------------
    if cfg.wandb.get("enabled", False):
        try:
            from eb_jepa.training_utils import setup_wandb
            import wandb

            if cfg.wandb.get("entity"):
                os.environ["WANDB_ENTITY"] = cfg.wandb.entity
            run = setup_wandb(cfg.wandb.project, cfg, out_dir, enabled=True)
            if run is not None:
                run.log({f"flow/{k}": v for k, v in metrics.items()
                         if isinstance(v, (int, float))})
                png = [p for p in paths_saved if p.endswith(".png")]
                if png:
                    run.log({"flow/latent_flow": wandb.Image(png[0])})
        except Exception as e:  # never fail the figure on a logging hiccup
            logger.warning("wandb logging skipped: %s", e)

    logger.info("=== Latent-flow figure done (%.1fs) ===", time.time() - t0)
    return summary


if __name__ == "__main__":
    import fire

    fire.Fire({"run": run})
