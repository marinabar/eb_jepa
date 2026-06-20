"""Dose-shift visualization for the perturbator (CLAUDE.md "Perturbator-specific").

The headline figure: for a chosen ``(cell_line, drug)`` we encode the control cells
of that line, fit a single 2-D projector (t-SNE or PCA) on them, then overlay the
**predicted centroid shift** as the dose increases — an arrow from the control
centroid to the predicted-perturbed centroid at each dose. Reading the arrow track
demonstrates two things the project wants to show:

- **dose monotonicity**: the centroid moves progressively further with dose;
- **collinearity**: successive dose-step vectors point the same way (a straight,
  ordered latent trajectory).

Multiple drugs share one panel (one arrow-track per drug). A cherry-picking utility
(:func:`monotonicity_score`, :func:`rank_combos`) scores ``(cell_line, drug)`` combos
so the best illustrative examples can be selected automatically.

All numeric work is in the **encoder latent space**; the 2-D projection is for
display only (centroids + arrows are projected through the *same* fitted transform so
the picture is geometrically faithful). House style is imported from the shared
``eb_jepa.singlecell.visualize`` module (palette / spine treatment) and never
modified here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# House palette (single source of truth — do not redefine).
from eb_jepa.singlecell.visualize import _INK, _SUB

_ACCENT = "#2a6f97"
_ACCENT2 = "#3f8bb5"
_GRID = "#e9edf2"
_AXIS = "#c7cfdb"


# --------------------------------------------------------------------------- #
# Monotonicity scoring (cherry-picking)                                       #
# --------------------------------------------------------------------------- #
def _as_np(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float64)


def monotonicity_score(
    control_centroid, dose_centroids, doses=None
) -> dict:
    """Score a dose trajectory of predicted centroids (in latent space).

    Args:
        control_centroid: ``[d]`` control (dose-0) centroid.
        dose_centroids: ``[K, d]`` predicted perturbed centroids, **ordered by
            increasing dose**.
        doses: optional ``[K]`` dose values (only used to assert ordering / report).
    Returns:
        dict with:
        - ``collinearity``: mean cosine alignment of successive step vectors
          (1 = perfectly straight track), in ``[-1, 1]``;
        - ``magnitude_monotonicity``: fraction of consecutive displacement
          magnitudes (from the control) that strictly increase, in ``[0, 1]``;
        - ``score``: combined cherry-pick score
          ``0.5*(collinearity_clamped) + 0.5*magnitude_monotonicity``;
        - ``displacements``: ``[K]`` ||centroid_k - control|| (for reporting).

    Trajectories with fewer than 2 doses get ``score = 0`` (nothing to rank).
    """
    c0 = _as_np(control_centroid).reshape(-1)
    cents = _as_np(dose_centroids).reshape(len(dose_centroids), -1)
    k = cents.shape[0]
    disp = np.linalg.norm(cents - c0[None, :], axis=1)  # [K]
    out = {
        "collinearity": 0.0,
        "magnitude_monotonicity": 0.0,
        "score": 0.0,
        "displacements": disp,
    }
    if k < 2:
        return out

    # step vectors along the track: control -> dose0 -> dose1 -> ... -> doseK-1
    pts = np.concatenate([c0[None, :], cents], axis=0)  # [K+1, d]
    steps = np.diff(pts, axis=0)  # [K, d]
    norms = np.linalg.norm(steps, axis=1)
    valid = norms > 1e-8
    if valid.sum() >= 2:
        unit = steps[valid] / norms[valid][:, None]
        cos = (unit[:-1] * unit[1:]).sum(axis=1)  # successive-step cosines
        out["collinearity"] = float(np.mean(cos))

    incr = np.diff(disp) > 0  # magnitude strictly increasing with dose
    out["magnitude_monotonicity"] = float(np.mean(incr)) if incr.size else 0.0

    out["score"] = 0.5 * max(0.0, out["collinearity"]) + 0.5 * out["magnitude_monotonicity"]
    return out


@dataclass
class DoseTrack:
    """A drug's predicted dose trajectory for one cell line (latent space)."""

    drug: str
    smiles: str | None
    doses: list[float]  # ascending log10 molar
    control_centroid: np.ndarray  # [d]
    dose_centroids: np.ndarray  # [K, d] (aligned with doses)
    metrics: dict  # monotonicity_score output


def build_dose_track(
    perturbator,
    featurizer,
    control_latents: torch.Tensor,
    drug: str,
    smiles: str | None,
    doses,
    objective: str = "flow_matching",
    ode_steps: int = 20,
    ode_method: str = "heun",
) -> DoseTrack:
    """Predict the perturbed-centroid track for ``drug`` over ascending ``doses``.

    For each dose the control cloud is mapped through the perturbator (ODE for
    flow-matching, one-shot map for direct); the per-dose **predicted centroid** is
    the mean of the predicted cloud. The control centroid is the mean of the input
    control latents.
    """
    from eb_jepa.singlecell.perturbator.flow import predict_perturbed

    doses = [float(d) for d in doses]
    order = np.argsort(doses)
    doses_sorted = [doses[i] for i in order]
    control_centroid = control_latents.detach().float().mean(0).cpu().numpy()

    cents = []
    for d in doses_sorted:
        action = featurizer.featurize(smiles, d).to(control_latents.device)
        pred = predict_perturbed(
            perturbator, control_latents, action, objective,
            n_steps=ode_steps, method=ode_method,
        )
        cents.append(pred.detach().float().mean(0).cpu().numpy())
    dose_centroids = np.stack(cents, axis=0)
    metrics = monotonicity_score(control_centroid, dose_centroids, doses_sorted)
    return DoseTrack(
        drug=drug,
        smiles=smiles,
        doses=doses_sorted,
        control_centroid=control_centroid,
        dose_centroids=dose_centroids,
        metrics=metrics,
    )


def rank_combos(tracks: list[DoseTrack]) -> list[DoseTrack]:
    """Sort dose tracks by descending monotonicity score (best illustrative first)."""
    return sorted(tracks, key=lambda t: t.metrics.get("score", 0.0), reverse=True)


# --------------------------------------------------------------------------- #
# 2-D projection (display only)                                               #
# --------------------------------------------------------------------------- #
def _fit_projector(control_latents: torch.Tensor, method: str, seed: int):
    """Fit a 2-D display projector on the control latents. Returns a transform fn.

    ``pca`` (default) fits a linear map so centroids/arrows project exactly through
    the same transform. ``tsne`` is non-parametric, so we project the control points
    *and* every centroid jointly by re-fitting on the stacked set (the centroids are
    appended; t-SNE has no out-of-sample transform). For arrow geometry a linear
    projector (PCA) is the honest choice; t-SNE is offered for cluster shape.
    """
    x = control_latents.detach().float().cpu().numpy()
    if method == "pca":
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=seed).fit(x)
        return ("pca", pca)
    if method == "tsne":
        return ("tsne", seed)
    raise ValueError(f"unknown projector {method!r}")


def _project(proj, control_points: np.ndarray, extra_points: np.ndarray):
    """Project control points + extra points (centroids) to 2-D via ``proj``.

    Returns ``(ctrl2d [Nc,2], extra2d [Ne,2])``.
    """
    kind = proj[0]
    if kind == "pca":
        pca = proj[1]
        return pca.transform(control_points), pca.transform(extra_points)
    # t-SNE: fit jointly on the stacked set, then split back.
    from sklearn.manifold import TSNE

    seed = proj[1]
    stacked = np.concatenate([control_points, extra_points], axis=0)
    perp = min(30.0, max(2.0, (stacked.shape[0] - 1) / 3.0))
    emb = TSNE(n_components=2, random_state=seed, perplexity=perp).fit_transform(stacked)
    nc = control_points.shape[0]
    return emb[:nc], emb[nc:]


# --------------------------------------------------------------------------- #
# Figure                                                                       #
# --------------------------------------------------------------------------- #
def _style_ax(ax):
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=_SUB, labelsize=8)
    ax.grid(True, axis="both", color=_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_dose_shift(
    control_latents: torch.Tensor,
    tracks: list[DoseTrack],
    out_prefix: str,
    cell_line: str,
    projector: str = "pca",
    seed: int = 0,
    formats=("png", "pdf", "svg"),
    max_control_points: int = 4000,
) -> list[str]:
    """Save the dose-shift figure (control cloud + per-drug dose-arrow tracks).

    Args:
        control_latents: ``[N, d]`` control latents of ``cell_line`` (latent space).
        tracks: dose tracks (one per drug) from :func:`build_dose_track`, all sharing
            the same ``control_centroid`` / control cloud.
        out_prefix: path prefix; ``"{prefix}.png"``, ``".pdf"``, ``".svg"`` are written.
        cell_line: cell-line name (title context).
        projector: ``"pca"`` (faithful linear, default) or ``"tsne"`` (cluster shape).
        formats: output formats to save.
        max_control_points: subsample the control scatter for legibility (display only).
    Returns:
        list of written file paths.
    """
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ctrl = control_latents.detach().float().cpu().numpy()
    rng = np.random.default_rng(seed)
    if ctrl.shape[0] > max_control_points:
        sub = rng.choice(ctrl.shape[0], max_control_points, replace=False)
        ctrl_show = ctrl[sub]
    else:
        ctrl_show = ctrl

    proj = _fit_projector(control_latents, projector, seed)

    # stack every centroid (control + each drug's dose centroids) for joint projection
    centroid_blocks, block_index = [], []
    if tracks:
        centroid_blocks.append(tracks[0].control_centroid[None, :])
        block_index.append(("control", 0))
    for ti, t in enumerate(tracks):
        centroid_blocks.append(t.dose_centroids)
        block_index.append(("track", ti))
    all_centroids = np.concatenate(centroid_blocks, axis=0) if centroid_blocks else np.zeros((0, ctrl.shape[1]))

    ctrl2d, cent2d = _project(proj, ctrl_show, all_centroids)

    # split cent2d back into control centroid + per-track centroids
    cursor = 0
    control_c2d = None
    track_c2d = {}
    for kind, idx in block_index:
        if kind == "control":
            control_c2d = cent2d[cursor]
            cursor += 1
        else:
            k = tracks[idx].dose_centroids.shape[0]
            track_c2d[idx] = cent2d[cursor:cursor + k]
            cursor += k

    fig, ax = plt.subplots(figsize=(8.2, 7.2), facecolor="white")
    _style_ax(ax)

    # control cloud (faint accent)
    ax.scatter(
        ctrl2d[:, 0], ctrl2d[:, 1], s=9, c=_ACCENT, alpha=0.18,
        linewidths=0, zorder=1, label=f"control cells (n={ctrl.shape[0]})",
    )
    if control_c2d is not None:
        ax.scatter(
            [control_c2d[0]], [control_c2d[1]], s=140, marker="*",
            c=_INK, edgecolors="white", linewidths=1.0, zorder=5,
            label="control centroid",
        )

    # one arrow track per drug
    from eb_jepa.singlecell.visualize import _palette

    drug_colors = _palette(max(1, len(tracks)))
    for ti, t in enumerate(tracks):
        col = drug_colors[ti]
        c2d = track_c2d[ti]
        pts = np.concatenate([control_c2d[None, :], c2d], axis=0)  # [K+1, 2]
        # ordered dose arrows
        for a, b in zip(pts[:-1], pts[1:]):
            ax.annotate(
                "", xy=(b[0], b[1]), xytext=(a[0], a[1]),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8, alpha=0.9,
                                shrinkA=0, shrinkB=0),
                zorder=4,
            )
        # dose markers sized by rank
        sizes = np.linspace(28, 90, len(c2d))
        ax.scatter(
            c2d[:, 0], c2d[:, 1], s=sizes, c=[col], edgecolors="white",
            linewidths=0.8, zorder=6,
            label=f"{t.drug}  (mono={t.metrics['score']:.2f})",
        )

    ax.set_title(
        f"Dose-shift trajectories — {cell_line}",
        loc="left", fontsize=16, fontweight="bold", color=_INK, pad=26,
    )
    n_drugs = len(tracks)
    dose_span = ""
    if tracks and tracks[0].doses:
        dose_span = f"  ·  {len(tracks[0].doses)} doses / drug"
    ax.text(
        0.0, 1.018,
        f"predicted control->perturbed centroid shift  ·  {n_drugs} drug(s){dose_span}  ·  {projector.upper()} projection",
        transform=ax.transAxes, fontsize=9, color=_SUB,
    )
    ax.set_xlabel(f"{projector.upper()}-1", color=_SUB, fontsize=9)
    ax.set_ylabel(f"{projector.upper()}-2", color=_SUB, fontsize=9)
    leg = ax.legend(
        loc="best", frameon=False, fontsize=8, markerscale=1.0,
        handletextpad=0.5, labelspacing=0.4,
    )
    for txt in leg.get_texts():
        txt.set_color(_INK)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    paths = []
    for fmt in formats:
        p = f"{out_prefix}.{fmt}"
        fig.savefig(p, dpi=220, bbox_inches="tight", facecolor="white")
        paths.append(p)
    plt.close(fig)
    return paths
