"""Comparison figures for the baseline benchmark (CLAUDE.md "Visualization design
system"). Reuses the shared ``visualize.py`` palette + t-SNE panel helper; never
modifies it. Every figure is publication-grade house style, saved as PNG + PDF to
``visualizations/benchmarks/``.

Three figures:
  - ``probe_bars``: grouped bar chart of probe balanced-accuracy per class across
    the models, with the per-class chance line annotated.
  - ``effrank``: effective-rank comparison bar (collapse diagnostic side by side).
  - ``tsne_grid``: 2x2 t-SNE grid on the SAME cells, colored by cell_line, one
    panel per model so collapse / structure is visible side by side.
"""
from __future__ import annotations

import os

import numpy as np
import torch

# Reuse the shared design system (palette + the single-panel t-SNE drawer).
from eb_jepa.singlecell.visualize import (
    _INK,
    _SUB,
    _draw_tsne_panel,
    _palette,
    tsne_embed,
)

_ACCENT = "#2a6f97"
_ACCENT2 = "#3f8bb5"
_GRID = "#e9edf2"
_AXIS = "#c7cfdb"
_CLF_CLASSES = ("organ", "cell_line_id", "drug", "moa_fine")


def _save(fig, path_no_ext: str):
    """Save PNG + PDF (house style); return the PNG path."""
    png = f"{path_no_ext}.png"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{path_no_ext}.pdf", bbox_inches="tight", facecolor="white")
    return png


def _style_axes(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=_SUB, labelsize=9)


def _title(ax, title, subtitle):
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=_INK, pad=18)
    ax.text(0.0, 1.012, subtitle, transform=ax.transAxes, fontsize=9, color=_SUB)


def plot_probe_bars(table: dict, out_dir: str):
    """Grouped bar chart: balanced accuracy per class across models."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(table.keys())
    classes = [c for c in _CLF_CLASSES
               if any(f"{c}/balanced_accuracy" in table[m] for m in models)]
    if not classes:
        return None
    colors = _palette(len(models))
    x = np.arange(len(classes))
    w = 0.8 / max(1, len(models))

    fig, ax = plt.subplots(figsize=(1.9 * len(classes) + 2.5, 5), facecolor="white")
    _style_axes(ax)
    ax.yaxis.grid(True, color=_GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    for mi, m in enumerate(models):
        vals = [table[m].get(f"{c}/balanced_accuracy", np.nan) for c in classes]
        ax.bar(x + mi * w, vals, w, label=m, color=colors[mi], edgecolor="white", linewidth=0.6)
    # chance line per class (markers, model-independent)
    for ci, c in enumerate(classes):
        ch = next((table[m].get(f"{c}/chance") for m in models
                   if table[m].get(f"{c}/chance") is not None), None)
        if ch is not None:
            ax.plot([x[ci] - 0.4 * w, x[ci] + (len(models) - 0.6) * w], [ch, ch],
                    color=_SUB, linestyle=(0, (3, 2)), linewidth=1.1,
                    label="chance" if ci == 0 else None)
    ax.set_xticks(x + (len(models) - 1) * w / 2)
    ax.set_xticklabels(classes, fontsize=10, color=_INK)
    ax.set_ylabel("balanced accuracy", fontsize=10, color=_SUB)
    ax.set_ylim(0, 1)
    _title(ax, "Linear-probe balanced accuracy",
           "detached probes on the fixed shared validation set · higher is better · dashed = chance")
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for t in leg.get_texts():
        t.set_color(_INK)
    fig.tight_layout()
    png = _save(fig, os.path.join(out_dir, "probe_balanced_accuracy"))
    plt.close(fig)
    return png


def plot_effrank(table: dict, out_dir: str):
    """Effective-rank comparison bar (collapse diagnostic)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(table.keys())
    eff = [table[m].get("effective_rank", np.nan) for m in models]
    dims = [table[m].get("latent_dim", np.nan) for m in models]
    colors = _palette(len(models))

    fig, ax = plt.subplots(figsize=(1.3 * len(models) + 3, 5), facecolor="white")
    _style_axes(ax)
    ax.yaxis.grid(True, color=_GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    bars = ax.bar(range(len(models)), eff, 0.62, color=colors, edgecolor="white", linewidth=0.6)
    for b, e, d in zip(bars, eff, dims):
        ax.text(b.get_x() + b.get_width() / 2, e, f"{e:.1f}\n/{int(d)}d",
                ha="center", va="bottom", fontsize=8.5, color=_INK)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=10, color=_INK)
    ax.set_ylabel("effective rank", fontsize=10, color=_SUB)
    _title(ax, "Representation effective rank",
           "exp(entropy of the normalized eigenspectrum) · ~latent_dim = rich, ~1 = collapsed")
    fig.tight_layout()
    png = _save(fig, os.path.join(out_dir, "effective_rank"))
    plt.close(fig)
    return png


def plot_tsne_grid(feats: dict, meta: dict, out_dir: str, color_key: str = "cell_line_id",
                   seed: int = 0):
    """2xK t-SNE grid: same cells, one panel per model, colored by ``color_key``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = meta.get(color_key)
    if labels is None:
        return None
    models = list(feats.keys())
    ncol = 2
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0 * ncol, 6.2 * nrow), facecolor="white")
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, models):
        emb = tsne_embed(feats[name], seed=seed)
        _draw_tsne_panel(ax, np.asarray(emb), labels, name)
    for ax in axes[len(models):]:
        ax.axis("off")
    fig.suptitle(f"t-SNE of cell representations · colored by {color_key}",
                 x=0.04, y=0.995, ha="left", fontsize=18, fontweight="bold", color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = _save(fig, os.path.join(out_dir, f"tsne_grid_{color_key}"))
    plt.close(fig)
    return png


def make_all_plots(table: dict, feats: dict, meta: dict, out_dir: str) -> dict:
    """Render all three comparison figures; return {tag -> png path} (None on skip)."""
    os.makedirs(out_dir, exist_ok=True)
    # Coerce feats to CPU tensors for t-SNE.
    feats = {k: (v if torch.is_tensor(v) else torch.as_tensor(v)) for k, v in feats.items()}
    return {
        "probe_bars": plot_probe_bars(table, out_dir),
        "effrank": plot_effrank(table, out_dir),
        "tsne_grid": plot_tsne_grid(feats, meta, out_dir),
    }
