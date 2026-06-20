"""House-style figures for the representation-QUALITY benchmark.

Reuses ``visualize.py``'s palette (never modifies it). Every per-metric subplot is
oriented so "up = better" via the direction registry; Subliminal14's bar is drawn
in the accent ink ``#2a6f97`` so the expected JEPA wins read at a glance. A final
"win summary" panel lists which oriented metrics sub14 wins. PNG + PDF, dpi>=200.
"""
from __future__ import annotations

import os

import numpy as np

from eb_jepa.singlecell.visualize import _INK, _OTHER, _SUB

_ACCENT = "#2a6f97"
_ACCENT2 = "#3f8bb5"
_GRID = "#e9edf2"
_AXIS = "#c7cfdb"
_WIN = "#2a6f97"
_LOSS = "#b56357"
_TARGET = "Subliminal14"


def _save(fig, path_no_ext: str):
    png = f"{path_no_ext}.png"
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{path_no_ext}.pdf", bbox_inches="tight", facecolor="white")
    return png


def _style_axes(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=_SUB, labelsize=8)


def _short(metric: str) -> str:
    return metric.split("/", 1)[-1]


def plot_metric_grid(table: dict, direction: dict, metrics, out_dir: str,
                     fname: str, suptitle: str):
    """Grid of per-metric bars across models; sub14 in accent, 'up=better' oriented."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(table.keys())
    metrics = [m for m in metrics
               if any(m in table[mm] and np.isfinite(table[mm][m]) for mm in models)]
    if not metrics:
        return None
    ncol = min(3, len(metrics))
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.4 * nrow),
                             facecolor="white")
    axes = np.atleast_1d(axes).ravel()

    for ax, metric in zip(axes, metrics):
        _style_axes(ax)
        ax.yaxis.grid(True, color=_GRID, linewidth=0.9)
        ax.set_axisbelow(True)
        d = direction.get(metric, +1)
        vals = [table[m].get(metric, np.nan) for m in models]
        colors = [_ACCENT if m == _TARGET else _OTHER for m in models]
        ax.bar(range(len(models)), vals, 0.66, color=colors,
               edgecolor="white", linewidth=0.6)
        if d < 0:
            ax.invert_yaxis()  # lower-is-better -> taller bar lower => flip so up=better
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=7.5, color=_INK)
        arrow = "higher better" if d > 0 else "lower better (axis flipped)"
        ax.set_title(_short(metric), loc="left", fontsize=11, fontweight="bold",
                     color=_INK, pad=12)
        ax.text(0.0, 1.01, arrow, transform=ax.transAxes, fontsize=7.5, color=_SUB)
    for ax in axes[len(metrics):]:
        ax.axis("off")
    fig.suptitle(suptitle, x=0.02, y=1.0, ha="left", fontsize=15,
                 fontweight="bold", color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    png = _save(fig, os.path.join(out_dir, fname))
    plt.close(fig)
    return png


def plot_win_summary(wins: dict, out_dir: str):
    """Win/loss panel: one row per oriented metric, sub14 win vs best-other."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not wins:
        return None
    metrics = list(wins.keys())
    y = np.arange(len(metrics))
    won = [wins[m]["sub14_wins"] for m in metrics]
    fig, ax = plt.subplots(figsize=(8.2, 0.5 * len(metrics) + 1.6), facecolor="white")
    _style_axes(ax)
    ax.barh(y, [1.0] * len(metrics), 0.7,
            color=[_WIN if w else _OTHER for w in won],
            edgecolor="white", linewidth=0.6)
    for yi, m in zip(y, metrics):
        w = wins[m]
        tag = "WIN" if w["sub14_wins"] else f"loses to {w['winner']}"
        ax.text(0.02, yi, f"{_short(m)}", va="center", ha="left",
                fontsize=8.5, color="white", fontweight="bold")
        ax.text(0.98, yi, tag, va="center", ha="right", fontsize=8,
                color="white" if w["sub14_wins"] else _INK)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    n_win = sum(won)
    ax.set_title("Subliminal14 representation-quality wins", loc="left",
                 fontsize=14, fontweight="bold", color=_INK, pad=14)
    ax.text(0.0, 1.02,
            f"{n_win}/{len(metrics)} oriented metrics won  ·  accent = sub14 best",
            transform=ax.transAxes, fontsize=9, color=_SUB)
    fig.tight_layout()
    png = _save(fig, os.path.join(out_dir, "win_summary"))
    plt.close(fig)
    return png


def make_all_plots(table: dict, wins: dict, direction: dict, headline, out_dir: str):
    """Render the headline grid, the full grid, and the win-summary panel."""
    os.makedirs(out_dir, exist_ok=True)
    all_metrics = [m for m in direction if any(m in table[mm] for mm in table)]
    return {
        "headline": plot_metric_grid(
            table, direction, headline, out_dir, "repr_quality_headline",
            "Where LeJEPA is expected to win"),
        "all_metrics": plot_metric_grid(
            table, direction, all_metrics, out_dir, "repr_quality_all",
            "Representation-quality metrics (oriented up = better)"),
        "win_summary": plot_win_summary(wins, out_dir),
    }
