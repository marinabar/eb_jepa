"""Scaling-law figure for the sub14 (Subliminal-1.4) recipe.

Each model size is run for the same wall-clock and logs (cumulative trained FLOPs,
loss, eval loss, probe balanced-accuracy, effective rank) THROUGHOUT — so every run
is a trajectory in compute. This pulls the per-run history, draws one curve per size
vs compute, traces the compute-optimal FRONTIER (lower envelope of eval loss) and
fits a power law L(C)=E+A·C^(-alpha), plus the downstream probe-vs-compute curves
(the headline for "do single-cell encoders scale?"). Publication-grade house style.

Run on the GPU node (wandb creds in ~/.netrc):
    PYTHONPATH=/data/eb_jepa .venv/bin/python scripts/fit_sub14_scaling.py
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

ENTITY, PROJECT, GROUP = "unaite", "hacktheworld", "sub14_scaling"
OUT = "/data/runs/sub14_scaling/scaling_law"

# house style (CLAUDE.md)
INK, MUTED, GRID = "#1d2433", "#7a8699", "#e9edf2"
QUAL = [
    "#2a6f97",
    "#3f8bb5",
    "#61a5c2",
    "#89c2d9",
    "#e09f3e",
    "#d4763a",
    "#9e2a2b",
    "#540b0e",
]
FK, LK, EK = "flops/cumulative", "loss", "eval/loss"
CLK = "probe/clf/cell_line_id/balanced_accuracy"
OGK = "probe/clf/organ/balanced_accuracy"
RK = "repr/effective_rank"


def fetch():
    runs = []
    for r in wandb.Api().runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP}):
        h = r.history(keys=[FK, LK, EK, CLK, OGK, RK], samples=4000, pandas=True)
        if h is None or h.empty or FK not in h:
            continue
        p = r.summary.get("model/trainable_params", float("nan"))
        runs.append((r.name, float(p), h.sort_values(FK)))
    runs.sort(key=lambda t: t[1])
    return runs


def xy(h, ycol):
    if ycol not in h:
        return np.array([]), np.array([])
    d = h[[FK, ycol]].dropna()
    return d[FK].to_numpy(float), d[ycol].to_numpy(float)


def frontier(pairs):  # list of (C, L) -> running-min envelope over sorted C
    pts = sorted((c, l) for c, l in pairs if c > 0 and np.isfinite(l))
    out, best = [], math.inf
    for c, l in pts:
        if l < best:
            best = l
            out.append((c, best))
    return np.array(out) if out else np.empty((0, 2))


def fit_powerlaw(C, L):  # L = E + A*C^-alpha ; grid E, log-log fit of (L-E)
    best = (0.0, 0.0, 0.0, -1e9)
    for E in np.linspace(0, 0.99 * L.min(), 200):
        y, x = np.log(np.clip(L - E, 1e-9, None)), np.log(C)
        a1, a0 = np.polyfit(x, y, 1)
        r2 = 1 - ((y - (a0 + a1 * x)) ** 2).sum() / max(
            ((y - y.mean()) ** 2).sum(), 1e-12
        )
        if r2 > best[3]:
            best = (E, -a1, math.exp(a0), r2)
    return best


def _style(ax):
    ax.set_xscale("log")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c7cfdb")
    ax.grid(True, which="both", axis="x", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED)
    ax.set_xlabel("training compute (FLOPs)", color=MUTED)


def main():
    runs = fetch()
    if len(runs) < 2:
        print(f"only {len(runs)} usable runs in group {GROUP}")
        return
    print(
        f"{'run':14s} {'params':>9s} {'PFLOP_end':>10s} {'eval_loss':>9s} {'cell_line':>9s} {'organ':>7s}"
    )
    allf = []
    for name, p, h in runs:
        ef, el = xy(h, EK)
        cf, cl = xy(h, CLK)
        of, og = xy(h, OGK)
        endC = h[FK].max()
        print(
            f"{name:14s} {p/1e6:8.1f}M {endC/1e15:10.2f} "
            f"{(el[-1] if len(el) else float('nan')):9.3f} "
            f"{(cl[-1] if len(cl) else float('nan')):9.3f} "
            f"{(og[-1] if len(og) else float('nan')):7.3f}"
        )
        allf += list(zip(ef, el))

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
    for i, (name, p, h) in enumerate(runs):
        c = QUAL[i % len(QUAL)]
        lab = f"{name.split('_',1)[-1]} ({p/1e6:.0f}M)"
        for j, k in enumerate((EK, CLK, OGK)):
            fx, fy = xy(h, k)
            if len(fx):
                ax[j].plot(
                    fx,
                    fy,
                    color=c,
                    lw=1.8,
                    marker="o",
                    ms=3,
                    alpha=0.9,
                    label=lab if j == 0 else None,
                )

    # frontier + power-law fit on the eval-loss envelope
    fr = frontier(allf)
    if len(fr) >= 4:
        E, alpha, A, r2 = fit_powerlaw(fr[:, 0], fr[:, 1])
        cc = np.geomspace(fr[:, 0].min(), fr[:, 0].max(), 100)
        ax[0].plot(
            cc,
            E + A * cc ** (-alpha),
            color=INK,
            lw=2.4,
            ls="--",
            label=f"frontier α={alpha:.2f}",
        )
        ax[0].scatter(fr[:, 0], fr[:, 1], color=INK, s=18, zorder=5)
        print(f"\nfrontier: L(C) = {E:.4f} + {A:.3g}·C^(-{alpha:.3f})   R2={r2:.3f}")

    titles = [
        "Held-out loss vs compute",
        "Cell-line probe vs compute",
        "Organ probe vs compute",
    ]
    subs = [
        "lower envelope = compute-optimal frontier",
        "balanced accuracy (46 lines)",
        "balanced accuracy (13 organs)",
    ]
    for j in range(3):
        _style(ax[j])
        ax[j].set_title(titles[j], color=INK, fontweight="bold", loc="left")
        ax[j].text(0, 1.02, subs[j], transform=ax[j].transAxes, color=MUTED, fontsize=9)
    ax[0].set_ylabel("eval loss", color=MUTED)
    ax[1].set_ylabel("balanced acc", color=MUTED)
    ax[2].set_ylabel("balanced acc", color=MUTED)
    ax[0].legend(frameon=False, fontsize=7.5, labelcolor=INK, ncol=2)
    fig.suptitle(
        "sub14 single-cell encoder — scaling with training compute",
        color=INK,
        fontweight="bold",
        x=0.5,
        y=1.0,
        ha="center",
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=220, bbox_inches="tight")
    print(f"saved {OUT}.png / .pdf")


if __name__ == "__main__":
    main()
