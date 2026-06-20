"""Compute scaling-law figure for the sub14 (Subliminal-1.4) recipe.

Each run is logged densely (loss + cumulative trained FLOPs every ~20 steps), so every
run is a high-resolution TRAJECTORY in compute. We draw one trajectory per run (loss vs
cumulative compute), and the compute-optimal scaling law is the lower envelope
(FRONTIER) over all trajectories — fitted as a power law L(C)=E+A·C^(-alpha). The
downstream cell-line probe vs compute accompanies it. House style.

Run on the GPU node (wandb creds in ~/.netrc):
    PYTHONPATH=/data/eb_jepa .venv/bin/python scripts/fit_sub14_scaling.py
"""
from __future__ import annotations

import math
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

ENTITY, PROJECT, GROUP = "unaite", "hacktheworld", "sub14_law"
OUT = "/data/runs/sub14_law/scaling_law"

INK, MUTED, GRID, ACC = "#1d2433", "#7a8699", "#e9edf2", "#2a6f97"
FK, LK = "flops/cumulative", "loss"
CLK = "probe/clf/cell_line_id/balanced_accuracy"
SMOOTH = 9  # display-only rolling mean over the minibatch-loss trajectory


def fetch():
    runs = []
    for r in wandb.Api().runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP}):
        if r.state == "running":  # only ended runs (LR fully decayed by the end)
            continue
        h = r.history(keys=[FK, LK, CLK], samples=8000, pandas=True)
        if h is None or h.empty or FK not in h or LK not in h:
            continue
        d = h[[FK, LK]].dropna().sort_values(FK)
        if len(d) < 3:
            continue
        s = d[LK].rolling(SMOOTH, min_periods=1, center=True).mean()
        p = h[[FK, CLK]].dropna().sort_values(FK) if CLK in h else None
        arch = r.name.split("_", 1)[1] if "_" in r.name else r.name
        runs.append(dict(
            name=r.name, arch=arch,
            params=float(r.summary.get("model/trainable_params", 0.0)),
            C=d[FK].to_numpy(float), L=s.to_numpy(float),
            pC=(p[FK].to_numpy(float) if p is not None else np.array([])),
            pV=(p[CLK].to_numpy(float) if p is not None else np.array([])),
        ))
    return runs


def frontier(allC, allL, nbins=26):  # min loss per log-compute bin, then make monotone
    bins = np.geomspace(allC.min(), allC.max(), nbins)
    idx = np.digitize(allC, bins)
    pts = []
    for b in range(1, len(bins)):
        m = allL[idx == b]
        if len(m):
            pts.append((math.sqrt(bins[b - 1] * bins[b]), float(np.min(m))))
    out, best = [], math.inf
    for c, l in pts:  # enforce non-increasing envelope
        best = min(best, l)
        out.append((c, best))
    return np.array(out)


def fit_powerlaw(C, L):  # L = E + A*C^-alpha ; grid floor E, log-log fit of (L-E)
    C, L = np.asarray(C, float), np.asarray(L, float)
    best = (0.0, 0.0, 0.0, -1e9)
    for E in np.linspace(0, 0.98 * L.min(), 300):
        y, x = np.log(L - E), np.log(C)
        a1, a0 = np.polyfit(x, y, 1)
        r2 = 1 - ((y - (a0 + a1 * x)) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
        if r2 > best[3]:
            best = (E, -a1, math.exp(a0), r2)
    return best


def _style(ax, xlab, ylab, title, sub, logy=False):
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c7cfdb")
    ax.grid(True, which="both", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED)
    ax.set_xlabel(xlab, color=MUTED)
    ax.set_ylabel(ylab, color=MUTED)
    ax.set_title(title, color=INK, fontweight="bold", loc="left", pad=18)
    ax.text(0, 1.015, sub, transform=ax.transAxes, color=MUTED, fontsize=8.5)


def main():
    runs = fetch()
    if len(runs) < 2:
        print(f"only {len(runs)} usable runs in group {GROUP}")
        return
    runs.sort(key=lambda d: d["params"])
    print(f"{'run':14s}{'params':>9s}{'PFLOP_end':>10s}{'loss_end':>9s}")
    for r in runs:
        print(f"{r['name']:14s}{r['params']/1e6:8.1f}M{r['C'][-1]/1e15:10.1f}{r['L'][-1]:9.2f}")

    # one color per architecture
    archs = sorted({(r["params"], r["arch"]) for r in runs})
    col = {a: c for a, c in zip(archs, plt.cm.viridis(np.linspace(0, 0.9, len(archs))))}

    allC = np.concatenate([r["C"] for r in runs])
    allL = np.concatenate([r["L"] for r in runs])
    fr = frontier(allC, allL)
    E, alpha, A, r2 = fit_powerlaw(fr[:, 0], fr[:, 1])
    print(f"\nfrontier L(C) = {E:.3f} + {A:.3g}·C^(-{alpha:.3f})   R2={r2:.3f}   ({len(fr)} bins)")

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
    cc = np.geomspace(allC.min(), allC.max(), 200)
    seen = set()
    for r in runs:
        a = (r["params"], r["arch"])
        lab = f"{r['arch']} ({r['params']/1e6:.0f}M)" if a not in seen else None
        seen.add(a)
        ax[0].plot(r["C"], r["L"], color=col[a], lw=1.3, alpha=0.85, label=lab)
        if len(r["pC"]):
            ax[2].plot(r["pC"], r["pV"], "-o", color=col[a], lw=1.3, ms=4, alpha=0.85, label=lab)
    ax[0].plot(cc, E + A * cc ** (-alpha), color=INK, lw=2.6, ls="--",
               label=f"frontier  α={alpha:.2f}\nE={E:.2f}  R²={r2:.3f}")
    ax[0].scatter(fr[:, 0], fr[:, 1], color=INK, s=16, zorder=6)
    ax[1].scatter(fr[:, 0], fr[:, 1] - E, color=ACC, s=42, zorder=4)
    ax[1].plot(cc, A * cc ** (-alpha), color=INK, lw=2.0, ls="--")

    _style(ax[0], "training compute (FLOPs)", "training loss (smoothed)",
           "Loss vs compute", "log–log; thin = per-run trajectories, dashed = frontier", logy=True)
    ax[0].legend(frameon=False, fontsize=7.5, labelcolor=INK)
    _style(ax[1], "training compute (FLOPs)", "loss − E",
           "Frontier, straightened", f"power law, slope −{alpha:.2f}", logy=True)
    _style(ax[2], "training compute (FLOPs)", "cell-line balanced acc",
           "Downstream probe vs compute", "representation quality scales with compute")
    ax[2].legend(frameon=False, fontsize=7.5, labelcolor=INK)

    fig.suptitle("sub14 single-cell encoder — compute scaling law",
                 color=INK, fontweight="bold", x=0.5, y=1.04, ha="center")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=220, bbox_inches="tight")
    print(f"saved {OUT}.png / .pdf")


if __name__ == "__main__":
    main()
