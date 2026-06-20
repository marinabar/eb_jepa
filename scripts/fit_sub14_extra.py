"""Extra scaling-law figures for the sub14 sweep (wandb group sub14_law).

Builds three figures from the dense per-run histories (no re-training):
  A. IsoFLOP   — loss vs model size at fixed compute budgets; the minimum of each
                 curve is the compute-optimal size (shifts right with compute).
  B. Data      — loss vs cells seen (log-log) + power-law fit (data scaling).
  C. Iso-arch  — each architecture's best-loss-vs-compute envelope, and the lowest
                 loss each architecture reaches vs its size.

Run on the GPU node:
    PYTHONPATH=/data/eb_jepa .venv/bin/python scripts/fit_sub14_extra.py
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
OUTDIR = "/data/runs/sub14_law"
INK, MUTED, GRID, ACC = "#1d2433", "#7a8699", "#e9edf2", "#2a6f97"
FK, DK, LK = "flops/cumulative", "data/cells_seen", "loss"
SMOOTH = 9


def fetch():
    runs = []
    for r in wandb.Api().runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP}):
        if r.state == "running":
            continue
        h = r.history(keys=[FK, DK, LK], samples=8000, pandas=True)
        if h is None or h.empty or FK not in h or LK not in h:
            continue
        d = h[[FK, DK, LK]].dropna().sort_values(FK)
        if len(d) < 3:
            continue
        s = d[LK].rolling(SMOOTH, min_periods=1, center=True).mean().to_numpy(float)
        arch = r.name.split("_", 1)[1] if "_" in r.name else r.name
        runs.append(dict(arch=arch,
                         params=float(r.summary.get("model/trainable_params", 0.0)),
                         C=d[FK].to_numpy(float), D=d[DK].to_numpy(float), L=s))
    return runs


def envelope(x, y):  # sorted running-min of y over x
    o = np.argsort(x)
    x, y = np.asarray(x)[o], np.asarray(y)[o]
    best, ex, ey = math.inf, [], []
    for xi, yi in zip(x, y):
        best = min(best, yi)
        ex.append(xi); ey.append(best)
    return np.array(ex), np.array(ey)


def by_arch(runs, xkey):
    """{(params,arch): (env_x, env_y)} lower-loss envelope over all runs of the arch."""
    g = defaultdict(lambda: ([], []))
    for r in runs:
        g[(r["params"], r["arch"])][0].extend(r[xkey])
        g[(r["params"], r["arch"])][1].extend(r["L"])
    return {k: envelope(np.array(xs), np.array(ys)) for k, (xs, ys) in g.items()}


def fit_powerlaw(C, L):
    C, L = np.asarray(C, float), np.asarray(L, float)
    best = (0.0, 0.0, 0.0, -1e9)
    for E in np.linspace(0, 0.98 * L.min(), 300):
        y, x = np.log(L - E), np.log(C)
        a1, a0 = np.polyfit(x, y, 1)
        r2 = 1 - ((y - (a0 + a1 * x)) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
        if r2 > best[3]:
            best = (E, -a1, math.exp(a0), r2)
    return best


def style(ax, xlab, ylab, title, sub, logx=True, logy=True):
    if logx:
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


def fig_isoflop(runs):
    env = by_arch(runs, "C")
    keys = sorted(env)  # by params
    P = np.array([k[0] for k in keys])
    budgets = [3e16, 1e17, 3e17, 6e17]  # 30, 100, 300, 600 PF
    cols = plt.cm.plasma(np.linspace(0.1, 0.85, len(budgets)))
    fig, ax = plt.subplots(figsize=(7.5, 6))
    opt = []
    for C0, c in zip(budgets, cols):
        xs, ys = [], []
        for k in keys:
            ex, ey = env[k]
            if ex.min() <= C0 <= ex.max():
                xs.append(k[0])
                ys.append(float(np.interp(math.log(C0), np.log(ex), ey)))
        if len(xs) >= 3:
            xs, ys = np.array(xs), np.array(ys)
            ax.plot(xs, ys, "-o", color=c, ms=6, lw=1.6, label=f"{C0/1e15:.0f} PF")
            j = int(np.argmin(ys))
            ax.scatter([xs[j]], [ys[j]], s=180, facecolor="none",
                       edgecolor=c, lw=2.2, zorder=5)
            opt.append((C0, xs[j], ys[j]))
    if len(opt) >= 2:
        oc, op, ol = np.array(opt).T
        order = np.argsort(op)
        ax.plot(op[order], ol[order], "--", color=INK, lw=2.4, zorder=6,
                label="compute-optimal")  # trendline through the minima
        p = np.polyfit(np.log(oc), np.log(op), 1)[0]
        ax.text(0.04, 0.06, f"optimal size ∝ C^{p:.2f}", transform=ax.transAxes,
                color=INK, fontsize=11, fontweight="bold")
    style(ax, "model size (params)", "loss at fixed compute", "IsoFLOP",
          "each curve = one compute budget; ◯ = compute-optimal size", logy=True)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, title="compute budget",
              title_fontproperties={"weight": "bold"})
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"{OUTDIR}/scaling_isoflop.{e}", dpi=220, bbox_inches="tight")
    print(f"isoflop: optimal points {[(f'{c/1e15:.0f}PF', f'{p/1e6:.0f}M') for c,p,_ in opt]}")


def fig_data(runs):
    archs = sorted({(r["params"], r["arch"]) for r in runs})
    col = {a: c for a, c in zip(archs, plt.cm.viridis(np.linspace(0, 0.9, len(archs))))}
    allD = np.concatenate([r["D"] for r in runs])
    allL = np.concatenate([r["L"] for r in runs])
    bins = np.geomspace(allD.min(), allD.max(), 26)
    idx = np.digitize(allD, bins)
    fr, best = [], math.inf
    for b in range(1, len(bins)):
        m = allL[idx == b]
        if len(m):
            best = min(best, float(m.min()))
            fr.append((math.sqrt(bins[b - 1] * bins[b]), best))
    fr = np.array(fr)
    E, alpha, A, r2 = fit_powerlaw(fr[:, 0], fr[:, 1])
    fig, ax = plt.subplots(figsize=(8, 6))
    seen = set()
    for r in runs:
        a = (r["params"], r["arch"])
        lab = f"{r['arch']} ({r['params']/1e6:.0f}M)" if a not in seen else None
        seen.add(a)
        ax.plot(r["D"], r["L"], color=col[a], lw=1.3, alpha=0.85, label=lab)
    dd = np.geomspace(allD.min(), allD.max(), 200)
    ax.plot(dd, E + A * dd ** (-alpha), color=INK, lw=2.6, ls="--",
            label=f"frontier  α={alpha:.2f}\nE={E:.2f}  R²={r2:.3f}")
    style(ax, "cells seen", "training loss (smoothed)", "Data scaling",
          "loss vs amount of data; dashed = frontier power law", logy=True)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, ncol=2)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"{OUTDIR}/scaling_data.{e}", dpi=220, bbox_inches="tight")
    print(f"data: L(D) = {E:.3f} + {A:.3g}·D^(-{alpha:.3f})  R2={r2:.3f}")


def fig_isoarch(runs):
    # one curve per model size; x = compute, y = converged loss (run endpoints, LR-decayed)
    g = defaultdict(list)
    for r in runs:
        g[(r["params"], r["arch"])].append((float(r["C"][-1]), float(np.min(r["L"][-3:]))))
    keys = sorted(g)
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(keys)))
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.8))
    bestloss = []
    for k, c in zip(keys, cols):
        pts = sorted(g[k])
        C = np.array([p[0] for p in pts])
        L = np.array([p[1] for p in pts])
        ax[0].plot(C, L, "-o", color=c, ms=7, lw=1.9, label=f"{k[1]} ({k[0]/1e6:.0f}M)")
        bestloss.append((k[0], float(L.min())))
    style(ax[0], "training compute (FLOPs)", "converged loss",
          "Loss vs compute, per model size", "each curve = one architecture; ● = a run (LR-decayed)", logy=True)
    ax[0].legend(frameon=False, fontsize=7.5, labelcolor=INK, ncol=2)
    bl = np.array(sorted(bestloss))
    ax[1].plot(bl[:, 0], bl[:, 1], "-o", color=ACC, ms=7, lw=1.6)
    style(ax[1], "model size (params)", "lowest loss achieved",
          "Best loss vs architecture size", "diminishing returns of width/depth", logy=True)
    fig.suptitle("sub14 — iso-architecture analysis", color=INK, fontweight="bold", y=1.02)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"{OUTDIR}/scaling_isoarch.{e}", dpi=220, bbox_inches="tight")
    print(f"isoarch: best loss per size {[(f'{p/1e6:.0f}M', round(l,3)) for p,l in bestloss]}")


def main():
    runs = fetch()
    print(f"{len(runs)} runs, {len({r['arch'] for r in runs})} architectures")
    if len(runs) < 4:
        return
    fig_isoflop(runs)
    fig_data(runs)
    fig_isoarch(runs)
    print("saved scaling_isoflop / scaling_data / scaling_isoarch (.png/.pdf)")


if __name__ == "__main__":
    main()
