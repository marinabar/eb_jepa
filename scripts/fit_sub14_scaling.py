"""Compute scaling-law figure for the sub14 (Subliminal-1.4) recipe.

Each run is trained to its OWN compute budget with the LR fully cosine-decayed, so its
final held-out loss is the converged loss at that compute. Within one architecture the
loss falls cleanly with compute; the compute-optimal scaling law is the lower envelope
(FRONTIER) over all (size, budget) runs. We plot eval loss vs compute (log-log), one
line per architecture, the frontier, and a fitted power law L(C)=E+A·C^(-alpha); plus
the downstream cell-line probe vs compute. House style.

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
FK, EK = "flops/cumulative", "eval/loss"
CLK = "probe/clf/cell_line_id/balanced_accuracy"


def fetch():
    rows = []
    for r in wandb.Api().runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP}):
        if r.state == "running":  # only ended (converged) runs; LR fully decayed
            continue
        h = r.history(keys=[FK, EK, CLK], samples=5000, pandas=True)
        if h is None or h.empty or FK not in h:
            continue

        def last(c):
            d = h[c].dropna() if c in h else []
            return float(d.iloc[-1]) if len(d) else float("nan")

        arch = r.name.split("_", 1)[1] if "_" in r.name else r.name  # e.g. d320_l5
        rows.append(dict(name=r.name, arch=arch,
                         params=float(r.summary.get("model/trainable_params", 0.0)),
                         C=float(h[FK].max()), loss=last(EK), cell=last(CLK)))
    return [r for r in rows if r["C"] > 0 and np.isfinite(r["loss"])]


def frontier(pts):  # [(C, L)] -> running-min envelope over sorted C
    out, best = [], math.inf
    for c, l in sorted(pts):
        if l < best - 1e-9:
            best = l
            out.append((c, l))
    return np.array(out) if out else np.empty((0, 2))


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
    ax.set_title(title, color=INK, fontweight="bold", loc="left")
    ax.text(0, 1.02, sub, transform=ax.transAxes, color=MUTED, fontsize=9)


def main():
    rows = fetch()
    if len(rows) < 3:
        print(f"only {len(rows)} usable runs in group {GROUP}")
        return
    rows.sort(key=lambda d: d["C"])
    print(f"{'run':14s}{'PFLOP':>9s}{'eval_loss':>10s}{'cell':>7s}")
    for r in rows:
        print(f"{r['name']:14s}{r['C']/1e15:9.1f}{r['loss']:10.2f}{r['cell']:7.3f}")

    # per-architecture grouping (ordered by size)
    archs = defaultdict(list)
    for r in rows:
        archs[(r["params"], r["arch"])].append(r)
    keys = sorted(archs)
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(keys)))

    fr = frontier([(r["C"], r["loss"]) for r in rows])
    E, alpha, A, r2 = fit_powerlaw(fr[:, 0], fr[:, 1]) if len(fr) >= 3 else (0, 0, 0, 0)
    print(f"\nfrontier L(C) = {E:.3f} + {A:.3g}·C^(-{alpha:.3f})   R2={r2:.3f}   ({len(fr)} pts)")

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
    cc = np.geomspace(min(r["C"] for r in rows), max(r["C"] for r in rows), 200)

    for (k, col) in zip(keys, cols):
        g = sorted(archs[k], key=lambda d: d["C"])
        lab = f"{k[1]} ({k[0]/1e6:.0f}M)"
        ax[0].plot([r["C"] for r in g], [r["loss"] for r in g], "-o", color=col,
                   ms=6, lw=1.4, alpha=0.85, label=lab)
        cg = [r for r in g if np.isfinite(r["cell"])]
        ax[2].plot([r["C"] for r in cg], [r["cell"] for r in cg], "-o", color=col,
                   ms=6, lw=1.4, alpha=0.85, label=lab)
    if len(fr) >= 3:
        ax[0].plot(cc, E + A * cc ** (-alpha), color=INK, lw=2.4, ls="--",
                   label=f"frontier  α={alpha:.2f}\nE={E:.2f}  R²={r2:.3f}")
        ax[0].scatter(fr[:, 0], fr[:, 1], color=INK, s=30, zorder=6)
        ax[1].scatter(fr[:, 0], fr[:, 1] - E, color=ACC, s=55, zorder=4)
        ax[1].plot(cc, A * cc ** (-alpha), color=INK, lw=2.0, ls="--")

    _style(ax[0], "training compute (FLOPs)", "held-out loss",
           "Loss vs compute", "log–log; lines = architectures, dashed = frontier", logy=True)
    ax[0].legend(frameon=False, fontsize=7.5, labelcolor=INK)
    _style(ax[1], "training compute (FLOPs)", "loss − E",
           "Frontier, straightened", f"power law, slope −{alpha:.2f}", logy=True)
    _style(ax[2], "training compute (FLOPs)", "cell-line balanced acc",
           "Downstream probe vs compute", "representation quality scales with compute")
    ax[2].legend(frameon=False, fontsize=7.5, labelcolor=INK)

    fig.suptitle("sub14 single-cell encoder — compute scaling law",
                 color=INK, fontweight="bold", x=0.5, y=1.02, ha="center")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=220, bbox_inches="tight")
    print(f"saved {OUT}.png / .pdf")


if __name__ == "__main__":
    main()
