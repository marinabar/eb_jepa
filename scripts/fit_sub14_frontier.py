"""Standalone compute-optimal frontier figure for the sub14 sweep (group sub14_law).

The frontier is the lower envelope of held-out loss over all runs' trajectories. With
the irreducible floor E removed, (loss - E) vs compute is a straight line on log-log
whose slope is the scaling exponent. Points are coloured by the model size that
achieves the frontier at that compute (small at low compute -> large at high compute).

    PYTHONPATH=/data/eb_jepa .venv/bin/python scripts/fit_sub14_frontier.py
"""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

ENTITY, PROJECT, GROUP = "unaite", "hacktheworld", "sub14_law"
OUT = "/data/runs/sub14_law/scaling_frontier"
INK, MUTED, GRID = "#1d2433", "#7a8699", "#e9edf2"
FK, LK = "flops/cumulative", "loss"
SMOOTH = 9


def fetch():
    C, L, P = [], [], []
    for r in wandb.Api().runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP}):
        if r.state == "running":
            continue
        h = r.history(keys=[FK, LK], samples=8000, pandas=True)
        if h is None or h.empty or FK not in h or LK not in h:
            continue
        d = h[[FK, LK]].dropna().sort_values(FK)
        if len(d) < 3:
            continue
        s = d[LK].rolling(SMOOTH, min_periods=1, center=True).mean().to_numpy(float)
        p = float(r.summary.get("model/trainable_params", 0.0))
        C.append(d[FK].to_numpy(float)); L.append(s); P.append(np.full(len(s), p))
    return np.concatenate(C), np.concatenate(L), np.concatenate(P)


def frontier(C, L, P, nbins=26):
    bins = np.geomspace(C.min(), C.max(), nbins)
    idx = np.digitize(C, bins)
    out, best = [], math.inf
    for b in range(1, len(bins)):
        m = idx == b
        if not m.any():
            continue
        j = np.argmin(L[m])
        lo = L[m][j]
        if lo < best:  # strict lower envelope = true frontier records
            best = lo
            out.append((math.sqrt(bins[b - 1] * bins[b]), lo, P[m][j]))
    return np.array(out)


def fit_powerlaw(C, L):
    best = (0.0, 0.0, 0.0, -1e9)
    for E in np.linspace(0, 0.98 * L.min(), 300):
        y, x = np.log(L - E), np.log(C)
        a1, a0 = np.polyfit(x, y, 1)
        r2 = 1 - ((y - (a0 + a1 * x)) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
        if r2 > best[3]:
            best = (E, -a1, math.exp(a0), r2)
    return best


def main():
    C, L, P = fetch()
    fr = frontier(C, L, P)
    fc, fl, fp = fr[:, 0], fr[:, 1], fr[:, 2]
    E, alpha, A, r2 = fit_powerlaw(fc, fl)
    print(f"frontier L(C)=E+A·C^-a : E={E:.3f} A={A:.3g} alpha={alpha:.3f} R2={r2:.3f} ({len(fr)} pts)")

    fig, ax = plt.subplots(figsize=(8, 6.2))
    cc = np.geomspace(fc.min(), fc.max(), 200)
    ax.plot(cc, A * cc ** (-alpha), color=INK, lw=2.4, ls="--", zorder=3,
            label=f"power law  slope −{alpha:.2f}\nE={E:.2f}  R²={r2:.3f}")
    sc = ax.scatter(fc, fl - E, c=fp / 1e6, cmap="viridis", s=85, zorder=4,
                    edgecolor="white", lw=0.7,
                    norm=plt.cm.colors.LogNorm(fp.min() / 1e6, fp.max() / 1e6))
    ax.set_xscale("log"); ax.set_yscale("log")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c7cfdb")
    ax.grid(True, which="both", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED)
    ax.set_xlabel("training compute (FLOPs)", color=MUTED)
    ax.set_ylabel("loss − E  (irreducible removed)", color=MUTED)
    ax.set_title("sub14 — compute-optimal frontier", color=INK, fontweight="bold",
                 loc="left", pad=18)
    ax.text(0, 1.015, "loss minus irreducible floor vs compute; a power law is a line",
            transform=ax.transAxes, color=MUTED, fontsize=8.5)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("model size at frontier (M params)", color=MUTED)
    cb.ax.tick_params(colors=MUTED)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"{OUT}.{e}", dpi=220, bbox_inches="tight")
    print(f"saved {OUT}.png / .pdf")


if __name__ == "__main__":
    main()
