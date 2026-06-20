"""Two standalone scaling figures for the sub14 sweep (group sub14_law):

  A. loss vs compute     — one point per run (converged loss at its compute budget),
                           coloured by model size, + power-law frontier fit.
  B. probe vs compute    — downstream cell-line balanced accuracy, with the
                           best-achievable (upper-envelope) frontier, coloured by size.

    PYTHONPATH=/data/eb_jepa .venv/bin/python scripts/fit_sub14_panels.py
"""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

ENTITY, PROJECT, GROUP = "unaite", "hacktheworld", "sub14_law"
OUTL, OUTP = "/data/runs/sub14_law/scaling_loss", "/data/runs/sub14_law/scaling_probe"
INK, MUTED, GRID = "#1d2433", "#7a8699", "#e9edf2"
FK, LK = "flops/cumulative", "loss"
CLK = "probe/clf/cell_line_id/balanced_accuracy"
SMOOTH = 9


def fetch():
    runs = []
    for r in wandb.Api().runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP}):
        if r.state == "running":
            continue
        h = r.history(keys=[FK, LK, CLK], samples=8000, pandas=True)
        if h is None or h.empty or FK not in h or LK not in h:
            continue
        d = h[[FK, LK]].dropna().sort_values(FK)
        if len(d) < 3:
            continue
        s = d[LK].rolling(SMOOTH, min_periods=1, center=True).mean().to_numpy(float)
        pe = h[[FK, CLK]].dropna().sort_values(FK) if CLK in h else None
        runs.append(dict(
            params=float(r.summary.get("model/trainable_params", 0.0)),
            Cend=float(d[FK].to_numpy()[-1]), Lend=float(np.min(s[-3:])),
            pC=(pe[FK].to_numpy(float) if pe is not None else np.array([])),
            pV=(pe[CLK].to_numpy(float) if pe is not None else np.array([])),
        ))
    return runs


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


def axstyle(ax, xlab, ylab, title, sub, logy):
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


def cbar(fig, ax, sc, label):
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label, color=MUTED)
    cb.ax.tick_params(colors=MUTED)


def fig_loss(runs):
    C = np.array([r["Cend"] for r in runs])
    L = np.array([r["Lend"] for r in runs])
    P = np.array([r["params"] for r in runs])
    o = np.argsort(C)
    Cs, Ls = C[o], L[o]
    fr, best = [], math.inf
    for c, l in zip(Cs, Ls):
        if l < best:
            best = l
            fr.append((c, l))
    fr = np.array(fr)
    E, alpha, A, r2 = fit_powerlaw(fr[:, 0], fr[:, 1])
    fig, ax = plt.subplots(figsize=(8, 6.2))
    cc = np.geomspace(C.min(), C.max(), 200)
    ax.plot(cc, E + A * cc ** (-alpha), color=INK, lw=2.4, ls="--", zorder=3,
            label=f"frontier  L = {E:.2f} + {A:.2g}·C^(−{alpha:.2f})\nR²={r2:.3f}")
    sc = ax.scatter(C, L, c=P / 1e6, cmap="viridis", s=85, zorder=4, edgecolor="white",
                    lw=0.7, norm=plt.cm.colors.LogNorm(P.min() / 1e6, P.max() / 1e6))
    axstyle(ax, "training compute (FLOPs)", "converged loss", "loss vs compute",
            "one point per run (LR-decayed); dashed = power-law frontier", logy=True)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK)
    cbar(fig, ax, sc, "model size (M params)")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"{OUTL}.{e}", dpi=220, bbox_inches="tight")
    print(f"loss: E={E:.3f} alpha={alpha:.3f} R2={r2:.3f}")


def _logistic(x, lo, hi, k, x0):  # S-curve in log10(compute): floor -> rise -> ceiling
    return lo + (hi - lo) / (1.0 + np.exp(-k * (x - x0)))


def fig_probe(runs):
    allC = np.concatenate([r["pC"] for r in runs if len(r["pC"])])
    allV = np.concatenate([r["pV"] for r in runs if len(r["pC"])])
    allP = np.concatenate([np.full(len(r["pC"]), r["params"]) for r in runs if len(r["pC"])])
    bins = np.geomspace(allC.min(), allC.max(), 40)
    idx = np.digitize(allC, bins)
    fc, fv, fp, best = [], [], [], -math.inf
    for b in range(1, len(bins)):
        m = idx == b
        if not m.any():
            continue
        j = np.argmax(allV[m])
        if allV[m][j] > best:  # running max = best-achievable frontier
            best = float(allV[m][j])
            bp = float(allP[m][j])
        fc.append(math.sqrt(bins[b - 1] * bins[b])); fv.append(best); fp.append(bp)
    fc, fv, fp = np.array(fc), np.array(fv), np.array(fp)
    from scipy.optimize import curve_fit

    x = np.log10(fc)
    p0 = [float(fv.min()), float(min(1.0, fv.max() + 0.03)), 2.0, float(np.median(x))]
    popt, _ = curve_fit(_logistic, x, fv, p0=p0, maxfev=40000,
                        bounds=([-0.2, 0.3, 0.2, x.min()], [0.4, 1.0, 12.0, x.max()]))
    r2 = 1 - ((fv - _logistic(x, *popt)) ** 2).sum() / max(((fv - fv.mean()) ** 2).sum(), 1e-12)
    fig, ax = plt.subplots(figsize=(8, 6.2))
    cc = np.geomspace(fc.min(), fc.max(), 200)
    ax.plot(cc, _logistic(np.log10(cc), *popt), color=INK, lw=2.4, ls="--", zorder=3,
            label=f"logistic fit  ceiling={popt[1]:.2f}\nR²={r2:.3f}")
    sc = ax.scatter(fc, fv, c=fp / 1e6, cmap="viridis", s=85, zorder=4,
                    edgecolor="white", lw=0.7,
                    norm=plt.cm.colors.LogNorm(allP.min() / 1e6, allP.max() / 1e6))
    axstyle(ax, "training compute (FLOPs)", "cell-line balanced accuracy",
            "downstream probe vs compute",
            "best-achievable frontier; colour = model size", logy=False)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK, loc="upper left")
    cbar(fig, ax, sc, "model size (M params)")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"{OUTP}.{e}", dpi=220, bbox_inches="tight")
    print(f"probe: logistic ceiling={popt[1]:.3f} R2={r2:.3f} max={fv.max():.3f}")


def main():
    runs = fetch()
    print(f"{len(runs)} runs")
    fig_loss(runs)
    fig_probe(runs)
    print("saved scaling_loss / scaling_probe (.png/.pdf)")


if __name__ == "__main__":
    main()
