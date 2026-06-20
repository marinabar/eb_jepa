"""Compute scaling-law figure for the sub14 (Subliminal-1.4) recipe.

Each run is trained to its OWN compute budget with the LR fully cosine-decayed, so its
final held-out loss is the converged loss at that compute. We pull the final
(compute, eval-loss) of every run, plot it log-log, and fit a power law
L(C) = E + A·C^(-alpha). The straightened panel (L-E) vs C makes the power law a line.
Downstream cell_line / organ probe-vs-compute panels accompany it. House style.

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

ENTITY, PROJECT, GROUP = "unaite", "hacktheworld", "sub14_law"
OUT = "/data/runs/sub14_law/scaling_law"

INK, MUTED, GRID, ACC = "#1d2433", "#7a8699", "#e9edf2", "#2a6f97"
FK, EK = "flops/cumulative", "eval/loss"
CLK = "probe/clf/cell_line_id/balanced_accuracy"
OGK = "probe/clf/organ/balanced_accuracy"


def fetch():
    rows = []
    for r in wandb.Api().runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP}):
        if r.state != "finished":  # only converged runs (LR fully decayed) on the curve
            continue
        h = r.history(keys=[FK, EK, CLK, OGK], samples=4000, pandas=True)
        if h is None or h.empty or FK not in h:
            continue
        C = float(h[FK].max())

        def last(col):
            if col not in h:
                return float("nan")
            d = h[col].dropna()
            return float(d.iloc[-1]) if len(d) else float("nan")

        rows.append(
            dict(
                name=r.name,
                params=float(r.summary.get("model/trainable_params", float("nan"))),
                C=C,
                loss=last(EK),
                cell=last(CLK),
                organ=last(OGK),
            )
        )
    rows.sort(key=lambda d: d["C"])
    return [r for r in rows if r["C"] > 0 and np.isfinite(r["loss"])]


def fit_powerlaw(C, L):  # L = E + A*C^-alpha ; grid the floor E, log-log fit of (L-E)
    C, L = np.asarray(C, float), np.asarray(L, float)
    best = (0.0, 0.0, 0.0, -1e9)
    for E in np.linspace(0, 0.98 * L.min(), 300):
        y, x = np.log(L - E), np.log(C)
        a1, a0 = np.polyfit(x, y, 1)
        r2 = 1 - ((y - (a0 + a1 * x)) ** 2).sum() / max(
            ((y - y.mean()) ** 2).sum(), 1e-12
        )
        if r2 > best[3]:
            best = (E, -a1, math.exp(a0), r2)
    return best  # E, alpha, A, r2


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
    C = np.array([r["C"] for r in rows])
    L = np.array([r["loss"] for r in rows])
    P = np.array([r["params"] for r in rows])
    print(
        f"{'run':12s} {'params':>8s} {'PFLOP':>9s} {'eval_loss':>9s} {'cell':>6s} {'organ':>6s}"
    )
    for r in rows:
        print(
            f"{r['name']:12s} {r['params']/1e6:7.1f}M {r['C']/1e15:9.2f} "
            f"{r['loss']:9.3f} {r['cell']:6.3f} {r['organ']:6.3f}"
        )
    E, alpha, A, r2 = fit_powerlaw(C, L)
    print(f"\nL(C) = {E:.4f} + {A:.4g}·C^(-{alpha:.3f})   R2={r2:.4f}")

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
    sizes = np.unique(P)
    cmap = {s: plt.cm.viridis(i / max(1, len(sizes) - 1)) for i, s in enumerate(sizes)}
    cc = np.geomspace(C.min(), C.max(), 200)

    # (0) loss vs compute, log-log + power-law fit
    for r in rows:
        ax[0].scatter(
            r["C"],
            r["loss"],
            color=cmap[r["params"]],
            s=55,
            zorder=4,
            edgecolor="white",
            lw=0.6,
        )
    ax[0].plot(
        cc,
        E + A * cc ** (-alpha),
        color=INK,
        lw=2.2,
        ls="--",
        label=f"L = {E:.2f} + {A:.2g}·C^(−{alpha:.2f})\nR²={r2:.3f}",
    )
    _style(
        ax[0],
        "training compute (FLOPs)",
        "held-out loss",
        "Loss vs compute",
        "log–log; dashed = fitted power law",
        logy=True,
    )
    ax[0].legend(frameon=False, fontsize=9, labelcolor=INK)

    # (1) straightened: (L - E) vs C -> a clean line confirms the power law
    ax[1].scatter(C, L - E, color=ACC, s=55, zorder=4, edgecolor="white", lw=0.6)
    ax[1].plot(cc, A * cc ** (-alpha), color=INK, lw=2.0, ls="--")
    _style(
        ax[1],
        "training compute (FLOPs)",
        "loss − E (irreducible removed)",
        "Power law, straightened",
        f"slope −{alpha:.2f} on log–log",
        logy=True,
    )

    # (2) downstream probe vs compute
    for r in rows:
        if np.isfinite(r["cell"]):
            ax[2].scatter(
                r["C"],
                r["cell"],
                color=cmap[r["params"]],
                s=50,
                zorder=4,
                edgecolor="white",
                lw=0.6,
            )
    _style(
        ax[2],
        "training compute (FLOPs)",
        "cell-line balanced acc",
        "Downstream probe vs compute",
        "representation quality scales with compute",
    )

    sm = plt.cm.ScalarMappable(
        cmap="viridis", norm=plt.Normalize(P.min() / 1e6, P.max() / 1e6)
    )
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("params (M)", color=MUTED)
    fig.suptitle(
        "sub14 single-cell encoder — compute scaling law",
        color=INK,
        fontweight="bold",
        x=0.5,
        y=1.02,
        ha="center",
    )
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=220, bbox_inches="tight")
    print(f"saved {OUT}.png / .pdf")


if __name__ == "__main__":
    main()
