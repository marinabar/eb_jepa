"""Fit + plot the sub14 scaling law from the wandb 'sub14_scaling' group.

Per rung pulls (params, cumulative trained FLOPs, held-out eval loss, cell_line /
organ probe balanced-accuracy, effective rank), fits eval-loss vs compute
L(C)=E+A·C^(-alpha), and plots the scaling curves vs FLOPs and vs params.

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
OUT = "/data/runs/sub14_scaling/scaling_law.png"


def fetch():
    rows = []
    for r in wandb.Api().runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP}):
        s = r.summary
        p, f = s.get("model/trainable_params"), s.get("flops/cumulative")
        if not (p and f):
            continue
        rows.append(
            dict(
                name=r.name,
                params=float(p),
                flops=float(f),
                eval_loss=float(s.get("eval/loss", float("nan"))),
                cell_line=float(
                    s.get("probe/clf/cell_line_id/balanced_accuracy", float("nan"))
                ),
                organ=float(s.get("probe/clf/organ/balanced_accuracy", float("nan"))),
                eff_rank=float(s.get("repr/effective_rank", float("nan"))),
            )
        )
    rows.sort(key=lambda d: d["params"])
    return rows


def fit_powerlaw(C, L):  # L = E + A*C^-alpha ; grid E, log-log fit of (L-E)
    C, L = np.asarray(C, float), np.asarray(L, float)
    best = (0.0, 0.0, 0.0, -1e9)
    for E in np.linspace(0, 0.99 * np.nanmin(L), 200):
        y, x = np.log(np.clip(L - E, 1e-9, None)), np.log(C)
        a1, a0 = np.polyfit(x, y, 1)
        r2 = 1 - ((y - (a0 + a1 * x)) ** 2).sum() / max(
            ((y - y.mean()) ** 2).sum(), 1e-12
        )
        if r2 > best[3]:
            best = (E, -a1, math.exp(a0), r2)
    return best


def main():
    rows = fetch()
    if len(rows) < 2:
        print(f"only {len(rows)} usable runs")
        return
    print(
        f"{'run':14s} {'params':>9s} {'PFLOP':>9s} {'eval_loss':>9s} {'cell_line':>9s} {'organ':>7s} {'rank':>6s}"
    )
    for r in rows:
        print(
            f"{r['name']:14s} {r['params']/1e6:8.1f}M {r['flops']/1e15:9.2f} "
            f"{r['eval_loss']:9.3f} {r['cell_line']:9.3f} {r['organ']:7.3f} {r['eff_rank']:6.1f}"
        )
    C, L = [r["flops"] for r in rows], [r["eval_loss"] for r in rows]
    if np.isfinite(L).all():
        E, alpha, A, r2 = fit_powerlaw(C, L)
        print(
            f"\nloss vs compute: L(C) = {E:.4f} + {A:.3g}·C^(-{alpha:.3f})   R2={r2:.3f}"
        )

    P = [r["params"] for r in rows]
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    panels = [
        ("eval_loss", "held-out eval loss"),
        ("cell_line", "cell_line bacc"),
        ("organ", "organ bacc"),
    ]
    for j, (key, lab) in enumerate(panels):
        y = [r[key] for r in rows]
        ax[0][j].scatter(C, y, c="crimson", zorder=3)
        ax[0][j].set_xscale("log")
        ax[0][j].set_xlabel("cumulative FLOPs")
        ax[0][j].set_ylabel(lab)
        ax[0][j].grid(alpha=0.3)
        ax[1][j].scatter(P, y, c="navy", zorder=3)
        ax[1][j].set_xscale("log")
        ax[1][j].set_xlabel("params")
        ax[1][j].set_ylabel(lab)
        ax[1][j].grid(alpha=0.3)
    fig.suptitle("sub14 scaling: top = vs FLOPs, bottom = vs params")
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
