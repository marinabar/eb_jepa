"""Fit + plot the encoder scaling law from the wandb 'scaling' group.

For each rung pulls (params, cumulative trained-parts FLOPs, final held-out eval
loss, probe metrics, effective rank), fits  L(C) = E + A * C**(-alpha)  (grid over
the irreducible term E, log-log linear fit for alpha), plots eval-loss vs FLOPs and
vs params, and appends a results table to research/scaling_laws.md.

Run on the GPU node (wandb creds in ~/.netrc):
    PYTHONPATH=/data/eb_jepa .venv/bin/python scripts/fit_scaling.py
"""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

ENTITY, PROJECT, GROUP = "unaite", "hacktheworld", "scaling"
OUT_PNG = "/data/runs/scaling/scaling_law.png"
MD = "research/scaling_laws.md"


def fetch():
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP})
    rows = []
    for r in runs:
        s = r.summary
        params = s.get("model/trainable_params")
        flops = s.get("flops/cumulative")
        loss = s.get("eval/loss")
        if not (params and flops and loss):
            continue
        rows.append(
            {
                "name": r.name,
                "params": float(params),
                "flops": float(flops),
                "eval_loss": float(loss),
                "eval_sigreg": float(s.get("eval/sigreg_loss", float("nan"))),
                "eff_rank": float(s.get("repr/effective_rank", float("nan"))),
                "organ_bacc": float(
                    s.get("probe/clf/organ/balanced_accuracy", float("nan"))
                ),
                "cellline_bacc": float(
                    s.get("probe/clf/cell_line_id/balanced_accuracy", float("nan"))
                ),
            }
        )
    rows.sort(key=lambda d: d["params"])
    return rows


def fit_powerlaw(C, L):
    """L = E + A C^-alpha. Grid E in [0, min(L)); log-log fit of (L-E) vs C for alpha.
    Returns (E, alpha, A, r2)."""
    C, L = np.asarray(C, float), np.asarray(L, float)
    best = (0.0, 0.0, 0.0, -1e9)
    for E in np.linspace(0, 0.99 * L.min(), 200):
        y = np.log(L - E)
        x = np.log(C)
        a1, a0 = np.polyfit(x, y, 1)  # y = a0 + a1 x  -> alpha=-a1, A=exp(a0)
        pred = a0 + a1 * x
        ss = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
        if ss > best[3]:
            best = (E, -a1, math.exp(a0), ss)
    return best


def main():
    rows = fetch()
    if len(rows) < 2:
        print(f"only {len(rows)} usable runs; need >=2"); return
    C = [r["flops"] for r in rows]
    L = [r["eval_loss"] for r in rows]
    E, alpha, A, r2 = fit_powerlaw(C, L)

    print(f"{'run':14s} {'params':>10s} {'PFLOP':>9s} {'eval_loss':>9s} {'eff_rank':>8s} {'organ_bacc':>10s}")
    for r in rows:
        print(f"{r['name']:14s} {r['params']/1e6:9.1f}M {r['flops']/1e15:9.2f} "
              f"{r['eval_loss']:9.4f} {r['eff_rank']:8.1f} {r['organ_bacc']:10.3f}")
    print(f"\nfit  L(C) = {E:.4f} + {A:.3g}*C^(-{alpha:.3f})   (R2={r2:.3f})")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(C, L, c="crimson", zorder=3)
    cc = np.logspace(math.log10(min(C)), math.log10(max(C)), 100)
    ax[0].plot(cc, E + A * cc ** (-alpha), "k--", lw=1,
               label=f"L={E:.3f}+{A:.2g}·C^-{alpha:.3f}\nR²={r2:.3f}")
    ax[0].set_xscale("log"); ax[0].set_xlabel("cumulative trained FLOPs")
    ax[0].set_ylabel("held-out eval loss"); ax[0].set_title("loss vs compute")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    P = [r["params"] for r in rows]
    ax[1].scatter(P, L, c="navy", zorder=3)
    ax[1].set_xscale("log"); ax[1].set_xlabel("trainable params")
    ax[1].set_ylabel("held-out eval loss"); ax[1].set_title("loss vs params")
    ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=130)
    print(f"saved {OUT_PNG}")

    lines = ["", "### Results (auto)", "",
             "| run | params | PFLOP | eval_loss | eff_rank | organ b-acc | cellline b-acc |",
             "|-----|--------|-------|-----------|----------|-------------|----------------|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['params']/1e6:.1f}M | {r['flops']/1e15:.2f} | "
                     f"{r['eval_loss']:.4f} | {r['eff_rank']:.1f} | {r['organ_bacc']:.3f} | "
                     f"{r['cellline_bacc']:.3f} |")
    lines += ["", f"**Fit:** `L(C) = {E:.4f} + {A:.3g}·C^(-{alpha:.3f})`, R²={r2:.3f}. "
              f"Plot: `{OUT_PNG}`.", ""]
    with open(MD, "a") as f:
        f.write("\n".join(lines))
    print(f"appended results to {MD}")


if __name__ == "__main__":
    main()
