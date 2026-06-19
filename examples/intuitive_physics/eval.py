"""Intuitive physics — violation-of-expectation PROBE (the exercise).

Loads a trained checkpoint, builds matched plausible/impossible clips, and compares
the video-JEPA's latent prediction energy (``predcost``) between them. The ONE
thing you implement is ``clip_energy`` (a ``# TODO``) — the per-clip latent
prediction energy. Everything else (stimuli, loading, AUROC, reporting) is given.

Run:  python -m examples.intuitive_physics.eval --ckpt <.../latest.pth.tar>
"""
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from eb_jepa.training_utils import load_checkpoint, load_config, setup_device
from examples.intuitive_physics.main import build_jepa
from examples.intuitive_physics.stimuli import VIOLATIONS, build_probe_pairs


@torch.no_grad()
def clip_energy(jepa, clips, nsteps, device, batch_size=32):
    """TODO: return a PER-CLIP latent prediction energy, shape ``[N]``.

    This is the heart of the track. For a batch of clips ``[B, 1, T, H, W]``:
      1. encode the ground truth:  ``state = jepa.encoder(clips)``  -> ``[B, D, T, H, W]``
      2. roll the predictor forward K=``nsteps`` steps in parallel mode, refeeding
         the ground-truth context each step. The exact loop is in
         ``eb_jepa.jepa.JEPA.unroll`` (``unroll_mode="parallel"``): predict, drop
         the last frame, ``torch.cat`` the first ``jepa.predictor.context_length``
         ground-truth latents back on the left.
      3. accumulate the prediction cost ``jepa.predcost(state, pred)`` between the
         (target-encoded) ground truth and the rollout, averaged over the K steps,
         but reduced PER CLIP (mean over D, T, H, W) so you can compare individual
         plausible vs impossible clips -- not the batch-mean scalar.

    Sanity check: ``jepa.unroll(clips, actions=None, nsteps=nsteps,
    unroll_mode="parallel", compute_loss=True)`` returns the batch-mean prediction
    loss as the 5th element of its loss tuple; your per-clip energies should
    average to it.

    Return a CPU tensor of shape ``[N]`` (one energy per clip).
    """
    raise NotImplementedError("TODO: implement the per-clip predcost energy (see docstring)")


def _auroc(e_pla, e_imp):
    y = np.r_[np.zeros(len(e_pla)), np.ones(len(e_imp))]
    return float(roc_auc_score(y, np.r_[e_pla.numpy(), e_imp.numpy()]))


def main():
    if "--ckpt" not in sys.argv:
        raise SystemExit("usage: python -m examples.intuitive_physics.eval --ckpt <path> "
                         "[--fname examples/intuitive_physics/cfgs/eval.yaml]")
    ckpt = sys.argv[sys.argv.index("--ckpt") + 1]
    fname = (sys.argv[sys.argv.index("--fname") + 1] if "--fname" in sys.argv
             else "examples/intuitive_physics/cfgs/eval.yaml")
    cfg = load_config(fname)
    device = setup_device(cfg.meta.get("device", "auto"))

    jepa = build_jepa(cfg, device)
    load_checkpoint(ckpt, jepa, device=device)
    jepa.eval()

    pairs = build_probe_pairs(n_pairs=cfg.probe.n_pairs, T=cfg.data.T, seed=cfg.probe.seed)
    nsteps = cfg.model.steps
    print(f"{'violation':12s} {'energy gap':>12s} {'AUROC':>7s}")
    results = {}
    for v in VIOLATIONS:
        e_pla = clip_energy(jepa, pairs[v]["plausible"], nsteps, device)
        e_imp = clip_energy(jepa, pairs[v]["impossible"], nsteps, device)
        gap = float(e_imp.mean() - e_pla.mean())
        results[v] = {"gap": gap, "auroc": _auroc(e_pla, e_imp)}
        print(f"{v:12s} {gap:>12.3e} {results[v]['auroc']:>7.3f}")
    return results


if __name__ == "__main__":
    main()
