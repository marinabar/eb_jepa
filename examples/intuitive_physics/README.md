# Track: Does intuitive physics emerge?

Test whether a video-JEPA flags physically **impossible** events through its
latent prediction energy (`predcost`): feed it matched **plausible vs impossible**
Moving-MNIST clips and ask whether the energy spikes on the impossible ones — a
violation-of-expectation signal — and how that gap grows over training.

## What's provided vs. what you implement

- **Provided:** the procedural stimulus generator (`stimuli.py`) — single bouncing
  digit, with *matched* plausible/impossible pairs (pixel-identical until a known
  violation frame, then teleport / instant reversal / wall pass-through) — and the
  training loop (`main.py`), reusing the eb_jepa video-JEPA core.
- **You implement (`# TODO`):** the per-clip energy probe `clip_energy` in
  `eval.py` — encode a clip, roll the predictor `K` steps in parallel mode, and
  reduce `predcost` per clip. That one function is the heart of the track.

## Run

```bash
# 1) train the video-JEPA on plausible clips
python -m examples.intuitive_physics.main --fname examples/intuitive_physics/cfgs/train.yaml \
    meta.ckpt_dir=checkpoints/intuitive_physics/seed1

# 2) probe the energy gap on held-out matched pairs (after you fill in clip_energy)
python -m examples.intuitive_physics.eval --ckpt checkpoints/intuitive_physics/seed1/latest.pth.tar
```

For multi-seed runs on the cluster, register the example in
`examples/launch_sbatch.py` (`EXAMPLE_CONFIGS`) and launch with
`python -m examples.launch_sbatch --example intuitive_physics ...`.

## What to look for

- **Energy gap:** `E(impossible) - E(plausible)` should be positive; report it and
  the detection **AUROC** per violation type. At random init the gap is ~0.
- **Collapse watch:** if the VICReg `std` term stays high while `pred` → 0, the
  encoder is collapsing and the gap becomes meaningless — fix collapse first.

## Don't stop at "AUROC ≈ 1.0" — controls decide the interpretation

The plausible generator caps per-frame motion, so the easy violations (large
teleports, wall pass-through) are partly just **out-of-distribution motion**. Worth
checking:

- **Trivial baseline:** does a one-line *max centroid displacement* detector match
  your JEPA? (It does on teleport/pass-through; it is at *chance* on an instant
  velocity reversal — that magnitude-matched case is where the JEPA earns its keep.)
- **Plausible-but-OOD control:** feed *physically legal* but unusually fast clips —
  if the energy flags them too, it is partly a novelty detector, not a physics
  checker.
- **Latent vs. pixels / generative:** is the gap larger in the JEPA latent than for
  a pixel-prediction model?
- **Regularization:** sweep the anti-collapse strength and watch the
  representation's effective rank (participation ratio), not just per-dim variance.

## Extend the stimuli

`stimuli.py` ships the three guide violations. Natural additions: a momentum
violation (instant speed change), a freeze (inertia), an appearance-only control
(swap the digit mid-flight), severity sweeps (dose-response), and multi-digit
scenes (is the surprise object-centric?).
