# Encoder scaling laws — Tahoe-100M LeJEPA

**Question.** A recent paper reports an *absence* of scaling laws for single-cell
encoders. Does our multi-view LeJEPA setup, with measured data diversity and a clean
trained-parts FLOP budget, reveal a scaling regime where they found none?

## Design

A ladder of model sizes trained on 8×B200, **everything fixed except model size**:

- **Shared held-out validation.** The held-out split is seeded (`seed=0`) with a
  fixed `eval_cells`, so the eval cells are *identical* across every rung. y-axis =
  held-out **LeJEPA eval loss** (same loss as training, on the held-out V-view batch)
  + detached probe metrics. This is the fixed validation set shared across all scales.
- **x-axis = cumulative trained-parts FLOPs** (FlopCounterMode on the encoder,
  fwd+bwd, summed across GPUs — logged live to wandb), and model parameter count.
- **Fixed across scales:** data + held-out split, batch (global 256), L=4096, V=4,
  λ=0.02 (LeJEPA reference, constant — varying it would confound scaling),
  count-mode A, grad-checkpoint, optimizer/warmup. Only `d_model`, `n_layers`,
  `n_heads` vary (head_dim=64 throughout).
- Per-rung LR cosine is sized to that rung's `max_steps`; `max_minutes` is a hard
  deadline cap.

| rung | d_model | layers | heads | ~params | max_steps | cap (min) |
|------|---------|--------|-------|---------|-----------|-----------|
| s1   | 256     | 4      | 4     | ~3.5M   | 600       | 12 |
| s2   | 384     | 6      | 6     | ~8M     | 700       | 16 |
| s3   | 512     | 8      | 8     | ~17M    | 800       | 23 |
| s4   | 768     | 10     | 12    | ~43M    | 900       | 33 |
| s5   | 1024    | 12     | 16    | ~97M    | 950       | 46 |

Run: `bash scripts/run_scaling.sh` (sequential, inside tmux). Config:
`examples/tahoe_jepa/cfgs/scaling_base.yaml`.

## GPU-saturation finding (informs the config)

At L=4096/d1024/12L the encoder is **compute-bound**: throughput is ~constant
(~94 cells/s global) across batch size, so larger batch does not raise throughput —
it only raises memory + the SIGReg global-batch quality. grad_checkpoint is required
(bs=16 without it already hit 148 GB; with it bs32→50 GB, bs48→74 GB, bs96→147 GB).
We fix global batch 256 (bs=32/GPU, 50 GB, large margin) across all rungs.

## Fit

For each rung take the final held-out eval loss `L` at cumulative compute `C`, fit
`L(C) = E + A · C^(-α)` (and `L(N)` vs params). Report α, the irreducible term E, and
whether a power-law trend is present (vs the no-scaling baseline).

## Results

_(filled in after the sweep)_
