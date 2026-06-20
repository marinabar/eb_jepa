#!/usr/bin/env bash
# Fast lambda calibration (+ torch.compile smoke) before the scaling sweep. Fixed
# medium model (d384/6L), short budget, vary loss.lamb over low->high to expose the
# collapse trade-off: low lambda -> invariance dominates -> views collapse (low
# repr/effective_rank); high lambda -> SIGReg dominates. Pick the lambda with high
# effective rank AND low held-out eval + probe loss. Logs to wandb group "lambda".
set -u
cd /data/eb_jepa
export PATH="$PATH:/root/.local/bin" GIT_LFS_SKIP_SMUDGE=1 PYTHONPATH=/data/eb_jepa
export WANDB_RUN_GROUP=lambda
CFG=examples/tahoe_jepa/cfgs/scaling_base.yaml

# name        lamb     max_steps max_minutes
RUNS=(
  "lam_0p005  0.005    250       7"
  "lam_0p02   0.02     250       7"
  "lam_0p08   0.08     250       7"
)
for r in "${RUNS[@]}"; do
  set -- $r; name=$1; lamb=$2; ms=$3; mm=$4
  echo "=== $(date -u +%T) START $name lamb=$lamb ==="
  WANDB_NAME=$name .venv/bin/torchrun --standalone --nproc_per_node=8 \
    -m examples.tahoe_jepa.main run --config "$CFG" \
    --model.d_model 384 --model.n_layers 6 --model.n_heads 6 --model.n_kv_heads 1 \
    --model.compile true \
    --loss.lamb "$lamb" --optim.max_steps "$ms" --training.max_minutes "$mm" \
    --eval.eval_every 100 \
    --meta.run_dir "/data/runs/lambda/$name" \
    > "/data/runs/lambda_$name.log" 2>&1
  echo "=== $(date -u +%T) END $name exit=$? ==="
done
echo "ALL_LAMBDA_DONE"
