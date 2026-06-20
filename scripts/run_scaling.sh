#!/usr/bin/env bash
# Sequential scaling-law sweep on 8x B200. Each rung is a full DDP run at a fixed
# model size, sharing the held-out validation set (seed=0). max_steps sizes the LR
# cosine; max_minutes is a hard deadline-safety cap. Logs to wandb group "scaling".
#
# Usage:  bash scripts/run_scaling.sh   (run inside tmux on the GPU node)
set -u
cd /data/eb_jepa
export PATH="$PATH:/root/.local/bin" GIT_LFS_SKIP_SMUDGE=1 PYTHONPATH=/data/eb_jepa
export WANDB_RUN_GROUP=scaling
CFG=examples/tahoe_jepa/cfgs/scaling_base.yaml

# name            d     L   heads kv  max_steps max_minutes
RUNS=(
  "s1_d256_l4    256   4   4     1   600       12"
  "s2_d384_l6    384   6   6     1   700       16"
  "s3_d512_l8    512   8   8     2   800       23"
  "s4_d768_l10   768   10  12    3   900       33"
  "s5_d1024_l12  1024  12  16    4   950       46"
)

for r in "${RUNS[@]}"; do
  set -- $r; name=$1; d=$2; l=$3; h=$4; kv=$5; ms=$6; mm=$7
  echo "=== $(date -u +%T) START $name  d=$d L=$l heads=$h kv=$kv max_steps=$ms max_minutes=$mm ==="
  WANDB_NAME=$name .venv/bin/torchrun --standalone --nproc_per_node=8 \
    -m examples.tahoe_jepa.main run --config "$CFG" \
    --model.d_model "$d" --model.n_layers "$l" --model.n_heads "$h" --model.n_kv_heads "$kv" \
    --optim.max_steps "$ms" --training.max_minutes "$mm" \
    --meta.run_dir "/data/runs/scaling/$name" \
    > "/data/runs/scaling_$name.log" 2>&1
  echo "=== $(date -u +%T) END $name exit=$? ==="
done
echo "ALL_SCALING_DONE"
