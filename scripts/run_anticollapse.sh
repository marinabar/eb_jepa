#!/usr/bin/env bash
# Anti-collapse candidate test: same model (d384/6L, lambda=0.02, 1500 steps), vary
# the fix, compare repr/effective_rank + cell_line probe vs the collapsing baseline.
#   base               : current (BN projector, no repr reg) -> collapses (~rank 2)
#   proj_none / proj_ln: candidate 2 (projector without BatchNorm / with LayerNorm)
#   repr_var           : candidate 1 (VICReg variance+cov on the pre-projection rep)
#   repr_var_projnone  : 1 + 2 combined
set -u
cd /data/eb_jepa
export PATH="$PATH:/root/.local/bin" GIT_LFS_SKIP_SMUDGE=1 PYTHONPATH=/data/eb_jepa
export WANDB_RUN_GROUP=anticollapse
CFG=examples/tahoe_jepa/cfgs/scaling_base.yaml
COMMON="--model.d_model 384 --model.n_layers 6 --model.n_heads 6 --model.n_kv_heads 1 --optim.max_steps 1500 --training.max_minutes 20 --eval.eval_every 250"

run() {
  local name="$1"; shift
  echo "=== $(date -u +%T) START $name : $* ==="
  WANDB_NAME=$name .venv/bin/torchrun --standalone --nproc_per_node=8 \
    -m examples.tahoe_jepa.main run --config "$CFG" $COMMON "$@" \
    --meta.run_dir "/data/runs/anticollapse/$name" \
    > "/data/runs/anticollapse_$name.log" 2>&1
  echo "=== $(date -u +%T) END $name exit=$? ==="
}

run base
run proj_none          --model.proj_norm none
run proj_ln            --model.proj_norm ln
run repr_var           --loss.repr_var_weight 1.0 --loss.repr_cov_weight 0.04
run repr_var_projnone  --loss.repr_var_weight 1.0 --loss.repr_cov_weight 0.04 --model.proj_norm none
echo "ALL_ANTICOLLAPSE_DONE"
