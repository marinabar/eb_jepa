#!/usr/bin/env bash
# Fast head-to-head of the anti-collapse fixes (d384/6L, 750 steps). Compare TRAINED
# repr/effective_rank + cell_line probe vs the collapsing baseline (rank ~2.3, cell
# line at chance). cand1 = VICReg variance/cov on the pre-projection rep (most
# direct); cand3 = embed_norm=ln (rebalance gene-token components).
set -u
cd /data/eb_jepa
export PATH="$PATH:/root/.local/bin" GIT_LFS_SKIP_SMUDGE=1 PYTHONPATH=/data/eb_jepa
export WANDB_RUN_GROUP=fixtest
CFG=examples/tahoe_jepa/cfgs/scaling_base.yaml
COMMON="--model.d_model 384 --model.n_layers 6 --model.n_heads 6 --model.n_kv_heads 1 --optim.max_steps 750 --training.max_minutes 12 --eval.eval_every 150"

run() {
  local name="$1"; shift
  echo "=== $(date -u +%T) START $name : $* ==="
  WANDB_NAME=$name .venv/bin/torchrun --standalone --nproc_per_node=8 \
    -m examples.tahoe_jepa.main run --config "$CFG" $COMMON "$@" \
    --meta.run_dir "/data/runs/fixtest/$name" \
    > "/data/runs/fixtest_$name.log" 2>&1
  echo "=== $(date -u +%T) END $name exit=$? ==="
}

run var1          --loss.repr_var_weight 1.0 --loss.repr_cov_weight 0.04
run var1_embedln  --loss.repr_var_weight 1.0 --loss.repr_cov_weight 0.04 --model.embed_norm ln
run embedln       --model.embed_norm ln
run var5_embedln  --loss.repr_var_weight 5.0 --loss.repr_cov_weight 0.2 --model.embed_norm ln
echo "ALL_FIXTEST_DONE"
