#!/usr/bin/env bash
# Pure-SIGReg lambda sweep (NO VICReg, NO embed_norm -> the base LeJEPA setting),
# vary only loss.lamb to find the lambda that maximises repr/effective_rank.
# d384/6L, short, eval every 200. Logs to wandb group "lambda_pure".
set -u
cd /data/eb_jepa
export PATH="$PATH:/root/.local/bin" GIT_LFS_SKIP_SMUDGE=1 PYTHONPATH=/data/eb_jepa
export WANDB_RUN_GROUP=lambda_pure
CFG=examples/tahoe_jepa/cfgs/scaling_base.yaml
COMMON="--model.d_model 384 --model.n_layers 6 --model.n_heads 6 --model.n_kv_heads 1 --optim.max_steps 800 --training.max_minutes 12 --eval.eval_every 200 --eval.eval_cells 512"

for lamb in 0.02 0.1 0.3 0.6; do
  name="pure_lam${lamb/./p}"
  echo "=== $(date -u +%T) START $name lamb=$lamb ==="
  WANDB_NAME=$name .venv/bin/torchrun --standalone --nproc_per_node=8 \
    -m examples.tahoe_jepa.main run --config "$CFG" $COMMON \
    --loss.lamb "$lamb" --meta.run_dir "/data/runs/lambda_pure/$name" \
    > "/data/runs/lambda_pure_$name.log" 2>&1
  echo "=== $(date -u +%T) END $name exit=$? ==="
done
echo "ALL_LAMBDA_PURE_DONE"
