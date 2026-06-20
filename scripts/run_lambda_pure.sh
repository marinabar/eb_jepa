#!/usr/bin/env bash
# Pure-SIGReg lambda sweep (NO VICReg) WITH embed_norm=ln (normalize-each-component-
# then-sum, so gene identity isn't swamped by the count term). Small/fast arch
# (d256/4L). Vary only loss.lamb to find the lambda maximising repr/effective_rank.
# eval every 200. Logs to wandb group "lambda_ln".
set -u
cd /data/eb_jepa
export PATH="$PATH:/root/.local/bin" GIT_LFS_SKIP_SMUDGE=1 PYTHONPATH=/data/eb_jepa
export WANDB_RUN_GROUP=lambda_ln
CFG=examples/tahoe_jepa/cfgs/scaling_base.yaml
COMMON="--model.d_model 256 --model.n_layers 4 --model.n_heads 4 --model.n_kv_heads 1 --model.embed_norm ln --optim.max_steps 800 --training.max_minutes 10 --eval.eval_every 200 --eval.eval_cells 512"

for lamb in 0.02 0.1 0.3 0.6; do
  name="ln_lam${lamb/./p}"
  echo "=== $(date -u +%T) START $name lamb=$lamb ==="
  WANDB_NAME=$name .venv/bin/torchrun --standalone --nproc_per_node=8 \
    -m examples.tahoe_jepa.main run --config "$CFG" $COMMON \
    --loss.lamb "$lamb" --meta.run_dir "/data/runs/lambda_ln/$name" \
    > "/data/runs/lambda_ln_$name.log" 2>&1
  echo "=== $(date -u +%T) END $name exit=$? ==="
done
echo "ALL_LAMBDA_LN_DONE"
