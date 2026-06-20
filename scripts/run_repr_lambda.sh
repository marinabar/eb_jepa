#!/usr/bin/env bash
# Representation-quality sweep: longer runs over a WIDE lambda range on a fixed
# medium model (d512/8L), to find how rich a representation we can get (effective
# rank + probe quality) and the lambda trade-off. Readout is set from the CLS test
# (env USE_CLS / READOUT). Logs to wandb group "repr_lambda".
set -u
cd /data/eb_jepa
export PATH="$PATH:/root/.local/bin" GIT_LFS_SKIP_SMUDGE=1 PYTHONPATH=/data/eb_jepa
export WANDB_RUN_GROUP=repr_lambda
CFG=examples/tahoe_jepa/cfgs/scaling_base.yaml
USE_CLS=${USE_CLS:-false}     # CLS didn't help (collapses like meanpool) -> meanpool
READOUT=${READOUT:-meanpool}

# Wider lambda span (incl. very low + high) + longer runs to find the richest rep
# reachable and the collapse trade-off. name  lamb  max_steps max_minutes
RUNS=(
  "lam_0p002  0.002  3000      26"
  "lam_0p01   0.01   3000      26"
  "lam_0p05   0.05   3000      26"
  "lam_0p2    0.2    3000      26"
  "lam_0p5    0.5    3000      26"
)
for r in "${RUNS[@]}"; do
  set -- $r; name=$1; lamb=$2; ms=$3; mm=$4
  echo "=== $(date -u +%T) START $name lamb=$lamb readout=$READOUT ==="
  WANDB_NAME=$name .venv/bin/torchrun --standalone --nproc_per_node=8 \
    -m examples.tahoe_jepa.main run --config "$CFG" \
    --model.d_model 384 --model.n_layers 6 --model.n_heads 6 --model.n_kv_heads 1 \
    --model.use_cls "$USE_CLS" --model.readout "$READOUT" \
    --loss.lamb "$lamb" --optim.max_steps "$ms" --training.max_minutes "$mm" \
    --eval.eval_every 250 \
    --meta.run_dir "/data/runs/repr_lambda/$name" \
    > "/data/runs/repr_lambda_$name.log" 2>&1
  echo "=== $(date -u +%T) END $name exit=$? ==="
done
echo "ALL_REPR_LAMBDA_DONE"
