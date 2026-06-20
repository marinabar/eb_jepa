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
USE_CLS=${USE_CLS:-true}      # set from the CLS-vs-meanpool test
READOUT=${READOUT:-cls}

# name        lamb   max_steps max_minutes
RUNS=(
  "lam_0p01   0.01   2000      34"
  "lam_0p05   0.05   2000      34"
  "lam_0p15   0.15   2000      34"
  "lam_0p40   0.40   2000      34"
  "lam_0p80   0.80   2000      34"
)
for r in "${RUNS[@]}"; do
  set -- $r; name=$1; lamb=$2; ms=$3; mm=$4
  echo "=== $(date -u +%T) START $name lamb=$lamb readout=$READOUT ==="
  WANDB_NAME=$name .venv/bin/torchrun --standalone --nproc_per_node=8 \
    -m examples.tahoe_jepa.main run --config "$CFG" \
    --model.d_model 512 --model.n_layers 8 --model.n_heads 8 --model.n_kv_heads 2 \
    --model.use_cls "$USE_CLS" --model.readout "$READOUT" \
    --loss.lamb "$lamb" --optim.max_steps "$ms" --training.max_minutes "$mm" \
    --eval.eval_every 250 \
    --meta.run_dir "/data/runs/repr_lambda/$name" \
    > "/data/runs/repr_lambda_$name.log" 2>&1
  echo "=== $(date -u +%T) END $name exit=$? ==="
done
echo "ALL_REPR_LAMBDA_DONE"
