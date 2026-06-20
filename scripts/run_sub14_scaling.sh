#!/usr/bin/env bash
# Scaling-law sweep for the sub14 (Subliminal-1.4) recipe. The model is single-GPU,
# so we run the WHOLE size ladder IN PARALLEL — one size per GPU — for a SHARED
# wall-clock budget (MINUTES). All sizes train simultaneously and finish together;
# each logs cumulative trained-parts FLOPs + params + probes -> a compute/size
# scaling law in one wall-clock window. Recipe fixed (cfgs/sub14_scaling.yaml); only
# d_model/n_heads/n_layers/d_ff vary (head_dim 64, d_ff = 4*d_model).
#
# Usage:  MINUTES=35 bash scripts/run_sub14_scaling.sh   (run in tmux on the 8-GPU node)
set -u
cd /data/eb_jepa
export PATH="$PATH:/root/.local/bin" PYTHONPATH=/data/eb_jepa GIT_LFS_SKIP_SMUDGE=1
export WANDB_RUN_GROUP=sub14_scaling
CFG=examples/tahoe_jepa/cfgs/sub14_scaling.yaml
MINUTES=${MINUTES:-35}

# name        d_model n_heads n_layers d_ff
LADDER=(
  "s1_d128_l2   128   2   2    512"
  "s2_d192_l3   192   3   3    768"
  "s3_d256_l4   256   4   4    1024"
  "s4_d384_l5   384   6   5    1536"
  "s5_d512_l6   512   8   6    2048"
  "s6_d640_l8   640   10  8    2560"
  "s7_d768_l10  768   12  10   3072"
  "s8_d768_l12  768   12  12   3072"
)

gpu=0
for r in "${LADDER[@]}"; do
  set -- $r; name=$1; d=$2; h=$3; l=$4; ff=$5
  echo "=== $(date -u +%T) GPU$gpu START $name d=$d h=$h l=$l ff=$ff mins=$MINUTES ==="
  CUDA_VISIBLE_DEVICES=$gpu WANDB_NAME=$name .venv/bin/python \
    -m examples.tahoe_jepa.sub14_main run --config "$CFG" \
    --model.d_model "$d" --model.n_heads "$h" --model.n_layers "$l" --model.d_ff "$ff" \
    --training.max_minutes "$MINUTES" \
    --meta.run_dir "/data/runs/sub14_scaling/$name" \
    > "/data/runs/sub14_scaling_$name.log" 2>&1 &
  gpu=$((gpu + 1))
done
wait                                   # all 8 sizes train in parallel, same wall-clock
echo "ALL_SUB14_SCALING_DONE $(date -u +%T)"
