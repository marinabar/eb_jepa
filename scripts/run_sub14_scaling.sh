#!/usr/bin/env bash
# Compute-spread scaling sweep for the sub14 (Subliminal-1.4) recipe. Each run has its
# OWN compute budget (max_steps) — small budgets and large budgets — and uses the
# LeJEPA warmup+cosine LR schedule with the cosine horizon = that run's max_steps (LR
# fully decays by the end -> clean loss-vs-compute points). Recipe fixed
# (cfgs/sub14_scaling.yaml); d_model/n_heads/n_layers/d_ff + max_steps vary per run.
#
# Single-GPU model -> an 8-worker flock job-queue packs the runs onto the 8 GPUs
# dynamically (a GPU grabs the next run when it frees up). Queue is longest-first so
# the big runs start immediately. Each run saves its FINAL checkpoint (encoder_final.pt).
#
# Usage:  bash scripts/run_sub14_scaling.sh    (in tmux on the 8-GPU node)
set -u
cd /data/eb_jepa
export PATH="$PATH:/root/.local/bin" PYTHONPATH=/data/eb_jepa GIT_LFS_SKIP_SMUDGE=1
export WANDB_RUN_GROUP=sub14_law
CFG=examples/tahoe_jepa/cfgs/sub14_scaling.yaml
RUNDIR=/data/runs/sub14_law
mkdir -p "$RUNDIR"
export QUEUE="$RUNDIR/queue.txt" LOCK="$RUNDIR/queue.lock"

# name        d   h  l  ff   steps   (longest-first; compute ~ params*steps spans 1->650 PF)
cat > "$QUEUE" <<'Q'
e3_d768_l12 768 12 12 3072 12000
d3_d512_l8  512 8  8  2048 24000
c3_d320_l5  320 5  5  1280 24000
e2_d768_l12 768 12 12 3072 4000
a4_d128_l2  128 2  2  512  96000
b3_d192_l3  192 3  3  768  48000
d2_d512_l8  512 8  8  2048 6000
c2_d320_l5  320 5  5  1280 8000
a3_d128_l2  128 2  2  512  32000
e1_d768_l12 768 12 12 3072 1200
b2_d192_l3  192 3  3  768  12000
d1_d512_l8  512 8  8  2048 1500
c1_d320_l5  320 5  5  1280 2000
a2_d128_l2  128 2  2  512  8000
b1_d192_l3  192 3  3  768  3000
a1_d128_l2  128 2  2  512  2000
Q

pop() {  # atomically print + remove the first queue line
  flock "$LOCK" -c '
    line=$(head -n1 "$QUEUE")
    [ -n "$line" ] && sed -i 1d "$QUEUE"
    printf "%s" "$line"'
}

worker() {
  local gpu=$1 line name d h l ff steps
  while :; do
    line=$(pop); [ -z "$line" ] && break
    set -- $line; name=$1; d=$2; h=$3; l=$4; ff=$5; steps=$6
    echo "$(date -u +%T) GPU$gpu START $name steps=$steps"
    CUDA_VISIBLE_DEVICES=$gpu WANDB_NAME=$name .venv/bin/python \
      -m examples.tahoe_jepa.sub14_main run --config "$CFG" \
      --model.d_model $d --model.n_heads $h --model.n_layers $l --model.d_ff $ff \
      --optim.max_steps $steps --meta.run_dir "$RUNDIR/$name" \
      > "$RUNDIR/log_$name.txt" 2>&1
    echo "$(date -u +%T) GPU$gpu DONE  $name (exit $?)"
  done
}

for g in 0 1 2 3 4 5 6 7; do worker $g & done
wait
echo "ALL_SUB14_LAW_DONE $(date -u +%T)"
