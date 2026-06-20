#!/usr/bin/env bash
# EXPANSION sweep — denser, more exhaustive compute scaling law. Same recipe + same
# wandb group (sub14_law) as run_sub14_scaling.sh, so the fit aggregates everything.
# Fills the architecture gaps (d256/384/448/576/640/704), adds a bigger model (d896,
# ~185M) to extend the size axis, and pushes high-compute budgets to ~1 EFLOP. Each run
# = the LeJEPA warmup+cosine schedule over its own max_steps; saves encoder_final.pt.
# 8-worker flock queue, longest-first.
set -u
cd /data/eb_jepa
export PATH="$PATH:/root/.local/bin" PYTHONPATH=/data/eb_jepa GIT_LFS_SKIP_SMUDGE=1
export WANDB_RUN_GROUP=sub14_law
CFG=examples/tahoe_jepa/cfgs/sub14_scaling.yaml
RUNDIR=/data/runs/sub14_law
mkdir -p "$RUNDIR"
export QUEUE="$RUNDIR/queue_expand.txt" LOCK="$RUNDIR/queue_expand.lock"

# name        d   h  l  ff   steps   (longest-first; new sizes + high-compute extensions)
cat > "$QUEUE" <<'Q'
xe_d768_l12 768 12 12 3072 24000
xd_d512_l8  512 8  8  2048 48000
j3_d640_l10 640 10 10 2560 24000
k3_d704_l11 704 11 11 2816 20000
m2_d896_l14 896 14 14 3584 12000
g3_d384_l6  384 6  6  1536 36000
f3_d256_l4  256 4  4  1024 48000
i3_d576_l9  576 9  9  2304 20000
h3_d448_l7  448 7  7  1792 28000
j2_d640_l10 640 10 10 2560 8000
k2_d704_l11 704 11 11 2816 7000
m1_d896_l14 896 14 14 3584 4000
i2_d576_l9  576 9  9  2304 8000
g2_d384_l6  384 6  6  1536 12000
f2_d256_l4  256 4  4  1024 16000
h2x_d448_l7 448 7  7  1792 10000
j1_d640_l10 640 10 10 2560 2000
k1_d704_l11 704 11 11 2816 1800
i1_d576_l9  576 9  9  2304 2000
f1_d256_l4  256 4  4  1024 4000
g1_d384_l6  384 6  6  1536 3000
h1_d448_l7  448 7  7  1792 2500
Q

pop() {
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
echo "ALL_SUB14_EXPAND_DONE $(date -u +%T)"
