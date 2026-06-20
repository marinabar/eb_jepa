#!/bin/bash
# Generic single-run launcher for one 4x GB200 node on Dalia. Driven by env vars set
# at submit time (scripts/dalia_collapse_sweep.sh): RUN_NAME, LAMB, optional
# MAX_STEPS / MAX_MIN / USE_PATHWAYS. One job == one node == one DDP run (torchrun
# --nproc_per_node=4). Caches are redirected off the tiny home (inode quota).
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech
#SBATCH --gres=gpu:b200:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/work/vivatech-unaite/ljung/runs/collapse/%x-%j.out

set -euo pipefail
WORK=/lustre/work/vivatech-unaite/ljung
export XDG_CACHE_HOME=$WORK/.cache UV_CACHE_DIR=$WORK/.uv-cache
export HF_HOME=$WORK/.hf XDG_DATA_HOME=$WORK/.local-share
mkdir -p "$XDG_CACHE_HOME" "$WORK/runs/collapse/$RUN_NAME"

export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-collapse}
export WANDB_NAME=$RUN_NAME
export WANDB_ENTITY=unaite

cd "$WORK/eb_jepa-repo"
export PYTHONPATH=$PWD

# pathway run: append the pathway-token flags (50 hallmark tokens + membership cache)
EXTRA=""
if [ "${USE_PATHWAYS:-0}" = "1" ]; then
  EXTRA="--data.use_pathways true --model.n_pathways 50 \
         --data.pathway_membership $WORK/eb_jepa-cache/pathway_membership.pt"
fi

echo "=== $(date -u +%T) START $RUN_NAME lamb=$LAMB steps=${MAX_STEPS:-6000} node=$(hostname) ==="
"$WORK/venv-arm/bin/torchrun" --standalone --nproc_per_node=4 \
  -m examples.tahoe_jepa.main run \
  --config examples/tahoe_jepa/cfgs/collapse_sweep.yaml \
  --meta.run_dir "$WORK/runs/collapse/$RUN_NAME" \
  --loss.lamb "$LAMB" \
  --optim.max_steps "${MAX_STEPS:-6000}" \
  --training.max_minutes "${MAX_MIN:-110}" \
  $EXTRA
echo "=== $(date -u +%T) END $RUN_NAME exit=$? ==="
