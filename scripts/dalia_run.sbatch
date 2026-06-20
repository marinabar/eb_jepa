#!/bin/bash
# Generic single-run launcher for ONE GB200 on Dalia. Driven by env vars set at submit
# time (scripts/dalia_collapse_sweep.sh): RUN_NAME, LAMB, optional MAX_STEPS / MAX_MIN
# / USE_PATHWAYS. Single-GPU (no DDP): the vivatech-unaite group has a hard cap of 3
# concurrent GPUs (gres/gpu=3), so 1 GPU/run lets up to 3 runs run at once while
# 4-GPU jobs could never schedule. batch_size is raised to 256 so the single-GPU
# global batch matches the original DDP target (SIGReg quality). Caches off home.
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-unaite
#SBATCH --gres=gpu:b200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
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

# Run from an isolated worktree of the feature branch (the shared eb_jepa-repo main
# worktree has unrelated uncommitted local work — left untouched).
cd "${REPO:-$WORK/eb_jepa-sweep}"
export PYTHONPATH=$PWD

# pathway run: append the pathway-token flags (50 hallmark tokens + membership cache)
EXTRA=""
if [ "${USE_PATHWAYS:-0}" = "1" ]; then
  EXTRA="--data.use_pathways true --model.n_pathways 50 \
         --data.pathway_membership $WORK/eb_jepa-cache/pathway_membership.pt"
fi

echo "=== $(date -u +%T) START $RUN_NAME lamb=$LAMB steps=${MAX_STEPS:-6000} node=$(hostname) ==="
"$WORK/venv-arm/bin/python" \
  -m examples.tahoe_jepa.main run \
  --config examples/tahoe_jepa/cfgs/collapse_sweep.yaml \
  --meta.run_dir "$WORK/runs/collapse/$RUN_NAME" \
  --data.batch_size "${BS:-256}" \
  --loss.lamb "$LAMB" \
  --optim.max_steps "${MAX_STEPS:-6000}" \
  --training.max_minutes "${MAX_MIN:-110}" \
  $EXTRA
echo "=== $(date -u +%T) END $RUN_NAME exit=$? ==="
