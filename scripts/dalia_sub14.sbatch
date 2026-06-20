#!/bin/bash
# Subliminal 1.4 (faithful port) — multi-GPU GB200 run on Dalia (single node, DDP
# via torchrun). Submit from the repo root on the login node (dalia2):
#
#   sbatch --job-name=sub14 scripts/dalia_sub14.sbatch
#   # override scale / steps via env:
#   NGPU=4 MAX_STEPS=8000 RUN_NAME=sub14_d768 sbatch scripts/dalia_sub14.sbatch
#
# NOTE on GPUs: dalia_full.yaml uses 4x GB200 on one node. The vivatech-unaite
# group has a concurrent-GPU cap — if a 4-GPU job stays PENDING ("ReqNodeNotAvail"
# / QOSGrpGpuLimit), drop to NGPU=3 (or fewer): override BOTH the --gres line below
# (sbatch --gres=gpu:b200:3) and NGPU. DDP works at any world size; batch_size is
# per-GPU so the SIGReg global batch = batch_size * NGPU.
#SBATCH --partition=defq
#SBATCH --reservation=Vivatech
#SBATCH --account=vivatech-unaite
#SBATCH --gres=gpu:b200:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=10:00:00
#SBATCH --output=/lustre/work/vivatech-unaite/ljung/runs/sub14/%x-%j.out

set -euo pipefail
WORK=/lustre/work/vivatech-unaite/ljung
NGPU=${NGPU:-4}
RUN_NAME=${RUN_NAME:-sub14}

export XDG_CACHE_HOME=$WORK/.cache UV_CACHE_DIR=$WORK/.uv-cache
export HF_HOME=$WORK/.hf XDG_DATA_HOME=$WORK/.local-share
mkdir -p "$XDG_CACHE_HOME" "$WORK/runs/sub14/$RUN_NAME"

export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-sub14}
export WANDB_NAME=$RUN_NAME
export WANDB_ENTITY=unaite
# avoid tokenizer / thread oversubscription with many dataloader workers
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

cd "${REPO:-$WORK/eb_jepa}"
export PYTHONPATH=$PWD

echo "=== $(date -u +%T) START $RUN_NAME ngpu=$NGPU steps=${MAX_STEPS:-8000} node=$(hostname) ==="
"$WORK/venv-arm/bin/python" -m torch.distributed.run \
  --standalone --nproc_per_node="$NGPU" \
  -m examples.tahoe_jepa.sub14_main run \
  --config examples/tahoe_jepa/cfgs/sub14.yaml \
  --meta.run_dir "$WORK/runs/sub14/$RUN_NAME" \
  --data.batch_size "${BS:-32}" \
  --optim.max_steps "${MAX_STEPS:-8000}" \
  --training.max_minutes "${MAX_MIN:-580}"
echo "=== $(date -u +%T) END $RUN_NAME exit=$? ==="
