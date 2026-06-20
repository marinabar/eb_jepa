#!/usr/bin/env bash
# Submit the PURE-LeJEPA collapse / lambda sweep on Dalia. Each run is an independent
# sbatch job (1 node = 4x GB200); SLURM schedules them as nodes free up in the
# Vivatech reservation — no node hogging, "do the most" automatically. Run from the
# repo root on the Dalia login node (dalia2):
#
#   bash scripts/dalia_collapse_sweep.sh
#
# Wide lambda grid (0.001 .. 0.5) probes the full collapse -> over-regularization
# curve; a pathway-token run at the reference lambda exercises the new hallmark
# tokens. Build the membership cache first (one-off):
#   srun --reservation=Vivatech --account=vivatech --partition=defq --gres=gpu:b200:1 \
#        --cpus-per-task=8 --time=00:10:00 \
#        /lustre/work/vivatech-unaite/ljung/venv-arm/bin/python \
#        scripts/build_pathway_membership.py \
#        --gene-metadata /lustre/work/vivatech-unaite/shared/tahoe-100m/metadata/gene_metadata.parquet \
#        --out /lustre/work/vivatech-unaite/ljung/eb_jepa-cache/pathway_membership.pt
set -euo pipefail
SBATCH=scripts/dalia_run.sbatch
mkdir -p /lustre/work/vivatech-unaite/ljung/runs/collapse

LAMBDAS=(0.001 0.005 0.02 0.05 0.1 0.5)
for L in "${LAMBDAS[@]}"; do
  name="lam_${L//./p}"
  echo "submit $name (lamb=$L)"
  sbatch --job-name="$name" --export=ALL,RUN_NAME="$name",LAMB="$L" "$SBATCH"
done

# pathway-token run at the reference lambda (50 hallmark tokens, per-pathway dropout)
echo "submit pathway_lam0p02 (lamb=0.02, pathways on)"
sbatch --job-name=pathway_lam0p02 \
  --export=ALL,RUN_NAME=pathway_lam0p02,LAMB=0.02,USE_PATHWAYS=1 "$SBATCH"

echo "submitted ${#LAMBDAS[@]} lambda runs + 1 pathway run"
squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %R"
