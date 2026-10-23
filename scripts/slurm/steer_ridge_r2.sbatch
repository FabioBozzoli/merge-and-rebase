#!/bin/bash

#SBATCH --job-name=steer_ridge_r2
#SBATCH --chdir=/homes/fbozzoli/merge-and-rebase
#SBATCH --output=.log/steer_ridge_r2/job_%j.out
#SBATCH --error=.log/steer_ridge_r2/job_%j.err
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=all_serial

# R^2 delle ridge dello Stage 2 di steer per granularita' (globale / 13 blocchi residual / 50 siti linear)
# sulla stessa cache di feature, ViT-B/16 datacomp -> ViT-B/16 laion2b. Solo CPU, legge la cache.
#   mkdir -p .log/steer_ridge_r2 && sbatch scripts/slurm/steer_ridge_r2.sbatch <TASK> <RESIDUAL_CACHE> <LINEAR_CACHE>

set -euo pipefail
TASK="${1:?task}"; RES="${2:?residual cache (.../<task>/linear)}"; LIN="${3:?linear cache (.../<task>/linear_lineargranularity)}"
source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python scripts/analyze_steer_ridge_r2.py --task "$TASK" --residual-cache "$RES" --linear-cache "$LIN" \
  --out "logs/steer_ridge_r2/${TASK}_vitb_vitb.json"
