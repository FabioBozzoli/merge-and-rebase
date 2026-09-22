#!/usr/bin/env bash
#
# Run one block_ridge experiment script over every diagnostic run, one Slurm job per run.
#
#   scripts/block_ridge_experiments/submit.sh <script.py> [extra args ...]
#
# e.g.  submit.sh trace_lambda.py
#       submit.sh joint_ridge.py
#       submit.sh alpha_sweep.py --alphas 0,0.2,0.4,0.6,0.8,1
#       submit.sh alpha_sweep.py --alphas 0,0.005,0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2,1 --out-subdir alpha_sweep_fine
#
# The runs are the artifact-saving reruns made by scripts/slurm/submit_block_ridge_diag.py.
# The scripts are CPU-bound; the GPU is requested only because the partition expects one.
# --wrap runs under /bin/sh, so the venv's python is called directly instead of sourcing activate.
set -euo pipefail

SCRIPT="${1:?usage: submit.sh <script.py> [args ...]}"; shift
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIAG_ROOT="${DIAG_ROOT:-/work/intesasanpaolo_phd/merge-and-rebase/t5enc_block_ridge_diag}"
LOGS="$DIAG_ROOT/_reports/logs/${SCRIPT%.py}"
mkdir -p "$LOGS"

for run in "$DIAG_ROOT"/*_seed*; do
  name="$(basename "$run")"
  sbatch --parsable --job-name="${SCRIPT%.py}-$name" \
    --partition=all_usr_prod --account=intesasanpaolo_phd --gres=gpu:1 \
    --cpus-per-task=4 --mem=24G --time=00:20:00 \
    --output="$LOGS/$name.out" --error="$LOGS/$name.err" \
    --wrap="cd $REPO_DIR && $REPO_DIR/.venv/bin/python scripts/block_ridge_experiments/$SCRIPT $run $*" \
    | sed "s|^|$name -> |"
done
