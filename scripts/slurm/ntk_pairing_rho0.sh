#!/bin/bash

#SBATCH --job-name=ntk_pairing_rho0
#SBATCH --chdir=/homes/fbozzoli/steer-unified
#SBATCH --output=/work/fbozzoli_phd/steer_pairing_rho0/_slurm/text_%A_%a.out
#SBATCH --error=/work/fbozzoli_phd/steer_pairing_rho0/_slurm/text_%A_%a.err
#SBATCH --time=1:30:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --constraint="gpu_2080_11G|gpu_2080Ti_11G|gpu_RTX5000_16G|gpu_RTX_A5000_24G|gpu_RTX6000_24G|gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=all_usr_prod

# Block-pairing shuffle ablation of steer_text, t5-base -> {t5-large, roberta-base}: cell lin_block_ridge_attn_seg of
# configs/ntk_rebasin_study (stage1_lambda=1, ridge_lambda=0.1 trace, smoothed_residual rho=0.9, attention blocks,
# segment pooling, alpha=1) + block_pairing in {reversed, random0..4}. Configs: configs/ntk_pairing_indep/make_configs.py.
# Features only from the cache .../text_feature/results/ntk_segment_pooling/feature_cache (force_recompute=false).
#
#   ls configs/ntk_pairing_indep/*/*/*.json | sort > <manifest>
#   sbatch --array=0-71 scripts/slurm/ntk_pairing.sh <manifest> 9
# With the sorted manifest and CHUNK=9 one job = one (target, task, pairing): 3 few_shot x 3 seeds.
# A config configs/ntk_pairing_indep/<target>/<task>/<stem>.json writes <ROOT>/<target>/<task>/<stem>/summary.json;
# existing summaries are skipped.

MANIFEST="$1"
CHUNK="${2:-1}"
ROOT=/work/fbozzoli_phd/steer_pairing_rho0/text

export HF_HOME=/work/intesasanpaolo_phd/merge-and-rebase/hf_cache
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src:$PWD"
echo "node:   $(hostname)  gpu: ${CUDA_VISIBLE_DEVICES}"
echo "commit: $(git rev-parse HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"

FIRST=$(( SLURM_ARRAY_TASK_ID * CHUNK + 1 ))
LAST=$(( FIRST + CHUNK - 1 ))
FAILED=0
while read -r CFG; do
  [ -n "$CFG" ] || continue
  REL=${CFG#configs/ntk_pairing_indep/}
  OUT="${ROOT}/${REL%.json}"
  if [ -f "$OUT/summary.json" ]; then echo "skip: $OUT/summary.json exists"; continue; fi
  mkdir -p "$OUT"
  echo "=== $(date +%T) config: $CFG -> $OUT"
  /homes/fbozzoli/merge-and-rebase/.venv/bin/python -m merge_and_rebase.eval.text_rebase \
    --config "$CFG" --local-log-dir "$OUT" --run-name summary || { echo "FAILED: $CFG"; FAILED=$(( FAILED + 1 )); }
done < <(sed -n "${FIRST},${LAST}p" "$MANIFEST")
echo "=== $(date +%T) done, failed=$FAILED"
[ "$FAILED" -eq 0 ]
