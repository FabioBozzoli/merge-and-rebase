#!/bin/bash

#SBATCH --chdir=/homes/fbozzoli/merge-and-rebase
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --constraint="gpu_2080_11G|gpu_2080Ti_11G|gpu_RTX5000_16G|gpu_RTX_A5000_24G|gpu_RTX6000_24G|gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=all_usr_prod

# Steer global_ridge on non-linear features (feature_regime=standard, full-FT checkpoints), vision8.
# Source ViT-B/16 datacomp; target per <pair>: vitb = ViT-B/16 laion2b, vitl = ViT-L/14 datacomp.
# stage1_lambda=1, ridge_lambda=1, alpha=1 fixed, seeds 33/54/89. Features are read from the cache in
# /work/intesasanpaolo_phd/merge-and-rebase/features/nonlinear_feature (nothing is re-extracted).
# Configs: configs/vision8_steer_global/globalridge_nonlinear_<pair>_fs<K>_seed<S>.json
#
# One array per (pair, dataset); the array index IS the few-shot K, one task runs the three seeds:
#
#   sbatch --job-name=grnl_<pair>_<Dataset> --array=1,2,5,10,20 --mem=.. --time=.. \
#     --output=<dir>/<Dataset>/K%a/slurm_%A_%a.out --error=<dir>/<Dataset>/K%a/slurm_%A_%a.err \
#     scripts/slurm/vision_globalridge_nonlinear.sh <pair> <Dataset>
#   (scripts/slurm/submit_vision_globalridge_nonlinear.sh does exactly this for the whole grid.)
#
# Output, <dir> = .log/steer_globalridge_nonlinear/<source>_to_<target>/:
#   <Dataset>/K<k>/seed<S>.log        full log of one run
#   <Dataset>/K<k>/slurm_<job>.out    job header and per-seed status
#   summaries/<Dataset>_K<k>_seed<S>.json   results (+ .events.jsonl)
# A run whose summary exists is skipped, so resubmitting only fills the gaps. A run already done with the
# same config by the earlier subset grid (.log/vitb_<pair>_globalridge_nonlinear_subset) is copied instead.

PAIR="$1"
TASK="$2"
K="$SLURM_ARRAY_TASK_ID"
SEEDS=(33 54 89)
case "$PAIR" in
  vitb) PAIR_DIR="vitb16-datacomp_to_vitb16-laion2b" ;;
  vitl) PAIR_DIR="vitb16-datacomp_to_vitl14-datacomp" ;;
  *) echo "usage: $0 <vitb|vitl> <Dataset> (array index = K)" >&2; exit 1 ;;
esac
[ -n "$TASK" ] && [ -n "$K" ] || { echo "usage: $0 <vitb|vitl> <Dataset> (array index = K)" >&2; exit 1; }

ROOT=".log/steer_globalridge_nonlinear/${PAIR_DIR}"
RUN_DIR="${ROOT}/${TASK}/K${K}"
SUMMARIES="${ROOT}/summaries"
OLD_SUBSET=".log/vitb_${PAIR}_globalridge_nonlinear_subset/summaries/vitb_${PAIR}_globalridge_nonlinear_subset_${TASK}_fs${K}"
mkdir -p "$RUN_DIR" "$SUMMARIES"

echo "global_ridge nonlinear | ${PAIR_DIR} | ${TASK} | K=${K} | node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES}"
echo "commit: $(git rev-parse HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"

source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/src"
FAILED=0
for SEED in "${SEEDS[@]}"; do
  NAME="${TASK}_K${K}_seed${SEED}"
  if [ -f "${SUMMARIES}/${NAME}.json" ]; then
    echo "seed ${SEED}: already done, skipped"
    continue
  fi
  if [ -f "${OLD_SUBSET}_seed${SEED}.json" ]; then
    cp "${OLD_SUBSET}_seed${SEED}.json" "${SUMMARIES}/${NAME}.json"
    [ -f "${OLD_SUBSET}_seed${SEED}.events.jsonl" ] && cp "${OLD_SUBSET}_seed${SEED}.events.jsonl" "${SUMMARIES}/${NAME}.events.jsonl"
    echo "seed ${SEED}: copied from the earlier subset grid (same config)"
    continue
  fi
  echo "seed ${SEED}: running ($(date +%T)), log ${RUN_DIR}/seed${SEED}.log"
  if python -m merge_and_rebase.eval.vision_rebase \
      --config "configs/vision8_steer_global/globalridge_nonlinear_${PAIR}_fs${K}_seed${SEED}.json" \
      --tasks "$TASK" --run-name "$NAME" --local-log-dir "$SUMMARIES" > "${RUN_DIR}/seed${SEED}.log" 2>&1; then
    echo "seed ${SEED}: done ($(date +%T))"
  else
    echo "seed ${SEED}: FAILED ($(date +%T)), see ${RUN_DIR}/seed${SEED}.log"
    FAILED=$(( FAILED + 1 ))
  fi
done
[ "$FAILED" -eq 0 ]
