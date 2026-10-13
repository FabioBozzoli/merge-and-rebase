#!/bin/bash

#SBATCH --chdir=/homes/fbozzoli/merge-and-rebase
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --constraint="gpu_2080_11G|gpu_2080Ti_11G|gpu_RTX5000_16G|gpu_RTX_A5000_24G|gpu_RTX6000_24G|gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=all_usr_prod

# Steer concat_ridge, vision8: one ridge from the concatenation of all of B's residual blocks + final feature
# (per-block trace scaling) to the global Stage 1 target. Same grid/caches as global_ridge
# (scripts/slurm/vision_globalridge_nonlinear.sh), for a direct comparison.
# <features>: linear (NTK ckpts, cache features/linear_feature) |
#             nonlinear (full-FT ckpts, cache features/nonlinear_feature; B's block activations are read from
#             features/linear_feature: B is the same pretrained model in both regimes).
# Only existing caches are read: nothing is extracted or written under features/.
# Code: frozen snapshot $CODE_ROOT (working tree of 2026-10-12 + concat_ridge), not the live working tree.
# SVHN: caches were built with resize 96 + pad, no augmentation, so the live eval uses the same transform
# (MR_SVHN_* only affect SVHN).
# Configs: configs/vision8_steer_concat/concatridge_<features>_<pair>_fs<K>_seed<S>.json
#
# One array per (features, pair, dataset); the array index IS the few-shot K, one task runs the three seeds:
#   bash scripts/slurm/submit_vision_concatridge.sh [linear|nonlinear ...]
#
# Output, <dir> = .log/steer_concatridge_<features>/<source>_to_<target>/:
#   <Dataset>/K<k>/seed<S>.log, <Dataset>/K<k>/slurm_<job>.out, summaries/<Dataset>_K<k>_seed<S>.json
# A run whose summary exists is skipped, so resubmitting only fills the gaps.

FEATURES="$1"
PAIR="$2"
TASK="$3"
K="$SLURM_ARRAY_TASK_ID"
SEEDS=(33 54 89)
USAGE="usage: $0 <linear|nonlinear> <vitb|vitl> <Dataset> (array index = K)"
case "$FEATURES" in linear|nonlinear) ;; *) echo "$USAGE" >&2; exit 1 ;; esac
case "$PAIR" in
  vitb) PAIR_DIR="vitb16-datacomp_to_vitb16-laion2b" ;;
  vitl) PAIR_DIR="vitb16-datacomp_to_vitl14-datacomp" ;;
  *) echo "$USAGE" >&2; exit 1 ;;
esac
[ -n "$TASK" ] && [ -n "$K" ] || { echo "$USAGE" >&2; exit 1; }

ROOT=".log/steer_concatridge_${FEATURES}/${PAIR_DIR}"
RUN_DIR="${ROOT}/${TASK}/K${K}"
SUMMARIES="${ROOT}/summaries"
mkdir -p "$RUN_DIR" "$SUMMARIES"

echo "concat_ridge ${FEATURES} | ${PAIR_DIR} | ${TASK} | K=${K} | node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES}"
CODE_ROOT=/work/intesasanpaolo_phd/merge-and-rebase/code_snapshots/concatridge_20261012
source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="${CODE_ROOT}/src"
export MR_SVHN_PREPROCESS=1 MR_SVHN_TARGET_SIZE=96 MR_SVHN_AUGMENTATION=none
echo "code: ${CODE_ROOT} (base commit $(cat ${CODE_ROOT}/BASE_COMMIT.txt) + worktree.patch)"
FAILED=0
for SEED in "${SEEDS[@]}"; do
  NAME="${TASK}_K${K}_seed${SEED}"
  if [ -f "${SUMMARIES}/${NAME}.json" ]; then
    echo "seed ${SEED}: already done, skipped"
    continue
  fi
  echo "seed ${SEED}: running ($(date +%T)), log ${RUN_DIR}/seed${SEED}.log"
  if python -m merge_and_rebase.eval.vision_rebase \
      --config "configs/vision8_steer_concat/concatridge_${FEATURES}_${PAIR}_fs${K}_seed${SEED}.json" \
      --tasks "$TASK" --run-name "$NAME" --local-log-dir "$SUMMARIES" > "${RUN_DIR}/seed${SEED}.log" 2>&1; then
    echo "seed ${SEED}: done ($(date +%T))"
  else
    echo "seed ${SEED}: FAILED ($(date +%T)), see ${RUN_DIR}/seed${SEED}.log"
    FAILED=$(( FAILED + 1 ))
  fi
done
[ "$FAILED" -eq 0 ]
