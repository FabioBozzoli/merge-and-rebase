#!/bin/bash

#SBATCH --job-name=vitl_50perclass
#SBATCH --chdir=/homes/fbozzoli/merge-and-rebase
#SBATCH --output=.log/vitl_50perclass_slurm/job_%A_%a.out
#SBATCH --error=.log/vitl_50perclass_slurm/job_%A_%a.err
#SBATCH --time=0:45:00

#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --constraint="gpu_RTX_A5000_24G|gpu_RTX6000_24G|gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=all_usr_prod

# ViT-B/16 datacomp -> ViT-L/14 datacomp, linear regime, features re-extracted from scratch into
#   /work/intesasanpaolo_phd/merge-and-rebase/features/linear_feature_vitl_50perclass
# train: at most 50 examples per class (train_features_per_class=50, smaller classes kept whole, fixed seed 0)
# test : full test split. SVHN: resize 96 + pad, no augmentation, on A and B (MR_SVHN_PREPROCESS=1).
# Method (configs/vision8_steer_vitl_50perclass/): block_ridge per-block (residual = output of the last layer
# of each block), smoothed_residual rho=0.9, stage1_lambda=1, lambda2 trace (ridge_lambda=0.1), alpha=1.
#
#   1) extraction, one task per dataset (the cache is written once per dataset; the fs1/seed33 run that
#      triggers it is a regular run too):
#        sbatch --array=0-7 --time=6:00:00 --mem=48G scripts/slurm/vitl_50perclass.sh extract
#   2) runs, array = 8 datasets x 5 few_shot x 3 seed (reads the cache only):
#        sbatch --array=0-119 --dependency=afterok:<extract job> scripts/slurm/vitl_50perclass.sh run
#   optional 2nd argument = config variant, e.g. "fullpinv" (Stage 1 exact pinv, stage1_energy_threshold=1.0):
#        sbatch --array=105-119 --mem=24G scripts/slurm/vitl_50perclass.sh run fullpinv
# GPU >= 24 GB: the cached-space stage-2 check moves all 24 ViT-L test blocks (float64) to the GPU.

MODE="${1:?usage: $0 extract|run [variant]}"
VARIANT="${2:-}"
EXP="vitb_vitl_50perclass${VARIANT:+_$VARIANT}"
PREFIX="vitl_50perclass${VARIANT:+_$VARIANT}"
CFG_DIR="configs/vision8_steer_vitl_50perclass"
DATASETS=(Cars DTD EuroSAT GTSRB MNIST RESISC45 SUN397 SVHN)
FEW_SHOTS=(1 2 5 10 20)
SEEDS=(33 54 89)

case "$MODE" in
  extract)
    SELECTED_DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}; FEW_SHOT=1; SEED=33
    LOG_DIR=".log/${EXP}/extract" ;;
  run)
    SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % 3 ))]}
    TEMP_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
    FEW_SHOT=${FEW_SHOTS[$(( TEMP_IDX % 5 ))]}
    SELECTED_DATASET=${DATASETS[$(( TEMP_IDX / 5 ))]}
    LOG_DIR=".log/${EXP}/${SELECTED_DATASET}/${FEW_SHOT}_shots" ;;
  *) echo "unknown mode $MODE" >&2; exit 1 ;;
esac

mkdir -p "$LOG_DIR" ".log/${EXP}/summaries"
exec > "${LOG_DIR}/${SELECTED_DATASET}_fs${FEW_SHOT}_seed_${SEED}_job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1
echo "${EXP} mode=${MODE} dataset=${SELECTED_DATASET} few_shot=${FEW_SHOT} seed=${SEED} node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES}"

source .venv/bin/activate
export MR_SVHN_PREPROCESS=1 MR_SVHN_TARGET_SIZE=96 MR_SVHN_AUGMENTATION=none
PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/src" python -m merge_and_rebase.eval.vision_rebase \
  --config "${CFG_DIR}/${PREFIX}_fs${FEW_SHOT}_seed${SEED}.json" \
  --tasks "${SELECTED_DATASET}" \
  --run-name "${EXP}_${SELECTED_DATASET}_fs${FEW_SHOT}_seed${SEED}" \
  --local-log-dir ".log/${EXP}/summaries"
