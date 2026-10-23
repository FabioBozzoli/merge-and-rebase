#!/bin/bash

#SBATCH --job-name=steer_perblock_trace96
#SBATCH --chdir=/homes/fbozzoli/merge-and-rebase
#SBATCH --output=.log/steer_perblock_trace96_slurm/job_%A_%a.out
#SBATCH --error=.log/steer_perblock_trace96_slurm/job_%A_%a.err
#SBATCH --time=0:20:00

#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --constraint="gpu_2080_11G|gpu_2080Ti_11G|gpu_RTX5000_16G|gpu_RTX_A5000_24G|gpu_RTX6000_24G|gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=all_usr_prod

# Ablation reference for block_ridge_input="final": steer block_ridge with per-block input (each ridge reads the
# output of the last layer of its own block of B, block_granularity=residual), current code (Stage 1 with 0.99
# energy truncation). Recipe: smoothed_residual rho=0.9, stage1_lambda=1, lambda2 trace-scaled (ridge_lambda=0.1),
# alpha=1 fixed. Configs: configs/vision8_steer_trace{,_vitl}/ (unchanged). SVHN: resize-96 + pad without
# augmentation on A and B (MR_SVHN_PREPROCESS=1), matching how features/linear_feature was built.
#
# Arguments: pair (vitb | vitl) then datasets; array = n_dataset x 5 few_shot x 3 seed.
#   vitb light : sbatch --array=0-74 scripts/slurm/steer_perblock_trace96.sh vitb Cars DTD EuroSAT GTSRB RESISC45
#   vitb heavy : sbatch --array=0-44 --mem=24G --time=0:45:00 scripts/slurm/steer_perblock_trace96.sh vitb MNIST SVHN SUN397
#   vitl needs a GPU >= 24 GB (the cached-space stage-2 check moves all 24 ViT-L test blocks, float64, to the GPU):
#   vitl light : sbatch --array=0-74 --mem=16G --constraint="$C24" scripts/slurm/steer_perblock_trace96.sh vitl ...
#   vitl heavy : sbatch --array=0-44 --mem=32G --time=0:45:00 --constraint="$C24" scripts/slurm/steer_perblock_trace96.sh vitl ...
#   with C24="gpu_RTX_A5000_24G|gpu_RTX6000_24G|gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G"

case "$1" in
  vitb) EXP="vitb_vitb_perblock_trace96"; CFG_DIR="configs/vision8_steer_trace"; PREFIX="vision8_steer_trace" ;;
  vitl) EXP="vitb_vitl_perblock_trace96"; CFG_DIR="configs/vision8_steer_trace_vitl"; PREFIX="vision8_steer_trace_vitl" ;;
  *) echo "usage: $0 vitb|vitl <datasets...>" >&2; exit 1 ;;
esac
shift

DATASETS=("$@")
FEW_SHOTS=(1 2 5 10 20)
SEEDS=(33 54 89)
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % 3 ))]}
TEMP_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
FEW_SHOT=${FEW_SHOTS[$(( TEMP_IDX % 5 ))]}
SELECTED_DATASET=${DATASETS[$(( TEMP_IDX / 5 ))]}
if [ -z "$SELECTED_DATASET" ]; then
  echo "Task ${SLURM_ARRAY_TASK_ID} out of range for datasets: ${DATASETS[*]}" >&2
  exit 1
fi

LOG_DIR=".log/${EXP}/${SELECTED_DATASET}/${FEW_SHOT}_shots"
mkdir -p "$LOG_DIR" ".log/${EXP}/summaries"
exec > "${LOG_DIR}/seed_${SEED}_job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1
echo "${EXP} dataset=${SELECTED_DATASET} few_shot=${FEW_SHOT} seed=${SEED} node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES}"

source .venv/bin/activate
export MR_SVHN_PREPROCESS=1 MR_SVHN_TARGET_SIZE=96 MR_SVHN_AUGMENTATION=none
PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/src" python -m merge_and_rebase.eval.vision_rebase \
  --config "${CFG_DIR}/${PREFIX}_fs${FEW_SHOT}_seed${SEED}.json" \
  --tasks "${SELECTED_DATASET}" \
  --run-name "${EXP}_${SELECTED_DATASET}_fs${FEW_SHOT}_seed${SEED}" \
  --local-log-dir ".log/${EXP}/summaries"
