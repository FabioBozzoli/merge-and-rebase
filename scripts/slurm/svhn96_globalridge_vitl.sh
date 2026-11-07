#!/bin/bash

#SBATCH --job-name=svhn96_globalridge_vitl
#SBATCH --chdir=/homes/fbozzoli/merge-and-rebase
#SBATCH --output=.log/svhn96_globalridge_vitl_slurm/job_%A_%a.out
#SBATCH --error=.log/svhn96_globalridge_vitl_slurm/job_%A_%a.err
#SBATCH --time=0:45:00

#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --constraint="gpu_2080_11G|gpu_2080Ti_11G|gpu_RTX5000_16G|gpu_RTX_A5000_24G|gpu_RTX6000_24G|gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=all_usr_prod

# Steer global_ridge (Stage 2 = one ridge from B's final feature to the Stage-1 target, i.e. A's final-feature
# delta mapped into B's space), ViT-B/16 datacomp -> ViT-L/14 datacomp, SVHN only, standard regime (nonlinear,
# full fine-tuning source checkpoint), cache features/nonlinear_feature, stage1_lambda=1, ridge_lambda=1.0,
# alpha=1 fixed. SVHN: resize 96 + pad, no augmentation, on A and B (MR_SVHN_PREPROCESS=1).
# Configs: configs/vision8_steer_global/globalridge_nonlinear_vitl_fs<K>_seed<S>.json (tasks narrowed to SVHN).
#   sbatch --array=0-14 scripts/slurm/svhn96_globalridge_vitl.sh     (array index = K x seed)

EXP="vitb_vitl_svhn96_globalridge_nonlinear"
FEW_SHOTS=(1 2 5 10 20)
SEEDS=(33 54 89)
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % 3 ))]}
FEW_SHOT=${FEW_SHOTS[$(( SLURM_ARRAY_TASK_ID / 3 ))]}
LOG_DIR=".log/${EXP}/SVHN/${FEW_SHOT}_shots"
mkdir -p "$LOG_DIR" ".log/${EXP}/summaries"
exec > "${LOG_DIR}/seed_${SEED}_job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1
echo "${EXP} few_shot=${FEW_SHOT} seed=${SEED} node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES}"

source .venv/bin/activate
export MR_SVHN_PREPROCESS=1 MR_SVHN_TARGET_SIZE=96 MR_SVHN_AUGMENTATION=none
PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/src" python -m merge_and_rebase.eval.vision_rebase \
  --config "configs/vision8_steer_global/globalridge_nonlinear_vitl_fs${FEW_SHOT}_seed${SEED}.json" \
  --tasks SVHN \
  --run-name "${EXP}_SVHN_fs${FEW_SHOT}_seed${SEED}" \
  --local-log-dir ".log/${EXP}/summaries"
