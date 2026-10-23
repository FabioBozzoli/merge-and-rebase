#!/bin/bash

#SBATCH --job-name=svhn96_source_only
#SBATCH --chdir=/homes/fbozzoli/merge-and-rebase
#SBATCH --output=.log/svhn96_source_only_slurm/job_%A_%a.out
#SBATCH --error=.log/svhn96_source_only_slurm/job_%A_%a.err
#SBATCH --time=0:45:00

#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --constraint="gpu_2080_11G|gpu_2080Ti_11G|gpu_RTX5000_16G|gpu_RTX_A5000_24G|gpu_RTX6000_24G|gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=all_usr_prod

# SVHN, ViT-B/16 datacomp -> ViT-B/16 laion2b: resize-96 + pad (no augmentation) only on the SOURCE,
# B keeps its stock CLIP preprocessing. Steer block_ridge per-block, the recipe that gives ~73% at K=20
# when both models see 96px (stage1_lambda=1, trace lambda2=0.1, smoothed_residual rho=0.9, residual, alpha=1).
#   1) cache: A's side copied from features/linear_feature, B's side recomputed with stock preprocessing
#        sbatch scripts/slurm/svhn96_source_only.sh build
#   2) runs: array index = few_shot x seed (15)
#        sbatch --array=0-14 --dependency=afterok:<build job> scripts/slurm/svhn96_source_only.sh run

MODE="${1:-run}"
EXP="svhn96_source_only"
FEATURES=/work/intesasanpaolo_phd/merge-and-rebase/features
mkdir -p ".log/${EXP}/summaries"
source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/src"
export MR_SVHN_PREPROCESS=1 MR_SVHN_TARGET_SIZE=96 MR_SVHN_AUGMENTATION=none MR_SVHN_PREPROCESS_SCOPE=source

if [ "$MODE" = "build" ]; then
  exec > ".log/${EXP}/build_${SLURM_JOB_ID}.log" 2>&1
  echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES}"
  python scripts/build_steer_target_cache.py \
    --src-root "${FEATURES}/linear_feature" --out-root "${FEATURES}/steer_svhn96_source_only"
  exit $?
fi

FEW_SHOTS=(1 2 5 10 20)
SEEDS=(33 54 89)
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % 3 ))]}
FEW_SHOT=${FEW_SHOTS[$(( SLURM_ARRAY_TASK_ID / 3 ))]}
LOG_DIR=".log/${EXP}/SVHN/${FEW_SHOT}_shots"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/seed_${SEED}_job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1
echo "SVHN few_shot=${FEW_SHOT} seed=${SEED} node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES}"
python -m merge_and_rebase.eval.vision_rebase \
  --config "configs/vision8_steer_svhn96_source_only/svhn96_source_only_fs${FEW_SHOT}_seed${SEED}.json" \
  --tasks SVHN \
  --run-name "${EXP}_SVHN_fs${FEW_SHOT}_seed${SEED}" \
  --local-log-dir ".log/${EXP}/summaries"
