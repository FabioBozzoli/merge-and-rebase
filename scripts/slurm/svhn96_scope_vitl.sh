#!/bin/bash

#SBATCH --job-name=svhn96_scope_vitl
#SBATCH --chdir=/homes/fbozzoli/merge-and-rebase
#SBATCH --output=.log/svhn96_scope_vitl_slurm/job_%A_%a.out
#SBATCH --error=.log/svhn96_scope_vitl_slurm/job_%A_%a.err
#SBATCH --time=0:25:00

#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=14G
#SBATCH --constraint="gpu_2080_11G|gpu_2080Ti_11G|gpu_RTX5000_16G|gpu_RTX_A5000_24G|gpu_RTX6000_24G|gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=all_usr_prod

# SVHN, ViT-B/16 datacomp -> ViT-L/14 datacomp: does B (ViT-L) benefit from A's resize-96 + pad preprocessing?
# Arms (configs/vision8_steer_svhn96_scope_vitl/<arm>_fs<K>_seed<S>.json), same trace recipe:
#   perblock_both    block_ridge per-block, 96px on A and B   (cache features/linear_feature)
#   perblock_source  block_ridge per-block, 96px on A only    (cache features/steer_svhn96_source_only)
#   final_source     block_ridge input=final, 96px on A only  (same cache; 96px-both is vitb_vitl_final_input)
#   final_source_rerun  identical rerun of final_source
#   global_source    global_ridge, ridge_lambda=0.1, 96px on A only (independent-mode limit of final_source)
#   globalridge_nonlinear_source  global_ridge, nonlinear (standard regime), ridge_lambda=1.0, 96px on A only
#   perblock_indep_source  block_ridge per-block, mode=independent (rho=0), 96px on A only (GPU >= 24 GB)
#   1) cache (B side recomputed with stock preprocessing, A side copied):
#        sbatch --mem=16G scripts/slurm/svhn96_scope_vitl.sh build
#   2) runs, array index = few_shot x seed (15):
#        sbatch --array=0-14 [--dependency=afterok:<build>] scripts/slurm/svhn96_scope_vitl.sh <arm>
#   perblock_* need a GPU >= 24 GB: the cached-space stage-2 check moves all 24 ViT-L test blocks (float64)
#   to the GPU (~10.7 GB) and OOMs on the 11 GB cards; add
#        --constraint="gpu_RTX_A5000_24G|gpu_RTX6000_24G|gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G"

ARM="${1:?usage: $0 build|perblock_both|perblock_source|final_source|final_source_rerun|global_source|globalridge_nonlinear_source|perblock_indep_source}"
EXP="svhn96_scope_vitl"
FEATURES=/work/intesasanpaolo_phd/merge-and-rebase/features
PAIR_DIR=ViT-B-16_datacomp_xl_s13b_b90k_to_ViT-L-14_datacomp_xl_s13b_b90k
mkdir -p ".log/${EXP}/summaries"
source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/src"
export MR_SVHN_PREPROCESS=1 MR_SVHN_TARGET_SIZE=96 MR_SVHN_AUGMENTATION=none
case "$ARM" in
  perblock_both) export MR_SVHN_PREPROCESS_SCOPE=both ;;
  build|perblock_source|final_source|final_source_rerun|global_source|globalridge_nonlinear_source|perblock_indep_source) export MR_SVHN_PREPROCESS_SCOPE=source ;;
  *) echo "unknown arm $ARM" >&2; exit 1 ;;
esac

if [ "$ARM" = "build" ]; then
  exec > ".log/${EXP}/build_${SLURM_JOB_ID}.log" 2>&1
  echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES}"
  python scripts/build_steer_target_cache.py \
    --src-root "${FEATURES}/linear_feature" --out-root "${FEATURES}/steer_svhn96_source_only" \
    --pair-dir "$PAIR_DIR" --target-model ViT-L-14 --target-pretrained datacomp_xl_s13b_b90k
  exit $?
fi

FEW_SHOTS=(1 2 5 10 20)
SEEDS=(33 54 89)
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % 3 ))]}
FEW_SHOT=${FEW_SHOTS[$(( SLURM_ARRAY_TASK_ID / 3 ))]}
LOG_DIR=".log/${EXP}/${ARM}/${FEW_SHOT}_shots"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/seed_${SEED}_job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1
echo "SVHN arm=${ARM} scope=${MR_SVHN_PREPROCESS_SCOPE} few_shot=${FEW_SHOT} seed=${SEED} node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES}"
python -m merge_and_rebase.eval.vision_rebase \
  --config "configs/vision8_steer_svhn96_scope_vitl/${ARM}_fs${FEW_SHOT}_seed${SEED}.json" \
  --tasks SVHN \
  --run-name "${EXP}_${ARM}_fs${FEW_SHOT}_seed${SEED}" \
  --local-log-dir ".log/${EXP}/summaries"
