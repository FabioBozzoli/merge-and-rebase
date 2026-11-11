#!/bin/bash

#SBATCH --job-name=steer_pairing_rho0
#SBATCH --chdir=/homes/fbozzoli/steer-unified
#SBATCH --output=/work/fbozzoli_phd/steer_pairing_rho0/_slurm/job_%A_%a.out
#SBATCH --error=/work/fbozzoli_phd/steer_pairing_rho0/_slurm/job_%A_%a.err
#SBATCH --time=0:15:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --constraint="gpu_2080_11G|gpu_2080Ti_11G|gpu_RTX5000_16G|gpu_RTX_A5000_24G|gpu_RTX6000_24G|gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=all_usr_prod

# Block-pairing shuffle ablation of steer block_ridge (vision8). Recipe = configs/vision8_steer_trace{,_vitl}:
# stage1_lambda=1, ridge_lambda=0.1 trace-scaled, smoothed_residual rho=0.9, residual blocks, alpha=1.
# Configs: configs/vision8_steer_pairing{,_vitl,_vitl_svhnsrc}/ (make_configs.py), block_pairing in
# {reversed, random0..4}. One array task = one run; index = ((dataset * 5 + few_shot) * 3 + seed) * 6 + pairing,
# over the datasets given as arguments (90 tasks per dataset; 15 with :identity). Resources are tiered on past runs of the same recipe:
#   C24="gpu_RTX_A5000_24G|gpu_RTX6000_24G|gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G"
#   vitb light: sbatch --array=0-539 scripts/slurm/steer_pairing_vision.sh vitb Cars DTD EuroSAT GTSRB MNIST RESISC45
#   vitb heavy: sbatch --array=0-179 --mem=16G --time=0:40:00 scripts/slurm/steer_pairing_vision.sh vitb SUN397 SVHN
#   vitl needs a GPU >= 24 GB (the cached-space stage-2 check moves all 24 ViT-L test blocks, float64, to the GPU):
#   vitl light: sbatch --array=0-539 --mem=20G --time=0:20:00 --constraint="$C24" ... vitl Cars DTD EuroSAT GTSRB MNIST RESISC45
#   vitl SVHN : sbatch --array=0-89 --mem=14G --time=0:35:00 --constraint="$C24" ... vitl SVHN
#   vitl SUN  : sbatch --array=0-89 --mem=24G --time=0:50:00 --constraint="$C24" ... vitl SUN397
#
# SVHN preprocessing (cached features verified: cached B zero-shot 62.46 vitb / 67.63 vitl):
#   vitb: resize-96 + pad, no augmentation, on A and B; cache features/linear_feature.
#   vitl: resize-96 + pad on A only (B stock); cache features/steer_svhn96_source_only.
# MR_SVHN_* only affect SVHN. The pair is positional because sbatch --export is reset on this cluster.

# Optional ":<pairing>[,<pairing>...]" suffix restricts the pairings (e.g. vitb:identity); the array index stride
# is the number of selected pairings.
PAIR_ARG="${1:?usage: $0 vitb|vitl[:pairings] <datasets...>}"
shift
PAIR="${PAIR_ARG%%:*}"
case "$PAIR" in
  vitb) CFG_DIR="configs/vision8_steer_pairing_indep"; PREFIX="vision8_steer_pairing_indep"; SCOPE=both ;;
  vitl) CFG_DIR="configs/vision8_steer_pairing_indep_vitl"; PREFIX="vision8_steer_pairing_indep_vitl"; SCOPE=source ;;
  *) echo "unknown pair $PAIR" >&2; exit 1 ;;
esac
EXP="vitb_${PAIR}"
ROOT=/work/fbozzoli_phd/steer_pairing_rho0

DATASETS=("$@")
FEW_SHOTS=(1 2 5 10 20)
SEEDS=(33 54 89)
PAIRINGS=(reversed random0 random1 random2 random3 random4)
if [ "$PAIR_ARG" != "$PAIR" ]; then IFS=, read -r -a PAIRINGS <<< "${PAIR_ARG#*:}"; fi
NP=${#PAIRINGS[@]}
I=$SLURM_ARRAY_TASK_ID
PAIRING=${PAIRINGS[$(( I % NP ))]}; I=$(( I / NP ))
SEED=${SEEDS[$(( I % 3 ))]}; I=$(( I / 3 ))
FEW_SHOT=${FEW_SHOTS[$(( I % 5 ))]}; I=$(( I / 5 ))
DATASET=${DATASETS[$I]}
if [ -z "$DATASET" ]; then echo "task ${SLURM_ARRAY_TASK_ID} out of range for: ${DATASETS[*]}" >&2; exit 1; fi
if [ "$PAIR" = "vitl" ] && [ "$DATASET" = "SVHN" ]; then
  CFG_DIR="configs/vision8_steer_pairing_indep_vitl_svhnsrc"; PREFIX="vision8_steer_pairing_indep_vitl_svhnsrc"
fi

NAME="${EXP}_${PAIRING}_${DATASET}_fs${FEW_SHOT}_seed${SEED}"
OUT="${ROOT}/${EXP}/${PAIRING}"
LOG_DIR="${OUT}/logs/${DATASET}/${FEW_SHOT}_shots"
mkdir -p "$LOG_DIR" "${OUT}/summaries"
if [ -f "${OUT}/summaries/${NAME}.json" ]; then echo "skip ${NAME}: summary exists"; exit 0; fi

export MR_SVHN_PREPROCESS=1 MR_SVHN_TARGET_SIZE=96 MR_SVHN_AUGMENTATION=none MR_SVHN_PREPROCESS_SCOPE=$SCOPE
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/src"
exec > "${LOG_DIR}/seed_${SEED}_job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1
echo "${NAME} svhn_scope=${SCOPE} node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES}"
echo "commit: $(git rev-parse HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
/homes/fbozzoli/merge-and-rebase/.venv/bin/python -m merge_and_rebase.eval.vision_rebase \
  --config "${CFG_DIR}/${PREFIX}_${PAIRING}_fs${FEW_SHOT}_seed${SEED}.json" \
  --tasks "${DATASET}" \
  --run-name "${NAME}" \
  --local-log-dir "${OUT}/summaries"
