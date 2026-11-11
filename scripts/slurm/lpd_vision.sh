#!/bin/bash

#SBATCH --job-name=lpd_vision
#SBATCH --chdir=/homes/fbozzoli/steer-unified
#SBATCH --output=/work/fbozzoli_phd/steer_lpd/_slurm/job_%A_%a.out
#SBATCH --error=/work/fbozzoli_phd/steer_lpd/_slurm/job_%A_%a.err
#SBATCH --time=0:10:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --constraint="gpu_2080_11G|gpu_2080Ti_11G|gpu_RTX5000_16G|gpu_RTX_A5000_24G|gpu_RTX6000_24G|gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G"
#SBATCH --account=fbozzoli_phd
#SBATCH --partition=all_usr_prod

# Linear probe distillation (stage1_mode=logit_distill; BASELINE, not steer) on vision8. Configs: configs/vision8_steer_lpd{,_vitl,_vitl_svhnsrc}/
# (make_configs.py): global_ridge on the logit residual, correction_space=logit, ridge_lambda in LAMBDAS, alpha=1.
# One array task = one run; index = ((dataset * 5 + few_shot) * 3 + seed) * NL + lambda (NL=1: lambda=1), over the datasets given as arguments
# (15 tasks per dataset). The pair is positional because sbatch --export is reset on this cluster.
#   sbatch --array=0-89 scripts/slurm/lpd_vision.sh vitb Cars DTD EuroSAT GTSRB MNIST RESISC45     (resources: see launch notes)
# SVHN preprocessing (cached features verified: cached B zero-shot 62.46 vitb / 67.63 vitl; vitl SVHN must NOT run on L40S):
#   vitb: resize-96 + pad on A and B, cache features/linear_feature.  vitl: on A only (B stock), cache features/steer_svhn96_source_only.

PAIR_ARG="${1:?usage: $0 vitb|vitl[:lam=<L>[,<L>..]][:k=<K>[,<K>..]][:seed=<S>[,<S>..]][:stage2=concat] <datasets...>}"
shift
PAIR="${PAIR_ARG%%:*}"
OPTS="${PAIR_ARG#"$PAIR"}"   # optional suffix, e.g. ":lam=0.1:k=1,2:seed=33:stage2=concat" (defaults: lambda 1, K 1 2 5 10 20, seeds 33 54 89, LPD on the final feature)
case "$PAIR" in
  vitb) SCOPE=both ;;
  vitl) SCOPE=source ;;
  *) echo "unknown pair $PAIR" >&2; exit 1 ;;
esac
EXP="vitb_${PAIR}"
# Outputs go to /work/fbozzoli_phd (the shared /work/intesasanpaolo_phd hit its disk quota on 2026-11-10); cached features are
# still READ from /work/intesasanpaolo_phd (config feature_cache_dir). LEGACY_ROOT holds the runs finished before the move.
ROOT=/work/fbozzoli_phd/steer_lpd
LEGACY_ROOT=/work/intesasanpaolo_phd/merge-and-rebase/steer_lpd

DATASETS=("$@")
FEW_SHOTS=(1 2 5 10 20)
SEEDS=(33 54 89)
LAMBDAS=(1)   # default: single ridge_lambda (configs for {0,.001,.01,.1,1} exist); override with the ":lam=" suffix
for o in ${OPTS//:/ }; do
  case "$o" in
    lam=*) IFS=, read -r -a LAMBDAS <<< "${o#lam=}" ;;
    k=*)   IFS=, read -r -a FEW_SHOTS <<< "${o#k=}" ;;
    seed=*) IFS=, read -r -a SEEDS <<< "${o#seed=}" ;;
    stage2=concat) STAGE2=concat ;;   # LPD with all of B's blocks concatenated as input (configs vision8_steer_lpdconcat*)
    *) echo "unknown option $o" >&2; exit 1 ;;
  esac
done
NL=${#LAMBDAS[@]}
NK=${#FEW_SHOTS[@]}
NS=${#SEEDS[@]}
# index = ((dataset * NK + few_shot) * NS + seed) * NL + lambda
I=$SLURM_ARRAY_TASK_ID
LAM=${LAMBDAS[$(( I % NL ))]}; I=$(( I / NL ))
SEED=${SEEDS[$(( I % NS ))]}; I=$(( I / NS ))
FEW_SHOT=${FEW_SHOTS[$(( I % NK ))]}; I=$(( I / NK ))
DATASET=${DATASETS[$I]}
if [ -z "$DATASET" ]; then echo "task ${SLURM_ARRAY_TASK_ID} out of range for: ${DATASETS[*]}" >&2; exit 1; fi
LPD=vision8_steer_lpd; TAG=lpd; DIRSUF=""
if [ "${STAGE2:-}" = concat ]; then LPD=vision8_steer_lpdconcat; TAG=lpdconcat; DIRSUF="_concat"; fi
case "$PAIR" in
  vitb) CFG_DIR="configs/${LPD}"; PREFIX="${LPD}" ;;
  vitl) CFG_DIR="configs/${LPD}_vitl"; PREFIX="${LPD}_vitl" ;;
esac
if [ "$PAIR" = "vitl" ] && [ "$DATASET" = "SVHN" ]; then
  CFG_DIR="configs/${LPD}_vitl_svhnsrc"; PREFIX="${LPD}_vitl_svhnsrc"
fi

NAME="${EXP}_${TAG}_lam${LAM}_${DATASET}_fs${FEW_SHOT}_seed${SEED}"
OUT="${ROOT}/${EXP}/lam${LAM}${DIRSUF}"
LOG_DIR="${OUT}/logs/${DATASET}/${FEW_SHOT}_shots"
mkdir -p "$LOG_DIR" "${OUT}/summaries"
if [ -f "${OUT}/summaries/${NAME}.json" ] || [ -f "${LEGACY_ROOT}/${EXP}/lam${LAM}${DIRSUF}/summaries/${NAME}.json" ]; then echo "skip ${NAME}: summary exists"; exit 0; fi

export MR_SVHN_PREPROCESS=1 MR_SVHN_TARGET_SIZE=96 MR_SVHN_AUGMENTATION=none MR_SVHN_PREPROCESS_SCOPE=$SCOPE
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/src"
exec > "${LOG_DIR}/seed_${SEED}_job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1
echo "${NAME} svhn_scope=${SCOPE} node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES}"
echo "commit: $(git rev-parse HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
/homes/fbozzoli/merge-and-rebase/.venv/bin/python -m merge_and_rebase.eval.vision_rebase \
  --config "${CFG_DIR}/${PREFIX}_lam${LAM}_fs${FEW_SHOT}_seed${SEED}.json" \
  --tasks "${DATASET}" \
  --run-name "${NAME}" \
  --local-log-dir "${OUT}/summaries"
