#!/bin/bash

#SBATCH --job-name=steer_integ
#SBATCH --chdir=/homes/fbozzoli/steer-unified
#SBATCH --output=/work/fbozzoli_phd/steer_integrated/_slurm/job_%A_%a.out
#SBATCH --error=/work/fbozzoli_phd/steer_integrated/_slurm/job_%A_%a.err
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --constraint="gpu_2080_11G|gpu_2080Ti_11G|gpu_RTX5000_16G|gpu_RTX_A5000_24G|gpu_RTX6000_24G|gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G"
#SBATCH --account=fbozzoli_phd
#SBATCH --partition=all_usr_prod

# steer with integrated per-block residuals (feature_regime=integrated, nonlinear source checkpoints), ViT-B/16 datacomp ->
# ViT-B/16 laion2b, vision8. Configs: configs/vision8_steer_integrated/vision8_steer_integrated_k<K>_fs<F>_seed<S>.json
# (make_configs.py): block_ridge, smoothed_residual rho=0.9, trace lambda 0.1, alpha=1, train_features_per_class=50.
#
# One array task = one run; index = (dataset * NF + few_shot) * 3 + seed over the datasets given as arguments (15 per dataset).
# Parameters are positional because sbatch --export is reset on this cluster. First argument: k=<K>[:fs=<F>[,<F>..]].
#
# The feature cache (/work/fbozzoli_phd/features/integrated_feature/<pair>/<Task>/integrated_k<K>) is extracted by the FIRST
# run of a dataset (K * 13 JVPs per batch on the train split): run ONE cell per dataset first, then the rest, so that
# concurrent tasks never extract the same cache:
#   W=$(sbatch --parsable --array=0 scripts/slurm/steer_integrated_vision.sh k=4 DTD)            # warm-up = cell fs1/seed33
#   sbatch --dependency=afterok:$W --array=1-14 --time=0:15:00 --mem=8G scripts/slurm/steer_integrated_vision.sh k=4 DTD
# Resources, measured on job 199230 (DTD, K=4, batch 32, gpu_2080_11G, 2350 train examples at 50/class):
#   peak GPU memory 3.89 GiB allocated / 4.89 GiB reserved, host MaxRSS 4.5 GiB, 23m47s in total, of which ~22 min are the
#   train extraction (~0.57 s per train example on a 2080; the test split has no JVPs, 36 s on DTD).
# The extraction time scales with 50 * num_classes, so the 1 h default does NOT cover the warm-up of the large datasets on
# a 2080 (estimates by extrapolation, not measured): Cars (9800 examples) ~1.6 h -> --time=3:00:00, SUN397 (19850) ~3.2 h
# -> --time=5:00:00; the others (<= 2350 train examples) fit in 1 h. Refit cells only load the cache: --time=0:15:00.
# GPUs: the venv's torch 2.11+cu128 is built for sm_75+ only, so gpu_K80_12G / gpu_P100_16G / gpu_1080_8G cannot run it
# (ignore sbatch's advice to add K80); the 2080 (11 GB) is the smallest usable card.
# integration_batch_size (make_configs.py --integration-batch-size 8) lowers the peak further; not needed on 11 GB at batch 32.
# The directories of --output/--error must exist before sbatch: mkdir -p /work/fbozzoli_phd/steer_integrated/_slurm
#
# SVHN: resize-96 + pad on A and B (MR_SVHN_* below only affect SVHN); the cached B zero-shot must be 62.46.

OPTS="${1:?usage: $0 k=<K>[:fs=<F>[,<F>..]] <datasets...>}"
shift
K=""
FEW_SHOTS=(1 2 5 10 20)
SEEDS=(33 54 89)
for o in ${OPTS//:/ }; do
  case "$o" in
    k=*)  K="${o#k=}" ;;
    fs=*) IFS=, read -r -a FEW_SHOTS <<< "${o#fs=}" ;;
    *) echo "unknown option $o" >&2; exit 1 ;;
  esac
done
[ -n "$K" ] || { echo "missing k=<K>" >&2; exit 1; }
DATASETS=("$@")
NF=${#FEW_SHOTS[@]}
I=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$(( I % 3 ))]}; I=$(( I / 3 ))
FEW_SHOT=${FEW_SHOTS[$(( I % NF ))]}; I=$(( I / NF ))
DATASET=${DATASETS[$I]}
if [ -z "$DATASET" ]; then echo "task ${SLURM_ARRAY_TASK_ID} out of range for: ${DATASETS[*]}" >&2; exit 1; fi

CFG_DIR="configs/vision8_steer_integrated"; PREFIX="vision8_steer_integrated"
EXP="vitb_vitb"
ROOT=/work/fbozzoli_phd/steer_integrated
NAME="${EXP}_integrated_k${K}_${DATASET}_fs${FEW_SHOT}_seed${SEED}"
OUT="${ROOT}/${EXP}/k${K}"
LOG_DIR="${OUT}/logs/${DATASET}/${FEW_SHOT}_shots"
mkdir -p "$LOG_DIR" "${OUT}/summaries"
if [ -f "${OUT}/summaries/${NAME}.json" ]; then echo "skip ${NAME}: summary exists"; exit 0; fi

export MR_SVHN_PREPROCESS=1 MR_SVHN_TARGET_SIZE=96 MR_SVHN_AUGMENTATION=none MR_SVHN_PREPROCESS_SCOPE=both
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/src"
exec > "${LOG_DIR}/seed_${SEED}_job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1
echo "${NAME} svhn_scope=both node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES}"
echo "commit: $(git rev-parse HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
/homes/fbozzoli/merge-and-rebase/.venv/bin/python -m merge_and_rebase.eval.vision_rebase \
  --config "${CFG_DIR}/${PREFIX}_k${K}_fs${FEW_SHOT}_seed${SEED}.json" \
  --tasks "${DATASET}" \
  --run-name "${NAME}" \
  --local-log-dir "${OUT}/summaries"
