#!/bin/bash

#SBATCH --job-name=vitb_vitb_final_input
#SBATCH --chdir=/homes/fbozzoli/merge-and-rebase
#SBATCH --output=.log/vitb_vitb_final_input_slurm/job_%A_%a.out
#SBATCH --error=.log/vitb_vitb_final_input_slurm/job_%A_%a.err
#SBATCH --time=0:20:00

#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --constraint="gpu_2080_11G|gpu_2080Ti_11G|gpu_RTX5000_16G|gpu_RTX_A5000_24G|gpu_RTX6000_24G|gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=all_usr_prod

# ViT-B/16 datacomp -> ViT-B/16 laion2b, steer block_ridge con block_ridge_input="final":
# ogni block ridge (13 = 12 resblock + final di A) prende in input la feature finale di B.
#   stage1_lambda=1.0, lambda2 trace-scaled (ridge_lambda=0.1; con input a norma 1 trace e' un no-op),
#   smoothed_residual rho=0.9, block_granularity=residual, alpha=1 fisso.
# Config: configs/vision8_steer_final_input/vision8_steer_final_input_fs<K>_seed<S>.json
# Feature linear gia' in cache (/work/intesasanpaolo_phd/merge-and-rebase/features/linear_feature):
# serve solo il forward di eval sulla GPU, e i blocchi di B non vengono caricati.
#
# Argomenti: coppia (vitb | vitl, default vitb) e poi i dataset; array = n_dataset x 5 few_shot x 3 seed.
# Risorse dai run trace (job 118969 vitb, 118990 vitl, che caricavano anche i blocchi di B -> limite superiore):
#   vitb leggeri (<= 8.5 GB RSS, <= 5 min):
#     sbatch --array=0-74 scripts/slurm/vitb_vitb_final_input.sh vitb Cars DTD EuroSAT GTSRB RESISC45
#   vitb pesanti (MNIST 13.5 GB, SVHN 20 GB, SUN397 11 GB / 17 min):
#     sbatch --array=0-44 --mem=24G --time=0:45:00 scripts/slurm/vitb_vitb_final_input.sh vitb MNIST SVHN SUN397
#   vitl: config configs/vision8_steer_final_input_vitl (stessi iperparametri; 13 ridge, nessun raggruppamento
#   dei 24 blocchi di B perche' l'input e' solo la feature finale), --mem=16G leggeri / 32G pesanti.

case "${1:-vitb}" in
  vitb) EXP="vitb_vitb_final_input"; CFG_DIR="configs/vision8_steer_final_input"; PREFIX="vision8_steer_final_input"; shift ;;
  vitl) EXP="vitb_vitl_final_input"; CFG_DIR="configs/vision8_steer_final_input_vitl"; PREFIX="vision8_steer_final_input_vitl"; shift ;;
  *) EXP="vitb_vitb_final_input"; CFG_DIR="configs/vision8_steer_final_input"; PREFIX="vision8_steer_final_input" ;;
esac

DATASETS=("$@")
FEW_SHOTS=(1 2 5 10 20)
SEEDS=(33 54 89)
N_SHOTS=${#FEW_SHOTS[@]}
N_SEEDS=${#SEEDS[@]}

SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))]}
TEMP_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
FEW_SHOT=${FEW_SHOTS[$(( TEMP_IDX % N_SHOTS ))]}
SELECTED_DATASET=${DATASETS[$(( TEMP_IDX / N_SHOTS ))]}
if [ -z "$SELECTED_DATASET" ]; then
  echo "Task ${SLURM_ARRAY_TASK_ID} out of range for datasets: ${DATASETS[*]}" >&2
  exit 1
fi

LOG_DIR=".log/${EXP}/${SELECTED_DATASET}/${FEW_SHOT}_shots"
mkdir -p "$LOG_DIR" ".log/${EXP}/summaries"

CUSTOM_LOG="${LOG_DIR}/seed_${SEED}_job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
exec > "$CUSTOM_LOG" 2>&1

echo "Running task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Selected dataset: ${SELECTED_DATASET}"
echo "Few-shot: ${FEW_SHOT}"
echo "Seed: ${SEED}"
echo "Node: $(hostname)"
echo "Allocated GPU: ${CUDA_VISIBLE_DEVICES}"

source .venv/bin/activate
# La cache SVHN e' costruita con resize 96 + pad e senza augmentation: la eval live deve usare la stessa
# trasformazione, altrimenti zero-shot cached (0.625) e live (0.50) divergono. Agisce solo su SVHN.
export MR_SVHN_PREPROCESS=1 MR_SVHN_TARGET_SIZE=96 MR_SVHN_AUGMENTATION=none
PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/src" python -m merge_and_rebase.eval.vision_rebase \
  --config "${CFG_DIR}/${PREFIX}_fs${FEW_SHOT}_seed${SEED}.json" \
  --tasks "${SELECTED_DATASET}" \
  --run-name "${EXP}_${SELECTED_DATASET}_fs${FEW_SHOT}_seed${SEED}" \
  --local-log-dir ".log/${EXP}/summaries"
