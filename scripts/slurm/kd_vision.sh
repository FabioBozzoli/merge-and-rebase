#!/bin/bash

#SBATCH --job-name=kd_vision
#SBATCH --output=.log/kd_vision/slurm_backup/job_%A_%a.out
#SBATCH --error=.log/kd_vision/slurm_backup/job_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --array=0-119

#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=36G
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=boost_usr_prod
#SBATCH --chdir=/homes/fbozzoli/merge-and-rebase

# Few-shot KD baseline per lo steering visione (eval/kd_rebase.py vision).
# Stessa griglia dei trace steer: dataset x few_shot x seed, 8 x 5 x 3 = 120 task.
#   PAIR=vitb_vitb  -> ViT-B/16 DataComp -> ViT-B/16 LAION      (configs/kd/vision8_kd_vitb_vitb.json)
#   PAIR=vitb_vitl  -> ViT-B/16 DataComp -> ViT-L/14 DataComp   (configs/kd/vision8_kd_vitb_vitl.json)
#   PAIR=vitb_vitl_laion -> ViT-B/16 DataComp -> ViT-L/14 LAION-2B (configs/kd/vision8_kd_vitb_vitl_laion.json)
# Da lanciare dalla root del repo:
#   mkdir -p .log/kd_vision/slurm_backup
#   sbatch scripts/slurm/kd_vision.sh vitb_vitl_laion [micro_batch]
# Iperparametri KD (lr, epochs, ...) nel blocco "kd" del config; few_shot/seed arrivano da --kd qui sotto.

PAIR="${1:-${PAIR:-vitb_vitl}}"
# Secondo argomento opzionale: micro-batch (gradient accumulation, batch effettivo invariato = kd.batch_size).
MICRO="${2:-null}"
EXP="kd_vision_${PAIR}"
CONFIG="configs/kd/vision8_kd_${PAIR}.json"

DATASETS=("EuroSAT" "RESISC45" "SVHN" "DTD" "MNIST" "GTSRB" "Cars" "SUN397")
FEW_SHOTS=(1 2 5 10 20)
SEEDS=(33 54 89)

N_SHOTS=${#FEW_SHOTS[@]}
N_SEEDS=${#SEEDS[@]}

SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
TEMP_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SHOT_IDX=$(( TEMP_IDX % N_SHOTS ))
DATASET_IDX=$(( TEMP_IDX / N_SHOTS ))

SELECTED_DATASET=${DATASETS[$DATASET_IDX]}
FEW_SHOT=${FEW_SHOTS[$SHOT_IDX]}
SEED=${SEEDS[$SEED_IDX]}

LOG_DIR=".log/${EXP}/${SELECTED_DATASET}/${FEW_SHOT}_shots"
mkdir -p "$LOG_DIR" ".log/${EXP}/summaries" ".log/kd_vision/slurm_backup"

CUSTOM_LOG="${LOG_DIR}/seed_${SEED}_job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
exec > "$CUSTOM_LOG" 2>&1

echo "Task ${SLURM_ARRAY_TASK_ID}: pair=${PAIR} dataset=${SELECTED_DATASET} few_shot=${FEW_SHOT} seed=${SEED}"
echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES}"

source .venv/bin/activate

PYTHONUNBUFFERED=1 PYTHONPATH="${SLURM_SUBMIT_DIR}/src" python -m merge_and_rebase.eval.kd_rebase vision \
  --config "${CONFIG}" \
  --tasks "${SELECTED_DATASET}" \
  --seed "${SEED}" \
  --kd "{\"few_shot\": ${FEW_SHOT}, \"seed\": ${SEED}, \"lr\": 1e-5, \"epoch_selection\": \"last\", \"micro_batch_size\": ${MICRO}}" \
  --run-name "${EXP}_${SELECTED_DATASET}_fs${FEW_SHOT}_seed${SEED}" \
  --local-log-dir ".log/${EXP}/summaries"
