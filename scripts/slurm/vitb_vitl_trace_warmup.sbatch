#!/bin/bash

#SBATCH --job-name=vitb_vitl_trace_warmup
# L'output di default di SLURM va in una cartella generica (funge da backup)
#SBATCH --output=.log/vitb_vitl_trace/slurm_backup/warmup_%A_%a.out
#SBATCH --error=.log/vitb_vitl_trace/slurm_backup/warmup_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --array=0-7

#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=boost_usr_prod

# Warm-up della cache delle feature, un array task per dataset. La cache e' indicizzata da
# sorgente/target/task/regime/split (non da few_shot ne' seed) e la scrittura non e' atomica:
# le feature vanno calcolate UNA volta per dataset prima che le 120 run le leggano.
# Usa il config few_shot=1, seed=33 ristretto al dataset; il risultato di questa run e' scartabile.

EXP="vitb_vitl_trace"
CFG_DIR="configs/vision8_steer_trace_vitl"
PREFIX="vision8_steer_trace_vitl"

DATASETS=(
"Cars"
"DTD"
"EuroSAT"
"GTSRB"
"MNIST"
"RESISC45"
"SUN397"
"SVHN"
)
SELECTED_DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

LOG_DIR=".log/${EXP}/warmup"
mkdir -p "$LOG_DIR"
mkdir -p ".log/${EXP}/slurm_backup"
mkdir -p ".log/${EXP}/summaries"

CUSTOM_LOG="${LOG_DIR}/${SELECTED_DATASET}_job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
exec > "$CUSTOM_LOG" 2>&1

echo "Warm-up task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Selected dataset: ${SELECTED_DATASET}"
echo "Node: $(hostname)"
echo "Allocated GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Log file: ${CUSTOM_LOG}"

PYTHONUNBUFFERED=1 PYTHONPATH="${SLURM_SUBMIT_DIR}/src" python -m merge_and_rebase.eval.vision_rebase \
  --config "${CFG_DIR}/${PREFIX}_fs1_seed33.json" \
  --tasks "${SELECTED_DATASET}" \
  --run-name "${EXP}_warmup_${SELECTED_DATASET}" \
  --local-log-dir ".log/${EXP}/summaries"
