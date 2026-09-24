#!/bin/bash

#SBATCH --job-name=vitb_vitl_trace
# L'output di default di SLURM va in una cartella generica (funge da backup)
#SBATCH --output=.log/vitb_vitl_trace/slurm_backup/job_%A_%a.out
#SBATCH --error=.log/vitb_vitl_trace/slurm_backup/job_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --array=0-119

#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=boost_usr_prod

# ViT-B/16 -> ViT-L/14: block_ridge con lambda2 scalato dalla traccia (trace), smoothed_residual rho=0.9,
# ridge_lambda=0.1, alpha=1 fisso, feature a fine blocco (block_granularity=residual).
# Tutti gli iperparametri stanno nei config configs/vision8_steer_trace_vitl/vision8_steer_trace_vitl_fs<K>_seed<S>.json
# (generati da configs/vision8_steer_trace/make_configs.py).
#
# Da lanciare dalla root del repo. Le feature sono gia' estratte in
# /work/intesasanpaolo_phd/merge-and-rebase/features/linear_feature (feature_cache_dir dei config):
# il warm-up (vitb_vitl_trace_warmup.sbatch) serve solo se manca la cache di un dataset.
#   scripts/run_vision_steer_trace_vitl.sh     # controlla la cache e sottomette (warm-up +) questo array
# Le cartelle .log/vitb_vitl_trace/slurm_backup devono esistere PRIMA di sbatch (SLURM non le crea).

EXP="vitb_vitl_trace"
CFG_DIR="configs/vision8_steer_trace_vitl"
PREFIX="vision8_steer_trace_vitl"

# 8 dataset del benchmark Vision8
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
FEW_SHOTS=(1 2 5 10 20)
SEEDS=(33 54 89)

N_DATASETS=${#DATASETS[@]}
N_SHOTS=${#FEW_SHOTS[@]}
N_SEEDS=${#SEEDS[@]}

# 1. Matematica per lo sweep 3D:
# task_id mappa la griglia dataset_idx x shot_idx x seed_idx
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
TEMP_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SHOT_IDX=$(( TEMP_IDX % N_SHOTS ))
DATASET_IDX=$(( TEMP_IDX / N_SHOTS ))

SELECTED_DATASET=${DATASETS[$DATASET_IDX]}
FEW_SHOT=${FEW_SHOTS[$SHOT_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# 2. Creazione cartelle dinamiche
LOG_DIR=".log/${EXP}/${SELECTED_DATASET}/${FEW_SHOT}_shots"
mkdir -p "$LOG_DIR"
mkdir -p ".log/${EXP}/slurm_backup"
mkdir -p ".log/${EXP}/summaries"

# 3. Reindirizzamento dell'output logico nei path specifici
# Tutto ciò che viene stampato da qui in poi finirà nel file di log strutturato
CUSTOM_LOG="${LOG_DIR}/seed_${SEED}_job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
exec > "$CUSTOM_LOG" 2>&1

echo "Running task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Selected dataset: ${SELECTED_DATASET}"
echo "Few-shot: ${FEW_SHOT}"
echo "Seed: ${SEED}"
echo "Node: $(hostname)"
echo "Allocated GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Log file: ${CUSTOM_LOG}"

# --run-name/--local-log-dir unici: il summary di default prende il nome dal secondo di avvio,
# quindi array task partiti insieme si sovrascriverebbero.
PYTHONUNBUFFERED=1 PYTHONPATH="${SLURM_SUBMIT_DIR}/src" python -m merge_and_rebase.eval.vision_rebase \
  --config "${CFG_DIR}/${PREFIX}_fs${FEW_SHOT}_seed${SEED}.json" \
  --tasks "${SELECTED_DATASET}" \
  --run-name "${EXP}_${SELECTED_DATASET}_fs${FEW_SHOT}_seed${SEED}" \
  --local-log-dir ".log/${EXP}/summaries"
