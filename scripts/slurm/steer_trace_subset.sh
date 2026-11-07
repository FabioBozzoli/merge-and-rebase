#!/bin/bash

#SBATCH --job-name=steer_trace_subset
#SBATCH --chdir=/homes/fbozzoli/merge-and-rebase
#SBATCH --output=.log/steer_trace_subset_slurm/job_%A_%a.out
#SBATCH --error=.log/steer_trace_subset_slurm/job_%A_%a.err
#SBATCH --time=0:45:00
#SBATCH --array=0-11

#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --constraint="gpu_2080_11G|gpu_2080Ti_11G|gpu_RTX5000_16G|gpu_RTX_A5000_24G|gpu_RTX6000_24G|gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=all_usr_prod

# Risorse dimensionate sui run precedenti con gli stessi config (job 118969 vitb, 118990 vitl):
#   vitb: MaxRSS <= 6 GB (SUN397 11 GB), elapsed <= 2.5 min (SUN397 17 min), VRAM ~3.6 GB.
#   vitl: MaxRSS <= 15 GB, elapsed <= 4 min (SUN397 23 min) -> lanciare con
#         sbatch --mem=24G --time=1:00:00 --constraint="<GPU >= 24GB>" scripts/slurm/steer_trace_subset.sh vitl
# Feature in cache -> la GPU serve solo per il forward di eval: non serve boost_usr_prod (>24GB VRAM).

# Steer block_ridge da ViT-B/16 datacomp:
#   arg vitb (default): -> ViT-B/16 laion2b, 12 resblock + final = 13 ridge.
#   arg vitl:            -> ViT-L/14 datacomp, 24 resblock di B concatenati a coppie (concat) in 12 gruppi + final = 13 ridge.
#   stage1_lambda=1.0, lambda2 trace-scaled (ridge_lambda=0.1), smoothed_residual rho=0.9,
#   block_granularity=residual, mean token pooling, alpha=1 fisso.
# Config: configs/vision8_steer_trace{,_vitl}/vision8_steer_trace{,_vitl}_fs<K>_seed<S>.json
# Feature linear gia' in cache: /work/intesasanpaolo_phd/merge-and-rebase/features/linear_feature
# Run: GTSRB/RESISC45/DTD con few_shot=10, SUN397 con few_shot=1; seed 33,54,89 -> 12 task.

# Coppia come argomento posizionale: sbatch --export=...,PAIR=... viene cancellato dal cluster (root) all avvio.
PAIR="${1:-vitb}"
case "$PAIR" in
  vitb) EXP="vitb_vitb_trace_subset"; CFG_DIR="configs/vision8_steer_trace"; PREFIX="vision8_steer_trace" ;;
  vitl) EXP="vitb_vitl_trace_subset"; CFG_DIR="configs/vision8_steer_trace_vitl"; PREFIX="vision8_steer_trace_vitl" ;;
  # global_mlp su checkpoint non lineari (feature standard): configs/vision8_steer_globalmlp_nonlinear/make_configs.py
  vitb_mlp) EXP="vitb_vitb_globalmlp_nonlinear_subset"; CFG_DIR="configs/vision8_steer_globalmlp_nonlinear"; PREFIX="globalmlp_nonlinear_vitb" ;;
  vitl_mlp) EXP="vitb_vitl_globalmlp_nonlinear_subset"; CFG_DIR="configs/vision8_steer_globalmlp_nonlinear"; PREFIX="globalmlp_nonlinear_vitl" ;;
  # varianti globali: <globalmlp|globalridge>_<linear|nonlinear>_<vitb|vitl>, configs/vision8_steer_global/make_configs.py
  global*_*linear_vit[bl])
    EXP="vitb_${PAIR##*_}_${PAIR%_*}_subset"; CFG_DIR="configs/vision8_steer_global"; PREFIX="$PAIR" ;;
  *) echo "Unknown PAIR=$PAIR" >&2; exit 1 ;;
esac

RUNS=(
"GTSRB:10"
"RESISC45:10"
"DTD:10"
"SUN397:1"
)
SEEDS=(33 54 89)

N_SEEDS=${#SEEDS[@]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))]}
RUN=${RUNS[$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))]}
SELECTED_DATASET=${RUN%%:*}
FEW_SHOT=${RUN##*:}

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
PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/src" python -m merge_and_rebase.eval.vision_rebase \
  --config "${CFG_DIR}/${PREFIX}_fs${FEW_SHOT}_seed${SEED}.json" \
  --tasks "${SELECTED_DATASET}" \
  --run-name "${EXP}_${SELECTED_DATASET}_fs${FEW_SHOT}_seed${SEED}" \
  --local-log-dir ".log/${EXP}/summaries"
