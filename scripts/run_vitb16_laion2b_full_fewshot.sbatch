#!/bin/bash

#SBATCH --job-name=vitb16_full_fewshot
#SBATCH --chdir=/homes/bcalderara/merge-and-rebase
#SBATCH --output=.log/vitb16_full_fewshot/slurm_backup/job_%A_%a.out
#SBATCH --error=.log/vitb16_full_fewshot/slurm_backup/job_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --array=0-14
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --constraint="gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G"
#SBATCH --account=bcalde_phd
#SBATCH --partition=all_usr_prod

set -euo pipefail

PROJECT_ROOT="/homes/bcalderara/merge-and-rebase"
cd "$PROJECT_ROOT"
WORK_ROOT="${WORK_ROOT:-/work/intesasanpaolo_phd/merge-and-rebase}"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
# Use the environment's Hugging Face cache, as sbatch_granularity.sh does.
export HF_HUB_DISABLE_PROGRESS_BARS=1
# Same deterministic preprocessing as existing calibration experiments.
export MR_SVHN_PREPROCESS=1
export MR_SVHN_TARGET_SIZE=96
export MR_SVHN_AUGMENTATION=none

TASKS=(GTSRB RESISC45 SUN397)
SHOTS=(1 2 5 10 20)
SEED=42
INDEX=${SLURM_ARRAY_TASK_ID:?Submit with sbatch}
N_SHOTS=${#SHOTS[@]}
TASK=${TASKS[$((INDEX / N_SHOTS))]}
K=${SHOTS[$((INDEX % N_SHOTS))]}

LOG_DIR=".log/vitb16_full_fewshot/${TASK}/${K}_shots"
mkdir -p "$LOG_DIR"
CUSTOM_LOG="${LOG_DIR}/seed_${SEED}_job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
exec >"$CUSTOM_LOG" 2>&1
CONFIG="${LOG_DIR}/seed_${SEED}_job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_config.yaml"

echo "Model: ViT-B-16/laion2b_s34b_b88k; full image encoder fine-tuning; fixed zero-shot text head"
echo "Dataset: ${TASK}; shots: ${K}; seed: ${SEED}"
echo "SVHN preprocess: ${MR_SVHN_PREPROCESS}; target: ${MR_SVHN_TARGET_SIZE}; augmentation: ${MR_SVHN_AUGMENTATION}"
echo "Node: $(hostname); GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Log file: ${CUSTOM_LOG}; config: ${CONFIG}"

# Materialize the exact config used by this run, also retained with the logs.
"$PYTHON" - "$CONFIG" "$K" "$SEED" "$WORK_ROOT" "$TASK" "$LOG_DIR" <<'PY'
import sys
from pathlib import Path
import yaml

output, k, seed, work, task, log_dir = sys.argv[1:]
base = Path('src/merge_and_rebase/finetune/configs/vision-full-vitb16-laion2b-fewshot.yaml')
cfg = yaml.safe_load(base.read_text())
common = cfg['common']
common['seed'] = int(seed)
common['data']['shots_per_class'] = int(k)
common['output']['out_dir'] = str(Path(work) / 'checkpoints/full_fewshot' / f'K{k}' / f'seed{seed}')
common['logging']['local_log_dir'] = log_dir
cfg['datasets_order'] = [task]
Path(output).write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

"$PYTHON" -m merge_and_rebase.finetune.train_vision \
  --vision-config "$CONFIG" --datasets "$TASK" --device cuda
