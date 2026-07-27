#!/bin/bash -l
#SBATCH --job-name=rebase_fix_test
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=250G
#SBATCH --time=06:00:00
#SBATCH --account=IscrB_MOSAIC
#SBATCH --output=slurm_logs/rebase_fix_test.out

set -e

HOME_DIR="/leonardo/home/userexternal/frinaldi/merge-and-rebase-dev"
MAIN_REPO="/leonardo/home/userexternal/frinaldi/merge-and-rebase"
PYTHON="$HOME_DIR/.venv/bin/python"
CONFIG="$MAIN_REPO/configs/vitb32_theseus_eurosat_test.json"

CACHE_ROOT="/leonardo_scratch/large/userexternal/frinaldi/.hf_cache"
export HF_HOME="$CACHE_ROOT"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

cd "$MAIN_REPO"

PYTHONPATH="$MAIN_REPO/src" $PYTHON -m merge_and_rebase.eval.vision_rebase \
  --config "$CONFIG" \
  --device cuda
