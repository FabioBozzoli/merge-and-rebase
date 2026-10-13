#!/bin/bash

#SBATCH --chdir=/homes/fbozzoli/merge-and-rebase
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --constraint="gpu_2080_11G|gpu_2080Ti_11G|gpu_RTX5000_16G|gpu_RTX_A5000_24G|gpu_RTX6000_24G|gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=all_usr_prod

# Stage 1 energy-threshold ablation for steer concat_ridge (cached-feature space, no model forward):
# thresholds 0.9/0.95/0.99/0.999/1.0 x K 1,2,5,10,20 x seeds 33/54/89, one task per (regime, dataset) of <pair>.
# Array index i in 0..15: regime = (linear, nonlinear)[i / 8], dataset = DATASETS[i % 8].
#   sbatch --array=0-15 --mem=.. --output=.log/steer_stage1_energy_ablation/slurm/%x_%a.out scripts/slurm/stage1_energy_ablation.sh <vitb|vitl>
# Output: .log/steer_stage1_energy_ablation/<pair>_<regime>_<Dataset>.json (+ reproduction check vs the grid logs).

PAIR="$1"
DATASETS=(EuroSAT SVHN Cars RESISC45 DTD GTSRB MNIST SUN397)
REGIMES=(linear nonlinear)
I="$SLURM_ARRAY_TASK_ID"
REGIME="${REGIMES[$((I / 8))]}"
TASK="${DATASETS[$((I % 8))]}"
CODE_ROOT=/work/intesasanpaolo_phd/merge-and-rebase/code_snapshots/concatridge_20261012
source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="${CODE_ROOT}/src"
echo "stage1 energy ablation | ${PAIR} ${REGIME} ${TASK} | node=$(hostname) | code ${CODE_ROOT}"
python scripts/analyze_stage1_energy.py --pair "$PAIR" --regime "$REGIME" --task "$TASK" \
  --out .log/steer_stage1_energy_ablation
