#!/bin/bash

#SBATCH --job-name=steer_repro_check
#SBATCH --chdir=/homes/fbozzoli/merge-and-rebase
#SBATCH --output=.log/repro_check/slurm_backup/job_%j.out
#SBATCH --error=.log/repro_check/slurm_backup/job_%j.err
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=boost_usr_prod

# Verifica di riproducibilita' del report "Reproducing the Vision8 results of the steer method":
# un run = il comando del report, `python -m merge_and_rebase.eval.vision_rebase --config <cfg>`, tutte le
# 8 task del config, nessun override dei parametri. --run-name/--local-log-dir cambiano solo dove finisce il
# summary (il nome di default dipende dal secondo di avvio e job concorrenti si sovrascriverebbero).
# Registra ambiente, hardware e commit accanto al log.
#   sbatch scripts/slurm/repro_check.sbatch <config.json> <nome>
# (niente --export=ALL,...: su questo cluster i job sottomessi cosi' vengono cancellati da root all'avvio)

set -uo pipefail
CFG="${1:?usage: sbatch repro_check.sbatch <config.json> <name>}"
NAME="${2:?usage: sbatch repro_check.sbatch <config.json> <name>}"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src"

OUT=".log/repro_check/${NAME}"
mkdir -p "$OUT"
exec > "${OUT}/run_${SLURM_JOB_ID}.log" 2>&1

echo "Config: ${CFG}"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
{
  echo "commit: $(git rev-parse HEAD)"; git status --short src configs
  python --version; python -c "import torch, open_clip; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version()); print('open_clip', open_clip.__version__)"
  nvidia-smi
  pip freeze
} > "${OUT}/environment_${SLURM_JOB_ID}.txt" 2>&1
sha256sum "$CFG" >> "${OUT}/environment_${SLURM_JOB_ID}.txt"
env | grep '^MR_SVHN' || echo "MR_SVHN_*: unset (default standard CLIP preprocessing)"

python -m merge_and_rebase.eval.vision_rebase --config "$CFG" \
  --run-name "${NAME}" --local-log-dir "${OUT}/summaries"
