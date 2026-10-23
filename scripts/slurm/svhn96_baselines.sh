#!/bin/bash

#SBATCH --job-name=svhn96_baselines
#SBATCH --chdir=/homes/fbozzoli/merge-and-rebase
#SBATCH --output=.log/svhn96_baselines_slurm/job_%A_%a.out
#SBATCH --error=.log/svhn96_baselines_slurm/job_%A_%a.err
#SBATCH --time=4:00:00

#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --constraint="gpu_RTX_A5000_24G|gpu_RTX6000_24G|gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=all_usr_prod

# gradfix / theseus / bico, ViT-B/16 datacomp -> ViT-B/16 laion2b, SVHN only, with resize-96 + pad (no
# augmentation) on source and target (MR_SVHN_PREPROCESS=1), K in {1,2,5,10,20} examples per class, seeds 33/54/89,
# alpha searched on val (0-10, step 0.2, shared). Configs: configs/vision8_svhn96_baselines/<method>_fs<K>_seed<S>.json.
#   sbatch --array=0-14 scripts/slurm/svhn96_baselines.sh <gradfix|theseus|bico>     (array index = K x seed)

METHOD="${1:?usage: $0 gradfix|theseus|bico}"
case "$METHOD" in gradfix|theseus|bico) ;; *) echo "unknown method $METHOD" >&2; exit 1 ;; esac
EXP="vitb_vitb_svhn96_${METHOD}"
FEW_SHOTS=(1 2 5 10 20)
SEEDS=(33 54 89)
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % 3 ))]}
FEW_SHOT=${FEW_SHOTS[$(( SLURM_ARRAY_TASK_ID / 3 ))]}

LOG_DIR=".log/${EXP}/SVHN/${FEW_SHOT}_shots"
mkdir -p "$LOG_DIR" ".log/${EXP}/summaries"
exec > "${LOG_DIR}/seed_${SEED}_job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>&1
echo "${EXP} few_shot=${FEW_SHOT} seed=${SEED} node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES}"

source .venv/bin/activate
export MR_SVHN_PREPROCESS=1 MR_SVHN_TARGET_SIZE=96 MR_SVHN_AUGMENTATION=none
PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/src" python -m merge_and_rebase.eval.vision_rebase \
  --config "configs/vision8_svhn96_baselines/${METHOD}_fs${FEW_SHOT}_seed${SEED}.json" \
  --tasks SVHN \
  --run-name "${EXP}_SVHN_fs${FEW_SHOT}_seed${SEED}" \
  --local-log-dir ".log/${EXP}/summaries"
