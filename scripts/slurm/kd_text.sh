#!/bin/bash

#SBATCH --job-name=kd_text
#SBATCH --output=.log/kd_text/slurm_backup/job_%A_%a.out
#SBATCH --error=.log/kd_text/slurm_backup/job_%A_%a.err
#SBATCH --time=3:00:00
#SBATCH --array=0-4

#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=boost_usr_prod

# Few-shot KD baseline per lo steering testo (eval/kd_rebase.py text), SNLI: few_shot x seed = 5 x 1 = 5 task.
#   PAIR=t5base_t5large      -> T5-base -> T5-large
#   PAIR=t5base_robertabase  -> T5-base -> RoBERTa-base
# L'unico teacher T5-base disponibile in questo checkout e' varun-v-rao/t5-base-snli (Hub): quindi solo SNLI.
# La head nearest-mean dello student (baseline + punto di partenza) dipende da few_shot e seed: se manca in
# HEADS_DIR la costruisce scripts/build_nearest_mean_head.py sul modello student, con lo stesso few_shot/seed.
# Da lanciare dalla root del repo:
#   mkdir -p .log/kd_text/slurm_backup
#   sbatch scripts/slurm/kd_text.sh t5base_robertabase
# lr e altri iperparametri: blocco "kd" del config (lr=1e-5); few_shot/seed arrivano da --kd qui sotto.

PAIR="${1:-${PAIR:-t5base_t5large}}"
EXP="kd_text_${PAIR}"
CONFIG="configs/kd/text_kd_${PAIR}.json"
HEADS_DIR="${HEADS_DIR:-.log/kd_text/heads/${PAIR}}"
case "$PAIR" in
  t5base_t5large)     TGT="google/t5-v1_1-large"; ARCH="t5" ;;
  t5base_robertabase) TGT="FacebookAI/roberta-base"; ARCH="auto" ;;
  *) echo "PAIR sconosciuto: $PAIR"; exit 2 ;;
esac

FEW_SHOTS=(1 2 5 10 20)
SEEDS=(33)
N_SEEDS=${#SEEDS[@]}

SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))]}
FEW_SHOT=${FEW_SHOTS[$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))]}
HEADS="${HEADS_DIR}/snli_nearest_mean_seed${SEED}_fewshot${FEW_SHOT}.pt"

LOG_DIR=".log/${EXP}/${FEW_SHOT}_shots"
mkdir -p "$LOG_DIR" ".log/${EXP}/summaries" ".log/kd_text/slurm_backup" "$HEADS_DIR"

CUSTOM_LOG="${LOG_DIR}/seed_${SEED}_job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
exec > "$CUSTOM_LOG" 2>&1

echo "Task ${SLURM_ARRAY_TASK_ID}: pair=${PAIR} few_shot=${FEW_SHOT} seed=${SEED} heads=${HEADS}"
echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES}"

source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTHONPATH="${SLURM_SUBMIT_DIR}/src:${SLURM_SUBMIT_DIR}"
export HF_HOME="${HF_HOME:-/work/intesasanpaolo_phd/merge-and-rebase/hf_cache}"

if [ ! -f "$HEADS" ]; then
  python -m scripts.build_nearest_mean_head --model-name-or-path "$TGT" --model-arch "$ARCH" \
    --task snli --few-shot "${FEW_SHOT}" --seed "${SEED}" --batch-size 16 --max-length 256 --output "$HEADS" || exit 1
fi

python -m merge_and_rebase.eval.kd_rebase text \
  --config "${CONFIG}" \
  --seed "${SEED}" \
  --batch-size 16 \
  --target-task-heads "${HEADS}" \
  --kd "{\"few_shot\": ${FEW_SHOT}, \"seed\": ${SEED}, \"lr\": 1e-5}" \
  --run-name "${EXP}_snli_fs${FEW_SHOT}_seed${SEED}" \
  --local-log-dir ".log/${EXP}/summaries"
