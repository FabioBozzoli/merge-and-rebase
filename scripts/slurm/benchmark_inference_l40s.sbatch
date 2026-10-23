#!/bin/bash

#SBATCH --job-name=bench_inference_l40s
#SBATCH --chdir=/homes/fbozzoli/merge-and-rebase
#SBATCH --output=.log/benchmark_inference/slurm_backup/job_%j.out
#SBATCH --error=.log/benchmark_inference/slurm_backup/job_%j.err
#SBATCH --time=4:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --constraint="gpu_L40S_45G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=boost_usr_prod

# Costo a inference time (latenza, memoria, FLOPs/TFLOPS) dei percorsi del repo su L40S, con pesi
# random (misura il costo dell'architettura, non l'accuratezza): zero-shot, steer global_ridge /
# global_mlp / block_ridge (residual, 13 blocchi; linear, 50 siti), forward linearized_ntk.
# Tutti i casi girano in sequenza sulla STESSA GPU, un processo per caso (niente stato condiviso).
# Ogni campione e' appaiato con un forward zero-shot (baseline) in ordine alternato.
#   mkdir -p .log/benchmark_inference/slurm_backup && sbatch scripts/slurm/benchmark_inference_l40s.sbatch
# Risultati: logs/benchmark_inference/<model>/<mode>_b<batch>.json

set -uo pipefail
source .venv/bin/activate
export PYTHONUNBUFFERED=1

MODELS=(ViT-B-16 ViT-L-14)
MODES=(zeroshot global_ridge global_mlp block_smoothed block_linear ntk ntk_single)
BATCHES=(1 32 128)
WARMUP=5
REPEAT=20
OUT_DIR="logs/benchmark_inference"
LOG_DIR=".log/benchmark_inference/job_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv

failed=0
for model in "${MODELS[@]}"; do
  for mode in "${MODES[@]}"; do
    for batch in "${BATCHES[@]}"; do
      tag="${model}/${mode}_b${batch}"
      echo "=== ${tag} ($(date +%T))"
      if ! python scripts/benchmark_inference.py --repo . --device cuda --model "$model" --mode "$mode" \
          --batch "$batch" --threads 4 --warmup "$WARMUP" --repeat "$REPEAT" --count-flops \
          --output "${OUT_DIR}/${tag}.json" > "${LOG_DIR}/${model}_${mode}_b${batch}.log" 2>&1; then
        echo "FAILED: ${tag} (see ${LOG_DIR}/${model}_${mode}_b${batch}.log)"
        failed=$((failed + 1))
      fi
    done
  done
done
echo "done, failed cases: ${failed}"
