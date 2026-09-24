#!/usr/bin/env bash
# Sottomette ViT-B/16 -> ViT-B/16 (block_ridge trace-lambda, 8 dataset x few_shot 1/2/5/10/20 x seed 33/54/89).
# Da lanciare dalla root del repo:  scripts/run_vision_steer_trace.sh
#
#   1. warm-up: array 0-7 (un job per dataset) che riempie la cache delle feature
#   2. run:     array 0-119 (dataset x few_shot x seed), parte quando TUTTO il warm-up e' finito (afterok)
#
# SKIP_WARMUP=1 se la cache e' gia' piena (nessuna dipendenza). Log: .log/vitb_vitb_trace/
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXP="vitb_vitb_trace"
# SLURM non crea le cartelle di --output/--error: devono esistere prima di sbatch.
mkdir -p ".log/${EXP}/slurm_backup" ".log/${EXP}/warmup" ".log/${EXP}/summaries"

dep=()
if [[ "${SKIP_WARMUP:-0}" != "1" ]]; then
  warm_id="$(sbatch --parsable "scripts/slurm/${EXP}_warmup.sbatch")"
  echo "warm-up array job: ${warm_id}"
  dep=(--dependency="afterok:${warm_id}" --kill-on-invalid-dep=yes)
fi

run_id="$(sbatch --parsable "${dep[@]}" "scripts/slurm/${EXP}.sbatch")"
echo "run array job:     ${run_id}"
echo "log strutturati:   .log/${EXP}/<DATASET>/<K>_shots/seed_<S>_job_${run_id}_<idx>.log"
echo "backup SLURM:      .log/${EXP}/slurm_backup/"
echo "monitor:           squeue -u \$USER"
