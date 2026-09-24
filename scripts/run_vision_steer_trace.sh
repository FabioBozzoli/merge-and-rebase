#!/usr/bin/env bash
# Sottomette ViT-B/16 -> ViT-B/16 (block_ridge trace-lambda, 8 dataset x few_shot 1/2/5/10/20 x seed 33/54/89).
# Da lanciare dalla root del repo:  scripts/run_vision_steer_trace.sh
#
#   - run:     array 0-119 (dataset x few_shot x seed) su scripts/slurm/vitb_vitb_trace.sbatch
#   - warm-up: array 0-7 (un job per dataset) che riempie la cache delle feature. E' AUTOMATICO:
#              il launcher controlla che in feature_cache_dir esistano i file di cache (train e test) di
#              ogni dataset; se ci sono salta il warm-up e sottomette solo le run, altrimenti sottomette
#              il warm-up e fa partire le run con dipendenza afterok (la scrittura della cache non e'
#              atomica: 120 job non devono ricalcolarla insieme).
#
#   WARMUP=1       forza il warm-up        SKIP_WARMUP=1   lo salta comunque
# Log: .log/vitb_vitb_trace/
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXP="vitb_vitb_trace"
CFG="configs/vision8_steer_trace/vision8_steer_trace_fs1_seed33.json"
# SLURM non crea le cartelle di --output/--error: devono esistere prima di sbatch.
mkdir -p ".log/${EXP}/slurm_backup" ".log/${EXP}/warmup" ".log/${EXP}/summaries"

# Dataset/split senza cache completa (le stesse chiavi che _load_cached_split richiede per block_ridge).
missing="$(python3 - "$CFG" <<'PY'
import json, pathlib, sys
c = json.load(open(sys.argv[1]))
root = pathlib.Path(c["method_params"]["feature_cache_dir"])
pair = f'{c["source_clip_model"]}_{c["source_clip_pretrained"]}_to_{c["target_clip_model"]}_{c["target_clip_pretrained"]}'
need = ["features_A", "delta_A", "features_B", "y_A", "delta_A_blocks", "features_B_blocks"]
for task in c["tuned_ckpts"]:
    for split in ("train", "test"):
        d = root / pair / task / "linear" / split
        if not all((d / f"{n}.pt").exists() for n in need):
            print(f"{task}/{split}")
PY
)"

warmup=0
if [[ "${WARMUP:-0}" == "1" ]]; then
  warmup=1
elif [[ "${SKIP_WARMUP:-0}" != "1" && -n "$missing" ]]; then
  warmup=1
  echo "cache incompleta, warm-up necessario per: $(echo "$missing" | tr '\n' ' ')"
fi

dep=()
if [[ "$warmup" == "1" ]]; then
  warm_id="$(sbatch --parsable "scripts/slurm/${EXP}_warmup.sbatch")"
  echo "warm-up array job: ${warm_id}"
  dep=(--dependency="afterok:${warm_id}" --kill-on-invalid-dep=yes)
elif [[ -n "$missing" ]]; then
  echo "ATTENZIONE: warm-up saltato (SKIP_WARMUP=1) ma la cache risulta incompleta per: $(echo "$missing" | tr '\n' ' ')"
else
  echo "cache gia' presente: nessun warm-up"
fi

run_id="$(sbatch --parsable "${dep[@]}" "scripts/slurm/${EXP}.sbatch")"
echo "run array job:     ${run_id}"
echo "log strutturati:   .log/${EXP}/<DATASET>/<K>_shots/seed_<S>_job_${run_id}_<idx>.log"
echo "backup SLURM:      .log/${EXP}/slurm_backup/"
echo "monitor:           squeue -u \$USER"
