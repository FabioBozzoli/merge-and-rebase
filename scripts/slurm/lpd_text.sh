#!/bin/bash

#SBATCH --job-name=lpd_text
#SBATCH --chdir=/homes/fbozzoli/steer-unified
#SBATCH --output=/work/fbozzoli_phd/steer_lpd/_slurm/text_%A_%a.out
#SBATCH --error=/work/fbozzoli_phd/steer_lpd/_slurm/text_%A_%a.err
#SBATCH --time=0:20:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --constraint="gpu_2080_11G|gpu_2080Ti_11G|gpu_RTX5000_16G|gpu_RTX_A5000_24G|gpu_RTX6000_24G|gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G"
#SBATCH --account=fbozzoli_phd
#SBATCH --partition=all_usr_prod

# Linear probe distillation (stage1_mode=logit_distill; BASELINE, not steer) for steer_text, t5-base -> {t5-large, roberta-base}.
# Configs: configs/ntk_lpd/<target>/<task>/lpd_lam<L>_fs<K>_seed<S>.json (make_configs.py), trace-scaled ridge_lambda.
# Features only from the cache .../text_feature/results/ntk_segment_pooling/feature_cache (force_recompute=false).
#   for t in t5-large roberta-base; do LC_ALL=C ls configs/ntk_lpd/$t/*/*.json > <manifest_$t>; done
#   sbatch --array=0-29 scripts/slurm/lpd_text.sh <manifest_$t> 9
# With the C-sorted per-target manifest and CHUNK=9 one job = one (task, lambda): 3 few_shot x 3 seeds.
# A config configs/ntk_lpd/<target>/<task>/<stem>.json writes <ROOT>/<target>/<task>/<stem>/summary.json; existing summaries are skipped.

MANIFEST="$1"
CHUNK="${2:-1}"
# Outputs go to /work/fbozzoli_phd (the shared /work/intesasanpaolo_phd hit its disk quota on 2026-11-10); features/HF cache are READ
# from /work/intesasanpaolo_phd. LEGACY_ROOT holds the runs finished before the move.
ROOT=/work/fbozzoli_phd/steer_lpd/text
LEGACY_ROOT=/work/intesasanpaolo_phd/merge-and-rebase/steer_lpd/text

export HF_HOME=/work/intesasanpaolo_phd/merge-and-rebase/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1   # the HF cache lives on the full /work/intesasanpaolo_phd: read only
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src:$PWD"
echo "node:   $(hostname)  gpu: ${CUDA_VISIBLE_DEVICES}"
echo "commit: $(git rev-parse HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"

FIRST=$(( SLURM_ARRAY_TASK_ID * CHUNK + 1 ))
LAST=$(( FIRST + CHUNK - 1 ))
# Fail loudly (never "succeed" on zero runs): the manifest must be on a shared filesystem (/work), not node-local /tmp.
[ -r "$MANIFEST" ] || { echo "FATAL: manifest not readable: $MANIFEST" >&2; exit 2; }
NCFG=$(sed -n "${FIRST},${LAST}p" "$MANIFEST" | grep -c .)
[ "$NCFG" -gt 0 ] || { echo "FATAL: no configs in lines ${FIRST}-${LAST} of $MANIFEST" >&2; exit 2; }
FAILED=0
RAN=0
while read -r CFG; do
  [ -n "$CFG" ] || continue
  REL=${CFG#configs/ntk_lpd/}
  OUT="${ROOT}/${REL%.json}"
  if [ -f "$OUT/summary.json" ] || [ -f "${LEGACY_ROOT}/${REL%.json}/summary.json" ]; then echo "skip: summary exists for ${REL%.json}"; continue; fi
  mkdir -p "$OUT"
  RAN=$(( RAN + 1 ))
  echo "=== $(date +%T) config: $CFG -> $OUT"
  /homes/fbozzoli/merge-and-rebase/.venv/bin/python -m merge_and_rebase.eval.text_rebase \
    --config "$CFG" --local-log-dir "$OUT" --run-name summary || { echo "FAILED: $CFG"; FAILED=$(( FAILED + 1 )); }
done < <(sed -n "${FIRST},${LAST}p" "$MANIFEST")
echo "=== $(date +%T) done, configs=$NCFG ran=$RAN failed=$FAILED"
[ "$FAILED" -eq 0 ]
