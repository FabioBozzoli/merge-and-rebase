#!/bin/bash

#SBATCH --job-name=text_probe_segments
#SBATCH --chdir=/homes/fbozzoli/merge-and-rebase-ntk
#SBATCH --output=/work/intesasanpaolo_phd/merge-and-rebase/text_probe_segments/_slurm/job_%A_%a.out
#SBATCH --error=/work/intesasanpaolo_phd/merge-and-rebase/text_probe_segments/_slurm/job_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --constraint="gpu_2080_11G|gpu_2080Ti_11G|gpu_RTX5000_16G|gpu_RTX_A5000_24G|gpu_RTX6000_24G|gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=all_usr_prod

# Segment-pooled vs global few-shot linear probe (configs/text_probe_segments/make_configs.py).
#
#   sbatch --array=0-<N-1>%<K> scripts/slurm/text_probe_segments.sh <manifest> [<results-root>]
#
# <manifest>: one config path per line (relative to the repo); array index i runs line i+1.
# A config at configs/text_probe_segments/<target>/<stem>.json writes to <results-root>/<target>/<stem>/summary.json;
# a config whose summary.json exists is skipped. No feature cache: the probe forwards the target itself.
# Arguments are positional because sbatch --export is reset on this cluster.
#
# Uses the venv of /homes/fbozzoli/merge-and-rebase (identical uv.lock); its editable install points at
# that checkout, so PYTHONPATH must put this clone's src first.

MANIFEST="$1"
RESULTS_ROOT="${2:-/work/intesasanpaolo_phd/merge-and-rebase/text_probe_segments}"
CHUNK=1
EXTRA=()

export HF_HOME=/work/intesasanpaolo_phd/merge-and-rebase/hf_cache
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src:$PWD"

echo "node:   $(hostname)  gpu: ${CUDA_VISIBLE_DEVICES}"
echo "commit: $(git rev-parse HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"

FIRST=$(( SLURM_ARRAY_TASK_ID * CHUNK + 1 ))
LAST=$(( FIRST + CHUNK - 1 ))
FAILED=0
while read -r CFG; do
  [ -n "$CFG" ] || continue
  REL=${CFG#configs/text_probe_segments/}
  OUT="${RESULTS_ROOT}/${REL%.json}"
  if [ -f "$OUT/summary.json" ]; then
    echo "skip: $OUT/summary.json exists"
    continue
  fi
  mkdir -p "$OUT"
  echo "=== $(date +%T) config: $CFG -> $OUT"
  /homes/fbozzoli/merge-and-rebase/.venv/bin/python -m merge_and_rebase.eval.text_linear_probe \
    --config "$CFG" --local-log-dir "$OUT" --run-name summary "${EXTRA[@]}" || { echo "FAILED: $CFG"; FAILED=$(( FAILED + 1 )); }
done < <(sed -n "${FIRST},${LAST}p" "$MANIFEST")
echo "=== $(date +%T) done, failed=$FAILED"
[ "$FAILED" -eq 0 ]
