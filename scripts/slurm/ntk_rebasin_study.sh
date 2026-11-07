#!/bin/bash

#SBATCH --job-name=ntk_rebasin_study
#SBATCH --chdir=/homes/fbozzoli/merge-and-rebase-ntk
#SBATCH --output=/work/intesasanpaolo_phd/merge-and-rebase/ntk_rebasin_study/_slurm/job_%A_%a.out
#SBATCH --error=/work/intesasanpaolo_phd/merge-and-rebase/ntk_rebasin_study/_slurm/job_%A_%a.err
#SBATCH --time=0:30:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --constraint="gpu_2080_11G|gpu_2080Ti_11G|gpu_RTX5000_16G|gpu_RTX_A5000_24G|gpu_RTX6000_24G|gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G"
#SBATCH --account=intesasanpaolo_phd
#SBATCH --partition=all_usr_prod

# NTK-vs-nonlinear steer_text study (configs/ntk_rebasin_study/make_configs.py).
#
#   sbatch --array=0-<ceil(N/CHUNK)-1>%<K> scripts/slurm/ntk_rebasin_study.sh <manifest> [<results-root>] [<feature-cache-dir>] [<chunk>]
#
# <manifest>: one config path per line (relative to the repo). Array index i runs lines
#   [i*CHUNK, (i+1)*CHUNK) one after the other (default CHUNK=1). With the sorted full manifest and CHUNK=9,
#   one job is one (target, task, cell): 3 few_shot x 3 seeds, sharing one GPU allocation.
# <results-root>: default /work/.../ntk_rebasin_study. A config at configs/ntk_rebasin_study/<target>/<task>/<stem>.json
#   writes to <results-root>/<target>/<task>/<stem>/summary.json; a cell whose summary.json exists is skipped.
# <feature-cache-dir>: optional override of method_params.feature_cache_dir (used by the cache-reproduction check).
# Arguments are positional because sbatch --export is reset on this cluster.
#
# Uses the venv of /homes/fbozzoli/merge-and-rebase (identical uv.lock); its editable install points at
# that checkout, so PYTHONPATH must put this clone's src first.

MANIFEST="$1"
RESULTS_ROOT="${2:-/work/intesasanpaolo_phd/merge-and-rebase/ntk_rebasin_study}"
CACHE_OVERRIDE="${3:-}"
CHUNK="${4:-1}"

EXTRA=()
[ -n "$CACHE_OVERRIDE" ] && EXTRA+=(--feature-cache-dir "$CACHE_OVERRIDE")

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
  REL=${CFG#configs/ntk_rebasin_study/}
  OUT="${RESULTS_ROOT}/${REL%.json}"
  if [ -f "$OUT/summary.json" ]; then
    echo "skip: $OUT/summary.json exists"
    continue
  fi
  mkdir -p "$OUT"
  echo "=== $(date +%T) config: $CFG -> $OUT"
  /homes/fbozzoli/merge-and-rebase/.venv/bin/python -m merge_and_rebase.eval.text_rebase \
    --config "$CFG" --local-log-dir "$OUT" --run-name summary "${EXTRA[@]}" || { echo "FAILED: $CFG"; FAILED=$(( FAILED + 1 )); }
done < <(sed -n "${FIRST},${LAST}p" "$MANIFEST")
echo "=== $(date +%T) done, failed=$FAILED"
[ "$FAILED" -eq 0 ]
