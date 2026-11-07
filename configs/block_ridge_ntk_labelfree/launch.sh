#!/usr/bin/env bash
#
# Submit the label-free ntk block_ridge / global_ridge runs in this directory, 3 seeds each.
#
#   configs/block_ridge_ntk_labelfree/launch.sh [task ...]      # default: all six tasks
#
# Environment (all optional):
#   RESULTS_ROOT  where the run directories go (default below; never the grid's root)
#   SEEDS         default "33 54 89"
#   CELLS         cell names to run (the part of the file name after "<task>_");
#                 default: every <task>_*.json in this directory
#   DRY_RUN=1     resolve and print, do not submit (passed through to submit_text_rebase.sh)
#
# Every config is already fully resolved; the only override added here is the seed.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
HERE=configs/block_ridge_ntk_labelfree

export RESULTS_ROOT="${RESULTS_ROOT:-/work/intesasanpaolo_phd/merge-and-rebase/t5enc_steer_text_labelfree}"
SEEDS="${SEEDS:-33 54 89}"
CELLS="${CELLS:-}"
TASKS=("$@")
[[ ${#TASKS[@]} -eq 0 ]] && TASKS=(mnli qnli snli scitail sick rte)

for task in "${TASKS[@]}"; do
  if [[ -n "$CELLS" ]]; then
    configs=(); for cell in $CELLS; do configs+=("$HERE/${task}_${cell}.json"); done
  else
    configs=("$HERE/${task}"_*.json)
  fi
  for config in "${configs[@]}"; do
    [[ -f "$config" ]] || { echo "missing $config" >&2; exit 1; }
    cell="$(basename "$config" .json)"; cell="${cell#"${task}"_}"
    # block_ridge/joint_ridge load and promote the per-block train/test tensors (see grid.py
    # RESOURCES); a non-mean block pooling also recollects B's blocks once per task (a plain
    # forward of B, then cached), and joint_ridge builds one [n, 25600] design matrix.
    case "$cell" in
      global_ridge) mem=8G; time=00:15:00 ;;
      *) mem=24G; time=00:45:00 ;;
    esac
    for seed in $SEEDS; do
      MEM="$mem" TIME="$time" scripts/slurm/submit_text_rebase.sh \
        "${task}_labelfree_${cell}_seed${seed}" "$config" "seed=${seed}"
    done
  done
done
