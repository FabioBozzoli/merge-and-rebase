#!/usr/bin/env bash
# Submit ALL vision steer-trace runs (few_shot x seed) to SLURM, in parallel.
#
#   scripts/run_vision_steer_trace.sh                 # ViT-B/16 -> ViT-B/16 (configs/vision8_steer_trace)
#   scripts/run_vision_steer_trace_vitl.sh            # ViT-B/16 -> ViT-L/14 (configs/vision8_steer_trace_vitl)
#
# Two stages, both fully parallel:
#   1. cache warm-up: one job per task (fs=1, seed=33 config restricted to that task). The feature cache
#      is keyed by source/target/task/regime/split -- not few_shot or seed -- and its writes are not
#      atomic, so the features must exist before anything reads them; per-task jobs write disjoint dirs.
#   2. the 15 runs, released together (afterok) once the warm-up jobs they need have finished. Each run
#      only refits Stage 2 and evaluates. SPLIT_TASKS=1 splits every run into one job per task
#      (120 jobs, each waiting only for its own task's warm-up); the default is one job per run (15).
#
# Every job gets a unique --run-name/--local-log-dir: the default summary file is named after the start
# second, so simultaneous jobs would otherwise overwrite each other's summary.
#
# Knobs (environment variables):
#   CFG_DIR PREFIX RESULTS_ROOT                    which configs / where logs and summaries go
#   PARTITION ACCOUNT GRES CPUS QOS                sbatch resources (defaults: this project's cluster)
#   MEM TIME                                       per-run job;  MEM_WARM TIME_WARM: warm-up jobs
#   VENV_ACTIVATE=/path/bin/activate PYTHON_BIN    environment inside the jobs
#   SKIP_WARMUP=1                                  cache already filled: no warm-up, no dependencies
#   SPLIT_TASKS=1   DRY_RUN=1 (print the sbatch commands, submit nothing)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CFG_DIR="${CFG_DIR:-configs/vision8_steer_trace}"
PREFIX="${PREFIX:-vision8_steer_trace}"
RESULTS_ROOT="${RESULTS_ROOT:-results/$PREFIX}"
PARTITION="${PARTITION:-all_usr_prod}"
ACCOUNT="${ACCOUNT:-intesasanpaolo_phd}"
GRES="${GRES:-gpu:1}"
CPUS="${CPUS:-4}"
QOS="${QOS:-}"
MEM="${MEM:-48G}"
TIME="${TIME:-12:00:00}"
MEM_WARM="${MEM_WARM:-64G}"
TIME_WARM="${TIME_WARM:-24:00:00}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VENV_ACTIVATE="${VENV_ACTIVATE:-}"
SKIP_WARMUP="${SKIP_WARMUP:-0}"
SPLIT_TASKS="${SPLIT_TASKS:-0}"
DRY_RUN="${DRY_RUN:-0}"

LOG_DIR="$RESULTS_ROOT/slurm_logs"
SUMMARY_DIR="$RESULTS_ROOT/summaries"
mkdir -p "$LOG_DIR" "$SUMMARY_DIR"

WARM_CFG="$CFG_DIR/${PREFIX}_fs1_seed33.json"
[[ -f "$WARM_CFG" ]] || { echo "missing $WARM_CFG (run make_configs.py first)" >&2; exit 1; }
mapfile -t TASKS < <(python3 -c "import json,sys; print('\n'.join(json.load(open(sys.argv[1]))['tuned_ckpts']))" "$WARM_CFG")
mapfile -t CONFIGS < <(ls "$CFG_DIR"/${PREFIX}_fs*_seed*.json | sort -V)

# submit NAME DEP MEM TIME CONFIG [TASK]  -> prints the job id
submit() {
  local name="$1" dep="$2" mem="$3" time="$4" cfg="$5" task="${6:-}"
  local run_cmd
  run_cmd="cd $(printf %q "$ROOT_DIR")"
  [[ -n "$VENV_ACTIVATE" ]] && run_cmd+=" && source $(printf %q "$VENV_ACTIVATE")"
  run_cmd+=" && PYTHONPATH=src $PYTHON_BIN -m merge_and_rebase.eval.vision_rebase --config $(printf %q "$cfg") --device cuda"
  [[ -n "$task" ]] && run_cmd+=" --tasks $(printf %q "$task")"
  run_cmd+=" --run-name $(printf %q "$name") --local-log-dir $(printf %q "$SUMMARY_DIR")"

  local args=(--parsable --job-name="$name" --partition="$PARTITION" --account="$ACCOUNT" --gres="$GRES"
              --cpus-per-task="$CPUS" --mem="$mem" --time="$time" --nodes=1
              --output="$LOG_DIR/%x_%j.out")
  [[ -n "$QOS" ]] && args+=(--qos="$QOS")
  [[ -n "$dep" ]] && args+=(--dependency="afterok:$dep" --kill-on-invalid-dep=yes)

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] sbatch ${args[*]} --wrap '$run_cmd'" >&2
    echo "dry$(printf %s "$name" | cksum | cut -d' ' -f1)"   # stable fake id (submit runs in a subshell)
  else
    sbatch "${args[@]}" --wrap "$run_cmd"
  fi
}

declare -A WARM_ID
if [[ "$SKIP_WARMUP" != "1" ]]; then
  echo "== stage 1: cache warm-up (${#TASKS[@]} tasks in parallel)"
  for task in "${TASKS[@]}"; do
    WARM_ID[$task]="$(submit "warm_${PREFIX}_${task}" "" "$MEM_WARM" "$TIME_WARM" "$WARM_CFG" "$task")"
    echo "   warm-up $task -> job ${WARM_ID[$task]}"
  done
fi

all_warm=""
for task in "${TASKS[@]}"; do
  [[ -n "${WARM_ID[$task]:-}" ]] && all_warm+="${all_warm:+:}${WARM_ID[$task]}"
done

echo "== stage 2: ${#CONFIGS[@]} runs$([[ "$SPLIT_TASKS" == "1" ]] && echo " x ${#TASKS[@]} tasks")"
submitted=0
for cfg in "${CONFIGS[@]}"; do
  name="$(basename "$cfg" .json)"
  if [[ "$SPLIT_TASKS" == "1" ]]; then
    for task in "${TASKS[@]}"; do
      submit "${name}_${task}" "${WARM_ID[$task]:-}" "$MEM" "$TIME" "$cfg" "$task" >/dev/null
      submitted=$((submitted + 1))
    done
  else
    submit "$name" "$all_warm" "$MEM" "$TIME" "$cfg" >/dev/null
    submitted=$((submitted + 1))
  fi
done

echo "submitted $submitted run job(s)$([[ "$SKIP_WARMUP" != "1" ]] && echo " + ${#TASKS[@]} warm-up job(s)")."
echo "logs:      $LOG_DIR"
echo "summaries: $SUMMARY_DIR"
echo "monitor:   squeue -u \$USER -o '%.10i %.40j %.8T %.10M %R'"
