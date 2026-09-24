#!/usr/bin/env bash
# Launch the 15 vision steer-trace runs (few_shot x seed), features shared through one cache dir.
#
#   DEVICES="cuda:0 cuda:1" scripts/run_vision_steer_trace.sh
#
# The first run (few_shot=1, seed=33) goes alone: it fills the feature cache, whose writes are not
# atomic, so no cold runs may share the cache dir concurrently. The other 14 only refit and are split
# across DEVICES (one sequential queue per device).
set -u
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CFG_DIR="${CFG_DIR:-configs/vision8_steer_trace}"
LOG_DIR="${LOG_DIR:-results/vision8_steer_trace/logs}"
read -r -a DEVICES <<< "${DEVICES:-cuda}"
mkdir -p "$LOG_DIR"

run_one() {
  local cfg="$1" device="$2" name
  name="$(basename "$cfg" .json)"
  [[ -n "${SKIP_DONE:-}" && -f "$LOG_DIR/$name.done" ]] && { echo "skip $name"; return 0; }
  echo "[$(date +%T)] start $name on $device"
  if "$PYTHON_BIN" -m merge_and_rebase.eval.vision_rebase --config "$cfg" --device "$device" >"$LOG_DIR/$name.log" 2>&1; then
    touch "$LOG_DIR/$name.done"
    echo "[$(date +%T)] done  $name"
  else
    echo "[$(date +%T)] FAIL  $name (see $LOG_DIR/$name.log)"
  fi
}

first="$CFG_DIR/vision8_steer_trace_fs1_seed33.json"
run_one "$first" "${DEVICES[0]}"

rest=()
for cfg in "$CFG_DIR"/vision8_steer_trace_fs*_seed*.json; do
  [[ "$cfg" == "$first" ]] || rest+=("$cfg")
done

for i in "${!DEVICES[@]}"; do
  (
    for j in "${!rest[@]}"; do
      (( j % ${#DEVICES[@]} == i )) && run_one "${rest[$j]}" "${DEVICES[$i]}"
    done
  ) &
done
wait
