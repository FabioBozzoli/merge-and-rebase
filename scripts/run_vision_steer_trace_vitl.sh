#!/usr/bin/env bash
# ViT-B/16 -> ViT-L/14: same runs as run_vision_steer_trace.sh, on configs/vision8_steer_trace_vitl.
# Accepts the same environment knobs (PARTITION, ACCOUNT, MEM, TIME, SPLIT_TASKS, DRY_RUN, ...).
export CFG_DIR="${CFG_DIR:-configs/vision8_steer_trace_vitl}"
export PREFIX="${PREFIX:-vision8_steer_trace_vitl}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_vision_steer_trace.sh" "$@"
