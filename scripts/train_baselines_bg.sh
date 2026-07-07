#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/outputs/fmt_simgen_v2_3k_20k/logs"
PID_FILE="$LOG_DIR/paper_baselines.pid"
MASTER_LOG="$LOG_DIR/paper_baselines_master.log"

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "paper baseline training is already running with PID $(cat "$PID_FILE")"
  exit 1
fi

nohup bash "$ROOT_DIR/scripts/train_paper_baselines_runner.sh" \
  >"$MASTER_LOG" 2>&1 &

echo "$!" >"$PID_FILE"
echo "started paper baseline training with PID $!"
echo "master log: $MASTER_LOG"
