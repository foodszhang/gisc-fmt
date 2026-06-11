#!/usr/bin/env bash
set -euo pipefail

ROOT="outputs/fmt_simgen_v2_3k_20k"
mkdir -p "$ROOT/logs"
nohup bash scripts/train_gisc_ablation_runner.sh \
  > "$ROOT/logs/gisc_ablation_master.log" 2>&1 &
echo $! > "$ROOT/logs/gisc_ablation.pid"
echo "started gisc ablation runner pid=$(cat "$ROOT/logs/gisc_ablation.pid")"
