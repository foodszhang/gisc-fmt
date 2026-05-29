#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/outputs/fmt_simgen_v2_3k_20k/current_code_runs"
LOG_DIR="$ROOT_DIR/outputs/fmt_simgen_v2_3k_20k/logs"
COMMON="exp=fmt_simgen_v2_3k_20k_common data.dataset_type=fmt_simgen data.batch_size=1 data.eval_batch_size=1 trainer.max_epochs=50"

mkdir -p "$RUN_DIR" "$LOG_DIR"

run_model() {
  local model="$1"
  shift
  local out_dir="$RUN_DIR/$model"
  local log_file="$LOG_DIR/${model}_train.log"
  mkdir -p "$out_dir"

  local ckpt_arg=""
  if [[ "$model" == "cnn3d_baseline" && -f "$ROOT_DIR/outputs/cnn3d_baseline/fit/2026-05-28/22-42-09/checkpoints/last.ckpt" ]]; then
    ckpt_arg="ckpt_path=$ROOT_DIR/outputs/cnn3d_baseline/fit/2026-05-28/22-42-09/checkpoints/last.ckpt"
  fi

  {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] start $model"
    echo "log: $log_file"
    echo "output: $out_dir"
  } | tee -a "$LOG_DIR/stable_voxel_baselines_master.log"

  uv run python "$ROOT_DIR/train.py" fit \
    model="$model" \
    $COMMON \
    +callbacks.early_stopping.check_on_train_epoch_end=false \
    paths.output_dir="$out_dir" \
    paths.checkpoint_dir="$out_dir/checkpoints" \
    paths.config_dir="$out_dir/config" \
    $ckpt_arg \
    "$@" \
    >"$log_file" 2>&1

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] finished $model" | tee -a "$LOG_DIR/stable_voxel_baselines_master.log"
}

run_model cnn3d_baseline
run_model transunet3d_baseline
run_model pah2t_former
run_model uhr_deepfmt

echo "[$(date '+%Y-%m-%d %H:%M:%S')] all stable voxel baselines finished" | tee -a "$LOG_DIR/stable_voxel_baselines_master.log"
