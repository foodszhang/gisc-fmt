#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/outputs/fmt_simgen_v2_3k_20k/current_code_runs"
LOG_DIR="$ROOT_DIR/outputs/fmt_simgen_v2_3k_20k/logs"
STATUS_FILE="$LOG_DIR/paper_baselines_status.tsv"

COMMON="exp=fmt_simgen_v2_3k_20k_common data.dataset_type=fmt_simgen data.batch_size=1 data.eval_batch_size=1 trainer.max_epochs=50"

mkdir -p "$RUN_DIR" "$LOG_DIR"
echo -e "model\tstatus\tstart_time\tend_time\tlog_file" >"$STATUS_FILE"

run_model() {
  local model="$1"
  shift
  local out_dir="$RUN_DIR/$model"
  local log_file="$LOG_DIR/${model}_train.log"
  local start_time
  local end_time

  mkdir -p "$out_dir"
  start_time="$(date '+%Y-%m-%d %H:%M:%S')"

  echo "[$start_time] start $model" | tee -a "$LOG_DIR/paper_baselines_master.log"
  echo "log: $log_file" | tee -a "$LOG_DIR/paper_baselines_master.log"
  echo "output: $out_dir" | tee -a "$LOG_DIR/paper_baselines_master.log"

  if uv run python "$ROOT_DIR/train.py" fit \
    model="$model" \
    $COMMON \
    +callbacks.early_stopping.check_on_train_epoch_end=false \
    paths.output_dir="$out_dir" \
    paths.checkpoint_dir="$out_dir/checkpoints" \
    paths.config_dir="$out_dir/config" \
    "$@" \
    >"$log_file" 2>&1; then

    end_time="$(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "$model\tok\t$start_time\t$end_time\t$log_file" >>"$STATUS_FILE"
    echo "[$end_time] finished $model" | tee -a "$LOG_DIR/paper_baselines_master.log"
  else
    end_time="$(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "$model\tfailed\t$start_time\t$end_time\t$log_file" >>"$STATUS_FILE"
    echo "[$end_time] failed $model, continue next" | tee -a "$LOG_DIR/paper_baselines_master.log"
  fi
}

# 1. Strong voxel / transformer baselines
run_model pah2t_former
run_model uhr_deepfmt

# 2. Optional sparse-view adapted baseline
run_model two_stage_deepfmt

# 3. PGDPNN only if real template exists
PGD_TEMPLATE="${PGD_TEMPLATE:-}"
if [[ -n "$PGD_TEMPLATE" && -f "$PGD_TEMPLATE" ]]; then
  run_model pgdpnn model.pgdpnn.template_path="$PGD_TEMPLATE" model.pgdpnn.allow_template_fallback=false
else
  echo "skip pgdpnn: set PGD_TEMPLATE=/path/to/real_pgdpnn_templates.npz" | tee -a "$LOG_DIR/paper_baselines_master.log"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] paper voxel baselines finished" | tee -a "$LOG_DIR/paper_baselines_master.log"
