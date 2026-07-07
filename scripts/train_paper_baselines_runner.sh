#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/outputs/fmt_simgen_v2_3k_20k/current_code_runs"
LOG_DIR="$ROOT_DIR/outputs/fmt_simgen_v2_3k_20k/logs"
STATUS_FILE="$LOG_DIR/paper_baselines_train_status.tsv"
COMMON=(exp=fmt_simgen_v2_3k_20k_common data.dataset_type=fmt_simgen data.batch_size=1 data.eval_batch_size=1)

mkdir -p "$RUN_DIR" "$LOG_DIR"
printf "model\tstatus\tstart_time\tend_time\tlog_file\n" >"$STATUS_FILE"

run_model() {
  local run_name="$1"
  local model_config="$2"
  local epochs="$3"
  shift 3
  local out_dir="$RUN_DIR/$run_name"
  local log_file="$LOG_DIR/${run_name}_train.log"
  local start_time end_time
  mkdir -p "$out_dir/checkpoints" "$out_dir/config"
  start_time="$(date '+%Y-%m-%d %H:%M:%S')"
  if PYTHONUNBUFFERED=1 uv run python "$ROOT_DIR/train.py" fit \
    model="$model_config" "${COMMON[@]}" \
    paths.output_dir="$out_dir" paths.checkpoint_dir="$out_dir/checkpoints" \
    paths.config_dir="$out_dir/config" trainer.max_epochs="$epochs" "$@" >"$log_file" 2>&1; then
    status=ok
  else
    status=failed
  fi
  end_time="$(date '+%Y-%m-%d %H:%M:%S')"
  printf "%s\t%s\t%s\t%s\t%s\n" "$run_name" "$status" "$start_time" "$end_time" "$log_file" \
    >>"$STATUS_FILE"
}

case "${1:-all}" in
  gaicn)
    run_model gaicn gaicn "${TRAIN_EPOCHS:-50}"
    ;;
  uhr)
    run_model uhr_deepfmt_lowres uhr_deepfmt_lowres "${TRAIN_EPOCHS:-50}"
    ;;
  uhr_paperlike)
    run_model uhr_deepfmt_paperlike uhr_deepfmt_paperlike "${TRAIN_EPOCHS:-60}"
    ;;
  two_stage_hw)
    run_model two_stage_deepfmt_hw two_stage_deepfmt "${TRAIN_EPOCHS:-10}" \
      model.two_stage_deepfmt.profile_axis=h model.two_stage_deepfmt.slice_axis=w \
      model.two_stage_deepfmt.iradon_hidden_dim=1024 model.two_stage_deepfmt.refine_channels=32 \
      model.two_stage_deepfmt.refine_blocks=4 model.two_stage_deepfmt.loss_type=default \
      model.two_stage_deepfmt.dice_weight=0.5
    ;;
  two_stage_wh)
    run_model two_stage_deepfmt_wh two_stage_deepfmt "${TRAIN_EPOCHS:-10}" \
      model.two_stage_deepfmt.profile_axis=w model.two_stage_deepfmt.slice_axis=h \
      model.two_stage_deepfmt.iradon_hidden_dim=1024 model.two_stage_deepfmt.refine_channels=32 \
      model.two_stage_deepfmt.refine_blocks=4 model.two_stage_deepfmt.loss_type=default \
      model.two_stage_deepfmt.dice_weight=0.5
    ;;
  *)
    echo "usage: $0 {gaicn|uhr|uhr_paperlike|two_stage_hw|two_stage_wh}" >&2
    exit 2
    ;;
esac
