#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT_DIR/outputs/fmt_simgen_v2_3k_20k/current_code_runs"
LOG_DIR="$ROOT_DIR/outputs/fmt_simgen_v2_3k_20k/logs"
STATUS_FILE="$LOG_DIR/paper_baselines_test_status.tsv"
EXP=fmt_simgen_v2_3k_20k_common
TRAINED=(pah2t_former uhr_deepfmt_lowres two_stage_deepfmt_fixed gaicn fem2vox_unet_residual)
ANALYTIC=(fem_coarse fem_to_voxel tikhonov_fem l1_fem elasticnet_fem fista_fem stomp_fem)

mkdir -p "$RUN_DIR" "$LOG_DIR"
printf "model\tstatus\tcheckpoint\tlog_file\tnotes\n" >"$STATUS_FILE"

best_ckpt() {
  local model="$1"
  uv run python - "$RUN_DIR/$model/checkpoints" <<'PY'
import re
import sys
from pathlib import Path
paths = list(Path(sys.argv[1]).glob("*.ckpt"))
def score(path):
    match = re.search(r"val_dice=([0-9]+(?:\.[0-9]+)?)", path.name)
    return (float(match.group(1)) if match else -1.0, path.stat().st_mtime)
print(max(paths, key=score) if paths else "")
PY
}

run_test() {
  local run_name="$1"
  local model_config="$2"
  local ckpt="$3"
  shift 3
  local out_dir="$RUN_DIR/$run_name"
  local save_dir="$out_dir/test300"
  local log_file="$LOG_DIR/${run_name}_test.log"
  mkdir -p "$save_dir" "$out_dir/checkpoints" "$out_dir/config"
  local cmd=(uv run python "$ROOT_DIR/scripts/eval_full_volume_fmt_simgen.py"
    model="$model_config" exp="$EXP" data.dataset_type=fmt_simgen
    --split test --threshold 0.5 --save_dir "$save_dir" --device auto --save_predictions)
  if [[ -n "$ckpt" ]]; then
    cmd+=(--ckpt_path "$ckpt")
  fi
  cmd+=("$@")
  if PYTHONUNBUFFERED=1 "${cmd[@]}" >"$log_file" 2>&1; then
    printf "%s\tok\t%s\t%s\t\n" "$run_name" "$ckpt" "$log_file" >>"$STATUS_FILE"
  else
    printf "%s\tfailed\t%s\t%s\tsee log\n" "$run_name" "$ckpt" "$log_file" >>"$STATUS_FILE"
  fi
}

for model in "${TRAINED[@]}"; do
  ckpt="$(best_ckpt "$model")"
  if [[ -z "$ckpt" ]]; then
    printf "%s\tskipped\t\t\tcheckpoint missing\n" "$model" >>"$STATUS_FILE"
    continue
  fi
  config="$model"
  if [[ "$model" == two_stage_deepfmt_fixed ]]; then
    config=two_stage_deepfmt
    run_test "$model" "$config" "$ckpt" \
      model.two_stage_deepfmt.profile_axis="${TWO_STAGE_PROFILE_AXIS:-h}" \
      model.two_stage_deepfmt.slice_axis="${TWO_STAGE_SLICE_AXIS:-w}" \
      model.two_stage_deepfmt.iradon_hidden_dim=1024 \
      model.two_stage_deepfmt.refine_channels=32 model.two_stage_deepfmt.refine_blocks=4 \
      model.two_stage_deepfmt.loss_type=default model.two_stage_deepfmt.dice_weight=0.5
    continue
  fi
  if [[ "$model" == fem2vox_unet_residual ]]; then
    config=fem2vox_unet
  fi
  run_test "$model" "$config" "$ckpt"
done
for model in "${ANALYTIC[@]}"; do
  run_test "$model" "$model" ""
done
