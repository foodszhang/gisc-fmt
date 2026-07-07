#!/usr/bin/env bash
set -u

ROOT="outputs/fmt_simgen_v2_3k_20k"
RUN_ROOT="$ROOT/ablation_runs"
LOG_ROOT="$ROOT/logs"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

COMMON=(
  "exp=fmt_simgen_v2_3k_20k_common"
  "data.dataset_type=fmt_simgen"
  "data.batch_size=1"
  "data.eval_batch_size=1"
  "trainer.max_epochs=50"
  "+callbacks.early_stopping.strict=false"
  "callbacks.early_stopping.patience=8"
)
MAX_EPOCH=49

variant_overrides() {
  case "$1" in
    gisc_point)
      printf '%s\n' \
        "model.gisc.footprint_mode=point" \
        "model.gisc.use_footprint=false" \
        "model.gisc.use_adaptive_footprint=false" \
        "model.gisc.use_physical_constraint=false"
      ;;
    gisc_fixed_footprint)
      printf '%s\n' \
        "model.gisc.footprint_mode=fixed" \
        "model.gisc.use_footprint=true" \
        "model.gisc.use_adaptive_footprint=false" \
        "model.gisc.use_physical_constraint=false" \
        "model.gisc.fixed_sigma=1.0"
      ;;
    gisc_depth_footprint)
      printf '%s\n' \
        "model.gisc.footprint_mode=depth" \
        "model.gisc.use_footprint=true" \
        "model.gisc.use_adaptive_footprint=false" \
        "model.gisc.use_physical_constraint=true" \
        "model.gisc.use_depth_cue=true" \
        "model.gisc.use_center_distance_cue=false"
      ;;
    gisc_center_distance)
      printf '%s\n' \
        "model.gisc.footprint_mode=center_distance" \
        "model.gisc.use_footprint=true" \
        "model.gisc.use_adaptive_footprint=false" \
        "model.gisc.use_physical_constraint=true" \
        "model.gisc.use_center_distance_cue=true"
      ;;
    gisc_adaptive_unconstrained)
      printf '%s\n' \
        "model.gisc.footprint_mode=adaptive_unconstrained" \
        "model.gisc.use_footprint=true" \
        "model.gisc.use_adaptive_footprint=true" \
        "model.gisc.use_physical_constraint=false"
      ;;
    gisc_full)
      printf '%s\n' \
        "model.gisc.footprint_mode=adaptive_constrained" \
        "model.gisc.use_footprint=true" \
        "model.gisc.use_adaptive_footprint=true" \
        "model.gisc.use_physical_constraint=true"
      ;;
    gisc_3view)
      printf '%s\n' \
        "model.gisc.footprint_mode=adaptive_constrained" \
        "model.gisc.view_subset=[-90,0,90]"
      ;;
    gisc_5view)
      printf '%s\n' \
        "model.gisc.footprint_mode=adaptive_constrained" \
        "model.gisc.view_subset=[-90,-60,0,60,90]"
      ;;
    gisc_7view)
      printf '%s\n' \
        "model.gisc.footprint_mode=adaptive_constrained" \
        "model.gisc.view_subset=[-90,-60,-30,0,30,60,90]"
      ;;
    *)
      return 1
      ;;
  esac
}

run_variant() {
  local variant="$1"
  local out_dir="$RUN_ROOT/$variant"
  local log_file="$LOG_ROOT/ablation_${variant}_train.log"
  mapfile -t overrides < <(variant_overrides "$variant")
  mkdir -p "$out_dir"
  if [[ -f "$log_file" ]] && grep -q "\[INFO\] ${variant} exit_status=0" "$log_file"; then
    echo "[SKIP] $variant already completed according to $log_file" | tee -a "$log_file"
    return 0
  fi
  local last_ckpt="$out_dir/checkpoints/last.ckpt"
  local resume_args=()
  if [[ -f "$last_ckpt" ]]; then
    local last_epoch
    last_epoch=$("$PYTHON_BIN" - "$last_ckpt" <<'PY'
import sys
import torch

ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(ckpt.get("epoch", -1)))
PY
)
    if [[ "$last_epoch" -ge "$MAX_EPOCH" ]]; then
      echo "[SKIP] $variant already reached epoch=$last_epoch" | tee -a "$log_file"
      return 0
    fi
    resume_args=("ckpt_path=$last_ckpt")
    echo "[INFO] resuming $variant from $last_ckpt epoch=$last_epoch" | tee -a "$log_file"
  else
    echo "[INFO] training $variant" | tee "$log_file"
  fi
  "$PYTHON_BIN" train.py fit model=gisc_fmt "${COMMON[@]}" "${resume_args[@]}" \
    "paths.output_dir=$out_dir" \
    "paths.checkpoint_dir=$out_dir/checkpoints" \
    "paths.config_dir=$out_dir/config" \
    "${overrides[@]}" >> "$log_file" 2>&1
  local status=$?
  echo "[INFO] $variant exit_status=$status" >> "$log_file"
  return "$status"
}

variants=(
  gisc_point
  gisc_fixed_footprint
  gisc_depth_footprint
  gisc_center_distance
  gisc_adaptive_unconstrained
  gisc_full
  gisc_3view
  gisc_5view
  gisc_7view
)

overall=0
for variant in "${variants[@]}"; do
  run_variant "$variant" || overall=1
done
exit "$overall"
