#!/usr/bin/env bash
set -u

ROOT="outputs/fmt_simgen_v2_3k_20k"
RUN_ROOT="$ROOT/ablation_runs"
LOG_ROOT="$ROOT/logs"
DATA_DIR="/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k"
mkdir -p "$LOG_ROOT"

variant_overrides() {
  case "$1" in
    gisc_point)
      printf '%s\n' "model.gisc.footprint_mode=point" "model.gisc.use_footprint=false" "model.gisc.use_adaptive_footprint=false" "model.gisc.use_physical_constraint=false" ;;
    gisc_fixed_footprint)
      printf '%s\n' "model.gisc.footprint_mode=fixed" "model.gisc.use_footprint=true" "model.gisc.use_adaptive_footprint=false" "model.gisc.use_physical_constraint=false" "model.gisc.fixed_sigma=1.0" ;;
    gisc_depth_footprint)
      printf '%s\n' "model.gisc.footprint_mode=depth" "model.gisc.use_footprint=true" "model.gisc.use_adaptive_footprint=false" "model.gisc.use_physical_constraint=true" "model.gisc.use_depth_cue=true" "model.gisc.use_center_distance_cue=false" ;;
    gisc_center_distance)
      printf '%s\n' "model.gisc.footprint_mode=center_distance" "model.gisc.use_footprint=true" "model.gisc.use_adaptive_footprint=false" "model.gisc.use_physical_constraint=true" "model.gisc.use_center_distance_cue=true" ;;
    gisc_adaptive_unconstrained)
      printf '%s\n' "model.gisc.footprint_mode=adaptive_unconstrained" "model.gisc.use_footprint=true" "model.gisc.use_adaptive_footprint=true" "model.gisc.use_physical_constraint=false" ;;
    gisc_full)
      printf '%s\n' "model.gisc.footprint_mode=adaptive_constrained" "model.gisc.use_footprint=true" "model.gisc.use_adaptive_footprint=true" "model.gisc.use_physical_constraint=true" ;;
    gisc_3view)
      printf '%s\n' "model.gisc.footprint_mode=adaptive_constrained" "model.gisc.view_subset=[-90,0,90]" ;;
    gisc_5view)
      printf '%s\n' "model.gisc.footprint_mode=adaptive_constrained" "model.gisc.view_subset=[-90,-60,0,60,90]" ;;
    gisc_7view)
      printf '%s\n' "model.gisc.footprint_mode=adaptive_constrained" "model.gisc.view_subset=[-90,-60,-30,0,30,60,90]" ;;
    *)
      return 1 ;;
  esac
}

best_ckpt() {
  local ckpt_dir="$1"
  python - "$ckpt_dir" <<'PY'
from pathlib import Path
import re
import sys

ckpt_dir = Path(sys.argv[1])
if not ckpt_dir.exists():
    sys.exit(1)
files = [p for p in ckpt_dir.glob("*.ckpt") if p.name != "last.ckpt"]
if not files and (ckpt_dir / "last.ckpt").exists():
    print(ckpt_dir / "last.ckpt")
    sys.exit(0)
def score(path: Path):
    m = re.search(r"val_dice=([0-9.]+)", path.name)
    if m:
        return (float(m.group(1).rstrip(".")), path.stat().st_mtime)
    return (-1.0, path.stat().st_mtime)
if not files:
    sys.exit(1)
print(max(files, key=score))
PY
}

run_variant() {
  local variant="$1"
  local out_dir="$RUN_ROOT/$variant"
  local log_file="$LOG_ROOT/ablation_${variant}_test.log"
  mapfile -t overrides < <(variant_overrides "$variant")
  echo "[INFO] testing $variant" > "$log_file"
  local ckpt
  ckpt=$(best_ckpt "$out_dir/checkpoints")
  local status=$?
  if [[ "$status" -ne 0 || -z "${ckpt:-}" ]]; then
    echo "[SKIPPED] no checkpoint under $out_dir/checkpoints" >> "$log_file"
    return 0
  fi
  echo "[INFO] ckpt=$ckpt" >> "$log_file"
  uv run python train.py test model=gisc_fmt \
    exp=fmt_simgen_v2_3k_20k_common \
    data.dataset_type=fmt_simgen \
    data.eval_batch_size=1 \
    "ckpt_path=$ckpt" \
    "paths.output_dir=$out_dir" \
    "paths.checkpoint_dir=$out_dir/checkpoints" \
    "paths.config_dir=$out_dir/config" \
    "${overrides[@]}" >> "$log_file" 2>&1
  echo "[INFO] train.py test exit_status=$?" >> "$log_file"

  uv run python scripts/eval_full_volume_fmt_simgen.py \
    --exp fmt_simgen_v2_3k_20k_common \
    --ckpt_path "$ckpt" \
    --split test \
    --threshold 0.5 \
    --save_dir "$out_dir" \
    --save_predictions \
    model=gisc_fmt \
    data.dataset_type=fmt_simgen \
    "${overrides[@]}" >> "$log_file" 2>&1
  local eval_status=$?
  echo "[INFO] full-volume eval exit_status=$eval_status" >> "$log_file"
  if [[ "$eval_status" -eq 0 ]]; then
    uv run python scripts/eval_components_fmt_simgen.py \
      --eval_dir "$out_dir" \
      --data_dir "$DATA_DIR" \
      --split test >> "$log_file" 2>&1
    echo "[INFO] component eval exit_status=$?" >> "$log_file"
  fi
  return 0
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

for variant in "${variants[@]}"; do
  run_variant "$variant"
done
