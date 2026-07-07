#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

WARMUP="outputs/view_complementary/long_1k_continue_candidate_warmup_seed42/checkpoints/epoch=07-val_dice=0.6460.ckpt"
A2U="outputs/view_complementary/long_1k_continue_a2u_seed42/checkpoints/epoch=04-val_dice=0.6474.ckpt"
GEOM_OUT="outputs/view_complementary/a3v2_fast_geometry_only_seed42_corrected"
A3V2_OUT="outputs/view_complementary/a3v2_fast_bounded_routing_seed42_corrected"
ROUTING_ONLY_OUT="outputs/view_complementary/a3v2_fast_routing_only_seed42"

select_checkpoint() {
  local requested="$1" directory
  if [[ -f "$requested" ]]; then printf '%s\n' "$requested"; return; fi
  directory="$(dirname "$requested")"
  local fallback
  fallback="$(find "$directory" -maxdepth 1 -type f -name 'epoch=*.ckpt' ! -name 'last.ckpt' | sort -t= -k3,3nr | head -1 || true)"
  [[ -n "$fallback" ]] || { echo "No checkpoint found in $directory" >&2; exit 2; }
  printf '%s\n' "$fallback"
}

run_stage() {
  local exp="$1" init="$2" output="$3"
  if [[ -f "$output/.complete" ]]; then
    echo "Skipping completed stage: $output"
    return
  fi
  mkdir -p "$output"
  local resume=() init_override
  if [[ -f "$output/checkpoints/last.ckpt" ]]; then
    resume=(ckpt_weights_only=false "ckpt_path=$output/checkpoints/last.ckpt")
  else
    init_override="ckpt_path=\"$(realpath "$init")\""
    resume=("$init_override")
  fi
  uv run python train.py fit model=ssq_fmt "exp=$exp" data.dataset_type=fmt_simgen \
    "paths.output_dir=$output" "${resume[@]}" 2>&1 | tee "$output/train.log"
  touch "$output/.complete"
}

WARMUP="$(select_checkpoint "$WARMUP")"
A2U="$(select_checkpoint "$A2U")"
run_stage fmt_simgen_v2_a3_geometry_only_fast "$WARMUP" "$GEOM_OUT"
run_stage fmt_simgen_v2_a3v2_fast "$A2U" "$A3V2_OUT"
BEST_A3V2="$(find "$A3V2_OUT/checkpoints" -maxdepth 1 -type f -name 'epoch=*.ckpt' | head -1)"
if uv run python - "$A3V2_OUT" <<'PY'
import sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
p = sorted((Path(sys.argv[1]) / "tensorboard").glob("version_*"))[-1]
e = EventAccumulator(str(p)); e.Reload()
best = max(x.value for x in e.Scalars("val_dice"))
raise SystemExit(0 if best >= 0.647369 - 0.0002 else 1)
PY
then
  run_stage fmt_simgen_v2_a3v2_routing_only_fast "$BEST_A3V2" "$ROUTING_ONLY_OUT"
else
  echo "Routing-only condition not met; skipping experiment 3."
fi
uv run python scripts/analysis/report_a3v2_fast.py
