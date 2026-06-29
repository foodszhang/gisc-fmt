#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

A2U_RUN="outputs/view_complementary/long_1k_continue_a2u_seed42"
A2U="$A2U_RUN/checkpoints/epoch=04-val_dice=0.6474.ckpt"
ROOT_OUT="outputs/view_complementary/isolation_and_stabilization_paired_seed42"
ROUTING_OUT="$ROOT_OUT/isolated_bounded_routing"
GRID_A_OUT="$ROOT_OUT/fixed_grid_stage_a"
GRID_B_OUT="$ROOT_OUT/fixed_grid_stage_b"

[[ -f "$A2U" ]] || { echo "Missing A2-U checkpoint: $A2U" >&2; exit 2; }
mkdir -p "$ROOT_OUT"

uv run python scripts/analysis/check_isolated_routing_equivalence.py \
  --run-dir "$A2U_RUN" --checkpoint "$A2U" \
  --output "$ROOT_OUT/zero_gain_equivalence.json"

run_stage() {
  local exp="$1" init="$2" output="$3"
  if [[ -f "$output/.complete" ]]; then
    echo "Skipping completed stage: $output"
    return
  fi
  mkdir -p "$output"
  local checkpoint="$init"
  local weights_only=true
  if [[ -f "$output/checkpoints/last.ckpt" ]]; then
    checkpoint="$output/checkpoints/last.ckpt"
    weights_only=false
  fi
  local checkpoint_override
  checkpoint_override="ckpt_path='$(realpath "$checkpoint")'"
  uv run python train.py fit model=ssq_fmt "exp=$exp" data.dataset_type=fmt_simgen \
    "paths.output_dir=$output" "$checkpoint_override" \
    "ckpt_weights_only=$weights_only" 2>&1 | tee "$output/train.log"
  touch "$output/.complete"
}

best_checkpoint() {
  local output="$1"
  find "$output/checkpoints" -maxdepth 1 -type f -name 'epoch=*.ckpt' | sort | head -1
}

run_stage fmt_simgen_v2_isolated_bounded_routing_fast "$A2U" "$ROUTING_OUT"
run_stage fmt_simgen_v2_fixed_grid_stabilized_stage_a "$A2U" "$GRID_A_OUT"
GRID_A_BEST="$(best_checkpoint "$GRID_A_OUT")"
[[ -n "$GRID_A_BEST" ]] || { echo "Stage A produced no best checkpoint" >&2; exit 3; }
run_stage fmt_simgen_v2_fixed_grid_stabilized_stage_b "$GRID_A_BEST" "$GRID_B_OUT"

uv run python scripts/analysis/report_isolation_and_stabilization.py --run-dir "$ROOT_OUT"
