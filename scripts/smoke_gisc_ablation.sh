#!/usr/bin/env bash
set -euo pipefail

ROOT="outputs/fmt_simgen_v2_3k_20k"
mkdir -p "$ROOT/logs"

COMMON=(
  "exp=fmt_simgen_v2_3k_20k_common"
  "data.dataset_type=fmt_simgen"
  "data.batch_size=1"
  "data.eval_batch_size=1"
  "trainer.max_epochs=1"
  "+trainer.limit_train_batches=2"
  "+trainer.limit_val_batches=2"
  "paths.output_dir=$ROOT/ablation_smoke"
  "paths.checkpoint_dir=$ROOT/ablation_smoke/checkpoints"
  "paths.config_dir=$ROOT/ablation_smoke/config"
)

uv run python train.py fit model=gisc_fmt "${COMMON[@]}" \
  model.gisc.footprint_mode=point \
  model.gisc.use_footprint=false

uv run python train.py fit model=gisc_fmt "${COMMON[@]}" \
  model.gisc.footprint_mode=fixed \
  model.gisc.use_footprint=true \
  model.gisc.use_adaptive_footprint=false

uv run python train.py fit model=gisc_fmt "${COMMON[@]}" \
  model.gisc.footprint_mode=center_distance \
  model.gisc.use_footprint=true

uv run python train.py fit model=gisc_fmt "${COMMON[@]}" \
  'model.gisc.view_subset=[-90,0,90]'
