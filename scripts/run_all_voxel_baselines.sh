#!/usr/bin/env bash
set -euo pipefail

models=(
  uhr_deepfmt
  vox_dmrn
  fem2vox_unet
  two_stage_deepfmt
  fmt_reconnet
  pgdpnn
  map_pgan
  d2_recst
  dspgn
)

for model_name in "${models[@]}"; do
  uv run python train.py fit "model=${model_name}" data.dataset_type=fmt_simgen "$@"
done

