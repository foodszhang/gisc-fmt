#!/usr/bin/env bash
set -euo pipefail

# Example evaluation command for GISC-FMT.
# Requires a checkpoint and the dataset.

CKPT=${1:-pretrained/gisc_fmt_brain1000_best.ckpt}

uv run python train.py test \
  model=gisc_fmt \
  ckpt_path=${CKPT}
