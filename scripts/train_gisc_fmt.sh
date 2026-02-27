#!/usr/bin/env bash
set -euo pipefail

# Example training command for GISC-FMT.
# You MUST set dataset paths (either edit configs/data/default.yaml or override here).

uv run python train.py fit \
  model=gisc_fmt \
  exp=default
