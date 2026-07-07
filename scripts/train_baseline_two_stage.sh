#!/usr/bin/env bash
set -euo pipefail

uv run python train.py fit model=two_stage_deepfmt data.dataset_type=fmt_simgen "$@"

