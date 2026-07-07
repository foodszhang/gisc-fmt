#!/usr/bin/env bash
set -euo pipefail

uv run python train.py fit model=uhr_deepfmt data.dataset_type=fmt_simgen "$@"

