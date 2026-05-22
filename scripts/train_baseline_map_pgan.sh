#!/usr/bin/env bash
set -euo pipefail

uv run python train.py fit model=map_pgan data.dataset_type=fmt_simgen "$@"

