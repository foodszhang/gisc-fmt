#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <ckpt_path> [hydra overrides...]" >&2
  exit 2
fi
ckpt_path="$1"
shift
uv run python train.py test model=uhr_deepfmt ckpt_path="${ckpt_path}" data.dataset_type=fmt_simgen "$@"

