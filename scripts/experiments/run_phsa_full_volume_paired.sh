#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export PYTHONUNBUFFERED=1

OUTPUT_DIR="${OUTPUT_DIR:-outputs/view_complementary/phsa_full_volume_paired_20260629}"
PROPOSAL_COUNT="${PROPOSAL_COUNT:-4096}"
PROPOSAL_SEED="${PROPOSAL_SEED:-42}"
CHUNK_SIZE="${CHUNK_SIZE:-32768}"
THRESHOLD="${THRESHOLD:-0.5}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"

mkdir -p "$OUTPUT_DIR"

COMMON=(
  --split test
  --proposal_count "$PROPOSAL_COUNT"
  --proposal_seed "$PROPOSAL_SEED"
  --chunk_size "$CHUNK_SIZE"
  --threshold "$THRESHOLD"
  --bootstrap_samples "$BOOTSTRAP_SAMPLES"
  --device cuda
)

if [[ -n "$MAX_SAMPLES" ]]; then
  COMMON+=(--max_samples "$MAX_SAMPLES")
fi

printf '\n[%s] Smoke test: 4 samples, 1024 proposal points\n' "$(date '+%F %T')"
uv run python scripts/eval_view_complementary_full_volume_paired.py \
  --output_dir "${OUTPUT_DIR}/smoke" \
  --proposal_count 1024 \
  --proposal_seed "$PROPOSAL_SEED" \
  --chunk_size 16384 \
  --max_samples 4 \
  --bootstrap_samples 1000 \
  --device cuda

printf '\n[%s] Formal paired full-volume evaluation\n' "$(date '+%F %T')"
uv run python scripts/eval_view_complementary_full_volume_paired.py \
  --output_dir "$OUTPUT_DIR" \
  "${COMMON[@]}"

printf '\n[%s] Completed. Summary: %s\n' "$(date '+%F %T')" "$OUTPUT_DIR/summary.md"
