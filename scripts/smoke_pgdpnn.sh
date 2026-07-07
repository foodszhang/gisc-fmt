#!/usr/bin/env bash
set -euo pipefail

COMMON="exp=fmt_simgen_v2_3k_20k_common data.dataset_type=fmt_simgen data.batch_size=1 data.eval_batch_size=1 trainer.max_epochs=1 +trainer.limit_train_batches=2 +trainer.limit_val_batches=2"
TEMPLATE="tests/assets/pgdpnn_template_common_190_200_104.npz"

if [[ ! -f "$TEMPLATE" ]]; then
  echo "missing template asset: $TEMPLATE" >&2
  exit 1
fi

set +e
MISSING_OUTPUT=$(uv run python train.py fit model=pgdpnn $COMMON model.pgdpnn.template_path="" model.pgdpnn.allow_template_fallback=false 2>&1)
MISSING_STATUS=$?
set -e

if [[ $MISSING_STATUS -eq 0 ]]; then
  echo "PGDPNN without template unexpectedly succeeded" >&2
  exit 1
fi

if [[ "$MISSING_OUTPUT" != *"requires a real template_path"* ]]; then
  echo "$MISSING_OUTPUT" >&2
  echo "PGDPNN missing-template failure did not report explicit template_path requirement" >&2
  exit 1
fi

uv run python train.py fit model=pgdpnn $COMMON model.pgdpnn.template_path="$TEMPLATE" model.pgdpnn.allow_template_fallback=false
