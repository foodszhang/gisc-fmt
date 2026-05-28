#!/usr/bin/env bash
set -euo pipefail

COMMON="exp=fmt_simgen_v2_3k_20k_common data.dataset_type=fmt_simgen data.batch_size=1 data.eval_batch_size=1 trainer.max_epochs=1 +trainer.limit_train_batches=2 +trainer.limit_val_batches=2"

uv run python train.py fit model=cnn3d_baseline $COMMON
uv run python train.py fit model=transunet3d_baseline $COMMON
uv run python train.py fit model=pah2t_former $COMMON
uv run python train.py fit model=uhr_deepfmt $COMMON
uv run python train.py fit model=two_stage_deepfmt $COMMON
