#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/vsc/logs

declare -A ROOTS=(
  [ssq_vsc_scratch_warmup]=outputs/vsc/vsc_warmup
  [ssq_vsc_scratch_nocons]=outputs/vsc/vsc_nocons
)

EXPERIMENTS=(
  ssq_vsc_scratch_warmup
  ssq_vsc_scratch_nocons
)

seed=42

for exp in "${EXPERIMENTS[@]}"; do
  root="${ROOTS[$exp]}/seed_${seed}"
  log="outputs/vsc/logs/${exp}_seed_${seed}.log"
  echo "[$(date '+%F %T %Z')] START exp=${exp} seed=${seed} root=${root}" | tee -a "${log}"
  uv run python train.py fit \
    exp="${exp}" \
    data.dataset_type=fmt_simgen \
    data.num_queries=16384 \
    data.sample_num=16384 \
    data.eval_sample_num=16384 \
    data.query_sampling.num_queries=16384 \
    data.num_workers=4 \
    trainer.accumulate_grad_batches=1 \
    seed="${seed}" \
    paths.root_dir="${root}" \
    2>&1 | tee -a "${log}"
  echo "[$(date '+%F %T %Z')] DONE exp=${exp} seed=${seed}" | tee -a "${log}"
done
