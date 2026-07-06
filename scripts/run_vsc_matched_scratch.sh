#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/vsc/logs

declare -A ROOTS=(
  [ssq_baseline_scratch_reproduce]=outputs/vsc/baseline_scratch
  [ssq_vsc_scratch_warmup]=outputs/vsc/vsc_warmup
  [ssq_vsc_scratch_nocons]=outputs/vsc/vsc_nocons
  [ssq_vsc_scratch_consonly]=outputs/vsc/vsc_consonly
)

EXPERIMENTS=(
  ssq_baseline_scratch_reproduce
  ssq_vsc_scratch_warmup
  ssq_vsc_scratch_nocons
  ssq_vsc_scratch_consonly
)

SEEDS=(42 43 44)

for seed in "${SEEDS[@]}"; do
  for exp in "${EXPERIMENTS[@]}"; do
    root="${ROOTS[$exp]}/seed_${seed}"
    log="outputs/vsc/logs/${exp}_seed_${seed}.log"
    echo "[$(date '+%F %T %Z')] START exp=${exp} seed=${seed} root=${root}" | tee -a "${log}"
    uv run python train.py fit \
      exp="${exp}" \
      data.dataset_type=fmt_simgen \
      seed="${seed}" \
      paths.root_dir="${root}" \
      2>&1 | tee -a "${log}"
    echo "[$(date '+%F %T %Z')] DONE exp=${exp} seed=${seed}" | tee -a "${log}"
  done
done
