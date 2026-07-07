#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export PYTHONUNBUFFERED=1

OUT_ROOT="${OUT_ROOT:-outputs/view_complementary/phsa_decisive_study/runs}"
SEEDS=(${SEEDS:-41 42 43})
STRATEGIES=(${STRATEGIES:-uniform support geometry full shuffled oracle})
SOURCES=(${SOURCES:-gt})

run_one() {
  local seed="$1" source="$2" strategy="$3"
  local out="${OUT_ROOT}/${source}_${strategy}_seed${seed}"
  if [[ -f "${out}/DONE" ]]; then
    return
  fi
  mkdir -p "$out"
  uv run python scripts/train_phsa_sample_level.py fit \
    model=ssq_fmt exp=fmt_simgen_v2_phsa_decisive data.dataset_type=fmt_simgen \
    "seed=${seed}" "data.subset_seed=${seed}" \
    "paths.output_dir=${out}" \
    "model.ssq_fmt.view_complementary.hypothesis_source=${source}" \
    "model.ssq_fmt.view_complementary.aggregation_strategy=${strategy}" \
    2>&1 | tee "${out}/train.log"
  touch "${out}/DONE"
}

for seed in "${SEEDS[@]}"; do
  for source in "${SOURCES[@]}"; do
    for strategy in "${STRATEGIES[@]}"; do
      run_one "$seed" "$source" "$strategy"
    done
  done
  run_one "$seed" learned full
done
