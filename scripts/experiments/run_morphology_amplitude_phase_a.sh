#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export PYTHONUNBUFFERED=1

: "${PHASE_A_CKPT:?Set PHASE_A_CKPT to the audited Phase-A checkpoint (Dice approximately 0.75)}"
[[ -f "$PHASE_A_CKPT" ]] || { echo "Missing Phase-A checkpoint: $PHASE_A_CKPT" >&2; exit 2; }
PHASE_A_CKPT="$(realpath "$PHASE_A_CKPT")"

SEED="${SEED:-42}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-2400}"
VAL_SAMPLES="${VAL_SAMPLES:-300}"
TEST_SAMPLES="${TEST_SAMPLES:-300}"
TRAIN_QUERIES="${TRAIN_QUERIES:-16384}"
EVAL_QUERIES="${EVAL_QUERIES:-16384}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-12}"
PATIENCE="${PATIENCE:-4}"
LR="${LR:-0.00003}"
VAL_EVERY="${VAL_EVERY:-1}"
RUN_FULL_EVAL="${RUN_FULL_EVAL:-1}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-100}"
CHUNK_SIZE="${CHUNK_SIZE:-32768}"
THRESHOLD="${THRESHOLD:-0.5}"
RUN_ROOT="${RUN_ROOT:-outputs/morphology_amplitude/phase_a_seed${SEED}}"

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }

best_checkpoint() {
  uv run python - "$1" <<'PY'
from pathlib import Path
import re
import sys
root = Path(sys.argv[1]) / "checkpoints"
files = [p for p in root.glob("*.ckpt") if p.name != "last.ckpt"]
if not files:
    last = root / "last.ckpt"
    if not last.exists():
        raise SystemExit(f"No checkpoint under {root}")
    print(last.resolve())
    raise SystemExit(0)

def score(path: Path):
    for pattern in (r"val_dice=([0-9.]+)", r"-([0-9]+\.[0-9]+)\.ckpt$"):
        match = re.search(pattern, path.name)
        if match:
            return float(match.group(1).rstrip(".")), path.stat().st_mtime
    return float("-inf"), path.stat().st_mtime

print(max(files, key=score).resolve())
PY
}

COMMON=(
  model=ssq_fmt
  exp=fmt_simgen_v2_morphology_amplitude_phase_a
  data.dataset_type=fmt_simgen
  "seed=${SEED}"
  "data.train_max_samples=${TRAIN_SAMPLES}"
  "data.val_max_samples=${VAL_SAMPLES}"
  data.subset_policy=random
  "data.subset_seed=${SEED}"
  data.projection_norm=raw
  "data.sample_num=${TRAIN_QUERIES}"
  "data.num_queries=${TRAIN_QUERIES}"
  "data.query_sampling.num_queries=${TRAIN_QUERIES}"
  "data.eval_sample_num=${EVAL_QUERIES}"
  "data.num_workers=${NUM_WORKERS}"
  "data.batch_size=${BATCH_SIZE}"
  data.eval_batch_size=1
  "data.pin_memory=true"
  data.persistent_workers=false
  "data.prefetch_factor=${PREFETCH_FACTOR}"
  data.resample_queries_each_epoch=true
  "++data.load_stage1_prior=false"
  "++data.load_stage1_mesh=false"
  model.ssq_fmt.view_complementary.training_phase=phase_a
  model.ssq_fmt.view_complementary.ablation=shared_only
  "++model.ssq_fmt.view_complementary.strong_shared_fusion=true"
  "++model.ssq_fmt.view_complementary.decoder_fusion_mode=joint_nonresidual"
  "++model.ssq_fmt.view_complementary.phase_a_aux_loss_scale=0.0"
  "++model.finetune.init_from_ckpt=${PHASE_A_CKPT}"
  "++model.finetune.train_modules_only=[complementary_aggregation,unified_density_decoder]"
  "optim.lr=${LR}"
  "trainer.max_epochs=${FINETUNE_EPOCHS}"
  "trainer.check_val_every_n_epoch=${VAL_EVERY}"
  trainer.num_sanity_val_steps=0
  trainer.enable_progress_bar=true
  trainer.gradient_clip_val=1.0
  callbacks.checkpoint.save_top_k=3
  callbacks.checkpoint.save_last=true
  callbacks.early_stopping.monitor=val_dice
  callbacks.early_stopping.mode=max
  "callbacks.early_stopping.patience=${PATIENCE}"
  "++callbacks.early_stopping.min_delta=0.0005"
)

evaluate_checkpoint() {
  local name="$1" checkpoint="$2" enabled="$3" compose="$4"
  local eval_dir="${RUN_ROOT}/${name}/full_volume_test"
  if [[ -f "${eval_dir}/metrics_summary.json" && -f "${eval_dir}/components/component_summary.json" ]]; then
    log "Evaluation already complete for ${name}"
    return
  fi
  mkdir -p "$eval_dir"
  local eval_cmd=(
    uv run python scripts/eval_morphology_amplitude_full_volume.py
    --exp fmt_simgen_v2_morphology_amplitude_phase_a
    --ckpt_path "$checkpoint"
    --split test
    --threshold "$THRESHOLD"
    --chunk_size "$CHUNK_SIZE"
    --max_samples "$EVAL_MAX_SAMPLES"
    --save_dir "$eval_dir"
    --save_predictions
    model=ssq_fmt
    data.dataset_type=fmt_simgen
    "data.test_max_samples=${TEST_SAMPLES}"
    data.projection_norm=raw
    "++data.load_stage1_prior=false"
    "++data.load_stage1_mesh=false"
    model.ssq_fmt.view_complementary.training_phase=phase_a
    model.ssq_fmt.view_complementary.ablation=shared_only
    "model.ssq_fmt.view_complementary.factorized_reconstruction.enabled=${enabled}"
    "model.ssq_fmt.view_complementary.factorized_reconstruction.compose_density=${compose}"
  )
  "${eval_cmd[@]}"
  uv run python scripts/eval_components_fmt_simgen.py \
    --eval_dir "$eval_dir" \
    --split test \
    --threshold "$THRESHOLD" \
    --max_samples "$EVAL_MAX_SAMPLES" \
    --save_dir "${eval_dir}/components"
}

run_variant() {
  local name="$1" enabled="$2" compose="$3" support_w="$4" amplitude_w="$5" component_w="$6"
  local out="${RUN_ROOT}/${name}"
  local last="${out}/checkpoints/last.ckpt"
  mkdir -p "$out"
  log "Training ${name}: enabled=${enabled}, compose=${compose}, weights=${support_w}/${amplitude_w}/${component_w}"
  local cmd=(
    uv run python scripts/train_morphology_amplitude.py fit
    "${COMMON[@]}"
    "paths.output_dir=${out}"
    "model.ssq_fmt.view_complementary.factorized_reconstruction.enabled=${enabled}"
    "model.ssq_fmt.view_complementary.factorized_reconstruction.compose_density=${compose}"
    "model.ssq_fmt.view_complementary.factorized_reconstruction.support_weight=${support_w}"
    "model.ssq_fmt.view_complementary.factorized_reconstruction.amplitude_weight=${amplitude_w}"
    "model.ssq_fmt.view_complementary.factorized_reconstruction.component_weight=${component_w}"
  )
  if [[ -f "$last" ]]; then
    # A resumed factorized checkpoint already contains the support head. Disable
    # baseline initialization and let Lightning restore model/optimizer state.
    cmd+=("model.finetune.init_from_ckpt=null" "ckpt_path=$(realpath "$last")" ckpt_weights_only=false)
  fi
  "${cmd[@]}"

  local best
  best="$(best_checkpoint "$out")"
  printf '%s\n' "$best" > "${out}/BEST_CHECKPOINT.txt"
  log "Best checkpoint for ${name}: ${best}"

  if [[ "$RUN_FULL_EVAL" == "1" ]]; then
    evaluate_checkpoint "$name" "$best" "$enabled" "$compose"
  fi
}

log "Running matched continuations from ${PHASE_A_CKPT}"
if [[ "$RUN_FULL_EVAL" == "1" ]]; then
  evaluate_checkpoint phase_a_reference "$PHASE_A_CKPT" false true
fi
run_variant scalar_control false true 0.0 0.0 0.0
run_variant support_aux true false 0.10 0.0 0.0
run_variant factorized_core true true 0.10 0.25 0.0
run_variant factorized_component true true 0.10 0.25 0.10

if [[ "$RUN_FULL_EVAL" == "1" ]]; then
  uv run python scripts/analysis/compare_morphology_amplitude.py --run_root "$RUN_ROOT"
fi
log "All variants complete. Results: ${RUN_ROOT}"
