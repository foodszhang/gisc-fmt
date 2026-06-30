#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export PYTHONUNBUFFERED=1

SEED="${SEED:-42}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-2400}"
VAL_SAMPLES="${VAL_SAMPLES:-300}"
TEST_SAMPLES="${TEST_SAMPLES:-300}"
TRAIN_QUERIES="${TRAIN_QUERIES:-16384}"
EVAL_QUERIES="${EVAL_QUERIES:-16384}"
HYPOTHESIS_POINTS="${HYPOTHESIS_POINTS:-4096}"
HYPOTHESIS_SEED="${HYPOTHESIS_SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PHASE_A_EPOCHS="${PHASE_A_EPOCHS:-40}"
PHASE_B_EPOCHS="${PHASE_B_EPOCHS:-15}"
FULL_EPOCHS="${FULL_EPOCHS:-30}"
VAL_EVERY="${VAL_EVERY:-2}"
SELECTION_SAMPLES="${SELECTION_SAMPLES:-16}"

RUN_ROOT="${RUN_ROOT:-outputs/view_complementary/phsa_strong_joint_2400_seed${SEED}}"
PHASE_A_DIR="${RUN_ROOT}_phase_a_e${PHASE_A_EPOCHS}"
PHASE_B_DIR="${RUN_ROOT}_phase_b_e${PHASE_B_EPOCHS}"
FULL_DIR="${RUN_ROOT}_full_e${FULL_EPOCHS}"
SELECTION_DIR="${RUN_ROOT}_checkpoint_selection"
OLD_EVAL_DIR="${OLD_EVAL_DIR:-outputs/view_complementary/phsa_full_volume_paired_20260629}"
FINAL_EVAL_DIR="${FINAL_EVAL_DIR:-${RUN_ROOT}_full_volume}"
RUN_FULL_EVAL="${RUN_FULL_EVAL:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-32768}"
PROPOSAL_COUNT="${PROPOSAL_COUNT:-${HYPOTHESIS_POINTS}}"
PROPOSAL_SEED="${PROPOSAL_SEED:-${HYPOTHESIS_SEED}}"
THRESHOLD="${THRESHOLD:-0.5}"

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

TOTAL_EPOCHS=$((PHASE_A_EPOCHS + PHASE_B_EPOCHS + FULL_EPOCHS))
for stage_epochs in "$PHASE_A_EPOCHS" "$PHASE_B_EPOCHS" "$FULL_EPOCHS"; do
  (( stage_epochs > 0 )) || die "Each stage length must be positive"
done
[[ "$PROPOSAL_COUNT" -eq "$HYPOTHESIS_POINTS" ]] || \
  die "PROPOSAL_COUNT must equal HYPOTHESIS_POINTS"
[[ "$PROPOSAL_SEED" -eq "$HYPOTHESIS_SEED" ]] || \
  die "PROPOSAL_SEED must equal HYPOTHESIS_SEED"
[[ "$SELECTION_SAMPLES" -gt 0 ]] || die "SELECTION_SAMPLES must be positive"

mkdir -p "$PHASE_A_DIR" "$PHASE_B_DIR" "$FULL_DIR" "$SELECTION_DIR" \
  "$FINAL_EVAL_DIR/predictions"

checkpoint_epoch() {
  uv run python - "$1" <<'PY'
import sys
import torch
obj = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(obj.get("epoch", -1)))
PY
}

stage_finished() {
  local dir="$1" target="$2" last="${1}/checkpoints/last.ckpt"
  [[ -f "${dir}/STAGE_DONE" ]] && return 0
  [[ -f "$last" ]] || return 1
  [[ "$(checkpoint_epoch "$last")" -ge $((target - 1)) ]]
}

best_sampled_checkpoint() {
  uv run python - "$1" <<'PY'
from pathlib import Path
import re
import sys
root = Path(sys.argv[1]) / "checkpoints"
files = [p for p in root.glob("*.ckpt") if p.name != "last.ckpt"]
if not files:
    last = root / "last.ckpt"
    if last.exists():
        print(last.resolve())
        raise SystemExit(0)
    raise SystemExit(f"No checkpoint under {root}")

def score(path):
    for pattern in (r"val_dice=([0-9.]+)", r"-([0-9]+\.[0-9]+)\.ckpt$"):
        match = re.search(pattern, path.name)
        if match:
            return float(match.group(1).rstrip(".")), path.stat().st_mtime
    return float("-inf"), path.stat().st_mtime

print(max(files, key=score).resolve())
PY
}

COMMON_ARGS=(
  model=ssq_fmt
  exp=fmt_simgen_v2_view_complementary
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
  trainer.accumulate_grad_batches=1
  data.pin_memory=true
  data.persistent_workers=false
  "data.prefetch_factor=${PREFETCH_FACTOR}"
  data.resample_queries_each_epoch=true
  "++data.load_stage1_prior=false"
  "++data.load_stage1_mesh=false"
  "trainer.check_val_every_n_epoch=${VAL_EVERY}"
  trainer.num_sanity_val_steps=0
  trainer.enable_progress_bar=true
  trainer.gradient_clip_val=1.0
  callbacks.checkpoint.save_top_k=3
  callbacks.checkpoint.save_last=true
  callbacks.early_stopping.monitor=val_dice
  callbacks.early_stopping.mode=max
  callbacks.early_stopping.check_finite=true
  model.ssq_fmt.view_complementary.ablation=full
  model.ssq_fmt.view_complementary.separability_mode=geometry_measurement
  "++model.ssq_fmt.view_complementary.strong_shared_fusion=true"
  "++model.ssq_fmt.view_complementary.decoder_fusion_mode=joint_nonresidual"
  "++model.ssq_fmt.view_complementary.support_weighted_reliability=false"
  "++model.ssq_fmt.view_complementary.sample_level_hypotheses.enabled=true"
  "++model.ssq_fmt.view_complementary.sample_level_hypotheses.count=${HYPOTHESIS_POINTS}"
  "++model.ssq_fmt.view_complementary.sample_level_hypotheses.seed=${HYPOTHESIS_SEED}"
  "++model.ssq_fmt.view_complementary.hypothesis_grid.enabled=false"
  "++model.ssq_fmt.view_complementary.routing.enabled=false"
  "++model.ssq_fmt.view_complementary.continuous_applicability=true"
  "++model.ssq_fmt.view_complementary.candidate_hidden_injection=true"
  model.ssq_fmt.view_complementary.lambda_separability_measurement=0.05
  "++model.ssq_fmt.view_complementary.phase_a_aux_loss_scale=0.25"
  "++model.ssq_fmt.memory.checkpoint_encoder=true"
)

run_phase_a() {
  local last="${PHASE_A_DIR}/checkpoints/last.ckpt"
  if stage_finished "$PHASE_A_DIR" "$PHASE_A_EPOCHS"; then
    log "Phase A already complete"
    return
  fi
  log "Phase A: strong shared reconstruction and sample-level source-hypothesis learning"
  log "Epochs=${PHASE_A_EPOCHS}; base LR=3e-4"
  local cmd=(
    uv run python scripts/train_phsa_sample_level.py fit
    "${COMMON_ARGS[@]}"
    "paths.output_dir=${PHASE_A_DIR}"
    model.ssq_fmt.view_complementary.training_phase=phase_a
    "++model.ssq_fmt.view_complementary.context_warmup_enabled=true"
    optim.lr=0.0003
    "++optim.scheduler.warmup_epochs=5"
    "++optim.scheduler.warmup_start_factor=0.2"
    callbacks.early_stopping.patience=6
    "++callbacks.early_stopping.min_delta=0.001"
    "trainer.max_epochs=${PHASE_A_EPOCHS}"
  )
  if [[ -f "$last" ]]; then
    cmd+=("ckpt_path=$(realpath "$last")" ckpt_weights_only=false)
  fi
  "${cmd[@]}"
  touch "${PHASE_A_DIR}/STAGE_DONE"
}

run_phase_b() {
  local init_ckpt="$1" last="${PHASE_B_DIR}/checkpoints/last.ckpt"
  if stage_finished "$PHASE_B_DIR" "$PHASE_B_EPOCHS"; then
    log "Phase B already complete"
    return
  fi
  log "Phase B: candidate-conditioned joint-head warm-up"
  local cmd=(
    uv run python scripts/train_phsa_sample_level.py fit
    "${COMMON_ARGS[@]}"
    "paths.output_dir=${PHASE_B_DIR}"
    model.ssq_fmt.view_complementary.training_phase=phase_b
    "++model.ssq_fmt.view_complementary.context_warmup_enabled=true"
    model.ssq_fmt.view_complementary.context_warmup_steps=500
    model.ssq_fmt.view_complementary.lr.encoder=0.0
    model.ssq_fmt.view_complementary.lr.constructor=0.00001
    model.ssq_fmt.view_complementary.lr.candidate_encoder=0.00003
    model.ssq_fmt.view_complementary.lr.candidate_context=0.00003
    model.ssq_fmt.view_complementary.lr.decoder=0.00003
    model.ssq_fmt.view_complementary.lr.separability=0.00001
    model.ssq_fmt.view_complementary.lr.shared=0.000005
    "++optim.scheduler.warmup_epochs=2"
    "++optim.scheduler.warmup_start_factor=0.5"
    callbacks.early_stopping.patience=4
    "++callbacks.early_stopping.min_delta=0.001"
    "trainer.max_epochs=${PHASE_B_EPOCHS}"
  )
  if [[ -f "$last" ]]; then
    cmd+=("ckpt_path=$(realpath "$last")" ckpt_weights_only=false)
  else
    cmd+=("ckpt_path=${init_ckpt}" ckpt_weights_only=true)
  fi
  "${cmd[@]}"
  touch "${PHASE_B_DIR}/STAGE_DONE"
}

run_full() {
  local init_ckpt="$1" last="${FULL_DIR}/checkpoints/last.ckpt"
  if stage_finished "$FULL_DIR" "$FULL_EPOCHS"; then
    log "Phase C already complete"
    return
  fi
  log "Phase C: joint end-to-end fine-tuning of the active PHSA path"
  local cmd=(
    uv run python scripts/train_phsa_sample_level.py fit
    "${COMMON_ARGS[@]}"
    "paths.output_dir=${FULL_DIR}"
    model.ssq_fmt.view_complementary.training_phase=full
    "++model.ssq_fmt.view_complementary.context_warmup_enabled=false"
    model.ssq_fmt.view_complementary.lr.encoder=0.00002
    model.ssq_fmt.view_complementary.lr.constructor=0.00002
    model.ssq_fmt.view_complementary.lr.candidate_encoder=0.00003
    model.ssq_fmt.view_complementary.lr.candidate_context=0.00003
    model.ssq_fmt.view_complementary.lr.decoder=0.00003
    model.ssq_fmt.view_complementary.lr.separability=0.00001
    model.ssq_fmt.view_complementary.lr.shared=0.00002
    "++optim.scheduler.warmup_epochs=3"
    "++optim.scheduler.warmup_start_factor=0.5"
    callbacks.early_stopping.patience=6
    "++callbacks.early_stopping.min_delta=0.001"
    "trainer.max_epochs=${FULL_EPOCHS}"
  )
  if [[ -f "$last" ]]; then
    cmd+=("ckpt_path=$(realpath "$last")" ckpt_weights_only=false)
  else
    cmd+=("ckpt_path=${init_ckpt}" ckpt_weights_only=true)
  fi
  "${cmd[@]}"
  touch "${FULL_DIR}/STAGE_DONE"
}

log "Audited strong-joint PHSA curriculum: ${PHASE_A_EPOCHS}+${PHASE_B_EPOCHS}+${FULL_EPOCHS}=${TOTAL_EPOCHS} epochs"
log "Train=${TRAIN_SAMPLES}, val=${VAL_SAMPLES}, test=${TEST_SAMPLES} samples"
log "Density queries=${TRAIN_QUERIES}; sample-level hypothesis points=${HYPOTHESIS_POINTS}"
log "Dataset supplies raw measurements; the network owns normalization"
log "Persistent workers are disabled so epoch-wise query resampling is effective"

uv run python scripts/preflight_phsa_curriculum.py \
  --train-samples "$TRAIN_SAMPLES" \
  --val-samples "$VAL_SAMPLES" \
  --test-samples "$TEST_SAMPLES" \
  --train-queries "$TRAIN_QUERIES" \
  --eval-queries "$EVAL_QUERIES" \
  --hypothesis-points "$HYPOTHESIS_POINTS" \
  --hypothesis-seed "$HYPOTHESIS_SEED" \
  --batch-size "$BATCH_SIZE" \
  --phase-a-epochs "$PHASE_A_EPOCHS" \
  --phase-b-epochs "$PHASE_B_EPOCHS" \
  --full-epochs "$FULL_EPOCHS" \
  --val-every "$VAL_EVERY" \
  --seed "$SEED"

run_phase_a
PHASE_A_BEST="$(best_sampled_checkpoint "$PHASE_A_DIR")"
printf '%s\n' "$PHASE_A_BEST" > "${PHASE_A_DIR}/BEST_CHECKPOINT.txt"

run_phase_b "$PHASE_A_BEST"
PHASE_B_BEST="$(best_sampled_checkpoint "$PHASE_B_DIR")"
printf '%s\n' "$PHASE_B_BEST" > "${PHASE_B_DIR}/BEST_CHECKPOINT.txt"

run_full "$PHASE_B_BEST"
SAMPLED_BEST="$(best_sampled_checkpoint "$FULL_DIR")"
printf '%s\n' "$SAMPLED_BEST" > "${FULL_DIR}/BEST_SAMPLED_QUERY_CHECKPOINT.txt"

if [[ "$RUN_FULL_EVAL" != "1" ]]; then
  log "RUN_FULL_EVAL=${RUN_FULL_EVAL}; curriculum complete without checkpoint selection/test"
  exit 0
fi

log "Selecting among top-k and last checkpoints on validation-only full volumes"
uv run python scripts/select_phsa_checkpoint_full_volume.py \
  --run-dir "$FULL_DIR" \
  --output-dir "$SELECTION_DIR" \
  --max-samples "$SELECTION_SAMPLES" \
  --proposal-count "$PROPOSAL_COUNT" \
  --proposal-seed "$PROPOSAL_SEED" \
  --chunk-size "$CHUNK_SIZE" \
  --threshold "$THRESHOLD" \
  --device cuda

SELECTED_RUN="$(cat "${SELECTION_DIR}/SELECTED_RUN.txt")"
SELECTED_CHECKPOINT="$(cat "${SELECTION_DIR}/SELECTED_CHECKPOINT.txt")"
printf '%s\n' "$SELECTED_CHECKPOINT" > "${FULL_DIR}/BEST_CHECKPOINT.txt"
log "Selected PHSA checkpoint: ${SELECTED_CHECKPOINT}"

log "Auditing reusable A2-U/A3-old prediction caches"
uv run python scripts/audit_phsa_reused_predictions.py \
  --output-dir "$OLD_EVAL_DIR" \
  --models a2u a3_old \
  --proposal-count "$PROPOSAL_COUNT" \
  --proposal-seed "$PROPOSAL_SEED" \
  --threshold "$THRESHOLD" \
  --expected-samples "$TEST_SAMPLES" \
  --volume-shape 190 200 104

[[ -d "${OLD_EVAL_DIR}/predictions/a2u" ]] || die "Missing prior A2-U predictions"
[[ -d "${OLD_EVAL_DIR}/predictions/a3_old" ]] || die "Missing prior A3-old predictions"
ln -sfn "$(realpath "${OLD_EVAL_DIR}/predictions/a2u")" "${FINAL_EVAL_DIR}/predictions/a2u"
ln -sfn "$(realpath "${OLD_EVAL_DIR}/predictions/a3_old")" "${FINAL_EVAL_DIR}/predictions/a3_old"

MANIFEST="${FINAL_EVAL_DIR}/PHSA_CHECKPOINT.txt"
if [[ -f "$MANIFEST" ]] && [[ "$(cat "$MANIFEST")" != "$SELECTED_CHECKPOINT" ]]; then
  rm -rf "${FINAL_EVAL_DIR}/predictions/phsa"
fi
printf '%s\n' "$SELECTED_CHECKPOINT" > "$MANIFEST"

log "Running aligned ${TEST_SAMPLES}-sample full-volume test evaluation"
uv run python scripts/eval_view_complementary_full_volume_paired_safe.py \
  --output_dir "$FINAL_EVAL_DIR" \
  --models a2u a3_old phsa \
  --phsa_run "$SELECTED_RUN" \
  --split test \
  --max_samples "$TEST_SAMPLES" \
  --proposal_count "$PROPOSAL_COUNT" \
  --proposal_seed "$PROPOSAL_SEED" \
  --chunk_size "$CHUNK_SIZE" \
  --threshold "$THRESHOLD" \
  --bootstrap_samples 10000 \
  --device cuda

log "Complete. Final summary: ${FINAL_EVAL_DIR}/summary.md"
