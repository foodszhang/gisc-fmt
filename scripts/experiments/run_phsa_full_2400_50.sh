#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export PYTHONUNBUFFERED=1

SEED="${SEED:-42}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-2400}"
VAL_SAMPLES="${VAL_SAMPLES:-300}"
TRAIN_QUERIES="${TRAIN_QUERIES:-8192}"
EVAL_QUERIES="${EVAL_QUERIES:-8192}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
VAL_INTERVAL="${VAL_INTERVAL:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-2}"

RUN_DIR="${RUN_DIR:-outputs/view_complementary/phsa_full_2400_seed${SEED}_e${MAX_EPOCHS}}"
OLD_EVAL_DIR="${OLD_EVAL_DIR:-outputs/view_complementary/phsa_full_volume_paired_20260629}"
FINAL_EVAL_DIR="${FINAL_EVAL_DIR:-outputs/view_complementary/phsa_full_2400_seed${SEED}_e${MAX_EPOCHS}_full_volume}"
CHUNK_SIZE="${CHUNK_SIZE:-32768}"
PROPOSAL_COUNT="${PROPOSAL_COUNT:-4096}"
PROPOSAL_SEED="${PROPOSAL_SEED:-42}"
THRESHOLD="${THRESHOLD:-0.5}"
RUN_FULL_EVAL="${RUN_FULL_EVAL:-1}"

mkdir -p "$RUN_DIR" "$FINAL_EVAL_DIR/predictions"

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

checkpoint_epoch() {
  uv run python - "$1" <<'PY'
import sys
import torch
obj = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(obj.get("epoch", -1)))
PY
}

best_checkpoint() {
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

resolve_init_checkpoint() {
  if [[ -n "${INIT_CKPT:-}" ]]; then
    [[ -f "$INIT_CKPT" ]] || die "Missing INIT_CKPT: $INIT_CKPT"
    realpath "$INIT_CKPT"
    return
  fi

  local phase_a
  phase_a="$({
    find outputs/view_complementary \
      -type f \
      -path "*phase_a_seed${SEED}/checkpoints/last.ckpt" \
      -printf '%T@ %p\n' 2>/dev/null || true
  } | sort -nr | head -n 1 | cut -d' ' -f2-)"
  if [[ -n "$phase_a" ]]; then
    realpath "$phase_a"
    return
  fi

  local warmup="outputs/view_complementary/long_1k_continue_candidate_warmup_seed${SEED}/checkpoints/last.ckpt"
  if [[ -f "$warmup" ]]; then
    realpath "$warmup"
    return
  fi

  die "No initialization checkpoint found. Set INIT_CKPT=/absolute/path/to/checkpoint.ckpt"
}

INIT_CKPT_RESOLVED="$(resolve_init_checkpoint)"
LAST_CKPT="${RUN_DIR}/checkpoints/last.ckpt"

log "PHSA full training"
log "Implementation: ablation=a3_geometry_only, separability=geometry_only"
log "Direct view reliability: valid_view * (epsilon + geometry_separability); no support multiplier"
log "Support remains through measurement-derived hypothesis construction and existence scoring"
log "Train samples=${TRAIN_SAMPLES}, val samples=${VAL_SAMPLES}, epochs=${MAX_EPOCHS}"
log "Data workers=${NUM_WORKERS}, prefetch=${PREFETCH_FACTOR}; unused Stage-1/descatter IO disabled"
log "Initialization checkpoint: ${INIT_CKPT_RESOLVED}"
log "Run directory: ${RUN_DIR}"

if [[ -f "$LAST_CKPT" ]] && [[ "$(checkpoint_epoch "$LAST_CKPT")" -ge $((MAX_EPOCHS - 1)) ]]; then
  log "Training already completed; reusing ${LAST_CKPT}"
else
  CMD=(
    uv run python train.py fit
    model=ssq_fmt
    exp=fmt_simgen_v2_view_complementary
    data.dataset_type=fmt_simgen
    "seed=${SEED}"
    "data.train_max_samples=${TRAIN_SAMPLES}"
    "data.val_max_samples=${VAL_SAMPLES}"
    data.subset_policy=random
    "data.subset_seed=${SEED}"
    "data.sample_num=${TRAIN_QUERIES}"
    "data.num_queries=${TRAIN_QUERIES}"
    "data.query_sampling.num_queries=${TRAIN_QUERIES}"
    "data.eval_sample_num=${EVAL_QUERIES}"
    "data.num_workers=${NUM_WORKERS}"
    "data.batch_size=${BATCH_SIZE}"
    data.eval_batch_size=1
    data.pin_memory=true
    data.persistent_workers=true
    "data.prefetch_factor=${PREFETCH_FACTOR}"
    data.resample_queries_each_epoch=true
    "data.descatter_target_files=[]"
    "++data.load_stage1_prior=false"
    "++data.load_stage1_mesh=false"
    trainer.check_val_every_n_epoch=1
    "+trainer.val_check_interval=${VAL_INTERVAL}"
    trainer.num_sanity_val_steps=0
    trainer.enable_progress_bar=true
    trainer.gradient_clip_val=1.0
    "trainer.max_epochs=${MAX_EPOCHS}"
    callbacks.checkpoint.save_top_k=3
    callbacks.checkpoint.save_last=true
    callbacks.early_stopping=null
    "paths.output_dir=${RUN_DIR}"
    model.ssq_fmt.view_complementary.training_phase=full
    model.ssq_fmt.view_complementary.ablation=a3_geometry_only
    model.ssq_fmt.view_complementary.separability_mode=geometry_only
    "++model.ssq_fmt.view_complementary.hypothesis_grid.enabled=false"
    "++model.ssq_fmt.view_complementary.routing.enabled=false"
    "++model.ssq_fmt.view_complementary.continuous_applicability=false"
    "++model.ssq_fmt.view_complementary.candidate_hidden_injection=true"
    "++model.ssq_fmt.view_complementary.context_warmup_enabled=false"
    model.ssq_fmt.view_complementary.lambda_separability_measurement=0.0
    model.ssq_fmt.view_complementary.lr.encoder=0.000005
    model.ssq_fmt.view_complementary.lr.constructor=0.00001
    model.ssq_fmt.view_complementary.lr.candidate_encoder=0.00003
    model.ssq_fmt.view_complementary.lr.candidate_context=0.00003
    model.ssq_fmt.view_complementary.lr.decoder=0.00002
    model.ssq_fmt.view_complementary.lr.separability=0.0
    model.ssq_fmt.view_complementary.lr.shared=0.000005
  )

  if [[ -f "$LAST_CKPT" ]]; then
    log "Resuming full trainer state from ${LAST_CKPT}"
    CMD+=("ckpt_path=$(realpath "$LAST_CKPT")" ckpt_weights_only=false)
  else
    log "Initializing weights from ${INIT_CKPT_RESOLVED}"
    CMD+=("ckpt_path=${INIT_CKPT_RESOLVED}" ckpt_weights_only=true)
  fi

  "${CMD[@]}"
fi

BEST_CKPT="$(best_checkpoint "$RUN_DIR")"
log "Best PHSA checkpoint: ${BEST_CKPT}"
printf '%s\n' "$BEST_CKPT" > "${RUN_DIR}/BEST_CHECKPOINT.txt"

if [[ "$RUN_FULL_EVAL" != "1" ]]; then
  log "RUN_FULL_EVAL=${RUN_FULL_EVAL}; training complete without test evaluation"
  exit 0
fi

[[ -d "${OLD_EVAL_DIR}/predictions/a2u" ]] || die "Missing prior A2-U predictions"
[[ -d "${OLD_EVAL_DIR}/predictions/a3_old" ]] || die "Missing prior A3-old predictions"

ln -sfn "$(realpath "${OLD_EVAL_DIR}/predictions/a2u")" "${FINAL_EVAL_DIR}/predictions/a2u"
ln -sfn "$(realpath "${OLD_EVAL_DIR}/predictions/a3_old")" "${FINAL_EVAL_DIR}/predictions/a3_old"

MANIFEST="${FINAL_EVAL_DIR}/PHSA_CHECKPOINT.txt"
if [[ -f "$MANIFEST" ]] && [[ "$(cat "$MANIFEST")" != "$BEST_CKPT" ]]; then
  log "Best checkpoint changed; removing stale PHSA full-volume predictions"
  rm -rf "${FINAL_EVAL_DIR}/predictions/phsa"
fi
printf '%s\n' "$BEST_CKPT" > "$MANIFEST"

log "Running aligned 300-sample full-volume evaluation"
uv run python scripts/eval_view_complementary_full_volume_paired_safe.py \
  --output_dir "$FINAL_EVAL_DIR" \
  --models a2u a3_old phsa \
  --phsa_run "$RUN_DIR" \
  --split test \
  --proposal_count "$PROPOSAL_COUNT" \
  --proposal_seed "$PROPOSAL_SEED" \
  --chunk_size "$CHUNK_SIZE" \
  --threshold "$THRESHOLD" \
  --bootstrap_samples 10000 \
  --device cuda

log "Complete. Final summary: ${FINAL_EVAL_DIR}/summary.md"
