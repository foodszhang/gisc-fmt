#!/usr/bin/env bash
set -Eeuo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

SEED="${SEED:-42}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-1000}"
VAL_SAMPLES="${VAL_SAMPLES:-64}"
TRAIN_QUERIES="${TRAIN_QUERIES:-8192}"
EVAL_QUERIES="${EVAL_QUERIES:-8192}"
VAL_INTERVAL="${VAL_INTERVAL:-1.0}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-8}"
BRANCH_EPOCHS="${BRANCH_EPOCHS:-10}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
RUN_ROOT="${RUN_ROOT:-outputs/view_complementary/long_1k_continue}"

WARMUP_DIR="${RUN_ROOT}_candidate_warmup_seed${SEED}"
A2U_DIR="${RUN_ROOT}_a2u_seed${SEED}"
A2S_DIR="${RUN_ROOT}_a2s_seed${SEED}"
A3_DIR="${RUN_ROOT}_a3_seed${SEED}"
SUMMARY_TXT="${RUN_ROOT}_summary.txt"

log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() {
  echo "ERROR: $*" >&2
  exit 1
}

resolve_phase_a_ckpt() {
  if [[ -n "${PHASE_A_CKPT:-}" ]]; then
    [[ -f "$PHASE_A_CKPT" ]] || die "Missing PHASE_A_CKPT: $PHASE_A_CKPT"
    realpath "$PHASE_A_CKPT"
    return
  fi
  local found
  found="$(
    find outputs/view_complementary \
      -type f \
      -path "*phase_a_seed${SEED}/checkpoints/last.ckpt" \
      -printf '%T@ %p\n' 2>/dev/null |
      sort -nr | head -n 1 | cut -d' ' -f2-
  )"
  [[ -n "$found" ]] || die "Set PHASE_A_CKPT=/abs/path/to/last.ckpt"
  realpath "$found"
}

checkpoint_epoch() {
  uv run python - "$1" <<'PY'
import sys, torch
obj = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(obj.get("epoch", -1)))
PY
}

stage_finished() {
  local dir="$1" target="$2" last="${1}/checkpoints/last.ckpt"
  [[ -f "$last" ]] || return 1
  [[ "$(checkpoint_epoch "$last")" -ge $((target - 1)) ]]
}

last_checkpoint() {
  local p="${1}/checkpoints/last.ckpt"
  [[ -f "$p" ]] || die "Missing checkpoint: $p"
  realpath "$p"
}

prune_stage_checkpoints() {
  local dir="$1"
  uv run python - "$dir" <<'PY'
from pathlib import Path
import re, sys

root = Path(sys.argv[1]) / "checkpoints"
if not root.exists():
    raise SystemExit(0)

files = list(root.glob("*.ckpt"))
if len(files) <= 2:
    print(f"[checkpoint-prune] {root}: {len(files)} file(s)")
    raise SystemExit(0)

last = root / "last.ckpt"
others = [p for p in files if p.name != "last.ckpt"]

def score(p):
    for pat in (r"val_dice=([0-9.]+)", r"-([0-9]+\.[0-9]+)\.ckpt$"):
        m = re.search(pat, p.name)
        if m:
            return float(m.group(1).rstrip("."))
    return float("-inf")

best = max(others, key=lambda p: (score(p), p.stat().st_mtime)) if others else None
keep = {p.resolve() for p in (last, best) if p is not None and p.exists()}

for p in files:
    if p.resolve() not in keep:
        print(f"[checkpoint-prune] delete {p}")
        p.unlink()
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
  "data.sample_num=${TRAIN_QUERIES}"
  "data.num_queries=${TRAIN_QUERIES}"
  "data.query_sampling.num_queries=${TRAIN_QUERIES}"
  "data.eval_sample_num=${EVAL_QUERIES}"
  "data.num_workers=${NUM_WORKERS}"
  "data.batch_size=2"
  data.pin_memory=true
  data.persistent_workers=true
  "data.prefetch_factor=${PREFETCH_FACTOR}"
  trainer.check_val_every_n_epoch=1
  "+trainer.val_check_interval=${VAL_INTERVAL}"
  trainer.num_sanity_val_steps=0
  trainer.enable_progress_bar=true
  callbacks.checkpoint.save_top_k=1
  callbacks.checkpoint.save_last=true
  callbacks.early_stopping=null
)

TRAIN_MODULES_CANDIDATE_ONLY='[complementary_aggregation.candidate_projection,unified_density_decoder.candidate_context,unified_density_decoder.candidate_norm,unified_density_decoder.candidate_input]'
TRAIN_MODULES_JOINT_LITE='[complementary_aggregation.candidate_projection,unified_density_decoder.candidate_context,unified_density_decoder.candidate_norm,unified_density_decoder.candidate_input,unified_density_decoder.head]'

run_warmup() {
  local init_ckpt="$1" dir="$WARMUP_DIR" last="${WARMUP_DIR}/checkpoints/last.ckpt"
  if stage_finished "$dir" "$WARMUP_EPOCHS"; then
    log "Warm-up already complete"
    prune_stage_checkpoints "$dir"
    return
  fi

  local cmd=(
    uv run python train.py fit
    "${COMMON_ARGS[@]}"
    "paths.output_dir=${dir}"
    model.ssq_fmt.view_complementary.training_phase=phase_b
    model.ssq_fmt.view_complementary.ablation=a2
    model.ssq_fmt.view_complementary.separability_mode=none
    model.ssq_fmt.view_complementary.lr.encoder=0.0
    model.ssq_fmt.view_complementary.lr.constructor=0.0
    model.ssq_fmt.view_complementary.lr.candidate_encoder=0.00005
    model.ssq_fmt.view_complementary.lr.candidate_context=0.00005
    model.ssq_fmt.view_complementary.lr.decoder=0.00005
    model.ssq_fmt.view_complementary.lr.separability=0.0
    model.ssq_fmt.view_complementary.lr.shared=0.0
    "+model.finetune.train_modules_only=${TRAIN_MODULES_CANDIDATE_ONLY}"
    "trainer.max_epochs=${WARMUP_EPOCHS}"
  )

  if [[ -f "$last" ]]; then
    cmd+=("ckpt_path=$(realpath "$last")" ckpt_weights_only=false)
  else
    cmd+=("ckpt_path=${init_ckpt}" ckpt_weights_only=true)
  fi

  "${cmd[@]}"
  prune_stage_checkpoints "$dir"
}

run_branch() {
  local label="$1" dir="$2" ablation="$3" separability="$4" init_ckpt="$5"
  local last="${dir}/checkpoints/last.ckpt"

  if stage_finished "$dir" "$BRANCH_EPOCHS"; then
    log "$label already complete"
    prune_stage_checkpoints "$dir"
    return
  fi

  local cmd=(
    uv run python train.py fit
    "${COMMON_ARGS[@]}"
    "paths.output_dir=${dir}"
    model.ssq_fmt.view_complementary.training_phase=phase_b
    "model.ssq_fmt.view_complementary.ablation=${ablation}"
    "model.ssq_fmt.view_complementary.separability_mode=${separability}"
    model.ssq_fmt.view_complementary.lr.encoder=0.0
    model.ssq_fmt.view_complementary.lr.constructor=0.0
    model.ssq_fmt.view_complementary.lr.candidate_encoder=0.00003
    model.ssq_fmt.view_complementary.lr.candidate_context=0.00003
    model.ssq_fmt.view_complementary.lr.decoder=0.000005
    model.ssq_fmt.view_complementary.lr.separability=0.0
    model.ssq_fmt.view_complementary.lr.shared=0.0
    "+model.finetune.train_modules_only=${TRAIN_MODULES_JOINT_LITE}"
    "trainer.max_epochs=${BRANCH_EPOCHS}"
  )

  if [[ -f "$last" ]]; then
    cmd+=("ckpt_path=$(realpath "$last")" ckpt_weights_only=false)
  else
    cmd+=("ckpt_path=${init_ckpt}" ckpt_weights_only=true)
  fi

  "${cmd[@]}"
  prune_stage_checkpoints "$dir"
}

write_report() {
  uv run python - "$A2U_DIR" "$A2S_DIR" "$A3_DIR" "$SUMMARY_TXT" <<'PY'
from collections import defaultdict
from pathlib import Path
import math, statistics, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

runs = {"A2-U": Path(sys.argv[1]), "A2-S": Path(sys.argv[2]), "A3": Path(sys.argv[3])}
out = Path(sys.argv[4])

def load(run):
    merged = defaultdict(list)
    for f in sorted(run.rglob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime):
        ea = EventAccumulator(str(f.parent), size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            for e in ea.Scalars(tag):
                merged[tag].append((e.step, e.value, e.wall_time))
    result = {}
    for tag, rows in merged.items():
        by_step = {}
        for row in rows:
            if row[0] not in by_step or row[2] > by_step[row[0]][2]:
                by_step[row[0]] = row
        result[tag] = [x[1] for x in sorted(by_step.values())]
    return result

def last(d, tag):
    return d[tag][-1] if tag in d and d[tag] else math.nan

def mean_tail(d, tag, n=6):
    return statistics.fmean(d[tag][-n:]) if tag in d and d[tag] else math.nan

data = {k: load(v) for k, v in runs.items()}
lines = [f"{'Run':8s} {'Last':>10s} {'Shared':>10s} {'Delta':>10s} {'T6 Dice':>10s} {'T6 Delta':>10s} {'T6 Loss':>10s}",
         "-" * 82]
rows = {}
for name in ("A2-U", "A2-S", "A3"):
    d = data[name]
    row = {
        "last": last(d, "val_dice"),
        "shared": last(d, "val_shared_dice"),
        "delta": last(d, "val_final_minus_shared_dice"),
        "t6_dice": mean_tail(d, "val_dice"),
        "t6_delta": mean_tail(d, "val_final_minus_shared_dice"),
        "t6_loss": mean_tail(d, "val_formal_density_loss"),
    }
    rows[name] = row
    lines.append(
        f"{name:8s} {row['last']:10.6f} {row['shared']:10.6f} {row['delta']:10.6f} "
        f"{row['t6_dice']:10.6f} {row['t6_delta']:10.6f} {row['t6_loss']:10.6f}"
    )

support = rows["A2-S"]["t6_dice"] - rows["A2-U"]["t6_dice"]
geometry = rows["A3"]["t6_dice"] - rows["A2-S"]["t6_dice"]
lines += ["", f"Tail-6 support gain:  {support:+.6f}",
          f"Tail-6 geometry gain: {geometry:+.6f}"]
report = "\n".join(lines)
print(report)
out.write_text(report + "\n")
PY
}

PHASE_A_CKPT_RESOLVED="$(resolve_phase_a_ckpt)"

log "Using Phase A checkpoint: $PHASE_A_CKPT_RESOLVED"
log "Validation is limited to ${VAL_SAMPLES} samples x ${EVAL_QUERIES} queries"
log "Checkpoint policy: best 1 + last"

run_warmup "$PHASE_A_CKPT_RESOLVED"
WARMUP_CKPT="$(last_checkpoint "$WARMUP_DIR")"

run_branch "A2-U" "$A2U_DIR" a2 none "$WARMUP_CKPT"
run_branch "A2-S" "$A2S_DIR" a2_s none "$WARMUP_CKPT"
run_branch "A3" "$A3_DIR" a3 geometry_only "$WARMUP_CKPT"

write_report
log "Done. Summary: $SUMMARY_TXT"
