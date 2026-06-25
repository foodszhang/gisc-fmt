#!/usr/bin/env bash
set -euo pipefail

cd /home/foods/pro/gisc_fmt_repo

old_pid="${1:?old training pid required}"
old_ckpt="${2:?old checkpoint path required}"

watch_log="logs/real_0622_mixed3foci_main/finetune_allcand_watcher.log"
run_log="logs/real_0622_mixed3foci_main/finetune_allcand_from_final.log"
mkdir -p logs/real_0622_mixed3foci_main

echo "[$(date '+%F %T')] watcher started, waiting for old pid ${old_pid}" >> "${watch_log}"
while kill -0 "${old_pid}" 2>/dev/null; do
  sleep 60
done

echo "[$(date '+%F %T')] old pid exited" >> "${watch_log}"
for _ in $(seq 1 60); do
  if [ -s "${old_ckpt}" ]; then
    break
  fi
  echo "[$(date '+%F %T')] waiting for checkpoint ${old_ckpt}" >> "${watch_log}"
  sleep 30
done

if [ ! -s "${old_ckpt}" ]; then
  echo "[$(date '+%F %T')] ERROR missing checkpoint ${old_ckpt}" >> "${watch_log}"
  exit 1
fi

echo "[$(date '+%F %T')] starting all-candidate finetune from ${old_ckpt}" >> "${watch_log}"
PYTHONUNBUFFERED=1 uv run python train.py fit \
  model=gisc_fmt \
  exp=fmt_simgen_real_20260622_luoshu1_mixed3foci_main \
  data.dataset_type=fmt_simgen \
  data.batch_size=1 data.eval_batch_size=1 \
  trainer.accumulate_grad_batches=4 \
  trainer.max_epochs=40 \
  data.sample_num=131072 data.num_queries=131072 data.eval_sample_num=131072 \
  data.val_max_samples=100 \
  trainer.check_val_every_n_epoch=3 \
  callbacks.early_stopping=null \
  ckpt_path="${old_ckpt}" \
  ckpt_weights_only=true \
  data.source_hypothesis.top_m=5 \
  +model.source_instance_cue.aggregation=all_candidates_null \
  +model.source_instance_cue.null_logit=0.0 \
  +model.source_instance_cue.score_weight=2.0 \
  +model.source_instance_cue.min_scale_mm=1.0 \
  +model.source_instance_cue.max_scale_mm=20.0 \
  2>&1 | tee "${run_log}"
