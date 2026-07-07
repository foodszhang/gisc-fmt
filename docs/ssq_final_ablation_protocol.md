# SSQ-FMT Final Ablation Protocol

All ablations derive from `configs/exp/fmt_simgen_v2_ssq_final.yaml` and keep the same
FMT-SimGen v2 split, non-GT query sampler, 32768 query count, optimizer, validation
checkpoint selection, and measurement-derived candidate inputs.

## Variants

- Full: `fmt_simgen_v2_ssq_final`
- Post-aggregation conditioning: `fmt_simgen_v2_ssq_ablate_post_aggregation`
- Shared cross-view fusion: `fmt_simgen_v2_ssq_ablate_shared_fusion`
- Assignment-only reliability: `fmt_simgen_v2_ssq_ablate_assignment_only`
- Fixed footprint: `fmt_simgen_v2_ssq_ablate_fixed_footprint`

## Gate Command

```bash
uv run python scripts/run_ssq_final_ablation_pipeline.py \
  --stage gate \
  --variants full post_aggregation shared_fusion assignment_only fixed_footprint
```

Use `data.batch_size=1 trainer.accumulate_grad_batches=4` manually if the 32768-query
run approaches GPU memory limits.

## Current Status

Only Full gates have been run after the final-method repair. The best short gate reached
0.1037 validation Dice, below the required threshold. The four ablations are configured
and importable but were not trained because the Full model did not pass gate.

Do not create an ablation summary table with test metrics until Full and all four
variants pass the same gate and complete formal validation-selected runs.
