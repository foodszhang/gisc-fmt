# Source-Separable Query Mainline

The formal GISC-FMT mainline targets multi-source FMT reconstruction. The central
failure mode is transport-induced source mixing: different internal fluorescent
sources scatter into overlapping multi-view surface measurements, so weak sources can
be suppressed by stronger sources and adjacent sources can merge into one density blob.
Density-only reconstruction remains vulnerable unless the query representation carries
measurement-derived source-separability cues.

## Formal Method

The main method is **source-separable query representation (SSQ)**, not a source-slot
decoder. SSQ keeps the strongest E15 reconstruction path and augments each query with
measurement-derived source-separable evidence:

- PTFA/PCFS provide measurement-aligned query evidence.
- Canonical reliability aggregation builds a query-canonical representation across
  views.
- Measurement-derived source hypotheses come only from proposal heatmaps computed from
  `proj.npz` and geometry.
- `source_instance_cue` encodes over-complete source-hypothesis geometry, score,
  ownership-like distance cues, and entropy/margin statistics into the query feature.
- Center and distance heads remain simulation-derived auxiliary training regularizers.
  They are not inference inputs and are not used for post-processing, component
  splitting, or threshold adjustment.
- The final density is still produced by the established E15 density head. The
  `source_instance_decoder` is disabled in the formal mainline.

Main training config:

```bash
uv run python train.py fit exp=fmt_simgen_v2_ssq_main data.dataset_type=fmt_simgen
```

Recommended full-volume and component evaluation:

```bash
uv run python scripts/eval_full_volume_fmt_simgen.py \
  --exp fmt_simgen_v2_ssq_main \
  --ckpt_path /path/to/checkpoint.ckpt \
  --split test \
  --threshold 0.5 \
  --save_predictions \
  --save_dir outputs/fmt_simgen_v2_source_slots_eval/ssq_main_test300

uv run python scripts/eval_components_fmt_simgen.py \
  --eval_dir outputs/fmt_simgen_v2_source_slots_eval/ssq_main_test300 \
  --split test \
  --save_dir outputs/fmt_simgen_v2_source_slots_eval/ssq_main_test300/components
```

Each formal run should report:

- `metrics_summary.json`
- `metrics_grouped.csv`, especially `num_foci`
- `component_summary.json`
- `component_by_num_foci.csv`

## Data-Leakage Boundary

- Proposal heatmaps come only from `proj.npz` and geometric projection.
- Source hypotheses and source cues do not use GT source centers, GT boxes, GT masks,
  GT foreground, or `tumor_params.json`.
- The Non-GT sampler keeps the existing no-GT-leakage query allocation policy.
- Center/distance targets are generated from synthetic source support only for training
  auxiliary supervision.
- Real-experiment inference requires only multi-view surface fluorescence measurements
  and geometric calibration. It does not require GT source centers, masks, boxes,
  center targets, distance targets, or GT component annotations.

## Ablations

Formal SSQ ablations:

- `exp=fmt_simgen_v2_ssq_no_ptfa`
- `exp=fmt_simgen_v2_ssq_no_canonical_reliability`
- `exp=fmt_simgen_v2_ssq_no_source_cue`
- `exp=fmt_simgen_v2_ssq_no_center_aux`
- `exp=fmt_simgen_v2_ssq_no_distance_aux`

Source-slot decoder experiments are retained only as ablations:

- `exp=fmt_simgen_v2_source_slots_soft_union`
- `exp=fmt_simgen_v2_source_slots_distance_ownership`
- `exp=fmt_simgen_v2_source_slots_weighted_sum`

The source-slot soft-union decoder did not outperform the E15/SSQ density path as a
final-output method on test300, so it is not the formal mainline.
