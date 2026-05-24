# FMT-SimGen Integration and Ablation Results

## Scope

This document records the FMT-SimGen data integration and the main ablation results before
the next method iteration. The current code supports:

- Direct loading of FMT-SimGen samples from
  `/home/foods/pro/FMT-SimGen/data/uniform_1000_20k`.
- FMT-SimGen trunk-local physical projection with `points_mm`.
- Non-GT fixed-budget query sampling with trunk-uniform and measurement-proposal branches.
- s3-only PTFA variants.
- Query-level residual scorer variants.
- High-resolution s1 PTFA feature-refinement variants.
- Measurement-candidate dense/semi-dense checkpoint evaluation.

The training/evaluation split used for the main results is the full `800/200` train/val split.
All full runs used `num_queries=16384`, FMT-SimGen physical projection, and the non-GT mixed
sampler unless stated otherwise.

## Data and Sampler Notes

The FMT-SimGen dataset contains 1000 samples under `samples/`. Each sample provides
`gt_voxels.npy` with shape `(190, 200, 104)` and `proj.npz` with seven views
`[-90, -60, -30, 0, 30, 60, 90]` plus `depth_*` maps.

The active sampler is non-GT:

- It does not use `gt_voxels`, `gt_nodes`, `tumor_params`, GT boxes, GT foreground masks, or
  `body_mask` for query allocation.
- `gt_voxels` is only used after sampling to look up labels.
- The mixed sampler uses 50% trunk-uniform queries and 50% measurement-proposal queries.

Proposal sanity before training:

- Proposal top-5% GT coverage: `0.68`.
- Mixed foreground ratio: `0.004`, about `3.4x` trunk-uniform.

## Validation Metric Naming

The historical `val_dice` values in this document are sampled-query validation metrics. They
should be interpreted as `val_query_dice`, not dense full-volume reconstruction Dice. Regular
FMT-SimGen training uses fixed-budget sampled queries from the same non-GT sampler for train and
validation, so this metric is useful for fast training monitoring but does not replace
checkpoint-level dense or semi-dense evaluation.

Newer code logs:

- `val_query_dice` for sampled-query validation.
- `val_full_dice` only when validation points cover the full voxel grid.
- `val_dice` as a backward-compatible checkpoint callback alias.

## Main Results

| Experiment | Description | Best val_query_dice | Epoch | Notes |
| --- | --- | ---: | ---: | --- |
| E1a | trunk-uniform only | 0.4144 | 25 | Non-GT, no proposal branch |
| E1b | trunk 50% + proposal 50% | 0.5629 | 25 | Baseline before PTFA/scorer |
| E2 | s3 fixed Gaussian PTFA, sigma=1.0 | 0.5768 | 26 | Small gain over E1b |
| E3 | s3 exit-depth PTFA, sigma=[0.8, 2.5] | 0.5448 | 24 | Worse than fixed PTFA |
| E3' | calibrated exit-depth PTFA, sigma=[0.6, 1.2] | 0.5395 | 26 | Worse again; not over-smoothing |
| E4 lambda=0.05 | residual scorer only | 0.6109 | 26 | Clear gain over E1b/E2 |
| E4 lambda=0.20 | residual scorer only | 0.6315 | 21 | Best sampled-query scorer run |
| E4' lambda=0.30 | residual scorer only | 0.6294 | 23 | Near lambda=0.20, slightly lower |
| E4' lambda=0.50 | residual scorer only | 0.6051 | 23 | Too strong; optimization/performance drops |
| E5 | residual scorer lambda=0.20 + fixed PTFA | 0.5904 | 23 | Corrected run; scorer and PTFA interfere |
| E6 | corrected exit-depth PTFA + geometry gate + residual scorer | 0.4645 | 23 | Geometry gate path underperformed |
| E7 | E4 scorer + s1 fixed-PTFA side evidence | 0.5794 | 35 | Continued to 60 epochs; below E4 |
| E8 | s1 fixed-PTFA feature refinement | 0.5827 | 27 | Residual scorer disabled |
| E8-cED | s1 corrected exit-depth PTFA feature refinement | 0.6048 | 27 | Best candidate-dense method |
| E9 | reliability-gated cED s1 PTFA refinement | 0.5745 | 25 | Learned weighted view sum hurts |
| E9a | stable full-geom reliability gate | 0.5547 | 25 | LayerNorm + residual mix; worse |
| E9b | compact stable reliability gate | 0.4985 | 12 | Stopped early; clearly weak |
| E9c | consensus-residual evidence calibration | 0.5269 | 26 | Residual confidence still hurts |

## Candidate-Dense Results

Candidate-dense evaluation uses `candidate_topk_ratio=0.10`, `coarse_cell_sample`,
`samples_per_candidate_cell=8`, `outside_sample_num=131072`, `threshold=0.5`, and `seed=0`.
It evaluates checkpoints on the full 200-sample validation split. Unlike `val_query_dice`, this
metric is tied to a measurement-derived candidate domain and should be used for method ranking.

| Experiment | Checkpoint | candidate_dice | Precision | Recall | outside_fp_rate | outside_p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| E4 lambda=0.20 rerun bs=4 | `2026-05-13/19-37-07/epoch=25-val_dice=0.5827.ckpt` | 0.57925 | 0.67424 | 0.60617 | 0.000148 | 2.97e-05 |
| E7 continued to 60 | `2026-05-13/18-02-58/epoch=35-val_dice=0.5794.ckpt` | 0.57239 | 0.64325 | 0.61296 | 0.000185 | 2.53e-05 |
| E8 fixed s1 PTFA refine | `2026-05-13/23-57-39/epoch=27-val_dice=0.5827.ckpt` | 0.58109 | 0.67621 | 0.59734 | 0.000144 | 2.35e-05 |
| E8-cED s1 PTFA refine | `2026-05-14/09-08-20/epoch=27-val_dice=0.6048.ckpt` | 0.60957 | 0.69455 | 0.62369 | 0.000176 | 1.63e-05 |
| E9 reliability gate | `2026-05-14/11-43-08/epoch=25-val_dice=0.5745.ckpt` | 0.57449 | 0.67805 | 0.59206 | 0.000152 | 3.84e-05 |
| E9a stable full gate | `2026-05-14/15-06-44/epoch=25-val_dice=0.5547.ckpt` | 0.54855 | 0.63912 | 0.57145 | 0.000152 | 4.38e-05 |
| E9c consensus residual | `2026-05-14/17-55-23/epoch=26-val_dice=0.5269.ckpt` | 0.52956 | 0.60529 | 0.56337 | 0.000129 | 4.65e-05 |

## Checkpoint Locations

- E1b: `outputs/gisc_fmt/fit/2026-05-11/17-07-50/checkpoints/epoch=25-val_dice=0.5629.ckpt`
- E2: `outputs/gisc_fmt/fit/2026-05-11/20-38-45/checkpoints/epoch=26-val_dice=0.5768.ckpt`
- E4 lambda=0.20: `outputs/gisc_fmt/fit/2026-05-12/17-54-52/checkpoints/epoch=21-val_dice=0.6315.ckpt`
- E5 corrected: `outputs/gisc_fmt/fit/2026-05-12/23-52-31/checkpoints/epoch=23-val_dice=0.5904.ckpt`
- E4' lambda=0.30: `outputs/gisc_fmt/fit/2026-05-13/02-37-22/checkpoints/epoch=23-val_dice=0.6294.ckpt`
- E4' lambda=0.50: `outputs/gisc_fmt/fit/2026-05-13/04-13-26/checkpoints/epoch=23-val_dice=0.6051.ckpt`
- E8-cED: `outputs/gisc_fmt/fit/2026-05-14/09-08-20/checkpoints/epoch=27-val_dice=0.6048.ckpt`
- E9: `outputs/gisc_fmt/fit/2026-05-14/11-43-08/checkpoints/epoch=25-val_dice=0.5745.ckpt`
- E9a: `outputs/gisc_fmt/fit/2026-05-14/15-06-44/checkpoints/epoch=25-val_dice=0.5547.ckpt`
- E9c: `outputs/gisc_fmt/fit/2026-05-14/17-55-23/checkpoints/epoch=26-val_dice=0.5269.ckpt`

## Diagnostics and Conclusions

### Exit-Depth PTFA

E3 and E3' show that the exit-depth PTFA issue is not simply excessive smoothing. Reducing the
sigma range from `[0.8, 2.5]` to `[0.6, 1.2]` made validation Dice worse, not closer to E2.

Forward-only diagnostics showed:

- Exit-depth PTFA with `sigma_min=sigma_max=1.0` exactly matches fixed PTFA.
- Projection centers match between fixed and exit-depth paths.
- The issue is therefore not a center mismatch or alternate projection path.

Depth-source sanity showed a foreground correlation of `-0.3145` between the sampled exit-depth
proxy and label z, indicating that the current depth source behaves like an inverted or
exit-depth-like quantity. This should be treated as a finding, not patched into the current
baseline without a targeted task.

E8-cED uses the corrected depth convention:

```text
raw_depth_like = clamp(query_depth - sampled_surface_depth, 0, exit_depth_max)
depth_eff = exit_depth_max - raw_depth_like
sigma_px = sigma_min + (sigma_max - sigma_min) * depth_eff / exit_depth_max
```

With high-resolution s1 PTFA feature refinement, this corrected mapping improves candidate Dice
from `0.58109` (fixed s1 PTFA) to `0.60957`.

### Residual Scorer

The residual scorer is the strongest current path. It is a small zero-initialized MLP that adds
a query-level correction to the base logit:

```text
final_logit = base_logit + lambda_R * residual
```

The best tested value is `lambda_R=0.20` with `val_dice=0.6315`. Increasing to `0.30` is nearly
flat but slightly lower, while `0.50` clearly drops. The practical range is therefore around
`0.20-0.30`, with `0.20` retained as the current main setting.

### E5 PTFA + Scorer

An implementation issue was found and corrected during E5 analysis: the residual scorer must
consume bilinear s3 features even when PTFA is enabled for the fusion s3 path. The corrected E5
keeps scorer input fixed to bilinear s3 and applies PTFA only to the fusion branch.

Corrected E5 reaches `val_dice=0.5904`, which is better than E2 but far below scorer-only
`0.6315`. This indicates that replacing the fusion s3 feature with fixed PTFA interferes with
the residual scorer rather than adding complementary fluorescence information.

### s1 PTFA Feature Refinement

E7 adds s1 fixed-PTFA evidence to the residual scorer input. It does not improve candidate Dice
over the scorer baseline and tends to trade precision for recall. E8 instead disables the
residual scorer and applies a zero-initialized feature-refinement block:

```text
f_refined = f_base + delta([f_base, f_ptfa, f_ptfa - f_base, geom])
```

E8-cED keeps this feature-refinement structure and switches the s1 PTFA footprint from fixed
Gaussian to corrected exit-depth Gaussian. This is the strongest checkpoint-level result so far:
`candidate_dice=0.60957`, with both precision and recall higher than E8 fixed.

### Learned View Aggregation

E9 variants tested whether learned view reliability can improve E8-cED's valid-view uniform
mean aggregation:

- E9 replaces uniform mean with a geometry-only learned softmax weighted sum.
- E9a adds LayerNorm, high temperature, and residual mixing around the uniform distribution.
- E9b uses a compact geometry feature set; it was stopped early because the sampled-query curve
  was clearly weak.
- E9c anchors on uniform mean and learns only a small confidence-gated residual correction.

All learned view-gating variants underperform E8-cED on candidate Dice. E9c is especially
diagnostic: even without replacing the uniform consensus, the learned residual confidence becomes
extreme (`c_v` spans approximately `0` to `1`) and reduces both precision and recall. The current
evidence therefore supports keeping valid-view uniform mean aggregation for cED s1 PTFA.

## Current Recommendation

Use E8-cED as the current checkpoint-level main method:

```bash
/home/foods/pro/minr_fmt/.venv/bin/python train.py fit \
  exp=fmt_simgen_e8_ced_s1_ptfa_feature_refine \
  trainer.accelerator=gpu \
  trainer.max_epochs=30 \
  data.num_queries=16384 \
  data.sample_num=16384 \
  data.batch_size=4 \
  data.eval_batch_size=4 \
  data.train_max_samples=null \
  data.val_max_samples=null \
  model.geometry.use_fmt_simgen_projection=true
```

Use E4 `lambda_R=0.20` as the sampled-query residual-scorer reference, but prefer candidate Dice
for method ranking. Do not continue learned view-gating or learned view-residual calibration
without a stronger constraint; all tested E9 variants reduce candidate performance. If PTFA is
revisited, prioritize non-learned physical rules or a small sigma/window sweep around E8-cED.

## Candidate-Dense Evaluation

FMT-SimGen does not provide a fixed generation ROI suitable for a single ROI-dense metric, and
full-trunk dense evaluation is expensive. Use measurement-candidate dense/semi-dense evaluation
for checkpoint-level reporting:

```bash
/home/foods/pro/minr_fmt/.venv/bin/python scripts/eval_candidate_dense_fmt_simgen.py \
  exp=fmt_simgen_e4_residual_scorer_lambda020 \
  ckpt_path=/abs/path/to/checkpoint.ckpt \
  split=val \
  eval.candidate_topk_ratio=0.10 \
  eval.candidate_mode=coarse_cell_sample \
  eval.samples_per_candidate_cell=8 \
  eval.outside_sample_num=131072 \
  eval.chunk_size=65536
```

The candidate domain is derived only from `proposal/meas_backproj_heatmap.npy`. It does not use
GT ROI, GT foreground, tumor parameters, or body masks. Candidate metrics report performance
inside measurement-derived cells, while outside metrics uniformly sample trunk voxels outside
the candidate cells to estimate false positives.

For controlled FMT-SimGen datasets such as `fmt_simgen_v2_3k_20k`, candidate evaluation also
writes grouped reporting files:

- `metrics_per_sample.csv`: per-sample metrics plus reporting metadata such as `depth_tier`,
  `num_foci`, `shape_set`, and per-shape flags.
- `metrics_grouped.json` / `metrics_grouped.csv`: metric means/stds grouped by depth tier,
  focus count, shape combination, and shape presence.

These metadata fields are used only after candidate/outside points are fixed. They are not used
for candidate construction, sampling, model input, or loss computation.

## FMT-SimGen v2 3k 20k protocol

Dataset path:

```bash
/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k
```

The fixed split keeps the original 2400-sample train split, backs up the original 600-sample
validation split as `splits/val_full_600.txt`, and stratifies it by `(num_foci, depth_tier)` with
seed `20260522` into `splits/val.txt` (300) and `splits/test.txt` (300). Regenerate and audit it
with:

```bash
uv run python scripts/prepare_fmt_simgen_v2_split.py
```

The shared exp is `exp=fmt_simgen_v2_3k_20k_common`. It fixes the v2 data path, FMT-SimGen
geometry `[190, 200, 104]`, seven views `[-90, -60, -30, 0, 30, 60, 90]`, non-GT mixed query
sampling, and descatter supervision through `proj_noscatter.npz` with `no_proj.npz` fallback.
GISC E12 uses `exp=fmt_simgen_v2_3k_20k_gisc_e12`, which layers PCFS/canonical reliability on top
of the common exp. Baselines should use the common exp so they do not inherit E12
`feature_refinement` or PCFS settings.

Measurement proposals are required before training with the common exp:

```bash
uv run python scripts/precompute_measurement_proposal.py \
  --data_dir /home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k \
  --num_workers 1
```

Training commands:

```bash
uv run python train.py fit model=gisc_fmt exp=fmt_simgen_v2_3k_20k_gisc_e12 data.dataset_type=fmt_simgen
uv run python train.py fit model=uhr_deepfmt exp=fmt_simgen_v2_3k_20k_common data.dataset_type=fmt_simgen
uv run python train.py fit model=vox_dmrn exp=fmt_simgen_v2_3k_20k_common data.dataset_type=fmt_simgen
```

Use the same common exp for `point_cqr`, `fixed_footprint_cqr`, `depth_footprint_cqr`,
`unconstrained_adaptive_cqr`, `fem2vox_unet`, `two_stage_deepfmt`, `fmt_reconnet`, `pgdpnn`,
`map_pgan`, `d2_recst`, and `dspgn`.

Final full-volume evaluation uses a fixed threshold of `0.5` and writes `metrics.csv`,
`metrics_per_sample.csv`, `metrics_grouped.csv`, `metrics_grouped.json`, and optional
`predictions/*.npz`:

```bash
uv run python scripts/eval_full_volume_fmt_simgen.py \
  model=gisc_fmt exp=fmt_simgen_v2_3k_20k_gisc_e12 data.dataset_type=fmt_simgen \
  --ckpt_path /path/to/checkpoint.ckpt \
  --split test \
  --threshold 0.5 \
  --save_predictions
```

Future result tables should separate:

- `val_query_dice`: fast sampled-query training monitor.
- `candidate_dice`: checkpoint-level candidate-domain metric.
- `outside_fp_rate`: false-positive rate outside the measurement candidate domain.
- Limited full-trunk sanity metrics, when affordable.
