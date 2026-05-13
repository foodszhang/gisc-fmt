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

## Main Results

| Experiment | Description | Best val_dice | Epoch | Notes |
| --- | --- | ---: | ---: | --- |
| E1a | trunk-uniform only | 0.4144 | 25 | Non-GT, no proposal branch |
| E1b | trunk 50% + proposal 50% | 0.5629 | 25 | Baseline before PTFA/scorer |
| E2 | s3 fixed Gaussian PTFA, sigma=1.0 | 0.5768 | 26 | Small gain over E1b |
| E3 | s3 exit-depth PTFA, sigma=[0.8, 2.5] | 0.5448 | 24 | Worse than fixed PTFA |
| E3' | calibrated exit-depth PTFA, sigma=[0.6, 1.2] | 0.5395 | 26 | Worse again; not over-smoothing |
| E4 lambda=0.05 | residual scorer only | 0.6109 | 26 | Clear gain over E1b/E2 |
| E4 lambda=0.20 | residual scorer only | 0.6315 | 21 | Current best |
| E4' lambda=0.30 | residual scorer only | 0.6294 | 23 | Near lambda=0.20, slightly lower |
| E4' lambda=0.50 | residual scorer only | 0.6051 | 23 | Too strong; optimization/performance drops |
| E5 | residual scorer lambda=0.20 + fixed PTFA | 0.5904 | 23 | Corrected run; scorer and PTFA interfere |

## Checkpoint Locations

- E1b: `outputs/gisc_fmt/fit/2026-05-11/17-07-50/checkpoints/epoch=25-val_dice=0.5629.ckpt`
- E2: `outputs/gisc_fmt/fit/2026-05-11/20-38-45/checkpoints/epoch=26-val_dice=0.5768.ckpt`
- E4 lambda=0.20: `outputs/gisc_fmt/fit/2026-05-12/17-54-52/checkpoints/epoch=21-val_dice=0.6315.ckpt`
- E5 corrected: `outputs/gisc_fmt/fit/2026-05-12/23-52-31/checkpoints/epoch=23-val_dice=0.5904.ckpt`
- E4' lambda=0.30: `outputs/gisc_fmt/fit/2026-05-13/02-37-22/checkpoints/epoch=23-val_dice=0.6294.ckpt`
- E4' lambda=0.50: `outputs/gisc_fmt/fit/2026-05-13/04-13-26/checkpoints/epoch=23-val_dice=0.6051.ckpt`

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

## Current Recommendation

Use E4 `lambda_R=0.20` as the main baseline:

```bash
/home/foods/pro/minr_fmt/.venv/bin/python train.py fit \
  exp=fmt_simgen_e4_residual_scorer_lambda020 \
  trainer.accelerator=gpu \
  trainer.max_epochs=30 \
  data.num_queries=16384 \
  data.sample_num=16384 \
  data.train_max_samples=null \
  data.val_max_samples=null \
  model.geometry.use_fmt_simgen_projection=true
```

Do not keep direct fixed-PTFA replacement as the default scorer path. If PTFA is revisited, use
a zero-initialized additive or gated design, for example:

```text
s3 = bilinear_s3 + alpha * (ptfa_s3 - bilinear_s3)
```

or expose PTFA as an additional residual-scorer input instead of replacing the base fusion
feature.

