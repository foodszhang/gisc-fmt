# GISC-FMT Ablation Protocol

This protocol keeps all ablation variants on the shared Hydra/Lightning entrypoint and
the common FMT-SimGen v2 output root:

`outputs/fmt_simgen_v2_3k_20k/ablation_runs/<variant_name>`

## Current Code Path

- Model factory name: `gisc_fmt`
- Model class: `minr_fmt.models.minr_fmt.GISCFMT`
- Query network: `PointDensityNet`
- Existing footprint implementation: `model.ptfa`
- E15 center-distance config chain: `configs/exp/fmt_simgen_v2_e15_center_distance.yaml`,
  inheriting `fmt_simgen_v2_e13_msq`, `fmt_simgen_v2_e12_mpb_auxnorm`,
  `fmt_simgen_v2_e12_mpb`, and `fmt_simgen_v2_3k_20k_gisc_e12`.

## Ablation Field Mapping

The ablation-facing fields live under `model.gisc` and are translated to existing PTFA
settings by `ConfigExtractor.extract_ptfa_config`.

| `model.gisc.footprint_mode` | Effective PTFA path |
| --- | --- |
| `point` | `model.ptfa.enabled=false`, point-wise bilinear sampling |
| `fixed` | `fixed_gaussian` on `s3`, `sigma_px=model.gisc.fixed_sigma` |
| `depth` | `exit_depth_gaussian` on `s3`, not inverted |
| `center_distance` | corrected exit-depth schedule on `s1`, no PCFS correction |
| `adaptive_unconstrained` | PCFS corrected exit-depth on `s1`, unbounded delta and no sigma upper bound |
| `adaptive_constrained` | existing bounded PCFS corrected exit-depth on `s1` |

Sparse-view variants use `model.gisc.view_subset`. The dataset still loads all views; the
model validates that selected angles exist in `data.view_angles`, then masks inactive
views in projection validity and feature tensors.

## Metrics

`train.py test` is still run for Lightning compatibility. Full-volume TMI metrics are
computed with `scripts/eval_full_volume_fmt_simgen.py`, which writes per-sample
`Dice`, `IoU`, `CLE`, `PLE`, `ASSD`, `HD95`, and `Volume Error` to the same variant
directory.
