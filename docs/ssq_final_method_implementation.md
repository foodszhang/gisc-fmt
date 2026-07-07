# SSQ-FMT Final Method Implementation

> Legacy implementation note: the formal method has moved to SHQ-FMT quotient residual
> composition. See `docs/shq_quotient_field_implementation.md`. This document describes the
> retained `legacy_branch_mixture` path and historical results only.

Status as of 2026-06-25: implemented and smoke-tested, but the short training gate has
not reached the quality threshold for formal runs.

## Critical Fixes

- `candidate_scalar_composition` is now the default density mode. Final density is
  `sum_m pi_m d_m` from branch scalar composition.
- `e15_backbone_baseline` is the only mode that may use the old E15/GISC query-density
  backbone as final density.
- Query-dependent footprints now predict `sigma_f` and use it to move `grid_sample`
  coordinates.
- Detector priors `K_all` are computed per local sample as `[B,V,N,K,M+1]`.
- Assignment mass, evidence, and reliability keep the view axis: `[B,V,N,M+1]`.
- Candidate density decoding uses Fourier-encoded relative query coordinates, not
  candidate score.
- SDF is disabled for this pass with `lambda_sdf=0`; the old mixed-z SDF head is not
  used as a paper implementation.

## Method To Code

- Normalization: `SurfaceMeasurementNormalizer`
- Detector encoder: `SharedSurfaceEncoder`
- Geometry mapping: `GeometryQueryMapper`
- Query footprint sampling: `QueryDependentSurfaceSampler`
- Candidate anchors: `MeasurementDerivedCandidateBuilder`
- Pre-aggregation routing and evidence: `CandidateSurfaceRouter`
- Candidate per-view representation: `CandidateViewEncoder`
- Candidate-specific cross-view fusion: `CandidateSpecificViewFusion`
- Query branch assignment: `CandidateAssignmentHead`
- Density decoders: `CompensationDensityDecoder`, `CandidateDensityDecoder`

## Tensor Shapes

- `A`: `[B,V,N,K]`
- `K_all`: `[B,V,N,K,M+1]`
- `zeta`: `[B,V,N,K,M+1]`
- `a`, `nu`, `r`: `[B,V,N,M+1]`
- `view_weights`: `[B,N,V,M+1]`
- `Lambda`: `[B,N,M+1]`
- `pi`: `[B,N,M+1]`
- `branch_density`: `[B,N,M+1,1]`
- `branch_contributions`: `[B,N,M+1,1]`

## Verification

- `uv run pytest tests/test_ssq_*.py -q`: 17 passed, 1 Lightning logging warning.
- `uv run ruff check minr_fmt/models/ssq_fmt.py minr_fmt/module.py minr_fmt/dataset/fmt_simgen_dataset.py tests scripts/run_ssq_final_ablation_pipeline.py scripts/export_ssq_mechanism_case.py`: passed.
- GPU smoke with 32768 queries passed with `model=ssq_fmt exp=fmt_simgen_v2_ssq_final`.

## Gate Results

- `outputs/ssq_fmt_final/full`: best val Dice 0.0968. Failed gate; compensation dominated.
- `outputs/ssq_fmt_final/full_gate_scale4`: best val Dice 0.0933. Candidate use improved but false positives remained high.
- `outputs/ssq_fmt_final/full_gate_scale4_sparse02`: best val Dice 0.0921. Failed gate.
- `outputs/ssq_fmt_final/full_gate_fourier4_bs1`: best val Dice 0.1037. Failed gate, but candidate branch use was active and no NaN/Inf was observed.

No formal full-data run or ablation result table should be started from these failed gates.

## Limitations

- `detector_side_path_proxy` uses the available depth-map proxy, not exact mesh
  entrance/exit intersections.
- Candidate scale is clipped to a broader lower bound because current proposal peaks can
  be several millimeters from sampled positives.
- SDF is disabled in the current implementation.
