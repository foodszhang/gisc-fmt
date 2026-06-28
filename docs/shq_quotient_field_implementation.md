# SHQ-FMT Source-Hypothesis Quotient Field

## Status

SHQ-FMT is the formal method name. The internal `SSQFMT` class name remains for checkpoint and
configuration compatibility. This implementation has unit and smoke coverage but has not yet been
validated by a same-protocol training run; all performance claims remain **to be verified**.

## Information Flow

The shared encoder and query-dependent Gaussian footprint sampler produce local samples
`f: [B,V,N,K,C]` and complete footprint weights `A: [B,V,N,K]`.

The shared evidence does not enter proposal competition:

```text
h_s^(v)(x) = Encode(sum_k A_k^(v)(x) f_k^(v)(x))
```

Candidate routing is normalized only over valid source hypotheses. For proposal
`P_m = (mu_m, b_m, Sigma_m)`:

```text
h_m^(v)(x) = Encode(sum_k A_k^(v)(x) zeta_km^(v)(x) f_k^(v)(x))
```

The router's proposal-independent slot is a complete-footprint slot in `proposal_only` mode. It is
not the old complement `1 - max_m K_m` and is not part of the candidate softmax.

## Quotient Aggregation

`SourceHypothesisQuotientAggregator` receives per-view evidence `[B,N,V,M,C]`, physical geometry
`[B,N,V,4]`, routed support `[B,N,V,M]`, and validity masks. Its reliability baseline is:

```text
log r_phy = log(Lambda + eps)
          + a_beta beta - a_xi xi - a_sigma sigma_norm - a_center d_center
log r = log r_phy + delta_r_max tanh(R_theta(h, geometry))
w_m^(v) = masked_softmax_v(log r / temperature)
```

The final layer of `R_theta` is zero-initialized. The query-hypothesis equivalence-class
representative and channel-normalized dispersion are:

```text
q_m = sum_v w_m^(v) h_m^(v)
u_m = sum_v w_m^(v) mean_c[(h_m^(v) - q_m)^2]
```

The same aggregator produces `(q_s, u_s)` from complete shared evidence. Invalid views have exactly
zero weight; all-invalid inputs produce finite zero quotient/support values.

## Continuous Density Reconstruction

`SharedDensityLogitDecoder` directly returns:

```text
l_s(x) = D_s(q_s(x), gamma_3D(x))
rho_s(x) = sigmoid(l_s(x))
```

There is no sigmoid-clamp-logit round trip. For each candidate:

```text
t_m = gamma_rel(Sigma_m^(-1/2) (x - mu_m))
Delta q_m = q_m - q_s
Delta l_m = delta_l_max tanh(D_r(q_s, Delta q_m, t_m, u_m, b_m, Lambda_m, eig(Sigma_m)))
```

The residual head's final layer is zero-initialized. Candidate applicability and composition are:

```text
p_m = exp(-0.5 (x-mu_m)^T Sigma_m^(-1) (x-mu_m))
c_m = valid_m b_m p_m Lambda_m exp(-u_m / tau_u)
alpha_m = c_m / (sum_n c_n + eps)
g_P = 1 - exp(-sum_m c_m)
rho_hat = sigmoid(l_s + g_P sum_m alpha_m Delta l_m)
```

This guarantees shared fallback for no candidates, all-invalid candidates, zero support, or a
zero residual. Candidate permutation does not change the output. No density envelope is applied.

## View-Subset Consistency

Training can deterministically split queries with at least two valid views into non-empty subsets
`A` and `B`. Validation and inference always use all valid views.

```text
L_quot = mean eta_m [
    ||q_m^A - stopgrad(q_m^B)||^2 +
    ||q_m^B - stopgrad(q_m^A)||^2
]
```

Eligibility requires both quotient subsets and sufficient support. The split seed and support
threshold are explicit configuration values.

## Loss

```text
L = L_final + lambda_shared L_density(rho_s, rho_gt)
              + lambda_quot L_quot + lambda_res L_res
L_res = mean[g_P sum_m alpha_m |Delta l_m|]
```

The first diagnostic settings are `lambda_shared=1.0`, `lambda_res=1e-3`,
`delta_l_max=3.0`, `delta_r_max=0.25`, and `tau_u=1.0`. The consistency experiment alone enables
`lambda_quot=0.05`. No GT assignment, matching, center, distance, or candidate-density target is
used by the SHQ configs.

## Code Map And Shapes

- Shared encoder and sampler: `minr_fmt/models/ssq_fmt.py`, `minr_fmt/network/ssq_sampler.py`
- Proposal-only routing: `minr_fmt/network/ssq_routing.py`
- Per-view evidence: `CandidateViewEncoder` in `minr_fmt/network/ssq_fusion.py`
- Quotient equations: `SourceHypothesisQuotientAggregator` in the same file
- Shared and residual decoders: `minr_fmt/network/ssq_decoder.py`
- Applicability and final equation: `compose_quotient_residual_density` in
  `minr_fmt/models/ssq_fmt.py`
- Loss composition: `MorphologyAwareDensityLoss` in `minr_fmt/loss.py`

Principal shapes are `q_s: [B,N,C]`, `q_m: [B,N,M,C]`, `u_m: [B,N,M]`,
`w_m: [B,N,V,M]`, `Lambda_m: [B,N,M]`, `Delta l_m: [B,N,M,1]`, and
`rho_hat: [B,N,1]`.

## Legacy Difference

`legacy_branch_mixture` retains the historical `sum_m pi_m d_m` implementation for checkpoint and
ablation compatibility. `quotient_residual` bypasses `CandidateSpecificViewFusion`,
`CandidateAssignmentHead`, occupancy posterior, compensation/candidate absolute-density branches,
`CandidateFieldCalibrator`, and density-envelope multiplication.

## GT Leakage Boundary

Formal SHQ configs explicitly load deterministic measurement-derived `candidate_anchors.npz`
(`ssq_candidate_anchors_v2`). Candidate tensors are detached before the SHQ router. Inference uses
only surface fluorescence, geometry/calibration, and deterministic proposal extraction. The formal
configs do not use `candidate_anchors_refiner_full_v1.npz`, `gt_voxels`, GT centers, GT boxes,
tumor metadata, center-distance targets, or component matching.

## Diagnostics

The SHQ path reports shared/final density and logit, residual correction, proposal gate, `alpha`,
applicability, shared/candidate quotient norms, quotient dispersion and view weights, aggregated
support, effective residual magnitude, final-minus-shared magnitude, valid candidate count, and
no-candidate fallback ratio. Compensation ratios, `pi0`, and branch-oracle metrics are legacy-only.

## First Training Command

```bash
uv run python train.py task=fit model=ssq_fmt exp=fmt_simgen_v2_shq_quotient_residual \
  data.dataset_type=fmt_simgen
```

The consistency config is a subsequent controlled experiment, not part of the first run. Whether
SHQ-FMT improves Dice or source separation is **to be verified by same-protocol evaluation**.
