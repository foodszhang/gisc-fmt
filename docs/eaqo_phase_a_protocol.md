# EAQO Phase A Protocol

EAQO is a training-time query-supervision weighting strategy for SSQ-FMT Phase A.
It does not define a decoder, routing path, candidate-composition rule, or inference
post-processing step. The reconstructed density is the density returned by the SSQ-FMT
model before EAQO is applied.

## Definition

For a query point `x`, EAQO computes

```text
A(x) = alpha A_pred(x) + beta A_view(x) + gamma A_geo(x) + delta A_comp(x)
w(x) = 1 + lambda A(x)
```

The supervised density loss is then evaluated with `w(x)` as a query loss weight.
The default model configuration keeps EAQO disabled.

## Phase A Density Path

EAQO must not change the Phase A final density path. In Phase A view-complementary
training, the expected density path remains:

```text
surface-response footprint aggregation
-> shared fusion
-> UnifiedDensityDecoder(ablation="shared_only", context_scale=0.0)
-> scalar density
```

The Phase A forward path exposes diagnostic fields:

```text
final_density_path: phase_a_shared_unified_decoder
decoder_ablation: shared_only
candidate_context_scale: 0.0
```

EAQO reads the returned density and auxiliary tensors, then writes
`query_loss_weight`, `eaqo_score`, and `eaqo_loss_weight` for training loss
evaluation. It does not create a replacement density.

## Ambiguity Terms

`A_pred` is prediction ambiguity:

```text
A_pred = 4 p (1 - p)
```

where `p` is the probability-domain density returned by the model.

`A_view` is the FMT-specific view conflict term. It uses query-level per-view
surface evidence:

```text
e = per_view_evidence          # [B, V, N]
valid = query_view_valid       # [B, V, N]
A_view = masked_var(abs(e)) / (masked_mean(abs(e)) + eps)
```

At least two valid views are required. If evidence is unavailable and
`require_view_evidence=true`, training raises an error.

`A_geo` uses existing center-distance targets when available:

```text
A_geo = 1 - distance_target
```

It is applied only on foreground/source-support queries.

`A_comp` uses existing `query_component_ids` and component valid masks. Queries
from smaller foreground components receive higher weights. This is not a topology
loss and should not be described as topology-preserving supervision.

## Commands

Baseline and EAQO runs use the shared entrypoint:

```bash
uv run python train.py fit exp=fmt_simgen_v2_ssq_final
uv run python train.py fit exp=fmt_simgen_v2_ssq_eaqo_off
uv run python train.py fit exp=fmt_simgen_v2_ssq_eaqo_view_only
uv run python train.py fit exp=fmt_simgen_v2_ssq_eaqo_pred_only
uv run python train.py fit exp=fmt_simgen_v2_ssq_eaqo_full
```

The compatibility entrypoint `train_eaqo.py` is retained, but `train.py` now selects
the EAQO Lightning module when `model.ssq_fmt.eaqo.enabled=true`.

One-batch diagnostics:

```bash
uv run python scripts/diagnostics/smoke_eaqo.py
```

## Diagnostic Logs

EAQO logs the following train-epoch metrics:

```text
train_eaqo/score_mean
train_eaqo/score_std
train_eaqo/weight_mean
train_eaqo/weight_max
train_eaqo/pred_ambiguity_mean
train_eaqo/view_ambiguity_mean
train_eaqo/geo_ambiguity_mean
train_eaqo/comp_ambiguity_mean
train_eaqo/high_ambiguity_query_foreground_ratio
train_eaqo/high_ambiguity_query_average_loss
train_eaqo/low_ambiguity_query_average_loss
```

High and low ambiguity subsets use the top and bottom 25% valid queries in each
batch. The average loss diagnostic uses an absolute-error proxy and is not a
separate optimization term.

## Contribution Criterion

EAQO can be treated as an FMT-specific training contribution only if the
view-evidence term is informative in controlled comparisons. The primary checks are:

```text
EAQO-ViewOnly > EAQO-PredOnly
EAQO-Full > EAQO-Off
```

If `EAQO-ViewOnly <= EAQO-PredOnly`, EAQO should be reported as a hard-query
emphasis strategy or as a negative/secondary result rather than as an FMT-specific
main contribution.

## Limitations

EAQO depends on the auxiliary tensors returned by the current SSQ-FMT path. The
view-only and full settings require Phase A per-view evidence. Geometry and component
terms are best-effort terms based on existing dataset fields and are zero when those
fields are unavailable. Validation and test are not EAQO-weighted unless explicitly
implemented in a future configuration.
