# A3-v2 fast feasibility report

## Scope and classification

- **Implementation issue:** standalone phase-B evaluation used step zero and disabled candidate context; evaluation now uses scale 1 while training warm-up is unchanged.
- **Method-intrinsic issue:** A3-v2 removes density-query-dependent hypothesis construction, makes confidence threshold analysis-only, and bounds centered hypothesis-conditioned routing around the A2-U uniform fallback.
- **Experiment-dependent conclusion:** conclusions below are limited to the 1000/64 fast protocol and are not a formal publication ablation.
- **Future discussion/limitation:** component recall, weak-component recall, merge rate, and centroid error remain unavailable unless reliable component-level prediction extraction is added.

## Modified files

See `git diff --stat` appended below. Core changes are in `ssq_fmt.py`, candidate evidence/constructor, unified decoder, bounded routing, configs, tests, and experiment/report scripts.
New files: `minr_fmt/network/a3v2_routing.py`, three fast experiment configs, `tests/test_a3v2_routing.py`, `scripts/experiments/run_a3v2_fast.sh`, and the two analysis/report scripts. Corrected runs use a `_corrected` suffix because failed diagnostic runs were preserved rather than deleted or overwritten.

## Compatibility and tests

Legacy modes default to fixed-grid off, routing off, and continuous applicability off. Old checkpoints receive only the zero routing scalar when the new mode requests it. Run: `uv run pytest`.

The A2-U standalone 64-sample/4096-query validation after the context fix produced Dice 0.625715, shared Dice 0.619362, and final-minus-shared +0.006353, confirming that old checkpoints now evaluate with the candidate path enabled.

## Fast comparison

| Model | Best val Dice | Last val Dice | Best final-shared |
|---|---:|---:|---:|
| A2-U existing | 0.647369 | 0.647144 | 0.010202 |
| A3-old existing | 0.647017 | 0.646198 | 0.009874 |
| A3-geometry-only | 0.617101 | 0.614492 | -0.001133 |
| A3-v2 | 0.622559 | 0.621878 | 0.000020 |

Completed rows: 4/4. Query-count and threshold invariance artifacts are stored beside the A3-v2 run when generated.

Geometry-only deltas: A3-G − A2-U = -0.030268; A3-G − A3-old = -0.029916. Geometry-only did not outperform A3-old, so this fast run does not support geometry separability as sufficient evidence.

## A3-v2 stability checks

- Query-count audit on the same 32 sample IDs: max center drift `0.0`, existence drift `0.0`, covariance drift `0.0`, support drift `0.0`, count drift `0`.
- Direct aligned 4096-query analysis: count 3.7812, coverage@6/8/10 0.9583/0.9583/0.9844, duplicate 0.4839, unmatched 0.0078, matched center error 1.7768 mm.
- Analysis-threshold rows: `[{"threshold": 0.2, "density_max_abs_diff": 0.0, "val_dice_32": 0.6226995936594903, "candidate_count_mean": 4.75}, {"threshold": 0.3, "density_max_abs_diff": 0.0, "val_dice_32": 0.6226995936594903, "candidate_count_mean": 4.3125}, {"threshold": 0.4, "density_max_abs_diff": 0.0, "val_dice_32": 0.6226995936594903, "candidate_count_mean": 3.78125}, {"threshold": 0.5, "density_max_abs_diff": 0.0, "val_dice_32": 0.6226995936594903, "candidate_count_mean": 2.8125}]`. Density max difference and 32-sample Dice change are exactly zero.
- TensorBoard candidate-count diagnostics disagree with the direct sample-aligned audit and are therefore treated as an implementation issue in aggregation/logging, not as valid method evidence.

## Decision

A3-v2 best Dice is 0.622559, below A2-U by -0.024810. Its best final-minus-shared is +0.000020, routing gain remains near zero, and the mean hypothesis gate is extremely small. It does not meet the routing-only trigger and should not enter a formal large experiment. Fixed-grid query invariance and analysis-threshold invariance are supported; reconstruction benefit is not.

Component recall, weak-component recall, merged-component rate, and component centroid error are marked **not implemented** because this point-query validation path does not produce a reliable connected-component prediction artifact.

## Reproduction

```bash
GPU_ID=0 bash scripts/experiments/run_a3v2_fast.sh 2>&1 | tee outputs/view_complementary/a3v2_fast.log
uv run pytest -q
uv run python scripts/analysis/check_a3v2_invariance.py   --run-dir outputs/view_complementary/a3v2_fast_bounded_routing_seed42_corrected   --checkpoint outputs/view_complementary/a3v2_fast_bounded_routing_seed42_corrected/checkpoints/epoch=00-val_dice=0.6226.ckpt
```

## Final checkpoints

- A2-U existing: `/home/foods/pro/gisc_fmt_repo/outputs/view_complementary/long_1k_continue_a2u_seed42/checkpoints/epoch=04-val_dice=0.6474.ckpt`
- A3-old existing: `/home/foods/pro/gisc_fmt_repo/outputs/view_complementary/long_1k_continue_a3_seed42/checkpoints/epoch=04-val_dice=0.6470.ckpt`
- A3-geometry-only: `/home/foods/pro/gisc_fmt_repo/outputs/view_complementary/a3v2_fast_geometry_only_seed42_corrected/checkpoints/epoch=01-val_dice=0.6171.ckpt`
- A3-v2: `/home/foods/pro/gisc_fmt_repo/outputs/view_complementary/a3v2_fast_bounded_routing_seed42_corrected/checkpoints/epoch=00-val_dice=0.6226.ckpt`

## Git diff --stat

```text
configs/callbacks/default.yaml                     |   1 +
 minr_fmt/model_factory.py                          |   3 +
 minr_fmt/models/ssq_fmt.py                         | 305 +++++++++++++--------
 minr_fmt/module.py                                 |  62 ++++-
 minr_fmt/network/complementary_aggregation.py      |   8 +-
 minr_fmt/network/diverse_candidate_constructor.py  |  36 ++-
 minr_fmt/network/unified_density_decoder.py        |   4 +
 minr_fmt/network/view_candidate_evidence.py        |  24 +-
 minr_fmt/network/view_separability.py              |   2 +
 outputs/paper_figures/ssq/audit_report.json        |   5 +-
 outputs/paper_figures/ssq/evidence_audit.md        |   5 +-
 outputs/paper_figures/ssq/figure_manifest.json     |  13 +-
 .../fig_delta_dice_supplementary.pdf               | Bin 10471 -> 10471 bytes
 .../fig_delta_dice_supplementary.svg               |  54 ++--
 .../fig_module_rescue_transitions.pdf              | Bin 13845 -> 13845 bytes
 .../fig_module_rescue_transitions.svg              |  20 +-
 .../fig_source_separation_profiles.pdf             | Bin 74146 -> 74146 bytes
 .../fig_source_separation_profiles.svg             | 258 ++++++++---------
 .../fig_three_source_candidate_contact_sheet.pdf   | Bin 71151 -> 71151 bytes
 .../fig_three_source_candidate_contact_sheet.svg   |  82 +++---
 outputs/paper_figures/ssq/redesign_audit.md        |   5 +-
 tests/test_diverse_candidate_constructor.py        |   4 +
 tests/test_view_complementary_model.py             |  25 ++
 23 files changed, 561 insertions(+), 355 deletions(-)
```

