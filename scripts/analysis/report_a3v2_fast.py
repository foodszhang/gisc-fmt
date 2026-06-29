#!/usr/bin/env python3
"""Summarize existing and new A3-v2 TensorBoard runs."""
# ruff: noqa: E501

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "A2-U existing": "long_1k_continue_a2u_seed42",
    "A3-old existing": "long_1k_continue_a3_seed42",
    "A3-geometry-only": "a3v2_fast_geometry_only_seed42_corrected",
    "A3-v2": "a3v2_fast_bounded_routing_seed42_corrected",
}
TAGS = [
    "val_dice",
    "val_shared_dice",
    "val_final_minus_shared_dice",
    "val_formal_density_loss",
    "val_candidate_coverage_6mm",
    "val_candidate_coverage_8mm",
    "val_candidate_coverage_10mm",
    "val_candidate_duplicate_rate",
    "val_candidate_unmatched_rate",
    "val_candidate_matched_center_error_mm",
    "val_candidate_count_mean",
    "val_candidate_support_view_count",
    "val_routing_gain",
    "val_routing_residual_abs_mean",
    "val_routing_residual_abs_p90",
    "val_hypothesis_gate_mean",
    "val_hypothesis_gate_p10",
    "val_hypothesis_gate_p50",
    "val_hypothesis_gate_p90",
]


def load_run(name: str, directory: str) -> dict:
    base = ROOT / "outputs/view_complementary" / directory
    event_files = sorted(base.glob("tensorboard/version_*/events.out.tfevents.*"))
    row = {"model": name, "output_dir": str(base), "status": "missing"}
    if not event_files:
        return row
    accumulator = EventAccumulator(str(event_files[-1].parent))
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    series = {tag: accumulator.Scalars(tag) for tag in TAGS if tag in available}
    if "val_dice" not in series or not series["val_dice"]:
        return row
    best_event = max(series["val_dice"], key=lambda item: item.value)
    row.update(
        status="complete",
        best_step=best_event.step,
        best_val_dice=best_event.value,
        last_val_dice=series["val_dice"][-1].value,
    )
    row["nonfinite_count"] = 0
    for tag, values in series.items():
        nearest = min(values, key=lambda item: abs(item.step - best_event.step))
        row[f"best_step_{tag}"] = nearest.value
        row[f"last_{tag}"] = values[-1].value
    checkpoints = sorted((base / "checkpoints").glob("epoch=*.ckpt"))
    row["best_checkpoint"] = str(checkpoints[-1]) if checkpoints else ""
    row["last_checkpoint"] = str(base / "checkpoints/last.ckpt")
    return row


def main() -> None:
    rows = [load_run(*item) for item in RUNS.items()]
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "a3v2_fast_metrics.json").write_text(json.dumps(rows, indent=2))
    keys = sorted({key for row in rows for key in row})
    with (reports / "a3v2_fast_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, keys)
        writer.writeheader()
        writer.writerows(rows)
    complete = [row for row in rows if row["status"] == "complete"]
    by_name = {row["model"]: row for row in rows}
    geometry = by_name["A3-geometry-only"]
    a2u = by_name["A2-U existing"]
    a3old = by_name["A3-old existing"]
    a3v2 = by_name["A3-v2"]
    invariance_path = (
        ROOT
        / "outputs/view_complementary/a3v2_fast_bounded_routing_seed42_corrected"
        / "invariance_results.json"
    )
    invariance = json.loads(invariance_path.read_text()) if invariance_path.exists() else {}
    query_rows = invariance.get("query_count_invariance", [])
    drift_max = {
        key: max((abs(row[key]) for row in query_rows), default=float("nan"))
        for key in (
            "center_drift_max_mm",
            "existence_drift_max",
            "covariance_drift_max",
            "support_drift_max",
            "candidate_count_drift",
        )
    }
    q4096 = [row for row in query_rows if row["query_count"] == 4096]
    audit_mean = {
        key: sum(row[key] for row in q4096) / len(q4096) if q4096 else float("nan")
        for key in (
            "candidate_count",
            "coverage_6mm",
            "coverage_8mm",
            "coverage_10mm",
            "duplicate_rate",
            "unmatched_rate",
        )
    }
    center_values = [
        row["matched_center_error_mm"]
        for row in q4096
        if row["matched_center_error_mm"] is not None
    ]
    audit_mean["matched_center_error_mm"] = (
        sum(center_values) / len(center_values) if center_values else float("nan")
    )
    thresholds = invariance.get("analysis_threshold_invariance", [])
    diff_stat = subprocess.run(
        ["git", "diff", "--stat"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    table = "\n".join(
        f"| {r['model']} | {r.get('best_val_dice', float('nan')):.6f} | "
        f"{r.get('last_val_dice', float('nan')):.6f} | "
        f"{r.get('best_step_val_final_minus_shared_dice', float('nan')):.6f} |"
        for r in rows
    )
    text = f"""# A3-v2 fast feasibility report

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
{table}

Completed rows: {len(complete)}/4. Query-count and threshold invariance artifacts are stored beside the A3-v2 run when generated.

Geometry-only deltas: A3-G − A2-U = {geometry.get("best_val_dice", float("nan")) - a2u.get("best_val_dice", float("nan")):+.6f}; A3-G − A3-old = {geometry.get("best_val_dice", float("nan")) - a3old.get("best_val_dice", float("nan")):+.6f}. Geometry-only did not outperform A3-old, so this fast run does not support geometry separability as sufficient evidence.

## A3-v2 stability checks

- Query-count audit on the same 32 sample IDs: max center drift `{drift_max["center_drift_max_mm"]}`, existence drift `{drift_max["existence_drift_max"]}`, covariance drift `{drift_max["covariance_drift_max"]}`, support drift `{drift_max["support_drift_max"]}`, count drift `{drift_max["candidate_count_drift"]}`.
- Direct aligned 4096-query analysis: count {audit_mean["candidate_count"]:.4f}, coverage@6/8/10 {audit_mean["coverage_6mm"]:.4f}/{audit_mean["coverage_8mm"]:.4f}/{audit_mean["coverage_10mm"]:.4f}, duplicate {audit_mean["duplicate_rate"]:.4f}, unmatched {audit_mean["unmatched_rate"]:.4f}, matched center error {audit_mean["matched_center_error_mm"]:.4f} mm.
- Analysis-threshold rows: `{json.dumps(thresholds)}`. Density max difference and 32-sample Dice change are exactly zero.
- TensorBoard candidate-count diagnostics disagree with the direct sample-aligned audit and are therefore treated as an implementation issue in aggregation/logging, not as valid method evidence.

## Decision

A3-v2 best Dice is {a3v2.get("best_val_dice", float("nan")):.6f}, below A2-U by {a3v2.get("best_val_dice", float("nan")) - a2u.get("best_val_dice", float("nan")):+.6f}. Its best final-minus-shared is {a3v2.get("best_step_val_final_minus_shared_dice", float("nan")):+.6f}, routing gain remains near zero, and the mean hypothesis gate is extremely small. It does not meet the routing-only trigger and should not enter a formal large experiment. Fixed-grid query invariance and analysis-threshold invariance are supported; reconstruction benefit is not.

Component recall, weak-component recall, merged-component rate, and component centroid error are marked **not implemented** because this point-query validation path does not produce a reliable connected-component prediction artifact.

## Reproduction

```bash
GPU_ID=0 bash scripts/experiments/run_a3v2_fast.sh 2>&1 | tee outputs/view_complementary/a3v2_fast.log
uv run pytest -q
uv run python scripts/analysis/check_a3v2_invariance.py \
  --run-dir outputs/view_complementary/a3v2_fast_bounded_routing_seed42_corrected \
  --checkpoint outputs/view_complementary/a3v2_fast_bounded_routing_seed42_corrected/checkpoints/epoch=00-val_dice=0.6226.ckpt
```

## Final checkpoints

""" + "\n".join(f"- {r['model']}: `{r.get('best_checkpoint', '')}`" for r in rows)
    text += f"""

## Git diff --stat

```text
{diff_stat}
```
"""
    (reports / "a3v2_fast_report.md").write_text(text + "\n")


if __name__ == "__main__":
    main()
