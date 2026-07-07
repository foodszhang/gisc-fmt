#!/usr/bin/env python3
"""Summarize isolated-routing and fixed-grid stabilization runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def scalar_summary(run_dir: Path) -> dict:
    versions = sorted((run_dir / "tensorboard").glob("version_*"))
    if not versions:
        return {"status": "missing"}
    accumulator = EventAccumulator(str(versions[-1]))
    accumulator.Reload()
    tags = set(accumulator.Tags()["scalars"])

    def values(tag: str) -> list[float]:
        return [event.value for event in accumulator.Scalars(tag)] if tag in tags else []

    dice = values("val_dice")
    result = {
        "status": "complete" if dice else "no_validation_metrics",
        "best_val_dice": max(dice) if dice else None,
        "last_val_dice": dice[-1] if dice else None,
    }
    for tag in (
        "val_shared_dice",
        "val_final_minus_shared_dice",
        "val_formal_density_loss",
        "val_routing_gain",
        "val_routing_residual_abs_mean",
        "val_candidate_count_mean",
        "val_candidate_coverage_8mm",
        "val_candidate_duplicate_rate",
        "val_candidate_unmatched_rate",
    ):
        series = values(tag)
        result[tag] = series[-1] if series else None
    checkpoints = sorted((run_dir / "checkpoints").glob("epoch=*.ckpt"))
    result["best_checkpoint"] = str(checkpoints[0]) if checkpoints else None
    result["last_checkpoint"] = str(run_dir / "checkpoints/last.ckpt")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = {
        "isolated_bounded_routing": scalar_summary(args.run_dir / "isolated_bounded_routing"),
        "fixed_grid_stage_a": scalar_summary(args.run_dir / "fixed_grid_stage_a"),
        "fixed_grid_stage_b": scalar_summary(args.run_dir / "fixed_grid_stage_b"),
    }
    equivalence_path = args.run_dir / "zero_gain_equivalence.json"
    payload = {
        "zero_gain_equivalence": json.loads(equivalence_path.read_text()),
        "runs": runs,
    }
    (args.run_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    lines = [
        "# Isolated routing and fixed-grid stabilization",
        "",
        f"Zero-gain max absolute density difference: "
        f"{payload['zero_gain_equivalence']['max_abs_density_difference']:.3e}.",
        "",
        "| Run | Best Dice | Last Dice | Final−shared | Routing gain |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in runs.items():
        lines.append(
            f"| {name} | {result.get('best_val_dice')} | {result.get('last_val_dice')} | "
            f"{result.get('val_final_minus_shared_dice')} | {result.get('val_routing_gain')} |"
        )
    (args.run_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
