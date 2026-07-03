#!/usr/bin/env python3
"""Collect matched morphology-amplitude experiment summaries into one table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

VARIANTS = (
    "scalar_control",
    "support_aux",
    "factorized_core",
    "factorized_component",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> None:
    args = parse_args()
    root = Path(args.run_root)
    rows = []
    for name in VARIANTS:
        eval_dir = root / name / "full_volume_test"
        volume = load_json(eval_dir / "metrics_summary.json")
        component = load_json(eval_dir / "components" / "component_summary.json")
        row = {"variant": name}
        for key in (
            "dice_mean",
            "iou_mean",
            "recall_mean",
            "precision_mean",
            "nrmse_mean",
            "assd_mean",
            "hd95_mean",
            "volume_error_mean",
        ):
            row[key] = volume.get(key)
        for key in (
            "component_recall_mean",
            "component_precision_mean",
            "mean_matched_iou_mean",
            "small_component_recall_mean",
            "merge_count_mean",
            "split_count_mean",
        ):
            row[key] = component.get(key)
        rows.append(row)

    output = Path(args.output) if args.output else root / "comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
