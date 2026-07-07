#!/usr/bin/env python3
"""Select a voxel operating threshold for continuous traditional FEM reconstructions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_THRESHOLDS = (0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    return parser.parse_args()


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    denominator = int(pred.sum()) + int(gt.sum())
    return 2.0 * float(np.logical_and(pred, gt).sum()) / max(denominator, 1)


def main() -> None:
    args = parse_args()
    predictions = []
    for path in sorted(args.prediction_dir.glob("*.npz")):
        with np.load(path) as data:
            predictions.append((data["pred"].astype(np.float32), data["gt"] >= 0.5))
    if not predictions:
        raise SystemExit(f"No prediction NPZ files found in {args.prediction_dir}")

    rows = []
    for threshold in args.thresholds:
        scores = [dice(pred >= threshold, gt) for pred, gt in predictions]
        rows.append(
            {
                "threshold": float(threshold),
                "dice_mean": float(np.mean(scores)),
                "dice_std": float(np.std(scores)),
            }
        )
    selected = max(rows, key=lambda row: row["dice_mean"])
    summary = {
        "prediction_dir": str(args.prediction_dir),
        "num_samples": len(predictions),
        "selected_threshold": selected["threshold"],
        "selected_dice_mean": selected["dice_mean"],
        "threshold_sweep": rows,
    }
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
