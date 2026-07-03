#!/usr/bin/env python
"""Predefine PHSA evaluation strata from GT sources and acquisition geometry only."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def projected_pair_separability(foci: list[dict], angles: list[float]) -> np.ndarray:
    """Return [V, P] normalized projected center separations for all source pairs."""
    rows: list[list[float]] = []
    for angle in angles:
        rad = math.radians(angle)
        horizontal = np.asarray([math.cos(rad), 0.0, math.sin(rad)], dtype=np.float64)
        vertical = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        values: list[float] = []
        for left in range(len(foci)):
            for right in range(left + 1, len(foci)):
                delta = np.asarray(foci[left]["center"], dtype=np.float64) - np.asarray(
                    foci[right]["center"], dtype=np.float64
                )
                distance = math.hypot(float(delta @ horizontal), float(delta @ vertical))
                # Conservative isotropic projected footprints from the largest semi-axis.
                radius_left = max(
                    float(foci[left].get(key) or 0.0) for key in ("radius", "rx", "ry", "rz")
                )
                radius_right = max(
                    float(foci[right].get(key) or 0.0)
                    for key in ("radius", "rx", "ry", "rz")
                )
                values.append(distance / max(radius_left + radius_right, 1.0e-6))
        rows.append(values)
    return np.asarray(rows, dtype=np.float64)


def sample_features(params: dict, angles: list[float]) -> dict[str, float | int]:
    foci = params.get("foci", [])
    if len(foci) < 2:
        return {
            "num_sources": len(foci),
            "view_sep_min": float("nan"),
            "view_sep_max": float("nan"),
            "view_sep_range": float("nan"),
            "depth_ratio": 1.0,
            "intensity_ratio": 1.0,
        }
    pair = projected_pair_separability(foci, angles)
    # A view is only as useful as its hardest-to-separate source pair.
    view_score = pair.min(axis=1)
    centers = np.asarray([focus["center"] for focus in foci], dtype=np.float64)
    depths = centers[:, 2]
    intensities = np.asarray(
        [float(focus.get("params", {}).get("intensity", 1.0)) for focus in foci]
    )
    return {
        "num_sources": len(foci),
        "view_sep_min": float(view_score.min()),
        "view_sep_max": float(view_score.max()),
        "view_sep_range": float(view_score.max() - view_score.min()),
        "depth_ratio": float((depths.max() + 1.0e-6) / (depths.min() + 1.0e-6)),
        "intensity_ratio": float(intensities.max() / max(float(intensities.min()), 1.0e-6)),
    }


def fit_thresholds(train: list[dict]) -> dict[str, float]:
    multi = [row for row in train if row["num_sources"] > 1]
    minimum = np.asarray([row["view_sep_min"] for row in multi])
    maximum = np.asarray([row["view_sep_max"] for row in multi])
    ranges = np.asarray([row["view_sep_range"] for row in multi])
    depth = np.asarray([row["depth_ratio"] for row in multi])
    strength = np.asarray([row["intensity_ratio"] for row in multi])
    return {
        "mixed_sep_low": float(np.quantile(minimum, 0.40)),
        "mixed_sep_high": float(np.quantile(maximum, 0.60)),
        "range_high": float(np.quantile(ranges, 0.60)),
        "depth_imbalance": float(np.quantile(depth, 0.75)),
        "intensity_imbalance": float(np.quantile(strength, 0.75)),
    }


def assign(row: dict, thresholds: dict[str, float]) -> tuple[str, bool]:
    if row["num_sources"] <= 1:
        return "single_source", False
    low = thresholds["mixed_sep_low"]
    high = thresholds["mixed_sep_high"]
    if row["view_sep_min"] >= high:
        geometry = "multi_all_views_separable"
    elif row["view_sep_max"] <= low:
        geometry = "multi_all_views_mixed"
    elif row["view_sep_min"] <= low and row["view_sep_max"] >= high:
        geometry = "multi_view_complementary"
    else:
        geometry = "multi_intermediate"
    imbalance = (
        row["depth_ratio"] >= thresholds["depth_imbalance"]
        or row["intensity_ratio"] >= thresholds["intensity_imbalance"]
    )
    return geometry, imbalance


def load_split(root: Path, split: str, angles: list[float]) -> list[dict]:
    ids = (root / "splits" / f"{split}.txt").read_text().splitlines()
    rows = []
    for sample_id in ids:
        params = json.loads((root / "samples" / sample_id / "tumor_params.json").read_text())
        rows.append({"sample_id": sample_id, "split": split, **sample_features(params, angles)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--angles", nargs="+", type=float, default=[-90, -60, -30, 0, 30, 60, 90]
    )
    args = parser.parse_args()
    split_rows = {
        split: load_split(args.data_root, split, args.angles)
        for split in ("train", "val", "test")
    }
    thresholds = fit_thresholds(split_rows["train"])
    all_rows = []
    manifests: dict[str, dict[str, list[str]]] = {}
    for split, rows in split_rows.items():
        manifests[split] = {}
        for row in rows:
            stratum, imbalance = assign(row, thresholds)
            row.update({"geometry_stratum": stratum, "source_imbalance": imbalance})
            all_rows.append(row)
            manifests[split].setdefault(stratum, []).append(row["sample_id"])
            if imbalance:
                manifests[split].setdefault("depth_or_strength_imbalanced", []).append(
                    row["sample_id"]
                )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "thresholds.json").write_text(json.dumps({
        "fit_split": "train",
        "angles_deg": args.angles,
        "definition": "min pairwise projected center distance / sum of max source semi-axes",
        **thresholds,
    }, indent=2) + "\n")
    (args.output_dir / "strata.json").write_text(json.dumps(manifests, indent=2) + "\n")
    with (args.output_dir / "sample_strata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    counts = {
        split: {name: len(ids) for name, ids in groups.items()}
        for split, groups in manifests.items()
    }
    (args.output_dir / "counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps({"thresholds": thresholds, "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
