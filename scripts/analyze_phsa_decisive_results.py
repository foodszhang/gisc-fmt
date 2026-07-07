#!/usr/bin/env python
"""Create hierarchical paired statistics for the decisive PHSA study."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METHODS = {
    "shared": ("shared", "only"),
    "uniform": ("gt", "uniform"),
    "support": ("gt", "support"),
    "geometry": ("gt", "geometry"),
    "full": ("gt", "full"),
    "shuffled": ("gt", "shuffled"),
    "oracle": ("gt", "oracle_from_uniform"),
    "learned": ("learned", "full"),
}
COMPARISONS = (
    ("full", "uniform"),
    ("full", "shuffled"),
    ("full", "support"),
    ("oracle", "uniform"),
    ("oracle", "full"),
    ("learned", "full"),
    ("learned", "uniform"),
    ("shared", "uniform"),
)
METRICS = (
    "dice",
    "component_recall",
    "weak_source_recall",
    "missed_component_rate",
    "tp",
    "fp",
    "fn",
)


def finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(array)) if np.isfinite(array).any() else float("nan")


def hierarchical_bootstrap(
    by_sample: dict[str, list[float]], draws: int, seed: int
) -> dict[str, float]:
    sample_ids = sorted(by_sample)
    sample_values = np.asarray(
        [finite_mean(by_sample[sample_id]) for sample_id in sample_ids], dtype=np.float64
    )
    sample_values = sample_values[np.isfinite(sample_values)]
    rng = np.random.default_rng(seed)
    bootstrap = sample_values[
        rng.integers(0, len(sample_values), size=(draws, len(sample_values)))
    ].mean(axis=1)
    return {
        "n_samples": int(len(sample_values)),
        "mean": float(sample_values.mean()),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-csv", type=Path, required=True)
    parser.add_argument("--strata-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()

    with args.strata_csv.open() as handle:
        strata = {row["sample_id"]: row for row in csv.DictReader(handle)}
    records: dict[tuple[str, str, int, str], dict[str, float]] = {}
    with args.paired_csv.open() as handle:
        for row in csv.DictReader(handle):
            key = (row["source"], row["strategy"], int(row["seed"]), row["sample_id"])
            records[key] = {metric: float(row[metric]) for metric in METRICS}

    stratum_names = (
        "all",
        "single_source",
        "multi_view_complementary",
        "multi_all_views_separable",
        "multi_intermediate",
        "depth_or_strength_imbalanced",
    )
    result: dict = {"absolute": {}, "paired": {}, "separability_range_quartiles": {}}
    for method, (source, strategy) in METHODS.items():
        result["absolute"][method] = {}
        selected = {
            (seed, sample_id): values
            for (row_source, row_strategy, seed, sample_id), values in records.items()
            if row_source == source and row_strategy == strategy
        }
        for stratum in stratum_names:
            values = [
                row
                for (_seed, sample_id), row in selected.items()
                if stratum == "all"
                or strata[sample_id]["geometry_stratum"] == stratum
                or (
                    stratum == "depth_or_strength_imbalanced"
                    and strata[sample_id]["source_imbalance"].lower() == "true"
                )
            ]
            if values:
                result["absolute"][method][stratum] = {
                    metric: finite_mean([row[metric] for row in values]) for metric in METRICS
                } | {"n_seed_samples": len(values)}

    for left, right in COMPARISONS:
        left_key, right_key = METHODS[left], METHODS[right]
        name = f"{left}_minus_{right}"
        result["paired"][name] = {}
        for stratum in stratum_names:
            by_metric: dict[str, dict[str, list[float]]] = {
                metric: defaultdict(list) for metric in METRICS
            }
            for seed in (41, 42, 43):
                for sample_id, stratum_row in strata.items():
                    include = (
                        stratum == "all"
                        or stratum_row["geometry_stratum"] == stratum
                        or (
                            stratum == "depth_or_strength_imbalanced"
                            and stratum_row["source_imbalance"].lower() == "true"
                        )
                    )
                    if not include:
                        continue
                    left_row = records.get((*left_key, seed, sample_id))
                    right_row = records.get((*right_key, seed, sample_id))
                    if left_row is None or right_row is None:
                        continue
                    for metric in METRICS:
                        by_metric[metric][sample_id].append(left_row[metric] - right_row[metric])
            result["paired"][name][stratum] = {
                metric: hierarchical_bootstrap(values, args.bootstrap_draws, 20260701)
                for metric, values in by_metric.items()
                if values
            }

    multi_ids = [
        sample_id
        for sample_id, row in strata.items()
        if row["split"] == "val" and int(row["num_sources"]) > 1
    ]
    ranges = np.asarray([float(strata[sample_id]["view_sep_range"]) for sample_id in multi_ids])
    boundaries = np.quantile(ranges, [0.25, 0.5, 0.75])
    for quartile in range(4):
        low = -np.inf if quartile == 0 else boundaries[quartile - 1]
        high = np.inf if quartile == 3 else boundaries[quartile]
        ids = [
            sample_id
            for sample_id in multi_ids
            if low < float(strata[sample_id]["view_sep_range"]) <= high
        ]
        deltas: dict[str, list[float]] = defaultdict(list)
        for seed in (41, 42, 43):
            for sample_id in ids:
                full = records.get((*METHODS["full"], seed, sample_id))
                uniform = records.get((*METHODS["uniform"], seed, sample_id))
                if full and uniform:
                    deltas[sample_id].append(full["dice"] - uniform["dice"])
        result["separability_range_quartiles"][f"q{quartile + 1}"] = {
            "range": [
                None if not np.isfinite(low) else float(low),
                None if not np.isfinite(high) else float(high),
            ],
            "dice_full_minus_uniform": hierarchical_bootstrap(
                deltas, args.bootstrap_draws, 20260701
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
