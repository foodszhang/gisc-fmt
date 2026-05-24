#!/usr/bin/env python
"""Summarize FMT-SimGen v2 multi-source experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

METHODS = [
    ("GISC baseline", "gisc_baseline", "outputs/fmt_simgen_v2_3k_20k/test/gisc_fmt"),
    ("E12-v2", "e12_v2", "outputs/fmt_simgen_v2_multisource/test/e12_v2"),
    ("E12-MPB", "e12_mpb", "outputs/fmt_simgen_v2_multisource/test/e12_mpb"),
    (
        "E12-MPB-Tversky",
        "e12_mpb_tversky",
        "outputs/fmt_simgen_v2_multisource/test/e12_mpb_tversky",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        default="outputs/fmt_simgen_v2_multisource/summary",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def read_grouped(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {(row["group_key"], row["group_value"]): row for row in rows}


def shape_class(shape_set: str) -> str:
    parts = [part for part in str(shape_set).split("+") if part]
    unique = sorted(set(parts))
    if len(unique) == 1:
        return unique[0]
    if len(unique) == 2:
        return "mixed_two_shape"
    if len(unique) >= 3:
        return "mixed_three_shape"
    return "unknown"


def shape_class_fallback(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    path = root / "metrics_per_sample.csv"
    if not path.exists():
        return {}
    buckets: dict[str, list[dict[str, Any]]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            buckets.setdefault(shape_class(row.get("shape_set", "")), []).append(row)
    out = {}
    for key, rows in buckets.items():
        values = {
            "group_key": "shape_class",
            "group_value": key,
            "dice_mean": mean([row.get("dice") for row in rows]),
            "recall_mean": mean([row.get("recall") for row in rows]),
        }
        out[("shape_class", key)] = values
    return out


def as_float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in {"", None}:
        return None
    return float(value)


def mean(values: list[Any]) -> float | None:
    vals = [float(v) for v in values if v not in {"", None}]
    return sum(vals) / len(vals) if vals else None


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def table_overall(method_rows: list[tuple[str, str, Path]]) -> list[dict[str, Any]]:
    rows = []
    for label, key, root in method_rows:
        summary = read_json(root / "metrics_summary.json")
        if not summary:
            continue
        rows.append(
            {
                "method": label,
                "method_key": key,
                "dice": fmt(summary.get("dice_mean")),
                "precision": fmt(summary.get("precision_mean")),
                "recall": fmt(summary.get("recall_mean")),
                "assd": fmt(summary.get("assd_mean")),
                "hd95": fmt(summary.get("hd95_mean")),
            }
        )
    return rows


def table_num_foci(method_rows: list[tuple[str, str, Path]]) -> list[dict[str, Any]]:
    rows = []
    for label, key, root in method_rows:
        grouped = read_grouped(root / "metrics_grouped.csv")
        if not grouped:
            continue
        grouped = {**shape_class_fallback(root), **grouped}
        row: dict[str, Any] = {"method": label, "method_key": key}
        for foci in ["1", "2", "3"]:
            group = grouped.get(("num_foci", foci), {})
            row[f"foci{foci}_dice"] = fmt(as_float(group, "dice_mean"))
            row[f"foci{foci}_recall"] = fmt(as_float(group, "recall_mean"))
        rows.append(row)
    return rows


def table_shape(method_rows: list[tuple[str, str, Path]]) -> list[dict[str, Any]]:
    rows = []
    for label, key, root in method_rows:
        grouped = read_grouped(root / "metrics_grouped.csv")
        if not grouped:
            continue
        row: dict[str, Any] = {"method": label, "method_key": key}
        for shape in [
            "ellipsoid",
            "sphere",
            "irregular",
            "mixed_two_shape",
            "mixed_three_shape",
        ]:
            group = grouped.get(("shape_class", shape), {})
            row[f"{shape}_dice"] = fmt(as_float(group, "dice_mean"))
            row[f"{shape}_recall"] = fmt(as_float(group, "recall_mean"))
        rows.append(row)
    return rows


def table_components(method_rows: list[tuple[str, str, Path]]) -> list[dict[str, Any]]:
    rows = []
    for label, key, root in method_rows:
        summary = read_json(root / "component_summary.json")
        grouped = read_grouped(root / "component_by_num_foci.csv")
        if not summary:
            continue
        foci3 = grouped.get(("num_foci", "3"), {})
        rows.append(
            {
                "method": label,
                "method_key": key,
                "component_recall": fmt(summary.get("component_recall_mean")),
                "missed_count": fmt(summary.get("missed_component_count_mean")),
                "merge_rate": fmt(summary.get("merge_count_mean")),
                "component_precision": fmt(summary.get("component_precision_mean")),
                "matched_iou": fmt(summary.get("mean_matched_iou_mean")),
                "foci3_component_recall": fmt(as_float(foci3, "component_recall_mean")),
                "foci3_missed_count": fmt(as_float(foci3, "missed_component_count_mean")),
                "foci3_merge_rate": fmt(as_float(foci3, "merge_count_mean")),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    method_rows = [(label, key, Path(root)) for label, key, root in METHODS]
    save_dir = Path(args.save_dir)
    write_csv(
        save_dir / "table1_overall_full_volume.csv",
        table_overall(method_rows),
        ["method", "method_key", "dice", "precision", "recall", "assd", "hd95"],
    )
    write_csv(
        save_dir / "table2_num_foci.csv",
        table_num_foci(method_rows),
        [
            "method",
            "method_key",
            "foci1_dice",
            "foci1_recall",
            "foci2_dice",
            "foci2_recall",
            "foci3_dice",
            "foci3_recall",
        ],
    )
    write_csv(
        save_dir / "table3_shape_mixed.csv",
        table_shape(method_rows),
        [
            "method",
            "method_key",
            "ellipsoid_dice",
            "ellipsoid_recall",
            "sphere_dice",
            "sphere_recall",
            "irregular_dice",
            "irregular_recall",
            "mixed_two_shape_dice",
            "mixed_two_shape_recall",
            "mixed_three_shape_dice",
            "mixed_three_shape_recall",
        ],
    )
    write_csv(
        save_dir / "table4_component_metrics.csv",
        table_components(method_rows),
        [
            "method",
            "method_key",
            "component_recall",
            "missed_count",
            "merge_rate",
            "component_precision",
            "matched_iou",
            "foci3_component_recall",
            "foci3_missed_count",
            "foci3_merge_rate",
        ],
    )
    print(f"wrote summary tables to {save_dir}")


if __name__ == "__main__":
    main()
