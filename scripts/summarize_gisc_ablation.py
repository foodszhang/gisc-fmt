#!/usr/bin/env python
"""Summarize GISC-FMT ablation metrics."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from statistics import mean, stdev


ROOT = Path("outputs/fmt_simgen_v2_3k_20k")
RUN_ROOT = ROOT / "ablation_runs"
CSV_OUT = ROOT / "ablation_summary.csv"
MD_OUT = ROOT / "ablation_summary.md"

VARIANTS = {
    "gisc_point": "Point sampling",
    "gisc_fixed_footprint": "Fixed footprint",
    "gisc_depth_footprint": "Depth-based footprint",
    "gisc_center_distance": "Center-distance schedule",
    "gisc_adaptive_unconstrained": "Adaptive w/o constraint",
    "gisc_full": "Full GISC-FMT",
    "gisc_3view": "3-view GISC",
    "gisc_5view": "5-view GISC",
    "gisc_7view": "7-view GISC",
}
MECHANISM = [
    "gisc_point",
    "gisc_fixed_footprint",
    "gisc_depth_footprint",
    "gisc_center_distance",
    "gisc_adaptive_unconstrained",
    "gisc_full",
]
SPARSE = ["gisc_3view", "gisc_5view", "gisc_7view"]

METRIC_MAP = {
    "Dice": ("Dice_mean", "Dice_std", ["Dice", "dice"]),
    "IoU": ("IoU_mean", "IoU_std", ["IoU", "iou"]),
    "CLE": ("CLE_mean", "CLE_std", ["CLE", "cle"]),
    "PLE": ("PLE_mean", "PLE_std", ["PLE", "ple"]),
    "ASSD": ("ASSD_mean", "ASSD_std", ["assd", "ASSD"]),
    "HD95": ("HD95_mean", "HD95_std", ["hd95", "HD95"]),
    "VolumeError": (
        "VolumeError_mean",
        "VolumeError_std",
        ["Volume Error", "volume_error", "VolumeError"],
    ),
}
COMPONENT_FIELDS = [
    "component_recall",
    "component_precision",
    "matched_component_count",
    "missed_component_count",
    "merge_count",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "nan", "null"}:
        return None
    try:
        val = float(text)
    except ValueError:
        return None
    return val if math.isfinite(val) else None


def mean_std(rows: list[dict[str, str]], fields: list[str]) -> tuple[float | None, float | None]:
    vals: list[float] = []
    for row in rows:
        for field in fields:
            val = to_float(row.get(field))
            if val is not None:
                vals.append(val)
                break
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return mean(vals), stdev(vals)


def best_ckpt(variant_dir: Path) -> str:
    ckpt_dir = variant_dir / "checkpoints"
    if not ckpt_dir.exists():
        return ""
    files = [p for p in ckpt_dir.glob("*.ckpt") if p.name != "last.ckpt"]
    if not files and (ckpt_dir / "last.ckpt").exists():
        return str(ckpt_dir / "last.ckpt")
    if not files:
        return ""

    def score(path: Path):
        match = re.search(r"val_dice=([0-9.]+)", path.name)
        val = float(match.group(1).rstrip(".")) if match else -1.0
        return val, path.stat().st_mtime

    return str(max(files, key=score))


def summarize_variant(name: str) -> dict[str, str | float | None]:
    variant_dir = RUN_ROOT / name
    metrics = read_rows(variant_dir / "metrics_per_sample.csv")
    components = read_rows(variant_dir / "component_per_sample.csv")
    ckpt = best_ckpt(variant_dir)
    row: dict[str, str | float | None] = {
        "variant": name,
        "description": VARIANTS[name],
        "best_ckpt": ckpt,
        "status": "completed" if metrics else ("skipped" if not ckpt else "pending_test"),
        "notes": "",
    }
    for _, (mean_key, std_key, source_fields) in METRIC_MAP.items():
        mu, sd = mean_std(metrics, source_fields)
        row[mean_key] = mu
        row[std_key] = sd
    for field in COMPONENT_FIELDS:
        mu, _sd = mean_std(components, [field])
        row[f"{field}_mean"] = mu
    if metrics and not components:
        row["notes"] = "component metrics missing"
    return row


def fmt(value) -> str:
    val = to_float(value)
    return "" if val is None else f"{val:.3f}"


def write_csv(rows: list[dict[str, str | float | None]]) -> None:
    fields = [
        "variant",
        "description",
        "best_ckpt",
        "Dice_mean",
        "Dice_std",
        "IoU_mean",
        "IoU_std",
        "CLE_mean",
        "CLE_std",
        "PLE_mean",
        "PLE_std",
        "ASSD_mean",
        "ASSD_std",
        "HD95_mean",
        "HD95_std",
        "VolumeError_mean",
        "VolumeError_std",
        "component_recall_mean",
        "component_precision_mean",
        "matched_component_count_mean",
        "missed_component_count_mean",
        "merge_count_mean",
        "status",
        "notes",
    ]
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def markdown_table(rows: list[dict[str, str | float | None]], sparse: bool = False) -> list[str]:
    if sparse:
        out = ["| Views | Dice | IoU | CLE | ASSD | HD95 | Status |", "| ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
        for row in rows:
            label = row["description"]
            out.append(
                "| "
                + " | ".join(
                    [
                        str(label).replace(" GISC", ""),
                        fmt(row.get("Dice_mean")),
                        fmt(row.get("IoU_mean")),
                        fmt(row.get("CLE_mean")),
                        fmt(row.get("ASSD_mean")),
                        fmt(row.get("HD95_mean")),
                        str(row.get("status", "")),
                    ]
                )
                + " |"
            )
        return out

    out = [
        "| Variant | Dice | IoU | CLE | PLE | ASSD | HD95 | Vol. Err. | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        out.append(
            "| "
            + " | ".join(
                [
                    str(row["description"]),
                    fmt(row.get("Dice_mean")),
                    fmt(row.get("IoU_mean")),
                    fmt(row.get("CLE_mean")),
                    fmt(row.get("PLE_mean")),
                    fmt(row.get("ASSD_mean")),
                    fmt(row.get("HD95_mean")),
                    fmt(row.get("VolumeError_mean")),
                    str(row.get("status", "")),
                ]
            )
            + " |"
        )
    return out


def write_md(rows: list[dict[str, str | float | None]]) -> None:
    by_name = {str(row["variant"]): row for row in rows}
    lines = [
        "# GISC-FMT Ablation Summary",
        "",
        "## Table A. Mechanism Ablation",
        "",
        *markdown_table([by_name[name] for name in MECHANISM]),
        "",
        "## Table B. Sparse-View Robustness",
        "",
        *markdown_table([by_name[name] for name in SPARSE], sparse=True),
        "",
    ]
    MD_OUT.write_text("\n".join(lines))


def main() -> None:
    rows = [summarize_variant(name) for name in VARIANTS]
    write_csv(rows)
    write_md(rows)
    print(f"wrote {CSV_OUT}")
    print(f"wrote {MD_OUT}")


if __name__ == "__main__":
    main()
