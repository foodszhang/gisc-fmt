#!/usr/bin/env python
"""Summarize unified paper-baseline test300 outputs and grouped comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "fmt_simgen_v2_3k_20k"
RUNS = OUT / "current_code_runs"
MODELS = [
    "gisc_fmt", "pah2t_former", "uhr_deepfmt", "uhr_deepfmt_lowres",
    "two_stage_deepfmt_fixed", "two_stage_deepfmt", "gaicn", "pgdpnn",
    "fem2vox_unet", "fem2vox_unet_residual", "vox_dmrn", "fmt_reconnet",
    "map_pgan", "d2_recst", "dspgn",
    "fem_coarse", "fem_to_voxel", "tikhonov_fem", "l1_fem", "elasticnet_fem",
    "fista_fem", "stomp_fem",
]
GROUP_KEYS = {"num_foci", "depth_tier", "shape_class", "shape_set", "num_foci x depth_tier"}
TRADITIONAL_FEM_THRESHOLDS = {
    "tikhonov_fem": 0.0075,
    "l1_fem": 0.01,
    "elasticnet_fem": 0.01,
    "fista_fem": 0.02,
    "stomp_fem": 0.001,
}
EXCLUDED_STALE_RESULTS = {
    "elasticnet_fem": "Formal rerun stopped by user request; exclude stale pre-fix metrics.",
    "fista_fem": "Post-fix test300 rerun not completed; exclude stale pre-fix metrics.",
    "stomp_fem": "Post-fix test300 rerun not completed; exclude stale pre-fix metrics.",
}
FIELDS = [
    "model", "status", "best_ckpt", "best_val_dice", "test_dice", "test_iou",
    "test_precision", "test_recall", "test_nrmse", "test_psnr", "test_ssim", "test_assd",
    "test_hd95", "test_cle", "test_ple", "test_cnr", "test_volume_error",
    "test_inference_time_ms", "notes",
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gisc_result_dir", type=Path, default=OUT / "test" / "gisc_fmt")
    parser.add_argument("--gisc_ckpt", type=Path, default=None)
    parser.add_argument("--gisc_label", default="GISC-FMT")
    return parser.parse_args()

def best_ckpt(model_dir: Path):
    paths = list((model_dir / "checkpoints").glob("*.ckpt"))
    def score(path):
        match = re.search(r"val_dice=([0-9]+(?:\.[0-9]+)?)", path.name)
        return (float(match.group(1)) if match else -1.0, path.stat().st_mtime)
    return max(paths, key=score) if paths else None

def metric(summary, key):
    for candidate in (f"{key}_mean", f"test_{key}", key):
        if candidate in summary:
            return summary[candidate]
    return ""

def result_dir(model, args):
    if model == "gisc_fmt":
        return args.gisc_result_dir
    legacy_test_models = {
        "uhr_deepfmt", "two_stage_deepfmt", "pgdpnn", "fem2vox_unet",
        "vox_dmrn", "fmt_reconnet", "map_pgan", "d2_recst", "dspgn",
    }
    if model in legacy_test_models:
        return OUT / "test" / model
    return RUNS / model / "test300"

def model_dir(model):
    legacy_run_models = {
        "gisc_fmt", "uhr_deepfmt", "two_stage_deepfmt", "pgdpnn", "fem2vox_unet",
        "vox_dmrn", "fmt_reconnet", "map_pgan", "d2_recst", "dspgn",
    }
    if model in legacy_run_models:
        return OUT / "runs" / model
    return RUNS / model

def derived_shape_groups(test_dir):
    sample_path = test_dir / "metrics_per_sample.csv"
    if not sample_path.exists():
        return []
    buckets = defaultdict(list)
    with sample_path.open(newline="") as f:
        for row in csv.DictReader(f):
            shape_set = row.get("shape_set", "")
            shapes = [shape for shape in shape_set.split("+") if shape]
            if len(shapes) == 1:
                shape_class = shapes[0]
            elif len(shapes) == 2:
                shape_class = "mixed_two_shape"
            elif len(shapes) >= 3:
                shape_class = "mixed_three_shape"
            else:
                shape_class = "unknown"
            buckets[("shape_set", shape_set or "unknown")].append(row)
            buckets[("shape_class", shape_class)].append(row)
    out = []
    for (group_key, group_value), bucket in sorted(buckets.items()):
        derived = {"group_key": group_key, "group_value": group_value, "num_samples": len(bucket)}
        for metric in ("dice", "precision", "recall", "assd", "hd95"):
            values = [float(row[metric]) for row in bucket if row.get(metric) not in {"", None}]
            derived[f"{metric}_mean"] = sum(values) / len(values) if values else ""
        out.append(derived)
    return out

args = parse_args()
rows = []
group_rows = []
for model in MODELS:
    run_dir = model_dir(model)
    test_dir = result_dir(model, args)
    ckpt = best_ckpt(run_dir)
    summary_path = test_dir / "metrics_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if model in EXCLUDED_STALE_RESULTS:
        summary = {}
    if summary.get("ckpt_path"):
        ckpt = Path(summary["ckpt_path"])
    if model == "gisc_fmt" and args.gisc_ckpt:
        ckpt = args.gisc_ckpt
    dice = metric(summary, "dice")
    notes = ""
    status = "pending"
    if model in EXCLUDED_STALE_RESULTS:
        status = "excluded_incomplete_rerun"
        notes = EXCLUDED_STALE_RESULTS[model]
    if summary:
        if float(dice) >= 0.4:
            status = "main_candidate"
        else:
            status = "below_0.4_supplementary"
            notes = "Retain metrics and figures; below main-table deep-baseline threshold."
    if model == "gisc_fmt" and summary:
        notes = f"Current paper result: {args.gisc_label}."
    elif model in {"fem_coarse", "fem_to_voxel"}:
        status = "internal_diagnostic_only"
        notes = (
            "Internal FEM-prior diagnostic only; exclude from the paper comparison. "
            "Evaluated after barycentric interpolation to [190, 200, 104]."
        )
    elif model == "gaicn" and summary:
        notes = "Full-data 2400-train/300-val cached mesh-space GAICN adaptation."
    elif model == "fem2vox_unet_residual" and summary:
        notes = "Full-data 2400-train/300-val residual-prior FEM2Vox adaptation."
    elif model in TRADITIONAL_FEM_THRESHOLDS and summary:
        notes = (
            "Traditional continuous FEM reconstruction; barycentric interpolation to "
            "[190, 200, 104]. Validation-selected voxel threshold="
            f"{TRADITIONAL_FEM_THRESHOLDS[model]}."
        )
    rows.append({
        "model": model,
        "status": status,
        "best_ckpt": str(ckpt) if ckpt else "",
        "best_val_dice": re.search(r"val_dice=([0-9]+(?:\.[0-9]+)?)", ckpt.name).group(1)
        if ckpt and "val_dice=" in ckpt.name else "",
        **{
            f"test_{key}": metric(summary, key)
            for key in (
                "dice", "iou", "precision", "recall", "nrmse", "psnr", "ssim", "assd",
                "hd95", "cle", "ple", "cnr", "volume_error", "inference_time_ms",
            )
        },
        "notes": notes,
    })
    grouped_path = test_dir / "metrics_grouped.csv"
    if grouped_path.exists():
        seen_group_keys = set()
        with grouped_path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row["group_key"] in GROUP_KEYS:
                    seen_group_keys.add(row["group_key"])
                    group_rows.append(
                        {
                            "model": model,
                            "group_key": row["group_key"],
                            "group_value": row["group_value"],
                            "num_samples": row["num_samples"],
                            "dice_mean": row.get("dice_mean", ""),
                            "precision_mean": row.get("precision_mean", ""),
                            "recall_mean": row.get("recall_mean", ""),
                            "assd_mean": row.get("assd_mean", ""),
                            "hd95_mean": row.get("hd95_mean", ""),
                        }
                    )
        if "shape_set" not in seen_group_keys or "shape_class" not in seen_group_keys:
            for row in derived_shape_groups(test_dir):
                group_rows.append({"model": model, **row})

csv_path = OUT / "baseline_summary.csv"
md_path = OUT / "baseline_summary.md"
grouped_csv_path = OUT / "baseline_grouped_summary.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
lines = ["# Paper Baseline Summary", "", "| " + " | ".join(FIELDS) + " |",
         "| " + " | ".join(["---"] * len(FIELDS)) + " |"]
for row in rows:
    lines.append("| " + " | ".join(str(row.get(field, "")) for field in FIELDS) + " |")
lines.extend(["", f"## Current Best: {args.gisc_label}", ""])
for group_key, title in (
    ("num_foci", "By Number of Foci"),
    ("depth_tier", "By Depth Tier"),
    ("shape_class", "By Shape Class"),
):
    lines.extend(
        [
            f"### {title}",
            "",
            "| group | samples | dice | precision | recall | assd | hd95 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in group_rows:
        if row["model"] == "gisc_fmt" and row["group_key"] == group_key:
            lines.append(
                "| "
                + " | ".join(
                    str(row.get(field, ""))
                    for field in (
                        "group_value", "num_samples", "dice_mean", "precision_mean",
                        "recall_mean", "assd_mean", "hd95_mean",
                    )
                )
                + " |"
            )
    lines.append("")

component_path = args.gisc_result_dir / "component_summary.json"
if not component_path.exists():
    component_path = args.gisc_result_dir / "components" / "component_summary.json"
if component_path.exists():
    component = json.loads(component_path.read_text())
    lines.extend(
        [
            "### Component Recovery",
            "",
            "| metric | value |",
            "| --- | --- |",
        ]
    )
    for key in (
        "gt_component_count_mean",
        "pred_component_count_mean",
        "matched_component_count_mean",
        "missed_component_count_mean",
        "merge_count_mean",
        "component_recall_mean",
        "component_precision_mean",
    ):
        lines.append(f"| {key} | {component.get(key, '')} |")
md_path.write_text("\n".join(lines) + "\n")
group_fields = [
    "model", "group_key", "group_value", "num_samples", "dice_mean", "precision_mean",
    "recall_mean", "assd_mean", "hd95_mean",
]
with grouped_csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=group_fields)
    writer.writeheader()
    writer.writerows(group_rows)
print(csv_path)
print(md_path)
print(grouped_csv_path)
