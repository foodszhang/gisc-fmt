#!/usr/bin/env python3
"""Select publication examples where GISC-FMT recovers all sources and baselines lag."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DEFAULT_METHODS = {
    "gisc_fmt": "test/gisc_fmt",
    "uhr_deepfmt": "test/uhr_deepfmt",
    "pah2t_former": "current_code_runs/pah2t_former/test300",
    "gaicn": "current_code_runs/gaicn/test300",
    "two_stage_deepfmt_fixed": "current_code_runs/two_stage_deepfmt_fixed/test300",
    "two_stage_deepfmt": "test/two_stage_deepfmt",
    "pgdpnn": "test/pgdpnn",
    "fem2vox_unet": "test/fem2vox_unet",
    "fem2vox_unet_residual": "current_code_runs/fem2vox_unet_residual/test300",
    "vox_dmrn": "test/vox_dmrn",
    "fmt_reconnet": "test/fmt_reconnet",
    "map_pgan": "test/map_pgan",
    "d2_recst": "test/d2_recst",
    "dspgn": "test/dspgn",
    "tikhonov_fem": "current_code_runs/tikhonov_fem/test300",
    "l1_fem": "current_code_runs/l1_fem/test300",
    "elasticnet_fem": "current_code_runs/elasticnet_fem/test300",
    "fista_fem": "current_code_runs/fista_fem/test300",
    "stomp_fem": "current_code_runs/stomp_fem/test300",
}
DEFAULT_COMPARISON_METHODS = [method for method in DEFAULT_METHODS if method != "gisc_fmt"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, default=Path("outputs/fmt_simgen_v2_3k_20k"))
    parser.add_argument("--save_dir", type=Path, default=None)
    parser.add_argument("--gisc_result_dir", type=Path, default=None)
    parser.add_argument("--gisc_component_csv", type=Path, default=None)
    parser.add_argument("--per_category", type=int, default=1)
    parser.add_argument("--min_gisc_dice", type=float, default=0.65)
    parser.add_argument(
        "--comparison_methods",
        nargs="*",
        default=DEFAULT_COMPARISON_METHODS,
        help=(
            "Methods that influence hard-case ranking. "
            "Methods without test300 metrics are skipped."
        ),
    )
    return parser.parse_args()


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        return {row["sample_id"]: row for row in csv.DictReader(f)}


def score_row(
    sample_id: str,
    gisc: dict[str, str],
    components: dict[str, str],
    methods: dict[str, dict[str, dict[str, str]]],
) -> dict[str, object]:
    method_dice = {
        name: float(rows[sample_id]["dice"])
        for name, rows in methods.items()
        if sample_id in rows
    }
    baselines = [value for name, value in method_dice.items() if name != "gisc_fmt"]
    best_baseline = max(baselines)
    return {
        "sample_id": sample_id,
        "num_foci": int(gisc["num_foci"]),
        "depth_tier": gisc["depth_tier"],
        "shape_set": gisc["shape_set"],
        "gisc_dice": method_dice["gisc_fmt"],
        "best_baseline_dice": best_baseline,
        "dice_gap": method_dice["gisc_fmt"] - best_baseline,
        "gt_component_count": int(components["gt_component_count"]),
        "gisc_pred_component_count": int(components["pred_component_count"]),
        "gisc_matched_component_count": int(components["matched_component_count"]),
        "gisc_missed_component_count": int(components["missed_component_count"]),
        "gisc_false_component_count": int(components["false_component_count"]),
        **{f"{name}_dice": value for name, value in method_dice.items()},
    }


def is_complete_recovery(row: dict[str, str]) -> bool:
    gt_count = int(row["gt_component_count"])
    return (
        int(row["matched_component_count"]) >= gt_count
        and int(row["missed_component_count"]) == 0
        and int(row["pred_component_count"]) >= gt_count
    )


def select_unique(
    candidates: list[dict[str, object]],
    predicate,
    count: int,
    used: set[str],
) -> list[dict[str, object]]:
    selected = []
    for row in sorted(candidates, key=lambda item: float(item["dice_gap"]), reverse=True):
        sample_id = str(row["sample_id"])
        if sample_id in used or not predicate(row):
            continue
        selected.append(row)
        used.add(sample_id)
        if len(selected) == count:
            break
    return selected


def main() -> None:
    args = parse_args()
    root = args.output_root
    save_dir = args.save_dir or root / "paper_figures" / "tmi_hard_cases"
    save_dir.mkdir(parents=True, exist_ok=True)
    gisc_result_dir = args.gisc_result_dir or root / DEFAULT_METHODS["gisc_fmt"]
    requested = ["gisc_fmt", *args.comparison_methods]
    methods = {}
    for method in requested:
        metrics_path = (
            gisc_result_dir / "metrics_per_sample.csv"
            if method == "gisc_fmt"
            else root / DEFAULT_METHODS[method] / "metrics_per_sample.csv"
        )
        if metrics_path.exists():
            methods[method] = read_rows(metrics_path)
        else:
            print(f"[WARN] Skipping {method}: missing {metrics_path}")
    component_csv = args.gisc_component_csv or gisc_result_dir / "component_per_sample.csv"
    if not component_csv.exists():
        component_csv = gisc_result_dir / "components" / "component_per_sample.csv"
    components = read_rows(component_csv)
    candidates = []
    for sample_id, gisc in methods["gisc_fmt"].items():
        component = components[sample_id]
        if float(gisc["dice"]) < args.min_gisc_dice or not is_complete_recovery(component):
            continue
        candidates.append(score_row(sample_id, gisc, component, methods))

    used: set[str] = set()
    categorized: dict[str, list[dict[str, object]]] = {}
    categorized["two_foci"] = select_unique(
        candidates, lambda row: row["num_foci"] == 2, args.per_category, used
    )
    categorized["three_foci"] = select_unique(
        candidates, lambda row: row["num_foci"] == 3, args.per_category, used
    )
    categorized["irregular_shape"] = select_unique(
        candidates,
        lambda row: "irregular" in str(row["shape_set"]),
        args.per_category,
        used,
    )
    selected = [
        {"category": category, **row}
        for category, rows in categorized.items()
        for row in rows
    ]
    if not selected:
        raise SystemExit("No hard cases matched the complete-recovery constraints.")

    fields = list(selected[0])
    with (save_dir / "selected_cases.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    (save_dir / "selected_cases.json").write_text(json.dumps(selected, indent=2))

    lines = [
        "# TMI Hard-Case Selection",
        "",
        "All selected cases require complete GISC-FMT connected-component recovery: "
        "`missed_component_count == 0` and matched components cover the GT components.",
        "",
        "| Category | Sample | Foci | Shape | GISC | Best baseline | Gap | Components GT/pred |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in selected:
        lines.append(
            f"| {row['category']} | {row['sample_id']} | {row['num_foci']} | "
            f"{row['shape_set']} | {row['gisc_dice']:.3f} | "
            f"{row['best_baseline_dice']:.3f} | {row['dice_gap']:+.3f} | "
            f"{row['gt_component_count']}/{row['gisc_pred_component_count']} |"
        )
    (save_dir / "selected_cases.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
