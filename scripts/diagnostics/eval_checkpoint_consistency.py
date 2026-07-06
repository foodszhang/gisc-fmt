#!/usr/bin/env python3
"""Compare sampled-query checkpoint scores with fixed-threshold full-volume metrics.

This evaluator is intentionally restricted to the Phase-A shared density path.  It
uses the sampled-query ``val_dice`` stored by Lightning at checkpoint time and either
reuses an existing full-volume evaluation or computes a small, aligned validation
subset with the deterministic sample-level proposal cache.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.dataset.fmt_simgen_dataset import FmtSimGenProjDataset  # noqa: E402
from minr_fmt.module import TrainingLightningModule  # noqa: E402
from minr_fmt.phsa_sample_level import activate_phsa_sample_level_hypotheses  # noqa: E402
from scripts.eval_components_fmt_simgen import evaluate_sample as component_metrics  # noqa: E402
from scripts.eval_full_volume_fmt_simgen import (  # noqa: E402
    load_gt,
    load_sample_statistics,
    sample_metadata,
    volume_metrics,
)
from scripts.eval_view_complementary_full_volume_paired import (  # noqa: E402
    deterministic_proposal_indices,
    linear_indices_to_points_mm,
)
from scripts.eval_view_complementary_full_volume_paired_safe import (  # noqa: E402
    predict_full_volume_low_memory,
)

activate_phsa_sample_level_hypotheses()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/third_contribution_diagnostics"
    )
    parser.add_argument("--split", choices=["val"], default="val")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--sample-ids", nargs="+", default=None)
    parser.add_argument("--proposal-count", type=int, default=4096)
    parser.add_argument("--proposal-seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=32768)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-region-size", type=int, default=10)
    parser.add_argument("--cc-connectivity", type=int, default=26)
    parser.add_argument("--iou-threshold", type=float, default=0.01)
    parser.add_argument("--centroid-threshold-vox", type=float, default=3.0)
    parser.add_argument("--small-component-max-voxels", type=int, default=64)
    parser.add_argument(
        "--existing-eval",
        action="append",
        default=[],
        metavar="CHECKPOINT=DIR",
        help="Reuse metrics_per_sample.csv from a fixed-threshold full-volume evaluation.",
    )
    return parser.parse_args()


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    sampled = None
    monitor = None
    for callback in obj.get("callbacks", {}).values():
        if not isinstance(callback, dict):
            continue
        if callback.get("monitor") == "val_dice":
            monitor = "val_dice"
            value = callback.get("current_score", callback.get("best_model_score"))
            if value is not None:
                sampled = float(value.item() if torch.is_tensor(value) else value)
                break
    if sampled is None:
        match = re.search(r"val_dice=([0-9.]+)", path.name)
        sampled = float(match.group(1).rstrip(".")) if match else math.nan
    return {
        "checkpoint_name": path.name,
        "checkpoint_path": str(path.resolve()),
        "epoch": int(obj.get("epoch", -1)),
        "global_step": int(obj.get("global_step", -1)),
        "sampled_query_dice": sampled,
        "sampled_query_source": monitor or "checkpoint_filename",
    }


def assert_phase_a_contract(cfg: Any) -> None:
    view = cfg.model.ssq_fmt.view_complementary
    checks = {
        "density_output_mode=view_complementary": str(cfg.model.ssq_fmt.density_output_mode)
        == "view_complementary",
        "training_phase=phase_a": str(view.training_phase) == "phase_a",
        "strong_shared_fusion=true": bool(view.get("strong_shared_fusion", False)),
        "routing.enabled=false": not bool(view.get("routing", {}).get("enabled", False)),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"Diagnostic is restricted to the Phase-A shared path; failed={failed}")


def load_model(cfg: Any, checkpoint: Path, device: torch.device):
    module = TrainingLightningModule(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    module.load_state_dict(state, strict=True)
    module.eval().to(device)
    module.net.set_view_training_phase("phase_a")
    return module.net


def load_existing(path: Path, case_ids: list[str] | None) -> list[dict[str, Any]]:
    metrics_path = path / "metrics_per_sample.csv"
    component_path = path / "components/component_per_sample.csv"
    with metrics_path.open(newline="") as handle:
        metrics = list(csv.DictReader(handle))
    components: dict[str, dict[str, str]] = {}
    if component_path.exists():
        with component_path.open(newline="") as handle:
            components = {row["sample_id"]: row for row in csv.DictReader(handle)}
    allowed = set(case_ids) if case_ids else None
    rows = []
    for row in metrics:
        if allowed is not None and row["sample_id"] not in allowed:
            continue
        merged: dict[str, Any] = {**row, **components.get(row["sample_id"], {})}
        rows.append(merged)
    return rows


def numeric_mean(rows: list[dict[str, Any]], key: str) -> float:
    values = []
    for row in rows:
        value = row.get(key)
        if value not in {None, ""}:
            values.append(float(value))
    return float(np.mean(values)) if values else math.nan


def evaluate_checkpoint(
    cfg: Any, checkpoint: Path, sample_dirs: list[Path], args: argparse.Namespace
) -> list[dict[str, Any]]:
    device = torch.device(args.device)
    net = load_model(cfg, checkpoint, device)
    data_dir = Path(str(cfg.data.val_dir))
    loader = FmtSimGenProjDataset(str(data_dir), config=cfg, split="all", is_training=False)
    statistics = load_sample_statistics(data_dir)
    voxel_size = float(cfg.data.voxel_size_mm)
    rows = []
    for index, sample_dir in enumerate(sample_dirs, start=1):
        gt = load_gt(sample_dir)
        shape = tuple(int(value) for value in gt.shape)
        _, surface, _, depth_maps, _ = loader._load_projection(sample_dir)
        surface = surface.unsqueeze(0).to(device)
        depth_maps = depth_maps.unsqueeze(0).to(device)
        valid = torch.isfinite(depth_maps)
        indices = deterministic_proposal_indices(
            shape, args.proposal_count, args.proposal_seed, sample_dir.name
        )
        proposal = linear_indices_to_points_mm(indices, shape, voxel_size).to(device)
        pred, _ = predict_full_volume_low_memory(
            net,
            surface,
            valid,
            depth_maps,
            shape,
            voxel_size,
            proposal,
            args.chunk_size,
            1.0e-6,
        )
        volume = volume_metrics(
            pred,
            gt,
            args.threshold,
            (voxel_size,) * 3,
            args.min_region_size,
            args.cc_connectivity,
        )
        components = component_metrics(
            pred,
            gt,
            args.threshold,
            args.min_region_size,
            args.cc_connectivity,
            args.iou_threshold,
            args.centroid_threshold_vox,
            args.small_component_max_voxels,
        )
        rows.append(
            {
                "sample_id": sample_dir.name,
                **sample_metadata(sample_dir, statistics),
                **volume,
                **components,
            }
        )
        print(
            f"[{checkpoint.name}] {index}/{len(sample_dirs)} {sample_dir.name}: "
            f"{volume['dice']:.6f}"
        )
    del net
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def correlation(rows: list[dict[str, Any]], left: str, right: str) -> tuple[float, float]:
    pairs = [(float(row[left]), float(row[right])) for row in rows]
    pairs = [(a, b) for a, b in pairs if math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 2:
        return math.nan, math.nan
    result = stats.spearmanr([a for a, _ in pairs], [b for _, b in pairs])
    return float(result.statistic), float(result.pvalue)


def write_report(
    rows: list[dict[str, Any]], output: Path, case_ids: list[str], threshold: float
) -> None:
    sampled_rank = sorted(rows, key=lambda row: float(row["sampled_query_dice"]), reverse=True)
    volume_rank = sorted(rows, key=lambda row: float(row["full_volume_dice"]), reverse=True)
    rho, pvalue = correlation(rows, "sampled_query_dice", "full_volume_dice")
    component_rho, component_p = correlation(rows, "sampled_query_dice", "component_recall")
    same_best = bool(
        sampled_rank
        and volume_rank
        and sampled_rank[0]["checkpoint_path"] == volume_rank[0]["checkpoint_path"]
    )
    lines = [
        "# Checkpoint consistency diagnostic",
        "",
        f"Fixed threshold: `{threshold}`. Full-volume validation case IDs: "
        + ", ".join(f"`{value}`" for value in case_ids)
        + ".",
        "",
        "Sampled-query values are the actual checkpoint-time `val_dice` monitor values. "
        "They use the configured fixed validation split; the full-volume subset is listed above.",
        "",
        "## Ranking comparison",
        "",
        "| sampled rank | checkpoint | sampled Dice | full-volume Dice | component recall |",
        "|---:|---|---:|---:|---:|",
    ]
    by_path = {row["checkpoint_path"]: row for row in rows}
    for rank, item in enumerate(sampled_rank, start=1):
        row = by_path[item["checkpoint_path"]]
        lines.append(
            f"| {rank} | `{row['checkpoint_name']}` | {row['sampled_query_dice']:.6f} | "
            f"{row['full_volume_dice']:.6f} | {row['component_recall']:.6f} |"
        )
    lines += [
        "",
        "Full-volume order: "
        + " > ".join(f"`{row['checkpoint_name']}`" for row in volume_rank)
        + ".",
        "",
        f"Sampled-best equals full-volume-best: **{same_best}**.",
        f"Spearman(sampled Dice, full-volume Dice): rho={rho:.6f}, p={pvalue:.6g}.",
        f"Spearman(sampled Dice, component recall): rho={component_rho:.6f}, p={component_p:.6g}.",
        "",
        "## Interpretation",
        "",
    ]
    if len(rows) < 2:
        lines.append(
            "Only one comparable Phase-A checkpoint was available, so a checkpoint-ranking "
            "mismatch and the 0–1 epoch phenomenon cannot be identified from this run."
        )
    elif same_best:
        lines.append(
            "The sampled-query best is also best on this full-volume subset. This does not support "
            "a selection mismatch on the evaluated cases; inspect precision/recall columns for a "
            "possible operating-point trade-off."
        )
    else:
        lines.append(
            "The sampled-query and full-volume best checkpoints differ on the evaluated cases, "
            "which is direct evidence of checkpoint-selection mismatch."
        )
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(args.run_config)
    cfg.data.num_workers = 0
    cfg.data.persistent_workers = False
    cfg.data.train_max_samples = None
    cfg.data.val_max_samples = None
    assert_phase_a_contract(cfg)
    dataset = FmtSimGenProjDataset(
        str(cfg.data.val_dir), config=cfg, split="val", is_training=False
    )
    sample_dirs = list(dataset.dirs)
    if args.sample_ids:
        wanted = set(args.sample_ids)
        sample_dirs = [path for path in sample_dirs if path.name in wanted]
    sample_dirs = sample_dirs[: args.max_cases]
    if not sample_dirs:
        raise RuntimeError("No validation cases selected")
    existing: dict[str, Path] = {}
    for value in args.existing_eval:
        key, directory = value.rsplit("=", 1)
        existing[str(Path(key).resolve())] = Path(directory)

    summary_rows = []
    per_case_rows = []
    for checkpoint in args.checkpoint:
        checkpoint = checkpoint.resolve()
        metadata = checkpoint_metadata(checkpoint)
        if str(checkpoint) in existing:
            cases = load_existing(existing[str(checkpoint)], [path.name for path in sample_dirs])
        else:
            cases = evaluate_checkpoint(cfg, checkpoint, sample_dirs, args)
        if not cases:
            raise RuntimeError(f"No full-volume rows for {checkpoint}")
        for case in cases:
            per_case_rows.append({**metadata, **case})
        summary_rows.append(
            {
                **metadata,
                "threshold": args.threshold,
                "num_full_volume_cases": len(cases),
                "case_ids": ";".join(str(row["sample_id"]) for row in cases),
                "full_volume_dice": numeric_mean(cases, "dice"),
                "full_volume_precision": numeric_mean(cases, "precision"),
                "full_volume_recall": numeric_mean(cases, "recall"),
                "volume_error": numeric_mean(cases, "volume_error"),
                "component_recall": numeric_mean(cases, "component_recall"),
                "component_precision": numeric_mean(cases, "component_precision"),
                "merge_count": numeric_mean(cases, "merge_count"),
            }
        )

    csv_path = args.output_dir / "checkpoint_consistency.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (args.output_dir / "checkpoint_consistency_per_case.csv").open("w", newline="") as handle:
        fields = sorted({key for row in per_case_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_case_rows)
    write_report(
        summary_rows,
        args.output_dir / "checkpoint_consistency_report.md",
        [path.name for path in sample_dirs],
        args.threshold,
    )
    print(csv_path)


if __name__ == "__main__":
    main()
