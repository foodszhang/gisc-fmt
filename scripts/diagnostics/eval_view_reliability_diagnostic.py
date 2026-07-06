#!/usr/bin/env python3
"""Offline leave-one-view diagnostic for the frozen Phase-A shared density path."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from scripts.eval_full_volume_fmt_simgen import load_gt, volume_metrics  # noqa: E402
from scripts.eval_view_complementary_full_volume_paired import (  # noqa: E402
    build_sample_cache,
    call_net,
    chunk_points_mm,
    deterministic_proposal_indices,
    frozen_sample_cache,
    linear_indices_to_points_mm,
)

activate_phsa_sample_level_hypotheses()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/third_contribution_diagnostics"
    )
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
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def assert_phase_a_contract(cfg: Any) -> None:
    view = cfg.model.ssq_fmt.view_complementary
    failed = []
    if str(cfg.model.ssq_fmt.density_output_mode) != "view_complementary":
        failed.append("density_output_mode=view_complementary")
    if str(view.training_phase) != "phase_a":
        failed.append("training_phase=phase_a")
    if not bool(view.get("strong_shared_fusion", False)):
        failed.append("strong_shared_fusion=true")
    if bool(view.get("routing", {}).get("enabled", False)):
        failed.append("routing.enabled=false")
    if failed:
        raise RuntimeError(f"Leave-one-view diagnostic is restricted to Phase A; failed={failed}")


def load_net(cfg: Any, checkpoint: Path, device: torch.device):
    module = TrainingLightningModule(cfg)
    obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    module.load_state_dict(obj["state_dict"], strict=True)
    module.eval().to(device)
    module.net.set_view_training_phase("phase_a")
    return module.net


def predict_with_cache(
    net,
    cache: dict[str, Any],
    surface: torch.Tensor,
    valid: torch.Tensor,
    depth_maps: torch.Tensor,
    shape: tuple[int, int, int],
    voxel_size: float,
    chunk_size: int,
) -> np.ndarray:
    pred = np.empty(int(np.prod(shape)), dtype=np.float32)
    with frozen_sample_cache(net, cache), torch.inference_mode():
        for start in range(0, pred.size, chunk_size):
            end = min(start + chunk_size, pred.size)
            points = chunk_points_mm(shape, start, end, voxel_size).to(surface.device)
            output = call_net(net, surface, points, valid, depth_maps, return_diagnostics=False)
            values = output["density"].squeeze(0).squeeze(-1)
            if not torch.isfinite(values).all():
                raise RuntimeError("Leave-one-view prediction contains NaN/Inf")
            pred[start:end] = values.float().cpu().numpy()
    return pred.reshape(shape)


def mean_candidate_signal(signal: torch.Tensor, candidate_valid: torch.Tensor) -> np.ndarray:
    # Signals use [B,V,M]; only valid candidate slots contribute.
    values = signal.detach().float()[0]
    valid = candidate_valid.detach().bool()[0]
    if values.ndim != 2:
        return np.full(values.shape[0], np.nan, dtype=np.float64)
    mask = valid[None].expand_as(values)
    denom = mask.sum(dim=1).clamp_min(1)
    result = (values.masked_fill(~mask, 0.0).sum(dim=1) / denom).cpu().numpy()
    return result.astype(np.float64)


def response_strength(surface: torch.Tensor, valid: torch.Tensor) -> np.ndarray:
    values = []
    for view in range(surface.shape[1]):
        measurement = surface[0, view].detach().float()
        mask = valid[0, view].detach().bool()
        if measurement.ndim == 3 and measurement.shape[0] == 1:
            measurement = measurement[0]
        selected = measurement.abs()[mask]
        values.append(float(selected.mean().cpu()) if selected.numel() else math.nan)
    return np.asarray(values, dtype=np.float64)


def metric_bundle(pred: np.ndarray, gt: np.ndarray, args: argparse.Namespace, voxel_size: float):
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
    return {**volume, **components}


def rank_correlation(rows: list[dict[str, Any]], key: str) -> tuple[float, float, int]:
    pairs = []
    for row in rows:
        left, right = row.get(key), row.get("usefulness_dice_drop")
        if left not in {None, ""} and right not in {None, ""}:
            left, right = float(left), float(right)
            if math.isfinite(left) and math.isfinite(right):
                pairs.append((left, right))
    if len(pairs) < 3 or len(set(a for a, _ in pairs)) < 2 or len(set(b for _, b in pairs)) < 2:
        return math.nan, math.nan, len(pairs)
    result = stats.spearmanr([a for a, _ in pairs], [b for _, b in pairs])
    return float(result.statistic), float(result.pvalue), len(pairs)


def within_case_correlation(rows: list[dict[str, Any]], key: str) -> tuple[float, float, int]:
    centered: list[tuple[float, float]] = []
    for case_id in sorted({str(row["case_id"]) for row in rows}):
        group = [row for row in rows if str(row["case_id"]) == case_id]
        signal = np.asarray([float(row[key]) for row in group])
        usefulness = np.asarray([float(row["usefulness_dice_drop"]) for row in group])
        finite = np.isfinite(signal) & np.isfinite(usefulness)
        centered.extend(
            zip(
                signal[finite] - signal[finite].mean(),
                usefulness[finite] - usefulness[finite].mean(),
                strict=True,
            )
        )
    if len(centered) < 3:
        return math.nan, math.nan, len(centered)
    result = stats.spearmanr([left for left, _ in centered], [right for _, right in centered])
    return float(result.statistic), float(result.pvalue), len(centered)


def write_report(rows: list[dict[str, Any]], path: Path, checkpoint: Path) -> None:
    correlations = {
        key: rank_correlation(rows, key)
        for key in (
            "response_strength",
            "raw_source_separability",
            "phsa_weight",
            "shared_attention_weight",
        )
    }
    within_case = {key: within_case_correlation(rows, key) for key in correlations}
    helpful = sorted(rows, key=lambda row: float(row["usefulness_dice_drop"]), reverse=True)[:10]
    harmful = sorted(rows, key=lambda row: float(row["usefulness_dice_drop"]))[:10]
    high_low = []
    low_high = []
    for case_id in sorted({str(row["case_id"]) for row in rows}):
        group = [row for row in rows if str(row["case_id"]) == case_id]
        response = np.asarray([float(row["response_strength"]) for row in group])
        lower, upper = np.nanquantile(response, [0.25, 0.75])
        high_low.extend(
            row
            for row in group
            if float(row["response_strength"]) >= upper and float(row["usefulness_dice_drop"]) <= 0
        )
        low_high.extend(
            row
            for row in group
            if float(row["response_strength"]) <= lower and float(row["usefulness_dice_drop"]) > 0
        )
    multi = [row for row in rows if int(row.get("num_foci", 0)) > 1]
    single = [row for row in rows if int(row.get("num_foci", 0)) == 1]

    def mean_abs(group: list[dict[str, Any]]) -> float:
        if not group:
            return math.nan
        return float(np.mean([abs(float(row["usefulness_dice_drop"])) for row in group]))

    lines = [
        "# View reliability diagnostic",
        "",
        f"Checkpoint: `{checkpoint}`. Fixed threshold: "
        f"`{rows[0]['threshold'] if rows else 'n/a'}`.",
        "",
        "The active Phase-A density uses `shared_attention_weight`. "
        "`raw_source_separability` and candidate `phsa_weight` are exported diagnostics only; "
        "they do not affect the Phase-A density call.",
        "",
        "## Correlation with empirical leave-one-view usefulness",
        "",
        "| signal | global rho | within-case rho | within p-value | n | active |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for key, (rho, _pvalue, count) in correlations.items():
        within_rho, within_pvalue, within_count = within_case[key]
        assert count == within_count
        active = "yes" if key == "shared_attention_weight" else "no"
        lines.append(
            f"| {key} | {rho:.6f} | {within_rho:.6f} | {within_pvalue:.6g} | {count} | {active} |"
        )
    lines += [
        "",
        f"High-response but non-helpful views: **{len(high_low)}**. "
        f"Low-response but helpful views: **{len(low_high)}**.",
        f"Mean absolute Dice effect, multi-source={mean_abs(multi):.6f}, "
        f"single-source={mean_abs(single):.6f}.",
        "Within-case p-values treat views from one case as independent and are exploratory.",
        "",
        "## Top helpful views",
        "",
        "| case | view | foci | response | separability | PHSA weight | Dice drop |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in helpful:
        lines.append(
            f"| {row['case_id']} | {row['view_id']} | {row['num_foci']} | "
            f"{float(row['response_strength']):.6g} | "
            f"{float(row['raw_source_separability']):.6g} | "
            f"{float(row['phsa_weight']):.6g} | {float(row['usefulness_dice_drop']):.6f} |"
        )
    lines += [
        "",
        "## Top failure cases (view removal improves Dice)",
        "",
        "| case | view | foci | response | separability | PHSA weight | Dice drop |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in harmful:
        lines.append(
            f"| {row['case_id']} | {row['view_id']} | {row['num_foci']} | "
            f"{float(row['response_strength']):.6g} | "
            f"{float(row['raw_source_separability']):.6g} | "
            f"{float(row['phsa_weight']):.6g} | {float(row['usefulness_dice_drop']):.6f} |"
        )
    response_rho = within_case["response_strength"][0]
    separability_rho = within_case["raw_source_separability"][0]
    phsa_rho = within_case["phsa_weight"][0]
    lines += [
        "",
        "## Required questions",
        "",
        f"1. Surface response strength has partial agreement with usefulness "
        f"(within-case centered rho={response_rho:.3f}), but case-wise behavior is heterogeneous.",
        f"2. Raw source separability is not more consistent (rho={separability_rho:.3f}).",
        f"3. Candidate PHSA weight is not consistent with usefulness (rho={phsa_rho:.3f}).",
        f"4. There are {len(high_low)} within-case high-response/non-helpful views.",
        f"5. There are {len(low_high)} within-case low-response/helpful views.",
        "6. Absolute leave-one-view effects are larger in the two multi-source cases, but three "
        "cases cannot establish concentration by source-response mixing.",
        "7. Current PHSA/separability values neither enter final Phase-A density nor show better "
        "empirical alignment on this subset.",
        "",
        "## Decision",
        "",
    ]
    if not math.isfinite(separability_rho) or not math.isfinite(phsa_rho):
        lines.append(
            "Separability/PHSA evidence is unavailable or degenerate; the claim is unsupported."
        )
    elif separability_rho > response_rho and phsa_rho > response_rho:
        lines.append(
            "Separability-derived signals align better than surface response strength on this "
            "diagnostic subset, but they are not causal in the current Phase-A density path. "
            "This supports calibration experiments, not a current-method aggregation claim."
        )
    else:
        lines.append(
            "Separability-derived signals do not consistently align better than surface response "
            "strength. The current evidence does not support a source-separability-guided view "
            "aggregation contribution."
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "view_reliability_diagnostic.csv"
    if args.report_only:
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        write_report(rows, args.output_dir / "view_reliability_report.md", args.checkpoint)
        print(args.output_dir / "view_reliability_report.md")
        return
    cfg = OmegaConf.load(args.run_config)
    cfg.data.num_workers = 0
    cfg.data.persistent_workers = False
    cfg.data.val_max_samples = None
    assert_phase_a_contract(cfg)
    device = torch.device(args.device)
    net = load_net(cfg, args.checkpoint, device)
    dataset = FmtSimGenProjDataset(
        str(cfg.data.val_dir), config=cfg, split="val", is_training=False
    )
    sample_dirs = list(dataset.dirs)
    if args.sample_ids:
        wanted = set(args.sample_ids)
        sample_dirs = [path for path in sample_dirs if path.name in wanted]
    sample_dirs = sample_dirs[: args.max_cases]
    loader = FmtSimGenProjDataset(str(cfg.data.val_dir), config=cfg, split="all", is_training=False)
    voxel_size = float(cfg.data.voxel_size_mm)
    angles = list(cfg.data.view_angles)
    rows = []
    for case_index, sample_dir in enumerate(sample_dirs, start=1):
        gt = load_gt(sample_dir)
        shape = tuple(int(value) for value in gt.shape)
        tumor = json.loads((sample_dir / "tumor_params.json").read_text())
        num_foci = int(tumor.get("num_foci", 0))
        _, surface, _, depth_maps, _ = loader._load_projection(sample_dir)
        surface = surface.unsqueeze(0).to(device)
        depth_maps = depth_maps.unsqueeze(0).to(device)
        valid = torch.isfinite(depth_maps)
        indices = deterministic_proposal_indices(
            shape, args.proposal_count, args.proposal_seed, sample_dir.name
        )
        proposal = linear_indices_to_points_mm(indices, shape, voxel_size).to(device)
        cache, error = build_sample_cache(net, surface, proposal, valid, depth_maps, 1.0e-6)
        with frozen_sample_cache(net, cache), torch.inference_mode():
            diagnostic = call_net(
                net,
                surface,
                proposal[:, : min(1024, proposal.shape[1])],
                valid,
                depth_maps,
                return_diagnostics=True,
            )["diagnostics"]
        candidate_valid = diagnostic["candidate_valid_mask"]
        separability = mean_candidate_signal(diagnostic["separability"], candidate_valid)
        phsa = mean_candidate_signal(diagnostic["view_weights"], candidate_valid)
        shared_weight = (
            diagnostic["shared_view_weights"].detach().float()[0].mean(dim=1).cpu().numpy()
        )
        strength = response_strength(surface, valid)
        pred_all = predict_with_cache(
            net, cache, surface, valid, depth_maps, shape, voxel_size, args.chunk_size
        )
        metrics_all = metric_bundle(pred_all, gt, args, voxel_size)
        for view in range(surface.shape[1]):
            masked_valid = valid.clone()
            masked_valid[:, view] = False
            pred_minus = predict_with_cache(
                net, cache, surface, masked_valid, depth_maps, shape, voxel_size, args.chunk_size
            )
            metrics_minus = metric_bundle(pred_minus, gt, args, voxel_size)
            rows.append(
                {
                    "case_id": sample_dir.name,
                    "view_id": int(angles[view]) if view < len(angles) else view,
                    "view_index": view,
                    "num_foci": num_foci,
                    "threshold": args.threshold,
                    "dice_all": metrics_all["dice"],
                    "dice_minus_view": metrics_minus["dice"],
                    "usefulness_dice_drop": metrics_all["dice"] - metrics_minus["dice"],
                    "response_strength": strength[view],
                    "raw_source_separability": separability[view],
                    "phsa_weight": phsa[view],
                    "shared_attention_weight": float(shared_weight[view]),
                    "full_volume_precision_all": metrics_all["precision"],
                    "full_volume_recall_all": metrics_all["recall"],
                    "volume_error_all": metrics_all["volume_error"],
                    "component_recall_all": metrics_all["component_recall"],
                    "component_precision_all": metrics_all["component_precision"],
                    "merge_count_all": metrics_all["merge_count"],
                    "cache_equivalence_max_abs": error,
                }
            )
            print(
                f"{case_index}/{len(sample_dirs)} {sample_dir.name} view={angles[view]} "
                f"drop={rows[-1]['usefulness_dice_drop']:+.6f}"
            )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_report(rows, args.output_dir / "view_reliability_report.md", args.checkpoint)
    print(csv_path)


if __name__ == "__main__":
    main()
