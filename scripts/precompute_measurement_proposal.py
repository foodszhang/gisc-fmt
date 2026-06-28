#!/usr/bin/env python
"""Precompute coarse measurement backprojection proposals for FMT-SimGen.

# NO-GT-LEAKAGE GUARANTEE:
# Reads ONLY:
#   - proj.npz (and view angle keys)
# Does NOT read:
#   - gt_voxels.npy
#   - gt_nodes.npy
#   - tumor_params.json
#   - body_mask
#
# The optional --gt_sanity branch reads gt_voxels.npy only for offline reporting.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.utils.fmt_simgen_projection import project_points_mm_to_detector  # noqa: E402
from minr_fmt.utils.ssq_candidate_extraction import (  # noqa: E402
    extract_candidate_anchors,
    write_candidate_cache,
)


def find_sample_dirs(data_dir: Path) -> list[Path]:
    direct = sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("sample_"))
    sample_root = data_dir if direct else data_dir / "samples"
    samples = sorted(
        p for p in sample_root.iterdir() if p.is_dir() and p.name.startswith("sample_")
    )
    return [p for p in samples if (p / "proj.npz").exists()]


def build_grid_points_mm(
    grid_size: tuple[int, int, int], trunk_size_mm: tuple[float, float, float]
):
    grid = np.stack(
        np.meshgrid(
            np.arange(grid_size[0], dtype=np.float32),
            np.arange(grid_size[1], dtype=np.float32),
            np.arange(grid_size[2], dtype=np.float32),
            indexing="ij",
        ),
        axis=-1,
    )
    cell_size_mm = np.asarray(trunk_size_mm, dtype=np.float32) / np.asarray(
        grid_size, dtype=np.float32
    )
    points_mm = (grid.reshape(-1, 3) + 0.5) * cell_size_mm
    return points_mm.astype(np.float32), cell_size_mm.astype(np.float32)


def normalize_projection(arr: np.ndarray, eps: float, clip_negative: bool) -> np.ndarray:
    proj = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if clip_negative:
        proj = np.clip(proj, 0.0, None)
    scale = max(float(np.abs(proj).max()), eps)
    return proj / scale


def normalize_projection_joint_percentile(
    projections: dict[int, np.ndarray],
    depth_maps: dict[int, np.ndarray],
    eps: float,
    percentile: float = 99.9,
) -> dict[int, np.ndarray]:
    vals = []
    for angle, proj in projections.items():
        valid = np.isfinite(depth_maps[angle])
        pos = np.clip(
            np.nan_to_num(proj.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0), 0.0, None
        )
        vals.append(pos[valid])
    merged = (
        np.concatenate([v.reshape(-1) for v in vals if v.size], axis=0)
        if vals
        else np.array([], dtype=np.float32)
    )
    scale = max(float(np.percentile(merged, percentile)), eps) if merged.size else 1.0
    return {
        angle: np.clip(
            np.nan_to_num(proj.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0),
            0.0,
            None,
        )
        / scale
        for angle, proj in projections.items()
    }


def compute_heatmap(sample_dir: Path, params: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    grid_size = tuple(int(v) for v in params["grid_size"])
    trunk_size_mm = tuple(float(v) for v in params["trunk_size_mm"])
    points_mm_np, cell_size_mm = build_grid_points_mm(grid_size, trunk_size_mm)
    points_mm = torch.from_numpy(points_mm_np).unsqueeze(0)

    views = [int(v) for v in params["views"]]
    heat_sum = torch.zeros(points_mm.shape[1], dtype=torch.float32)
    valid_count = torch.zeros(points_mm.shape[1], dtype=torch.float32)
    sampled_views = []
    valid_views = []

    z = np.load(sample_dir / "proj.npz")
    raw_proj: dict[int, np.ndarray] = {}
    depth_maps: dict[int, np.ndarray] = {}
    for angle in views:
        key = str(angle)
        if key not in z.files:
            raise KeyError(f"{sample_dir / 'proj.npz'} missing key {key!r}")
        raw_proj[angle] = z[key].astype(np.float32)
        depth_key = f"depth_{angle}"
        if depth_key in z.files:
            depth_maps[angle] = z[depth_key].astype(np.float32)
        else:
            depth_maps[angle] = np.full_like(raw_proj[angle], np.inf, dtype=np.float32)

    if params["normalization"] == "joint_sample_percentile_99.9":
        normalized = normalize_projection_joint_percentile(
            raw_proj, depth_maps, params["eps"], 99.9
        )
    else:
        normalized = {
            angle: normalize_projection(raw_proj[angle], params["eps"], params["clip_negative"])
            for angle in views
        }

    for angle in views:
        proj = normalized[angle]
        image = torch.from_numpy(proj).view(1, 1, proj.shape[0], proj.shape[1])

        grid, _depth, valid, _uv_px, _uv_phys = project_points_mm_to_detector(
            points_mm,
            angle,
            camera_distance_mm=params["camera_distance_mm"],
            fov_mm=params["fov_mm"],
            detector_resolution=tuple(params["detector_resolution"]),
            volume_center_world=tuple(params["volume_center_world"]),
            align_corners=True,
        )
        sampled = F.grid_sample(
            image,
            grid.unsqueeze(1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).view(-1)
        depth_valid_image = torch.from_numpy(
            np.isfinite(depth_maps[angle]).astype(np.float32)
        ).view(1, 1, proj.shape[0], proj.shape[1])
        sampled_depth_valid = F.grid_sample(
            depth_valid_image,
            grid.unsqueeze(1),
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        ).view(-1)
        valid_f = valid.view(-1).to(torch.float32)
        valid_f = valid_f * sampled_depth_valid
        heat_sum += sampled * valid_f
        valid_count += valid_f
        sampled_views.append(sampled.clamp_min(0.0))
        valid_views.append(valid_f)

    c_prop = (valid_count / max(float(len(views)), 1.0)).clamp(0.0, 1.0)
    fusion_mode = str(params.get("view_fusion", "arithmetic_mean"))
    if fusion_mode == "arithmetic_mean":
        fused = heat_sum / torch.clamp(valid_count, min=1.0)
    else:
        values = torch.stack(sampled_views, dim=0)
        validity = torch.stack(valid_views, dim=0)
        eps = float(params["eps"])
        if fusion_mode == "geometric_mean":
            fused = torch.exp(
                (torch.log(values.clamp_min(eps)) * validity).sum(dim=0)
                / valid_count.clamp_min(1.0)
            )
        elif fusion_mode == "harmonic_mean":
            fused = valid_count / (
                (validity / values.clamp_min(eps)).sum(dim=0).clamp_min(eps)
            )
        else:
            raise ValueError(f"Unsupported proposal view_fusion={fusion_mode!r}")
    heat = (c_prop.pow(float(params["coverage_gamma"])) * fused).numpy()
    heat = np.clip(heat.reshape(grid_size).astype(np.float32), 0.0, None)
    gamma = float(params["gamma"])
    if gamma != 1.0:
        heat = np.power(heat, gamma, dtype=np.float32)
    blur_sigma = float(params["blur_sigma"])
    if blur_sigma > 0:
        heat = gaussian_filter(heat, sigma=blur_sigma).astype(np.float32)
    heat = np.clip(heat, 0.0, None)
    max_val = float(heat.max())
    if max_val > params["eps"]:
        heat = heat / max_val

    meta = {
        "grid_size": list(grid_size),
        "axis_order": "grid[i,j,k] == (x_mm,y_mm,z_mm)",
        "trunk_size_mm": list(trunk_size_mm),
        "cell_size_mm": [float(v) for v in cell_size_mm],
        "voxel_size_mm": float(params["voxel_size_mm"]),
        "gamma": gamma,
        "blur_sigma": blur_sigma,
        "proposal_version": "ssq_joint_percentile_v2"
        if params["normalization"] == "joint_sample_percentile_99.9"
        else "legacy_per_view_max",
        "normalization": params["normalization"],
        "view_fusion": fusion_mode,
        "percentile": 99.9 if params["normalization"] == "joint_sample_percentile_99.9" else None,
        "views": views,
        "camera_distance_mm": float(params["camera_distance_mm"]),
        "fov_mm": float(params["fov_mm"]),
        "detector_resolution": [int(v) for v in params["detector_resolution"]],
        "volume_center_world": [float(v) for v in params["volume_center_world"]],
        "uses_body_mask": False,
        "uses_gt": False,
    }
    return heat.astype(np.float32), meta


def process_sample(sample_dir: Path, params: dict[str, Any]) -> dict[str, Any]:
    out_dir = sample_dir / "proposal"
    out_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = out_dir / "meas_backproj_heatmap.npy"
    meta_path = out_dir / "meas_backproj_meta.json"
    cand_path = out_dir / "candidate_anchors.npz"
    if (
        heatmap_path.exists()
        and meta_path.exists()
        and cand_path.exists()
        and not params["overwrite"]
    ):
        try:
            heat = np.load(heatmap_path)
            return {
                "sample_id": sample_dir.name,
                "status": "skipped",
                "min": float(heat.min()),
                "max": float(heat.max()),
                "mean": float(heat.mean()),
            }
        except Exception as exc:
            print(f"[WARN] Recomputing unreadable proposal for {sample_dir.name}: {exc}")

    heat, meta = compute_heatmap(sample_dir, params)
    anchors = extract_candidate_anchors(
        heat,
        meta,
        top_m=int(params["candidate_top_m"]),
        smoothing_sigma_mm=params.get("candidate_smoothing_sigma_mm"),
        smoothing_sigma_cells=params.get("candidate_smoothing_sigma_cells"),
        nms_radius_mm=float(params["candidate_nms_radius_mm"]),
        support_moment_radius_mm=float(params["candidate_support_moment_radius_mm"]),
        min_value_ratio=float(params["candidate_min_value_ratio"]),
    )
    candidate_meta = {
        **meta,
        "smoothing_sigma_mm": params.get("candidate_smoothing_sigma_mm"),
        "smoothing_sigma_cells": params.get("candidate_smoothing_sigma_cells"),
        "nms_radius_mm": float(params["candidate_nms_radius_mm"]),
        "support_moment_radius_mm": float(params["candidate_support_moment_radius_mm"]),
        "min_value_ratio": float(params["candidate_min_value_ratio"]),
        "top_m": int(params["candidate_top_m"]),
    }
    write_candidate_cache(sample_dir, anchors, candidate_meta)
    tmp_heatmap = heatmap_path.with_suffix(".tmp.npy")
    np.save(tmp_heatmap, heat)
    tmp_heatmap.replace(heatmap_path)
    tmp_meta = meta_path.with_suffix(".tmp.json")
    tmp_meta.write_text(json.dumps(meta, indent=2))
    tmp_meta.replace(meta_path)
    return {
        "sample_id": sample_dir.name,
        "status": "written",
        "min": float(heat.min()),
        "max": float(heat.max()),
        "mean": float(heat.mean()),
    }


def proposal_gt_sanity(sample_dirs: list[Path], voxel_size_mm: float, max_samples: int = 8) -> None:
    print("[INFO] GT sanity is for offline reporting only. Disable in training/eval.")
    center_hit_ratios = []
    gt_coverage_ratios = []
    for sample_dir in sample_dirs[:max_samples]:
        heatmap_path = sample_dir / "proposal" / "meas_backproj_heatmap.npy"
        meta_path = sample_dir / "proposal" / "meas_backproj_meta.json"
        gt_path = sample_dir / "gt_voxels.npy"
        if not heatmap_path.exists() or not meta_path.exists() or not gt_path.exists():
            continue
        heat = np.load(heatmap_path)
        meta = json.loads(meta_path.read_text())
        gt = np.load(gt_path) > 0
        flat = heat.reshape(-1)
        k = max(1, int(np.ceil(flat.size * 0.05)))
        top_idx = np.argpartition(flat, -k)[-k:]
        top_mask = np.zeros(flat.size, dtype=bool)
        top_mask[top_idx] = True
        grid_size = tuple(int(v) for v in meta["grid_size"])
        cell_size = np.asarray(meta["cell_size_mm"], dtype=np.float32)
        cell_ijk = np.stack(np.unravel_index(top_idx, grid_size), axis=-1).astype(np.float32)
        points_mm = (cell_ijk + 0.5) * cell_size
        ijk = np.floor(points_mm / voxel_size_mm).astype(np.int64)
        ijk[:, 0] = np.clip(ijk[:, 0], 0, gt.shape[0] - 1)
        ijk[:, 1] = np.clip(ijk[:, 1], 0, gt.shape[1] - 1)
        ijk[:, 2] = np.clip(ijk[:, 2], 0, gt.shape[2] - 1)
        center_hit_ratio = float(gt[ijk[:, 0], ijk[:, 1], ijk[:, 2]].mean())
        gt_idx = np.argwhere(gt)
        if len(gt_idx):
            gt_mm = (gt_idx.astype(np.float32) + 0.5) * voxel_size_mm
            gt_cell = np.floor(gt_mm / cell_size).astype(np.int64)
            grid_size_arr = np.asarray(grid_size, dtype=np.int64)
            gt_cell = np.clip(gt_cell, 0, grid_size_arr - 1)
            gt_lin = np.ravel_multi_index(gt_cell.T, grid_size)
            gt_coverage = float(top_mask[gt_lin].mean())
        else:
            gt_coverage = 0.0
        argmax_idx = np.asarray(np.unravel_index(int(flat.argmax()), grid_size), dtype=np.float32)
        argmax_mm = (argmax_idx + 0.5) * cell_size
        print(
            f"{sample_dir.name}: heat min/max/mean="
            f"{heat.min():.6g}/{heat.max():.6g}/{heat.mean():.6g} "
            f"argmax_mm={argmax_mm.tolist()} "
            f"top5_cell_center_hit={center_hit_ratio:.4f} "
            f"top5_gt_coverage={gt_coverage:.4f}"
        )
        center_hit_ratios.append(center_hit_ratio)
        gt_coverage_ratios.append(gt_coverage)
    if center_hit_ratios:
        print(
            "top-5% cell-center hit ratio "
            f"mean={np.mean(center_hit_ratios):.4f} std={np.std(center_hit_ratios):.4f}"
        )
        print(
            "top-5% GT foreground coverage "
            f"mean={np.mean(gt_coverage_ratios):.4f} std={np.std(gt_coverage_ratios):.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--grid_size", nargs=3, type=int, default=[50, 60, 50])
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--blur_sigma", type=float, default=1.0)
    parser.add_argument("--candidate_top_m", type=int, default=5)
    parser.add_argument("--candidate_smoothing_sigma_cells", type=float, default=1.0)
    parser.add_argument("--candidate_smoothing_sigma_mm", type=float)
    parser.add_argument("--candidate_nms_radius_mm", type=float, default=3.0)
    parser.add_argument("--candidate_support_moment_radius_mm", type=float, default=4.0)
    parser.add_argument("--candidate_min_value_ratio", type=float, default=0.1)
    parser.add_argument("--views", nargs="+", type=int, default=[-90, -60, -30, 0, 30, 60, 90])
    parser.add_argument("--camera_distance", type=float, default=200.0)
    parser.add_argument("--fov_mm", type=float, default=80.0)
    parser.add_argument("--detector_resolution", nargs=2, type=int, default=[256, 256])
    parser.add_argument("--volume_center_world", nargs=3, type=float, default=[19.0, 20.0, 10.4])
    parser.add_argument("--voxel_size_mm", type=float, default=0.2)
    parser.add_argument("--trunk_size_mm", nargs=3, type=float, default=[38.0, 40.0, 20.8])
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--only_sample")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gt_sanity", action="store_true")
    parser.add_argument(
        "--normalization",
        choices=["joint_sample_percentile_99.9", "per_view_max"],
        default="joint_sample_percentile_99.9",
    )
    parser.add_argument("--coverage_gamma", type=float, default=1.0)
    parser.add_argument(
        "--view_fusion",
        choices=["arithmetic_mean", "geometric_mean", "harmonic_mean"],
        default="arithmetic_mean",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    sample_dirs = find_sample_dirs(data_dir)
    if args.only_sample:
        sample_dirs = [p for p in sample_dirs if p.name == args.only_sample]
    if args.limit:
        sample_dirs = sample_dirs[: args.limit]
    if not sample_dirs:
        raise FileNotFoundError(f"No sample_XXXX/proj.npz found under {data_dir}")

    params = {
        "grid_size": args.grid_size,
        "gamma": args.gamma,
        "blur_sigma": args.blur_sigma,
        "views": args.views,
        "camera_distance_mm": args.camera_distance,
        "fov_mm": args.fov_mm,
        "detector_resolution": args.detector_resolution,
        "volume_center_world": args.volume_center_world,
        "voxel_size_mm": args.voxel_size_mm,
        "trunk_size_mm": args.trunk_size_mm,
        "eps": 1.0e-6,
        "clip_negative": True,
        "overwrite": args.overwrite,
        "normalization": args.normalization,
        "coverage_gamma": args.coverage_gamma,
        "view_fusion": args.view_fusion,
        "candidate_top_m": args.candidate_top_m,
        "candidate_smoothing_sigma_cells": args.candidate_smoothing_sigma_cells,
        "candidate_smoothing_sigma_mm": args.candidate_smoothing_sigma_mm,
        "candidate_nms_radius_mm": args.candidate_nms_radius_mm,
        "candidate_support_moment_radius_mm": args.candidate_support_moment_radius_mm,
        "candidate_min_value_ratio": args.candidate_min_value_ratio,
    }

    t0 = time.perf_counter()
    print(f"precomputing proposals: samples={len(sample_dirs)} workers={args.num_workers}")
    results = []
    if args.num_workers <= 1:
        for i, sample_dir in enumerate(sample_dirs, 1):
            result = process_sample(sample_dir, params)
            results.append(result)
            if i == 1 or i % 25 == 0 or i == len(sample_dirs):
                print(f"[{i}/{len(sample_dirs)}] {result}")
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            future_to_sample = {ex.submit(process_sample, p, params): p for p in sample_dirs}
            for i, fut in enumerate(as_completed(future_to_sample), 1):
                result = fut.result()
                results.append(result)
                if i == 1 or i % 25 == 0 or i == len(sample_dirs):
                    print(f"[{i}/{len(sample_dirs)}] {result}")

    dt = time.perf_counter() - t0
    written = sum(1 for r in results if r["status"] == "written")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    print(
        f"done: written={written} skipped={skipped} total_time={dt:.2f}s "
        f"per_sample={dt / max(len(sample_dirs), 1):.3f}s"
    )
    if args.gt_sanity:
        proposal_gt_sanity(sample_dirs, args.voxel_size_mm)


if __name__ == "__main__":
    main()
