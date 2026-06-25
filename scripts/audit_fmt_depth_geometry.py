#!/usr/bin/env python
"""Audit FMT-SimGen depth-map semantics for SSQ detector-side path proxy."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.network.ssq_geometry import sample_finite_scalar_map  # noqa: E402
from minr_fmt.utils.fmt_simgen_projection import project_points_mm_to_detector  # noqa: E402


def find_samples(data_dir: Path, limit: int) -> list[Path]:
    root = data_dir if any(data_dir.glob("sample_*/proj.npz")) else data_dir / "samples"
    return sorted(p for p in root.glob("sample_*") if (p / "proj.npz").exists())[:limit]


def sample_depth(depth: np.ndarray, grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    image = torch.from_numpy(depth.astype(np.float32)).view(1, 1, *depth.shape)
    surf, valid = sample_finite_scalar_map(image, grid[:, None], align_corners=True)
    return surf.view(-1), valid.view(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default="/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k"
    )
    parser.add_argument("--out_csv", default="outputs/ssq_fmt_rebuild/depth_geometry_audit.csv")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--views", nargs="+", type=int, default=[-90, -60, -30, 0, 30, 60, 90])
    parser.add_argument("--camera_distance", type=float, default=200.0)
    parser.add_argument("--fov_mm", type=float, default=80.0)
    parser.add_argument("--detector_resolution", nargs=2, type=int, default=[256, 256])
    parser.add_argument("--volume_center_world", nargs=3, type=float, default=[19.0, 20.0, 10.4])
    args = parser.parse_args()

    rows = []
    path_q_minus_s: list[float] = []
    path_s_minus_q: list[float] = []
    center = torch.tensor([[[19.0, 20.0, 10.4], [19.0, 20.0, 16.0], [19.0, 20.0, 4.0]]])
    for sample_dir in find_samples(Path(args.data_dir), args.limit):
        z = np.load(sample_dir / "proj.npz")
        for angle in args.views:
            key = f"depth_{angle}"
            if key not in z.files:
                continue
            depth = z[key].astype(np.float32)
            finite = np.isfinite(depth)
            grid, q_depth, valid, _px, _phys = project_points_mm_to_detector(
                center,
                angle,
                camera_distance_mm=args.camera_distance,
                fov_mm=args.fov_mm,
                detector_resolution=tuple(args.detector_resolution),
                volume_center_world=tuple(args.volume_center_world),
                align_corners=True,
            )
            surf, surf_valid = sample_depth(depth, grid)
            for idx, label in enumerate(["center", "deep_z16", "shallow_z4"]):
                q_minus_s = float(q_depth.view(-1)[idx] - surf[idx])
                s_minus_q = float(surf[idx] - q_depth.view(-1)[idx])
                if bool(surf_valid[idx]):
                    path_q_minus_s.append(max(q_minus_s, 0.0))
                    path_s_minus_q.append(max(s_minus_q, 0.0))
                rows.append(
                    {
                        "sample_id": sample_dir.name,
                        "view": angle,
                        "query": label,
                        "finite_depth_min": float(np.nanmin(depth[finite]))
                        if finite.any()
                        else np.nan,
                        "finite_depth_max": float(np.nanmax(depth[finite]))
                        if finite.any()
                        else np.nan,
                        "finite_depth_mean": float(np.nanmean(depth[finite]))
                        if finite.any()
                        else np.nan,
                        "query_camera_depth": float(q_depth.view(-1)[idx]),
                        "sampled_surface_depth": float(surf[idx]),
                        "query_minus_surface": q_minus_s,
                        "surface_minus_query": s_minus_q,
                        "projection_valid": bool(valid.view(-1)[idx]),
                        "surface_depth_finite": bool(surf_valid[idx]),
                    }
                )

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["empty"])
        writer.writeheader()
        writer.writerows(rows)
    for name, values in (
        ("query_minus_surface", path_q_minus_s),
        ("surface_minus_query", path_s_minus_q),
    ):
        arr = np.asarray(values, dtype=np.float32)
        if arr.size:
            print(
                f"{name} clipped path mm: "
                f"P50={np.percentile(arr, 50):.4f} "
                f"P90={np.percentile(arr, 90):.4f} "
                f"P95={np.percentile(arr, 95):.4f} "
                f"P99={np.percentile(arr, 99):.4f}"
            )
    print(f"wrote {out} rows={len(rows)}")


if __name__ == "__main__":
    main()
