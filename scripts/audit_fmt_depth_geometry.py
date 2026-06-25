#!/usr/bin/env python
"""Audit FMT-SimGen depth-map semantics for SSQ detector-side path proxy."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.utils.fmt_simgen_projection import project_points_mm_to_detector  # noqa: E402


def find_samples(data_dir: Path, limit: int) -> list[Path]:
    root = data_dir if any(data_dir.glob("sample_*/proj.npz")) else data_dir / "samples"
    return sorted(p for p in root.glob("sample_*") if (p / "proj.npz").exists())[:limit]


def sample_depth(depth: np.ndarray, grid: torch.Tensor) -> torch.Tensor:
    image = torch.from_numpy(depth.astype(np.float32)).view(1, 1, *depth.shape)
    return F.grid_sample(
        image, grid.unsqueeze(1), mode="bilinear", padding_mode="zeros", align_corners=True
    ).view(-1)


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
            surf = sample_depth(depth, grid)
            for idx, label in enumerate(["center", "deep_z16", "shallow_z4"]):
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
                        "query_minus_surface": float(q_depth.view(-1)[idx] - surf[idx]),
                        "surface_minus_query": float(surf[idx] - q_depth.view(-1)[idx]),
                        "projection_valid": bool(valid.view(-1)[idx]),
                        "surface_depth_finite": bool(torch.isfinite(surf[idx])),
                    }
                )

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["empty"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} rows={len(rows)}")


if __name__ == "__main__":
    main()
