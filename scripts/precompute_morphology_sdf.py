#!/usr/bin/env python
"""Precompute truncated normalized morphology SDF targets for FMT-SimGen samples."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import ndimage


def find_samples(data_dir: Path, limit: int | None) -> list[Path]:
    root = data_dir if any(data_dir.glob("sample_*/gt_voxels.npy")) else data_dir / "samples"
    samples = sorted(p for p in root.glob("sample_*") if (p / "gt_voxels.npy").exists())
    return samples[:limit] if limit else samples


def process_sample(
    sample_dir: Path,
    occupancy_threshold: float,
    truncation_distance_mm: float,
    voxel_spacing: tuple[float, float, float],
    overwrite: bool,
) -> dict[str, object]:
    out_dir = sample_dir / "morphology"
    out_path = out_dir / "sdf_target.npy"
    meta_path = out_dir / "sdf_meta.json"
    if out_path.exists() and meta_path.exists() and not overwrite:
        return {"sample_id": sample_dir.name, "status": "skipped"}
    gt = np.load(sample_dir / "gt_voxels.npy").astype(np.float32)
    gt_norm = gt / max(float(np.nanmax(gt)), 1.0e-8)
    mask = gt_norm > float(occupancy_threshold)
    spacing = tuple(float(v) for v in voxel_spacing)
    if mask.any():
        inside = ndimage.distance_transform_edt(mask, sampling=spacing)
        outside = ndimage.distance_transform_edt(~mask, sampling=spacing)
        sdf = inside - outside
    else:
        sdf = -np.full(gt.shape, truncation_distance_mm, dtype=np.float32)
    sdf = np.clip(sdf / max(float(truncation_distance_mm), 1e-6), -1.0, 1.0).astype(np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_path, sdf)
    meta = {
        "version": "ssq_candidate_conditioned_sdf_v2",
        "occupancy_threshold": float(occupancy_threshold),
        "truncation_distance_mm": float(truncation_distance_mm),
        "voxel_spacing": list(spacing),
        "sign_convention": "positive_inside_negative_outside",
        "source_normalization": "sample_peak",
        "target_shape": list(gt.shape),
        "source": "gt_voxels.npy",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return {
        "sample_id": sample_dir.name,
        "status": "written",
        "min": float(sdf.min()),
        "max": float(sdf.max()),
        "mean": float(sdf.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default="/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k"
    )
    parser.add_argument("--occupancy_threshold", type=float, default=0.5)
    parser.add_argument("--truncation_distance_mm", type=float, default=3.0)
    parser.add_argument("--voxel_spacing", nargs=3, type=float, default=[0.2, 0.2, 0.2])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    samples = find_samples(Path(args.data_dir), args.limit)
    common = (
        args.occupancy_threshold,
        args.truncation_distance_mm,
        tuple(args.voxel_spacing),
        args.overwrite,
    )
    if args.num_workers <= 1:
        for idx, sample_dir in enumerate(samples, 1):
            result = process_sample(sample_dir, *common)
            if idx == 1 or idx % 25 == 0 or idx == len(samples):
                print(f"[{idx}/{len(samples)}] {result}")
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = {
                ex.submit(process_sample, sample_dir, *common): sample_dir
                for sample_dir in samples
            }
            for idx, fut in enumerate(as_completed(futures), 1):
                result = fut.result()
                if idx == 1 or idx % 25 == 0 or idx == len(samples):
                    print(f"[{idx}/{len(samples)}] {result}")


if __name__ == "__main__":
    main()
