#!/usr/bin/env python
"""Precompute center and distance auxiliary targets for FMT-SimGen samples."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage


def find_sample_dirs(data_dir: Path) -> list[Path]:
    direct = sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("sample_"))
    sample_root = data_dir if direct else data_dir / "samples"
    if not sample_root.exists():
        raise FileNotFoundError(f"Sample root does not exist: {sample_root}")
    return sorted(
        p
        for p in sample_root.iterdir()
        if p.is_dir() and p.name.startswith("sample_") and (p / "gt_voxels.npy").exists()
    )


def build_center_distance_targets(
    gt: np.ndarray, voxel_size_mm: float, center_sigma_mm: float
) -> dict[str, np.ndarray]:
    structure = ndimage.generate_binary_structure(3, 1)
    labeled, num = ndimage.label(gt > 0.0, structure=structure)
    center_target = np.zeros_like(gt, dtype=np.float32)
    distance_target = np.zeros_like(gt, dtype=np.float32)
    fg_mask = np.zeros_like(gt, dtype=np.float32)
    if num <= 0:
        return {
            "center_target": center_target,
            "distance_target": distance_target,
            "fg_mask": fg_mask,
        }

    xs = np.arange(gt.shape[0], dtype=np.float32)[:, None, None]
    ys = np.arange(gt.shape[1], dtype=np.float32)[None, :, None]
    zs = np.arange(gt.shape[2], dtype=np.float32)[None, None, :]
    sigma_vox = max(float(center_sigma_mm) / max(float(voxel_size_mm), 1e-6), 1e-6)

    for label_id in range(1, num + 1):
        mask = labeled == label_id
        if not mask.any():
            continue
        coords = np.argwhere(mask).astype(np.float32)
        centroid = coords.mean(axis=0)
        dist_sq = (xs - centroid[0]) ** 2 + (ys - centroid[1]) ** 2 + (zs - centroid[2]) ** 2
        center = np.exp(-dist_sq / (2.0 * sigma_vox**2)).astype(np.float32)
        center_target = np.maximum(center_target, center)

        dist_map = ndimage.distance_transform_edt(mask).astype(np.float32)
        radius = max(float(dist_map[mask].max()), 1.0)
        fg = mask.astype(np.float32)
        fg_mask = np.maximum(fg_mask, fg)
        distance = np.where(mask, dist_map / radius, 0.0).astype(np.float32)
        distance_target = np.maximum(distance_target, distance)

    return {
        "center_target": center_target,
        "distance_target": distance_target,
        "fg_mask": fg_mask,
    }


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    np.save(tmp_path, array)
    saved_path = tmp_path.with_suffix(tmp_path.suffix + ".npy")
    saved_path.replace(path)


def process_sample(sample_dir: Path, params: dict[str, Any]) -> dict[str, Any]:
    out_dir = sample_dir / params["subdir"]
    paths = {
        "center_target": out_dir / params["center_filename"],
        "distance_target": out_dir / params["distance_filename"],
        "fg_mask": out_dir / params["fg_filename"],
    }
    meta_path = out_dir / "meta.json"
    if (
        all(path.exists() for path in paths.values())
        and meta_path.exists()
        and not params["overwrite"]
    ):
        return {"sample_id": sample_dir.name, "status": "skipped"}

    gt = np.load(sample_dir / "gt_voxels.npy").astype(np.float32)
    gt = np.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0)
    gt = np.clip(gt, 0.0, None)
    gt_max = float(gt.max())
    if gt_max > 0:
        gt = gt / gt_max

    targets = build_center_distance_targets(
        gt,
        voxel_size_mm=float(params["voxel_size_mm"]),
        center_sigma_mm=float(params["center_sigma_mm"]),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for key, path in paths.items():
        atomic_save_npy(path, targets[key])

    meta = {
        "sample_id": sample_dir.name,
        "shape": [int(v) for v in gt.shape],
        "voxel_size_mm": float(params["voxel_size_mm"]),
        "center_sigma_mm": float(params["center_sigma_mm"]),
        "files": {key: path.name for key, path in paths.items()},
        "source": "gt_voxels.npy",
    }
    tmp_meta = meta_path.with_name(".meta.json.tmp")
    tmp_meta.write_text(json.dumps(meta, indent=2))
    tmp_meta.replace(meta_path)

    return {
        "sample_id": sample_dir.name,
        "status": "written",
        "fg_voxels": int(targets["fg_mask"].sum()),
        "center_max": float(targets["center_target"].max()),
        "distance_max": float(targets["distance_target"].max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k"),
    )
    parser.add_argument("--subdir", default="center_distance")
    parser.add_argument("--center_filename", default="center_target.npy")
    parser.add_argument("--distance_filename", default="distance_target.npy")
    parser.add_argument("--fg_filename", default="fg_mask.npy")
    parser.add_argument("--voxel_size_mm", type=float, default=0.2)
    parser.add_argument("--center_sigma_mm", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_dirs = find_sample_dirs(args.data_dir)
    if args.max_samples is not None:
        sample_dirs = sample_dirs[: int(args.max_samples)]
    params = {
        "subdir": args.subdir,
        "center_filename": args.center_filename,
        "distance_filename": args.distance_filename,
        "fg_filename": args.fg_filename,
        "voxel_size_mm": args.voxel_size_mm,
        "center_sigma_mm": args.center_sigma_mm,
        "overwrite": args.overwrite,
    }

    print(
        "precomputing center-distance targets: "
        f"samples={len(sample_dirs)} workers={args.num_workers}"
    )
    start = time.time()
    counts: dict[str, int] = {}
    if args.num_workers <= 1:
        for sample_dir in sample_dirs:
            result = process_sample(sample_dir, params)
            counts[result["status"]] = counts.get(result["status"], 0) + 1
            print(result)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = {
                ex.submit(process_sample, sample_dir, params): sample_dir
                for sample_dir in sample_dirs
            }
            for future in as_completed(futures):
                result = future.result()
                counts[result["status"]] = counts.get(result["status"], 0) + 1
                print(result)
    elapsed = time.time() - start
    print(f"done in {elapsed:.1f}s counts={counts}")


if __name__ == "__main__":
    main()
