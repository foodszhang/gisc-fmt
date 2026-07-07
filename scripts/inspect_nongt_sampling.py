#!/usr/bin/env python
"""Inspect Non-GT query sampling behavior.

GT reads in this script are reporting-only. Training/eval allocation must not use them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.dataset.query_sampler import NonGTQuerySampler  # noqa: E402


def find_sample_dirs(data_dir: Path) -> list[Path]:
    direct = sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("sample_"))
    sample_root = data_dir if direct else data_dir / "samples"
    return sorted(
        p
        for p in sample_root.iterdir()
        if p.is_dir() and p.name.startswith("sample_") and (p / "gt_voxels.npy").exists()
    )


def base_cfg(num_queries: int, trunk_ratio: float, proposal_ratio: float) -> dict:
    return {
        "num_queries": num_queries,
        "voxel_size_mm": 0.2,
        "trunk_size_mm": [38.0, 40.0, 20.8],
        "proposal_subdir": "proposal",
        "proposal_filename": "meas_backproj_heatmap.npy",
        "proposal_meta_filename": "meas_backproj_meta.json",
        "query_sampling": {
            "type": "nongt_mixed",
            "num_queries": num_queries,
            "trunk_uniform_ratio": trunk_ratio,
            "meas_proposal_ratio": proposal_ratio,
            "use_gt_bbox": False,
            "use_gt_foreground_oversampling": False,
            "use_body_mask": False,
            "use_depth_silhouette_mask": False,
        },
    }


def old_uniform_sample(
    gt_shape: tuple[int, int, int], n_queries: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = int(np.prod(gt_shape))
    choice = rng.choice(total, size=n_queries, replace=n_queries > total)
    ix, iy, iz = np.unravel_index(choice, gt_shape)
    return ix.astype(np.int64), iy.astype(np.int64), iz.astype(np.int64)


def foreground_ratio(gt: np.ndarray, ix: np.ndarray, iy: np.ndarray, iz: np.ndarray) -> float:
    return float((gt[ix, iy, iz] > 0).mean())


def summarize(name: str, values: list[float]) -> None:
    arr = np.asarray(values, dtype=np.float64)
    print(f"{name:18s} foreground ratio mean={arr.mean():.6f} std={arr.std():.6f}")


def proposal_gt_sanity(sample_dirs: list[Path], voxel_size_mm: float) -> None:
    print("[INFO] GT sanity is for offline reporting only. Disable in training/eval.")
    top5_center_hit_ratios = []
    top5_gt_coverage_ratios = []
    all_zero = 0
    for sample_dir in sample_dirs:
        heatmap_path = sample_dir / "proposal" / "meas_backproj_heatmap.npy"
        meta_path = sample_dir / "proposal" / "meas_backproj_meta.json"
        if not heatmap_path.exists() or not meta_path.exists():
            print(f"{sample_dir.name}: proposal missing")
            continue
        heat = np.load(heatmap_path)
        if float(np.abs(heat).sum()) <= 0:
            all_zero += 1
        meta = json.loads(meta_path.read_text())
        gt = np.load(sample_dir / "gt_voxels.npy") > 0

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

        argmax_idx = np.asarray(np.unravel_index(int(flat.argmax()), grid_size), dtype=np.float32)
        argmax_mm = (argmax_idx + 0.5) * cell_size
        gt_idx = np.argwhere(gt)
        if len(gt_idx):
            gt_centroid_mm = ((gt_idx.astype(np.float32).mean(axis=0)) + 0.5) * voxel_size_mm
            centroid_msg = gt_centroid_mm.tolist()
            gt_mm = (gt_idx.astype(np.float32) + 0.5) * voxel_size_mm
            gt_cell = np.floor(gt_mm / cell_size).astype(np.int64)
            grid_size_arr = np.asarray(grid_size, dtype=np.int64)
            gt_cell = np.clip(gt_cell, 0, grid_size_arr - 1)
            gt_lin = np.ravel_multi_index(gt_cell.T, grid_size)
            gt_coverage = float(top_mask[gt_lin].mean())
        else:
            centroid_msg = None
            gt_coverage = 0.0
        top5_center_hit_ratios.append(center_hit_ratio)
        top5_gt_coverage_ratios.append(gt_coverage)
        print(
            f"{sample_dir.name}: heat min/max/mean="
            f"{heat.min():.6g}/{heat.max():.6g}/{heat.mean():.6g} "
            f"argmax_mm={argmax_mm.tolist()} gt_centroid_mm={centroid_msg} "
            f"top5_cell_center_hit={center_hit_ratio:.4f} "
            f"top5_gt_coverage={gt_coverage:.4f}"
        )

    if top5_center_hit_ratios:
        center_arr = np.asarray(top5_center_hit_ratios)
        coverage_arr = np.asarray(top5_gt_coverage_ratios)
        print(
            "proposal top-5% cell-center hit ratio "
            f"mean={center_arr.mean():.4f} std={center_arr.std():.4f}"
        )
        print(
            "proposal top-5% GT foreground coverage "
            f"mean={coverage_arr.mean():.4f} std={coverage_arr.std():.4f}"
        )
    print(f"proposal all-zero samples: {all_zero}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--num_queries", type=int, default=16384)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--gt_sanity", action="store_true")
    args = parser.parse_args()

    sample_dirs = find_sample_dirs(Path(args.data_dir).expanduser())[: args.num_samples]
    if not sample_dirs:
        raise FileNotFoundError(f"No FMT-SimGen sample dirs found under {args.data_dir}")

    trunk_sampler = NonGTQuerySampler(base_cfg(args.num_queries, 1.0, 0.0))
    mixed_sampler = NonGTQuerySampler(base_cfg(args.num_queries, 0.5, 0.5))
    uniform_ratios, trunk_ratios, mixed_ratios = [], [], []
    proposal_src_ratios = []

    for i, sample_dir in enumerate(sample_dirs):
        rng = np.random.default_rng(i)
        gt = np.load(sample_dir / "gt_voxels.npy").astype(np.float32)
        gt = np.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0)
        gt = np.clip(gt, 0.0, None)

        ix, iy, iz = old_uniform_sample(gt.shape, args.num_queries, rng)
        uniform_ratios.append(foreground_ratio(gt, ix, iy, iz))

        trunk = trunk_sampler.sample(
            sample_dir, gt.shape, args.num_queries, np.random.default_rng(i)
        )
        trunk_ratios.append(foreground_ratio(gt, trunk["ix"], trunk["iy"], trunk["iz"]))

        mixed = mixed_sampler.sample(
            sample_dir, gt.shape, args.num_queries, np.random.default_rng(1000 + i)
        )
        mixed_ratios.append(foreground_ratio(gt, mixed["ix"], mixed["iy"], mixed["iz"]))
        proposal_src_ratios.append(float((mixed["src_tag"] == 1).mean()))

    print(f"num samples inspected: {len(sample_dirs)}")
    summarize("uniform voxel", uniform_ratios)
    summarize("trunk-uniform", trunk_ratios)
    summarize("proposal mixed", mixed_ratios)
    print(
        "proposal mixed src_tag proposal ratio "
        f"mean={np.mean(proposal_src_ratios):.4f} std={np.std(proposal_src_ratios):.4f}"
    )

    if args.gt_sanity:
        proposal_gt_sanity(sample_dirs, voxel_size_mm=0.2)


if __name__ == "__main__":
    main()
