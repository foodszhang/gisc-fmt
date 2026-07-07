#!/usr/bin/env python
"""Evaluate measurement-derived SSQ candidate coverage against GT components."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.utils.ssq_candidate_extraction import load_candidate_cache  # noqa: E402


def find_samples(data_dir: Path, limit: int | None) -> list[Path]:
    root = data_dir if any(data_dir.glob("sample_*/gt_voxels.npy")) else data_dir / "samples"
    samples = sorted(p for p in root.glob("sample_*") if (p / "gt_voxels.npy").exists())
    return samples[:limit] if limit else samples


def load_candidates(sample_dir: Path) -> np.ndarray:
    cache = load_candidate_cache(sample_dir)
    return cache["centers_mm"][cache["valid"]].astype(np.float32)


def component_centers(gt: np.ndarray, voxel_size_mm: float) -> np.ndarray:
    gt_norm = gt / max(float(np.nanmax(gt)), 1.0e-8)
    labels, count = ndimage.label(gt_norm > 0.5)
    centers = []
    for idx in range(1, count + 1):
        coords = np.argwhere(labels == idx)
        if len(coords):
            centers.append((coords.astype(np.float32).mean(axis=0) + 0.5) * voxel_size_mm)
    return np.asarray(centers, dtype=np.float32)


def process_sample(sample_dir: Path, voxel_size_mm: float) -> dict[str, object]:
    gt = np.load(sample_dir / "gt_voxels.npy").astype(np.float32)
    comp = component_centers(gt, voxel_size_mm)
    cand = load_candidates(sample_dir)
    if len(comp) and len(cand):
        dist = np.linalg.norm(comp[:, None] - cand[None], axis=-1)
        min_comp = dist.min(axis=1)
        min_cand = dist.min(axis=0)
        row_ind, col_ind = linear_sum_assignment(dist)
        matched_dist = dist[row_ind, col_ind]
    else:
        min_comp = np.full(len(comp), np.inf, dtype=np.float32)
        min_cand = np.full(len(cand), np.inf, dtype=np.float32)
        matched_dist = np.asarray([], dtype=np.float32)
    matched3 = matched_dist <= 3.0
    one_to_one_matches_3mm = int(matched3.sum())
    duplicate_count = int(max((min_cand <= 3.0).sum() - one_to_one_matches_3mm, 0))
    tumor_path = sample_dir / "tumor_params.json"
    tumor = json.loads(tumor_path.read_text()) if tumor_path.exists() else {}
    return {
        "sample_id": sample_dir.name,
        "num_foci": tumor.get("num_foci", len(comp)),
        "candidate_count": len(cand),
        "component_count": len(comp),
        "candidate_recall_2mm": float((min_comp <= 2.0).mean()) if len(comp) else 1.0,
        "candidate_recall_3mm": float((min_comp <= 3.0).mean()) if len(comp) else 1.0,
        "candidate_recall_5mm": float((min_comp <= 5.0).mean()) if len(comp) else 1.0,
        "candidate_precision_2mm": float((min_cand <= 2.0).mean()) if len(cand) else 0.0,
        "candidate_precision_3mm": float((min_cand <= 3.0).mean()) if len(cand) else 0.0,
        "candidate_precision_5mm": float((min_cand <= 5.0).mean()) if len(cand) else 0.0,
        "one_to_one_matched_distance_mean": float(np.mean(matched_dist))
        if len(matched_dist)
        else float("inf"),
        "one_to_one_matched_distance_p95": float(np.percentile(matched_dist, 95))
        if len(matched_dist)
        else float("inf"),
        "duplicate_candidate_count_3mm": duplicate_count,
        "duplicate_candidate_ratio": float(duplicate_count / max(len(cand), 1)),
        "mean_anchor_to_component_distance": float(np.mean(min_cand))
        if len(cand)
        else float("inf"),
        "uncovered_component_count_3mm": int((min_comp > 3.0).sum()),
        "unmatched_candidate_count_3mm": int((min_cand > 3.0).sum()),
        "candidate_count_error": int(len(cand) - len(comp)),
        "min_inter_source_distance": tumor.get("min_inter_source_distance_mm", None),
        "weak_to_dominant_intensity_ratio": tumor.get("weak_to_dominant_intensity_ratio", None),
        "source_depth": tumor.get("source_depth", None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default="/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k"
    )
    parser.add_argument("--out_csv", default="outputs/ssq_fmt_rebuild/candidate_coverage.csv")
    parser.add_argument("--voxel_size_mm", type=float, default=0.2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()
    sample_dirs = find_samples(Path(args.data_dir), args.limit)
    rows = []
    if args.num_workers <= 1:
        for idx, sample_dir in enumerate(sample_dirs, 1):
            rows.append(process_sample(sample_dir, args.voxel_size_mm))
            if idx == 1 or idx % 25 == 0 or idx == len(sample_dirs):
                print(f"[{idx}/{len(sample_dirs)}] {sample_dir.name}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = {
                ex.submit(process_sample, sample_dir, args.voxel_size_mm): sample_dir
                for sample_dir in sample_dirs
            }
            for idx, fut in enumerate(as_completed(futures), 1):
                rows.append(fut.result())
                if idx == 1 or idx % 25 == 0 or idx == len(sample_dirs):
                    print(f"[{idx}/{len(sample_dirs)}] {futures[fut].name}", flush=True)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["empty"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} rows={len(rows)}")


if __name__ == "__main__":
    main()
