#!/usr/bin/env python
"""Evaluate measurement-derived SSQ candidate coverage against GT components."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


def find_samples(data_dir: Path, limit: int | None) -> list[Path]:
    root = data_dir if any(data_dir.glob("sample_*/gt_voxels.npy")) else data_dir / "samples"
    samples = sorted(p for p in root.glob("sample_*") if (p / "gt_voxels.npy").exists())
    return samples[:limit] if limit else samples


def load_candidates(sample_dir: Path) -> np.ndarray:
    cand_path = sample_dir / "proposal" / "candidate_anchors.npz"
    if not cand_path.exists():
        return np.zeros((0, 3), dtype=np.float32)
    z = np.load(cand_path)
    for key in ("candidate_centers_mm", "centers_mm", "centers"):
        if key in z.files:
            return z[key].astype(np.float32)
    return np.zeros((0, 3), dtype=np.float32)


def component_centers(gt: np.ndarray, voxel_size_mm: float) -> np.ndarray:
    labels, count = ndimage.label(gt > 0.5)
    centers = []
    for idx in range(1, count + 1):
        coords = np.argwhere(labels == idx)
        if len(coords):
            centers.append((coords.astype(np.float32).mean(axis=0) + 0.5) * voxel_size_mm)
    return np.asarray(centers, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default="/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k"
    )
    parser.add_argument("--out_csv", default="outputs/ssq_fmt_rebuild/candidate_coverage.csv")
    parser.add_argument("--voxel_size_mm", type=float, default=0.2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    rows = []
    for sample_dir in find_samples(Path(args.data_dir), args.limit):
        gt = np.load(sample_dir / "gt_voxels.npy").astype(np.float32)
        comp = component_centers(gt, args.voxel_size_mm)
        cand = load_candidates(sample_dir)
        if len(comp) and len(cand):
            dist = np.linalg.norm(comp[:, None] - cand[None], axis=-1)
            min_comp = dist.min(axis=1)
            min_cand = dist.min(axis=0)
        else:
            min_comp = np.full(len(comp), np.inf, dtype=np.float32)
            min_cand = np.full(len(cand), np.inf, dtype=np.float32)
        tumor_path = sample_dir / "tumor_params.json"
        tumor = json.loads(tumor_path.read_text()) if tumor_path.exists() else {}
        rows.append(
            {
                "sample_id": sample_dir.name,
                "num_foci": tumor.get("num_foci", len(comp)),
                "candidate_count": len(cand),
                "component_count": len(comp),
                "candidate_recall_2mm": float((min_comp <= 2.0).mean()) if len(comp) else 1.0,
                "candidate_recall_3mm": float((min_comp <= 3.0).mean()) if len(comp) else 1.0,
                "candidate_recall_5mm": float((min_comp <= 5.0).mean()) if len(comp) else 1.0,
                "candidate_precision_3mm": float((min_cand <= 3.0).mean()) if len(cand) else 0.0,
                "duplicate_candidate_ratio": float(
                    max(len(cand) - len(comp), 0) / max(len(cand), 1)
                ),
                "mean_anchor_to_component_distance": float(np.mean(min_cand))
                if len(cand)
                else float("inf"),
                "uncovered_component_count_3mm": int((min_comp > 3.0).sum()),
                "candidate_count_error": int(len(cand) - len(comp)),
                "min_inter_source_distance": tumor.get("min_inter_source_distance_mm", None),
                "weak_to_dominant_intensity_ratio": tumor.get(
                    "weak_to_dominant_intensity_ratio", None
                ),
                "source_depth": tumor.get("source_depth", None),
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
