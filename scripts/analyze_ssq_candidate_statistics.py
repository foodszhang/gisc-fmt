#!/usr/bin/env python
"""Analyze cached SSQ candidate anchors and recommend data-driven scale bounds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def find_samples(data_dir: Path, limit: int | None) -> list[Path]:
    root = data_dir if any(data_dir.glob("sample_*/proposal")) else data_dir / "samples"
    samples = sorted(p for p in root.glob("sample_*") if (p / "proposal").exists())
    return samples[:limit] if limit else samples


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            k: float("nan") for k in ["min", "P05", "P10", "P25", "P50", "P75", "P90", "P95", "max"]
        }
    arr = np.asarray(values, dtype=np.float32)
    return {
        "min": float(np.min(arr)),
        "P05": float(np.percentile(arr, 5)),
        "P10": float(np.percentile(arr, 10)),
        "P25": float(np.percentile(arr, 25)),
        "P50": float(np.percentile(arr, 50)),
        "P75": float(np.percentile(arr, 75)),
        "P90": float(np.percentile(arr, 90)),
        "P95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", default="/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k"
    )
    parser.add_argument("--out_dir", default="outputs/ssq_fmt_rebuild/candidate_statistics")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = []
    scales: list[float] = []
    scores: list[float] = []
    counts: list[int] = []
    nn_dists: list[float] = []
    for sample_dir in find_samples(Path(args.data_dir), args.limit):
        cand_path = sample_dir / "proposal" / "candidate_anchors.npz"
        meta_path = sample_dir / "proposal" / "meas_backproj_meta.json"
        if cand_path.exists():
            z = np.load(cand_path)
            centers = z["centers_mm"].astype(np.float32)
            score = (
                z["scores"].astype(np.float32)
                if "scores" in z.files
                else np.ones(len(centers), dtype=np.float32)
            )
            scale = (
                z["scales_mm"].astype(np.float32)
                if "scales_mm" in z.files
                else np.full(len(centers), np.nan, dtype=np.float32)
            )
        else:
            continue
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            version = meta.get("proposal_version", "unknown")
        else:
            version = "unknown"
        finite_scale = scale[np.isfinite(scale)]
        scales.extend(float(x) for x in finite_scale)
        scores.extend(float(x) for x in score)
        counts.append(int(len(centers)))
        if len(centers) > 1:
            dist = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
            dist[dist == 0] = np.inf
            nn_dists.extend(float(x) for x in np.min(dist, axis=1))
        rows.append(
            {
                "sample_id": sample_dir.name,
                "proposal_version": version,
                "candidate_count": int(len(centers)),
                "scale_min": float(np.nanmin(scale)) if len(scale) else np.nan,
                "scale_median": float(np.nanmedian(scale)) if len(scale) else np.nan,
                "score_max": float(np.max(score)) if len(score) else np.nan,
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "candidate_statistics_per_sample.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["empty"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "raw_scale_mm": summarize(scales),
        "candidate_count": summarize([float(x) for x in counts]),
        "candidate_score": summarize(scores),
        "candidate_nearest_neighbor_distance_mm": summarize(nn_dists),
    }
    p05 = summary["raw_scale_mm"]["P05"]
    p95 = summary["raw_scale_mm"]["P95"]
    summary["recommended_scale_bounds_mm"] = {
        "ell_min": max(0.5, p05) if np.isfinite(p05) else 1.0,
        "ell_max": min(15.0, p95) if np.isfinite(p95) else 12.0,
    }
    (out_dir / "candidate_statistics_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
