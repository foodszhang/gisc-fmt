#!/usr/bin/env python
"""Analyze cached SSQ candidate anchors and recommend data-driven scale bounds."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.utils.ssq_candidate_extraction import load_candidate_cache  # noqa: E402


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
        try:
            cache = load_candidate_cache(sample_dir)
        except FileNotFoundError:
            continue
        centers_all = cache["centers_mm"].astype(np.float32)
        valid = cache["valid"].astype(bool)
        centers = centers_all[valid]
        score = cache["scores"].astype(np.float32)[valid]
        raw_scale = cache["raw_support_scales_mm"].astype(np.float32)[valid]
        clipped_scale = np.clip(raw_scale, 1.0, 12.0)
        version = cache["metadata"].get("version", "unknown")
        finite_scale = raw_scale[np.isfinite(raw_scale)]
        scales.extend(float(x) for x in finite_scale)
        scores.extend(float(x) for x in score)
        counts.append(int(len(centers)))
        if len(centers) > 1:
            dist = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
            np.fill_diagonal(dist, np.inf)
            nearest = np.min(dist, axis=1)
            nn_dists.extend(float(x) for x in nearest)
            nearest_min = float(np.min(nearest))
        else:
            nearest_min = np.nan
        rows.append(
            {
                "sample_id": sample_dir.name,
                "proposal_version": version,
                "candidate_count": int(len(centers)),
                "raw_scale_min": float(np.nanmin(raw_scale)) if len(raw_scale) else np.nan,
                "raw_scale_median": float(np.nanmedian(raw_scale)) if len(raw_scale) else np.nan,
                "clipped_scale_min": float(np.nanmin(clipped_scale))
                if len(clipped_scale)
                else np.nan,
                "clipped_scale_median": float(np.nanmedian(clipped_scale))
                if len(clipped_scale)
                else np.nan,
                "scale_clip_ratio": float(np.mean(raw_scale != clipped_scale))
                if len(raw_scale)
                else np.nan,
                "nearest_neighbor_distance_mm": nearest_min,
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
