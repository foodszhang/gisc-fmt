#!/usr/bin/env python
"""Component-level evaluation for FMT-SimGen full-volume predictions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

METRIC_KEYS = [
    "gt_component_count",
    "pred_component_count",
    "matched_component_count",
    "missed_component_count",
    "false_component_count",
    "merge_count",
    "split_count",
    "component_recall",
    "component_precision",
    "mean_matched_iou",
    "small_component_recall",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", required=True, help="Full-volume eval directory.")
    parser.add_argument(
        "--data_dir",
        default="/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min_region_size", type=int, default=10)
    parser.add_argument("--connectivity", type=int, default=26)
    parser.add_argument("--centroid_threshold_vox", type=float, default=3.0)
    parser.add_argument("--iou_threshold", type=float, default=0.01)
    parser.add_argument("--small_component_max_voxels", type=int, default=64)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def split_ids(data_dir: Path, split: str) -> list[str]:
    path = data_dir / "splits" / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def sample_metadata(sample_dir: Path) -> dict[str, Any]:
    meta = {
        "num_foci": "",
        "depth_tier": "",
        "source_type": "",
        "shape_set": "",
    }
    tumor_path = sample_dir / "tumor_params.json"
    if not tumor_path.exists():
        return meta
    obj = json.loads(tumor_path.read_text())
    meta["num_foci"] = obj.get("num_foci", "")
    meta["depth_tier"] = obj.get("depth_tier", "")
    meta["source_type"] = obj.get("source_type", "")
    shapes = [str(f.get("shape", "unknown")) for f in obj.get("foci", [])]
    if shapes:
        meta["shape_set"] = "+".join(sorted(set(shapes)))
    return meta


def binary_components(mask: np.ndarray, min_region_size: int, connectivity: int):
    conn = {6: 1, 18: 2, 26: 3}.get(int(connectivity), int(connectivity))
    conn = max(1, min(conn, mask.ndim))
    structure = ndimage.generate_binary_structure(mask.ndim, conn)
    labeled, num = ndimage.label(mask.astype(bool), structure=structure)
    if num == 0:
        return labeled.astype(np.int32), []
    objects = ndimage.find_objects(labeled)
    components = []
    out = np.zeros_like(labeled, dtype=np.int32)
    new_id = 1
    for old_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        region = labeled[slc] == old_id
        size = int(region.sum())
        if size < min_region_size:
            continue
        coords = np.argwhere(region) + np.asarray([s.start for s in slc])
        centroid = coords.mean(axis=0)
        out[slc][region] = new_id
        components.append({"id": new_id, "size": size, "centroid": centroid})
        new_id += 1
    return out, components


def overlap_matrix(gt_labels: np.ndarray, pred_labels: np.ndarray, n_gt: int, n_pred: int):
    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt, n_pred), dtype=np.int64)
    valid = (gt_labels > 0) & (pred_labels > 0)
    if not valid.any():
        return np.zeros((n_gt, n_pred), dtype=np.int64)
    flat = (gt_labels[valid] - 1) * n_pred + (pred_labels[valid] - 1)
    return np.bincount(flat, minlength=n_gt * n_pred).reshape(n_gt, n_pred)


def match_components(
    gt_components: list[dict[str, Any]],
    pred_components: list[dict[str, Any]],
    overlaps: np.ndarray,
    iou_threshold: float,
    centroid_threshold_vox: float,
) -> list[dict[str, Any]]:
    n_gt = len(gt_components)
    n_pred = len(pred_components)
    gt_sizes = np.asarray([c["size"] for c in gt_components], dtype=np.float64)
    pred_sizes = np.asarray([c["size"] for c in pred_components], dtype=np.float64)
    candidates = []
    for gi in range(n_gt):
        for pi in range(n_pred):
            inter = float(overlaps[gi, pi])
            union = float(gt_sizes[gi] + pred_sizes[pi] - inter)
            iou = inter / union if union > 0 else 0.0
            dist = float(
                np.linalg.norm(gt_components[gi]["centroid"] - pred_components[pi]["centroid"])
            )
            if iou >= iou_threshold or dist <= centroid_threshold_vox:
                candidates.append((iou, -dist, gi, pi, dist))
    candidates.sort(reverse=True)
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches = []
    for iou, neg_dist, gi, pi, dist in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append(
            {
                "gt_id": gt_components[gi]["id"],
                "pred_id": pred_components[pi]["id"],
                "iou": float(iou),
                "centroid_distance_vox": float(dist),
                "weak_match": bool(iou < iou_threshold),
            }
        )
    return matches


def evaluate_sample(
    pred: np.ndarray,
    gt: np.ndarray,
    threshold: float,
    min_region_size: int,
    connectivity: int,
    iou_threshold: float,
    centroid_threshold_vox: float,
    small_component_max_voxels: int,
) -> dict[str, Any]:
    pred_labels, pred_components = binary_components(
        pred >= threshold, min_region_size, connectivity
    )
    gt_labels, gt_components = binary_components(gt > 0, min_region_size, connectivity)
    overlaps = overlap_matrix(gt_labels, pred_labels, len(gt_components), len(pred_components))
    matches = match_components(
        gt_components,
        pred_components,
        overlaps,
        iou_threshold=iou_threshold,
        centroid_threshold_vox=centroid_threshold_vox,
    )
    matched_gt = {m["gt_id"] for m in matches}
    gt_count = len(gt_components)
    pred_count = len(pred_components)
    matched_count = len(matches)

    gt_overlap_pred_counts = (overlaps > 0).sum(axis=1) if overlaps.size else np.zeros(gt_count)
    pred_overlap_gt_counts = (overlaps > 0).sum(axis=0) if overlaps.size else np.zeros(pred_count)
    merge_count = int((pred_overlap_gt_counts >= 2).sum())
    split_count = int((gt_overlap_pred_counts >= 2).sum())

    small_gt_ids = {
        c["id"] for c in gt_components if int(c["size"]) <= int(small_component_max_voxels)
    }
    small_recall = (
        len(small_gt_ids & matched_gt) / len(small_gt_ids) if small_gt_ids else None
    )

    return {
        "gt_component_count": gt_count,
        "pred_component_count": pred_count,
        "matched_component_count": matched_count,
        "missed_component_count": max(0, gt_count - matched_count),
        "false_component_count": max(0, pred_count - matched_count),
        "merge_count": merge_count,
        "split_count": split_count,
        "component_recall": matched_count / gt_count if gt_count else None,
        "component_precision": matched_count / pred_count if pred_count else None,
        "mean_matched_iou": float(np.mean([m["iou"] for m in matches])) if matches else None,
        "small_component_recall": small_recall,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def mean(values: list[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None and v != ""]
    return float(np.mean(vals)) if vals else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"num_samples": len(rows)}
    for key in METRIC_KEYS:
        out[f"{key}_mean"] = mean([row.get(key) for row in rows])
    return out


def grouped(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = row.get(group_key, "")
        buckets.setdefault(str(value if value not in {"", None} else "unknown"), []).append(row)
    out = []
    for value, group_rows in sorted(buckets.items()):
        summary = summarize(group_rows)
        summary.update({"group_key": group_key, "group_value": value})
        out.append(summary)
    return out


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    pred_dir = eval_dir / "predictions"
    if not pred_dir.exists():
        raise FileNotFoundError(f"Missing predictions directory: {pred_dir}")
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir) if args.save_dir else eval_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    ids = split_ids(data_dir, args.split)
    if args.max_samples is not None:
        ids = ids[: args.max_samples]
    for sample_id in ids:
        pred_path = pred_dir / f"{sample_id}.npz"
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)
        payload = np.load(pred_path)
        pred = np.asarray(payload["pred"], dtype=np.float32)
        gt = np.asarray(payload["gt"], dtype=np.float32)
        threshold = (
            float(args.threshold)
            if args.threshold is not None
            else float(payload["threshold"]) if "threshold" in payload else 0.5
        )
        sample_dir = data_dir / "samples" / sample_id
        row = {
            "sample_id": sample_id,
            **sample_metadata(sample_dir),
            **evaluate_sample(
                pred,
                gt,
                threshold=threshold,
                min_region_size=args.min_region_size,
                connectivity=args.connectivity,
                iou_threshold=args.iou_threshold,
                centroid_threshold_vox=args.centroid_threshold_vox,
                small_component_max_voxels=args.small_component_max_voxels,
            ),
        }
        rows.append(row)

    fields = [
        "sample_id",
        "num_foci",
        "depth_tier",
        "source_type",
        "shape_set",
        *METRIC_KEYS,
    ]
    write_csv(save_dir / "component_per_sample.csv", rows, fields)
    summary = summarize(rows)
    summary.update(
        {
            "eval_dir": str(eval_dir),
            "data_dir": str(data_dir),
            "split": args.split,
            "threshold": args.threshold,
            "min_region_size": args.min_region_size,
            "connectivity": args.connectivity,
            "iou_threshold": args.iou_threshold,
            "centroid_threshold_vox": args.centroid_threshold_vox,
        }
    )
    (save_dir / "component_summary.json").write_text(json.dumps(summary, indent=2))

    group_fields = ["group_key", "group_value", "num_samples"]
    group_fields.extend(f"{key}_mean" for key in METRIC_KEYS)
    by_num_foci = grouped(rows, "num_foci")
    by_shape = grouped(rows, "shape_set")
    write_csv(save_dir / "component_by_num_foci.csv", by_num_foci, group_fields)
    write_csv(save_dir / "component_by_shape_combo.csv", by_shape, group_fields)
    write_csv(
        save_dir / "component_grouped.csv",
        [*by_num_foci, *grouped(rows, "depth_tier"), *by_shape],
        group_fields,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
