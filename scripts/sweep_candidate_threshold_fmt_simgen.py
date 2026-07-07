#!/usr/bin/env python
"""One-pass threshold sweep for FMT-SimGen measurement-candidate evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from eval_candidate_dense_fmt_simgen import (
    binary_metrics,
    candidate_cells,
    compose_cfg,
    forward_points,
    load_gt,
    load_net,
    pack_projection_input,
    sample_candidate_points,
    sample_dirs_for_split,
    sample_outside_points,
)
from minr_fmt.dataset.fmt_simgen_dataset import FmtSimGenProjDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--candidate_topk_ratio", type=float, default=0.10)
    parser.add_argument("--candidate_mode", default="coarse_cell_sample")
    parser.add_argument("--samples_per_candidate_cell", type=int, default=16)
    parser.add_argument("--outside_sample_num", type=int, default=131072)
    parser.add_argument("--chunk_size", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--thresholds", type=float, nargs="+", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def mean_std(rows: list[dict], key: str) -> tuple[float, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return float(values.mean()), float(values.std())


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = compose_cfg(args.exp, args.overrides)
    net = load_net(cfg, Path(args.ckpt_path), device)
    sample_dirs = sample_dirs_for_split(
        cfg.data.val_dir if args.split == "val" else cfg.data.test_dir,
        cfg,
        args.split,
        args.max_samples,
    )

    thresholds = [float(t) for t in args.thresholds]
    rows_by_threshold: dict[float, list[dict]] = {t: [] for t in thresholds}
    print(
        f"[INFO] One-pass threshold sweep for {len(sample_dirs)} samples on {device}; "
        f"thresholds={thresholds}"
    )

    for idx, sample_dir in enumerate(sample_dirs):
        rng = np.random.default_rng(int(args.seed) + idx)
        loader = FmtSimGenProjDataset(
            str(sample_dir.parent), config=cfg, split="all", is_training=False
        )
        loader.dirs = [sample_dir]
        _projections, projections_packed, _scales, depth_maps_tensor, _descatter = (
            loader._load_projection(sample_dir)
        )
        proj_in = pack_projection_input(projections_packed).to(device)
        depth_maps = depth_maps_tensor.unsqueeze(0).to(device)
        gt_shape = tuple(int(v) for v in cfg.model.geometry.global_voxel_shape)
        cells, _heatmap, meta = candidate_cells(sample_dir, args.candidate_topk_ratio, None)
        cand_ijk, cand_mm = sample_candidate_points(
            cells,
            meta,
            gt_shape,
            args.candidate_mode,
            args.samples_per_candidate_cell,
            rng,
        )
        candidate_pred = forward_points(
            net, proj_in, depth_maps, cand_ijk, cand_mm, gt_shape, args.chunk_size, device
        )
        candidate_cell_set = set(map(tuple, cells.tolist()))
        out_ijk, out_mm = sample_outside_points(
            candidate_cell_set, meta, gt_shape, args.outside_sample_num, rng
        )
        outside_pred = forward_points(
            net, proj_in, depth_maps, out_ijk, out_mm, gt_shape, args.chunk_size, device
        )
        gt = load_gt(sample_dir)
        candidate_labels = gt[cand_ijk[:, 0], cand_ijk[:, 1], cand_ijk[:, 2]] > 0
        for threshold in thresholds:
            row = binary_metrics(candidate_pred, candidate_labels, threshold)
            row.update(
                {
                    "sample_id": sample_dir.name,
                    "outside_fp_rate": float((outside_pred >= threshold).mean()),
                    "outside_mean_pred": float(outside_pred.mean()),
                    "outside_max_pred": float(outside_pred.max()),
                    "outside_p95_pred": float(np.percentile(outside_pred, 95)),
                    "outside_num_points": int(outside_pred.size),
                    "candidate_topk_ratio": float(args.candidate_topk_ratio),
                    "candidate_mode": args.candidate_mode,
                    "threshold": float(threshold),
                }
            )
            rows_by_threshold[threshold].append(row)
        if (idx + 1) % 20 == 0 or idx == len(sample_dirs) - 1:
            print(f"[INFO] processed {idx + 1}/{len(sample_dirs)} samples")

    summary_rows = []
    for threshold, rows in rows_by_threshold.items():
        per_path = save_dir / f"metrics_per_sample_threshold_{threshold:.2f}.csv"
        with per_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        dice_mean, dice_std = mean_std(rows, "candidate_dice")
        prec_mean, prec_std = mean_std(rows, "candidate_precision")
        rec_mean, rec_std = mean_std(rows, "candidate_recall")
        fp_mean, fp_std = mean_std(rows, "outside_fp_rate")
        p95_mean, p95_std = mean_std(rows, "outside_p95_pred")
        summary_rows.append(
            {
                "threshold": threshold,
                "mean_candidate_dice": dice_mean,
                "std_candidate_dice": dice_std,
                "mean_candidate_precision": prec_mean,
                "std_candidate_precision": prec_std,
                "mean_candidate_recall": rec_mean,
                "std_candidate_recall": rec_std,
                "mean_outside_fp_rate": fp_mean,
                "std_outside_fp_rate": fp_std,
                "mean_outside_p95_pred": p95_mean,
                "std_outside_p95_pred": p95_std,
            }
        )
    summary_rows = sorted(summary_rows, key=lambda row: row["threshold"])
    with (save_dir / "threshold_sweep_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    payload = {
        "exp": args.exp,
        "ckpt_path": args.ckpt_path,
        "split": args.split,
        "num_samples": len(sample_dirs),
        "candidate_topk_ratio": args.candidate_topk_ratio,
        "candidate_mode": args.candidate_mode,
        "samples_per_candidate_cell": args.samples_per_candidate_cell,
        "outside_sample_num": args.outside_sample_num,
        "chunk_size": args.chunk_size,
        "seed": args.seed,
        "thresholds": thresholds,
        "summary": summary_rows,
        "best_by_candidate_dice": max(summary_rows, key=lambda row: row["mean_candidate_dice"]),
    }
    (save_dir / "threshold_sweep_summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["best_by_candidate_dice"], indent=2))


if __name__ == "__main__":
    main()
