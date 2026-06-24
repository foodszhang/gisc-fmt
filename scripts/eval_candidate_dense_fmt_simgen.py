#!/usr/bin/env python
"""Measurement-candidate dense/semi-dense evaluation for FMT-SimGen checkpoints.

Candidate allocation uses only measurement-derived proposal heatmaps. GT voxels are read only
after candidate/outside points are fixed, to compute labels and metrics.

Reporting metadata such as depth tier, focus count, and shape labels is read only after
candidate/outside points are fixed. It is used exclusively for grouped metric summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.dataset.fmt_simgen_dataset import FmtSimGenProjDataset  # noqa: E402
from minr_fmt.model_factory import ModelFactory  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", help="Hydra-style overrides, e.g. exp=...")
    parser.add_argument("--exp", default=None)
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--split", default=None, choices=["val", "test"])
    parser.add_argument("--candidate_topk_ratio", type=float, default=None)
    parser.add_argument("--candidate_min_value", type=float, default=None)
    parser.add_argument(
        "--candidate_mode",
        default=None,
        choices=["coarse_cell_sample", "coarse_cell_dense"],
    )
    parser.add_argument("--samples_per_candidate_cell", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--outside_sample_num", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--debug_npz_count", type=int, default=0)
    return parser.parse_args()


def split_overrides(raw: list[str]) -> tuple[list[str], dict[str, str]]:
    hydra_overrides = []
    eval_overrides = {}
    for item in raw:
        if item.startswith("eval."):
            key, value = item.split("=", 1)
            eval_overrides[key.removeprefix("eval.")] = value
        elif item.startswith("ckpt_path=") or item.startswith("split="):
            key, value = item.split("=", 1)
            eval_overrides[key] = value
        else:
            hydra_overrides.append(item)
    return hydra_overrides, eval_overrides


def parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null", ""}:
        return None
    return float(value)


def eval_config(args: argparse.Namespace, eval_overrides: dict[str, str]) -> dict[str, Any]:
    def get(name: str, default: Any) -> Any:
        cli_value = getattr(args, name, None)
        if cli_value is not None:
            return cli_value
        return eval_overrides.get(name, default)

    return {
        "ckpt_path": get("ckpt_path", None),
        "split": str(get("split", "val")),
        "candidate_topk_ratio": float(get("candidate_topk_ratio", 0.10)),
        "candidate_min_value": parse_optional_float(get("candidate_min_value", None)),
        "candidate_mode": str(get("candidate_mode", "coarse_cell_sample")),
        "samples_per_candidate_cell": int(get("samples_per_candidate_cell", 8)),
        "chunk_size": int(get("chunk_size", 65536)),
        "threshold": float(get("threshold", 0.5)),
        "max_samples": None if get("max_samples", None) is None else int(get("max_samples", 0)),
        "outside_sample_num": int(get("outside_sample_num", 131072)),
        "seed": int(get("seed", 0)),
        "save_dir": get("save_dir", None),
    }


def compose_cfg(exp: str, hydra_overrides: list[str]):
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return hydra.compose(
            config_name="config",
            overrides=[
                f"exp={exp}",
                "data.train_max_samples=null",
                "data.val_max_samples=null",
                "data.test_max_samples=null",
                *hydra_overrides,
            ],
        )


def load_net(cfg, ckpt_path: Path, device: torch.device):
    net = ModelFactory.create_model(cfg.model.name, config=cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = {
        key.removeprefix("net."): value
        for key, value in ckpt["state_dict"].items()
        if key.startswith("net.")
    }
    strict = str(cfg.model.name).lower() == "ssq_fmt"
    missing, unexpected = net.load_state_dict(state, strict=strict)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:10]}")
    if missing:
        print(f"[WARN] Missing checkpoint keys: {missing[:10]}")
    net.eval()
    return net


def pack_projection_input(projections_packed: torch.Tensor) -> torch.Tensor:
    p = projections_packed.unsqueeze(0) if projections_packed.dim() == 4 else projections_packed
    return p.permute(1, 0, 2, 3, 4).reshape(p.shape[0] * p.shape[1], 1, p.shape[-2], p.shape[-1])


def load_gt(sample_dir: Path) -> np.ndarray:
    gt = np.load(sample_dir / "gt_voxels.npy").astype(np.float32)
    gt = np.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0)
    gt = np.clip(gt, 0.0, None)
    gt_max = float(gt.max())
    if gt_max > 0:
        gt = gt / gt_max
    return gt


def load_sample_statistics(data_dir: Path) -> dict[str, dict[str, str]]:
    """Load generation-control metadata for grouped reporting only."""
    stats_path = data_dir / "sample_statistics.csv"
    if not stats_path.exists():
        return {}
    with stats_path.open(newline="") as f:
        return {row["sample_id"]: row for row in csv.DictReader(f)}


def sample_report_metadata(sample_dir: Path, stats: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Return non-allocation metadata used only in metrics CSV/grouped summaries."""
    row = dict(stats.get(sample_dir.name, {}))
    meta: dict[str, Any] = {
        "num_foci": row.get("num_foci", ""),
        "depth_tier": row.get("depth_tier", ""),
        "depth_mm": row.get("depth_mm", ""),
        "source_type": row.get("source_type", ""),
        "shape_set": "",
        "shape_counts": "",
        "has_sphere": 0,
        "has_ellipsoid": 0,
        "has_irregular": 0,
    }
    tumor_path = sample_dir / "tumor_params.json"
    if tumor_path.exists():
        obj = json.loads(tumor_path.read_text())
        meta["num_foci"] = obj.get("num_foci", meta["num_foci"])
        meta["depth_tier"] = obj.get("depth_tier", meta["depth_tier"])
        meta["depth_mm"] = obj.get("depth_mm", meta["depth_mm"])
        meta["source_type"] = obj.get("source_type", meta["source_type"])
        shapes = [str(f.get("shape", "unknown")) for f in obj.get("foci", [])]
        if shapes:
            counts = {shape: shapes.count(shape) for shape in sorted(set(shapes))}
            meta["shape_set"] = "+".join(sorted(set(shapes)))
            meta["shape_counts"] = ";".join(f"{k}:{v}" for k, v in counts.items())
            for shape in ["sphere", "ellipsoid", "irregular"]:
                meta[f"has_{shape}"] = int(shape in counts)
    return meta


def candidate_cells(
    sample_dir: Path,
    ratio: float,
    min_value: float | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    proposal_dir = sample_dir / "proposal"
    heatmap = np.load(proposal_dir / "meas_backproj_heatmap.npy").astype(np.float32)
    meta = json.loads((proposal_dir / "meas_backproj_meta.json").read_text())
    flat = np.nan_to_num(heatmap.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if min_value is not None:
        mask = flat >= float(min_value)
    else:
        k = max(1, int(math.ceil(flat.size * float(ratio))))
        if float(flat.max()) <= 0:
            print(
                f"[WARN] Empty proposal heatmap for {sample_dir.name}; using arbitrary top-k cells."
            )
        kth = np.partition(flat, flat.size - k)[flat.size - k]
        mask = flat >= kth
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        raise RuntimeError(f"No candidate cells selected for {sample_dir}")
    cells = np.stack(np.unravel_index(idx, heatmap.shape), axis=-1).astype(np.int64)
    return cells, heatmap, meta


def sample_candidate_points(
    cells: np.ndarray,
    meta: dict,
    gt_shape: tuple[int, int, int],
    mode: str,
    samples_per_cell: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    voxel_size = float(meta.get("voxel_size_mm", 0.2))
    trunk_size = np.asarray(meta.get("trunk_size_mm", [38.0, 40.0, 20.8]), dtype=np.float32)
    grid_size = np.asarray(meta.get("grid_size", cells.max(axis=0) + 1), dtype=np.int64)
    cell_size = np.asarray(meta.get("cell_size_mm", trunk_size / grid_size), dtype=np.float32)

    if mode == "coarse_cell_sample":
        repeated = np.repeat(cells.astype(np.float32), int(samples_per_cell), axis=0)
        jitter = rng.uniform(-0.5, 0.5, size=repeated.shape).astype(np.float32)
        points_mm = (repeated + 0.5 + jitter) * cell_size
        points_mm = np.clip(points_mm, 0.0, trunk_size - 1.0e-6)
        ijk = np.floor(points_mm / voxel_size).astype(np.int64)
    elif mode == "coarse_cell_dense":
        chunks = []
        for cell in cells:
            lo_mm = cell.astype(np.float32) * cell_size
            hi_mm = (cell.astype(np.float32) + 1.0) * cell_size
            lo = np.floor(lo_mm / voxel_size).astype(np.int64)
            hi = np.ceil(hi_mm / voxel_size).astype(np.int64)
            lo = np.maximum(lo, 0)
            hi = np.minimum(hi, np.asarray(gt_shape, dtype=np.int64))
            xs = np.arange(lo[0], hi[0], dtype=np.int64)
            ys = np.arange(lo[1], hi[1], dtype=np.int64)
            zs = np.arange(lo[2], hi[2], dtype=np.int64)
            if xs.size and ys.size and zs.size:
                chunks.append(
                    np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
                )
        if not chunks:
            raise RuntimeError("coarse_cell_dense produced no voxels")
        ijk = np.concatenate(chunks, axis=0)
        points_mm = (ijk.astype(np.float32) + 0.5) * voxel_size
    else:
        raise ValueError(f"Unknown candidate_mode: {mode}")

    ijk[:, 0] = np.clip(ijk[:, 0], 0, gt_shape[0] - 1)
    ijk[:, 1] = np.clip(ijk[:, 1], 0, gt_shape[1] - 1)
    ijk[:, 2] = np.clip(ijk[:, 2], 0, gt_shape[2] - 1)
    return ijk.astype(np.int64), points_mm.astype(np.float32)


def cell_mask_for_voxels(
    ijk: np.ndarray,
    cell_size_mm: np.ndarray,
    voxel_size_mm: float,
) -> set[tuple[int, int, int]]:
    cell = np.floor((ijk.astype(np.float32) + 0.5) * voxel_size_mm / cell_size_mm).astype(np.int64)
    return set(map(tuple, cell.tolist()))


def sample_outside_points(
    candidate_cell_set: set[tuple[int, int, int]],
    meta: dict,
    gt_shape: tuple[int, int, int],
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    voxel_size = float(meta.get("voxel_size_mm", 0.2))
    trunk_size = np.asarray(meta.get("trunk_size_mm", [38.0, 40.0, 20.8]), dtype=np.float32)
    grid_size = np.asarray(meta.get("grid_size"), dtype=np.int64)
    cell_size = np.asarray(meta.get("cell_size_mm", trunk_size / grid_size), dtype=np.float32)

    accepted = []
    attempts = 0
    while sum(len(x) for x in accepted) < n and attempts < 20:
        attempts += 1
        need = n - sum(len(x) for x in accepted)
        draw = max(need * 2, 4096)
        ijk = np.column_stack(
            [
                rng.integers(0, gt_shape[0], size=draw, dtype=np.int64),
                rng.integers(0, gt_shape[1], size=draw, dtype=np.int64),
                rng.integers(0, gt_shape[2], size=draw, dtype=np.int64),
            ]
        )
        cells = np.floor((ijk.astype(np.float32) + 0.5) * voxel_size / cell_size).astype(np.int64)
        keep = np.asarray([tuple(c) not in candidate_cell_set for c in cells], dtype=bool)
        if keep.any():
            accepted.append(ijk[keep][:need])

    if accepted:
        out = np.concatenate(accepted, axis=0)[:n]
    else:
        print("[WARN] No outside points available; sampling with replacement from full trunk.")
        out = np.column_stack(
            [
                rng.integers(0, gt_shape[0], size=n, dtype=np.int64),
                rng.integers(0, gt_shape[1], size=n, dtype=np.int64),
                rng.integers(0, gt_shape[2], size=n, dtype=np.int64),
            ]
        )
    if out.shape[0] < n:
        print(f"[WARN] Outside points shortfall {out.shape[0]}/{n}; resampling with replacement.")
        extra_idx = rng.integers(0, out.shape[0], size=n - out.shape[0], dtype=np.int64)
        out = np.concatenate([out, out[extra_idx]], axis=0)
    points_mm = (out.astype(np.float32) + 0.5) * voxel_size
    return out.astype(np.int64), points_mm.astype(np.float32)


def points_to_norm(ijk: np.ndarray, gt_shape: tuple[int, int, int]) -> torch.Tensor:
    denom = np.maximum(np.asarray(gt_shape, dtype=np.float32) - 1.0, 1.0)
    return torch.tensor(ijk.astype(np.float32) / denom, dtype=torch.float32).unsqueeze(0)


def forward_points(
    net,
    cfg,
    proj_in: torch.Tensor,
    depth_maps: torch.Tensor,
    source_hypotheses: dict[str, torch.Tensor] | None,
    ijk: np.ndarray,
    points_mm: np.ndarray,
    gt_shape: tuple[int, int, int],
    chunk_size: int,
    device: torch.device,
) -> np.ndarray:
    preds = []
    is_ssq = str(getattr(cfg.model, "name", "")).lower() == "ssq_fmt"
    surface = None
    detector_valid_mask = None
    if is_ssq:
        # ``proj_in`` is legacy view-major input for GISC-FMT. For SSQ-FMT callers pass
        # packed [B,V,C,H,W] surface measurements through this argument.
        surface = proj_in
        detector_valid_mask = torch.isfinite(
            depth_maps.squeeze(1) if depth_maps.dim() == 5 else depth_maps
        )
    with torch.no_grad():
        for start in range(0, len(ijk), chunk_size):
            end = min(start + chunk_size, len(ijk))
            points = points_to_norm(ijk[start:end], gt_shape).to(device)
            mm = torch.tensor(points_mm[start:end], dtype=torch.float32).unsqueeze(0).to(device)
            if is_ssq:
                out = net(
                    surface,
                    mm,
                    detector_valid_mask=detector_valid_mask,
                    depth_maps=depth_maps,
                    batch={
                        "surface_measurements_packed": surface,
                        "query_coordinates_mm": mm,
                        "detector_valid_mask": detector_valid_mask,
                        "depth_maps": depth_maps,
                    },
                )
                value = out["density"].squeeze(0).squeeze(-1)
            else:
                pred, _aux = net(
                    proj_in,
                    points,
                    points_mm=mm,
                    depth_maps=depth_maps,
                    source_hypotheses=source_hypotheses,
                )
                value = torch.sigmoid(pred.squeeze(0).squeeze(-1))
            if not torch.isfinite(value).all():
                raise RuntimeError("Model prediction contains NaN/Inf")
            preds.append(value.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def source_hypotheses_for_sample(
    loader: FmtSimGenProjDataset,
    sample_dir: Path,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    if not getattr(loader, "source_hypothesis_enabled", False):
        return None
    source_hyp = loader._load_source_hypotheses(sample_dir)
    return {
        "centers": torch.tensor(
            source_hyp["centers"], dtype=torch.float32, device=device
        ).unsqueeze(0),
        "peak_scores": torch.tensor(
            source_hyp["peak_scores"], dtype=torch.float32, device=device
        ).unsqueeze(0),
        "scales": torch.tensor(source_hyp["scales"], dtype=torch.float32, device=device).unsqueeze(
            0
        ),
        "valid": torch.tensor(source_hyp["valid"], dtype=torch.float32, device=device).unsqueeze(0),
    }


def binary_metrics(pred: np.ndarray, label: np.ndarray, threshold: float) -> dict[str, float]:
    pred_bin = pred >= threshold
    gt_bin = label > 0
    tp = float(np.logical_and(pred_bin, gt_bin).sum())
    fp = float(np.logical_and(pred_bin, ~gt_bin).sum())
    fn = float(np.logical_and(~pred_bin, gt_bin).sum())
    pred_pos = float(pred_bin.sum())
    gt_pos = float(gt_bin.sum())
    eps = 1.0e-8
    if pred_pos == 0 and gt_pos == 0:
        dice = 1.0
    elif gt_pos == 0:
        dice = 0.0
    else:
        dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    return {
        "candidate_dice": float(dice),
        "candidate_precision": float((tp + eps) / (tp + fp + eps)),
        "candidate_recall": float((tp + eps) / (tp + fn + eps)),
        "candidate_pred_positive_count": int(pred_pos),
        "candidate_gt_positive_count": int(gt_pos),
        "candidate_num_points": int(label.size),
        "candidate_positive_ratio": float(gt_pos / max(label.size, 1)),
    }


def sample_dirs_for_split(data_dir: str, cfg, split: str, max_samples: int | None) -> list[Path]:
    ds = FmtSimGenProjDataset(data_dir, config=cfg, split=split, is_training=False)
    dirs = list(ds.dirs)
    return dirs[:max_samples] if max_samples is not None else dirs


def evaluate_sample(
    sample_dir: Path,
    net,
    cfg,
    ev: dict[str, Any],
    device: torch.device,
    idx: int,
    save_dir: Path,
    stats: dict[str, dict[str, str]],
):
    rng = np.random.default_rng(int(ev["seed"]) + idx)
    loader = FmtSimGenProjDataset(
        str(sample_dir.parent), config=cfg, split="all", is_training=False
    )
    loader.dirs = [sample_dir]
    _projections, projections_packed, _projection_scales, depth_maps_tensor, _descatter_targets = (
        loader._load_projection(sample_dir)
    )
    proj_in = pack_projection_input(projections_packed).to(device)
    query_input = (
        projections_packed.unsqueeze(0).to(device)
        if str(cfg.model.name).lower() == "ssq_fmt"
        else proj_in
    )
    depth_maps = depth_maps_tensor.unsqueeze(0).to(device)
    source_hypotheses = source_hypotheses_for_sample(loader, sample_dir, device)

    gt_shape = tuple(int(v) for v in cfg.model.geometry.global_voxel_shape)
    cells, _heatmap, meta = candidate_cells(
        sample_dir, ev["candidate_topk_ratio"], ev["candidate_min_value"]
    )
    cand_ijk, cand_mm = sample_candidate_points(
        cells,
        meta,
        gt_shape,
        ev["candidate_mode"],
        ev["samples_per_candidate_cell"],
        rng,
    )
    candidate_pred = forward_points(
        net,
        cfg,
        query_input,
        depth_maps,
        source_hypotheses,
        cand_ijk,
        cand_mm,
        gt_shape,
        ev["chunk_size"],
        device,
    )

    cell_size = np.asarray(meta.get("cell_size_mm"), dtype=np.float32)
    if cell_size.shape != (3,):
        trunk_size = np.asarray(meta.get("trunk_size_mm", [38.0, 40.0, 20.8]), dtype=np.float32)
        grid_size = np.asarray(meta.get("grid_size"), dtype=np.float32)
        cell_size = trunk_size / grid_size
    candidate_cell_set = set(map(tuple, cells.tolist()))
    out_ijk, out_mm = sample_outside_points(
        candidate_cell_set,
        meta,
        gt_shape,
        ev["outside_sample_num"],
        rng,
    )
    outside_pred = forward_points(
        net,
        cfg,
        query_input,
        depth_maps,
        source_hypotheses,
        out_ijk,
        out_mm,
        gt_shape,
        ev["chunk_size"],
        device,
    )

    gt = load_gt(sample_dir)
    if tuple(int(v) for v in gt.shape) != gt_shape:
        raise RuntimeError(f"{sample_dir.name} gt shape {gt.shape} != config shape {gt_shape}")
    candidate_labels = gt[cand_ijk[:, 0], cand_ijk[:, 1], cand_ijk[:, 2]] > 0
    row = binary_metrics(candidate_pred, candidate_labels, ev["threshold"])
    row.update(
        {
            "sample_id": sample_dir.name,
            **sample_report_metadata(sample_dir, stats),
            "outside_fp_rate": float((outside_pred >= ev["threshold"]).mean()),
            "outside_mean_pred": float(outside_pred.mean()),
            "outside_max_pred": float(outside_pred.max()),
            "outside_p95_pred": float(np.percentile(outside_pred, 95)),
            "outside_num_points": int(outside_pred.size),
            "candidate_topk_ratio": float(ev["candidate_topk_ratio"]),
            "candidate_mode": ev["candidate_mode"],
            "threshold": float(ev["threshold"]),
        }
    )

    if idx < int(ev.get("debug_npz_count", 0)):
        np.savez_compressed(
            save_dir / f"debug_{sample_dir.name}.npz",
            candidate_points_ijk=cand_ijk,
            candidate_pred=candidate_pred,
            candidate_label=candidate_labels.astype(np.uint8),
            outside_pred=outside_pred,
        )
    return row


def write_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "num_foci",
        "depth_tier",
        "depth_mm",
        "source_type",
        "shape_set",
        "shape_counts",
        "has_sphere",
        "has_ellipsoid",
        "has_irregular",
        "candidate_dice",
        "candidate_precision",
        "candidate_recall",
        "candidate_pred_positive_count",
        "candidate_gt_positive_count",
        "candidate_num_points",
        "candidate_positive_ratio",
        "outside_fp_rate",
        "outside_mean_pred",
        "outside_max_pred",
        "outside_p95_pred",
        "outside_num_points",
        "candidate_topk_ratio",
        "candidate_mode",
        "threshold",
    ]
    with (save_dir / "metrics_per_sample.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (save_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    grouped = grouped_summaries(rows)
    (save_dir / "metrics_grouped.json").write_text(json.dumps(grouped, indent=2))
    write_grouped_csv(grouped, save_dir / "metrics_grouped.csv")


def mean_std(rows: list[dict[str, Any]], key: str) -> tuple[float, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return float(values.mean()), float(values.std())


def grouped_summaries(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for key in [
        "depth_tier",
        "num_foci",
        "shape_set",
        "has_sphere",
        "has_ellipsoid",
        "has_irregular",
    ]:
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            value = row.get(key, "")
            if value in {"", None}:
                value = "unknown"
            buckets.setdefault(str(value), []).append(row)
        groups[key] = [
            summarize_group(key, value, bucket) for value, bucket in sorted(buckets.items())
        ]
    return groups


def summarize_group(group_key: str, group_value: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "group_key": group_key,
        "group_value": group_value,
        "num_samples": len(rows),
    }
    for key in [
        "candidate_dice",
        "candidate_precision",
        "candidate_recall",
        "outside_fp_rate",
        "outside_p95_pred",
        "candidate_positive_ratio",
        "candidate_num_points",
    ]:
        mean, std = mean_std(rows, key)
        out[f"mean_{key}"] = mean
        out[f"std_{key}"] = std
    return out


def write_grouped_csv(grouped: dict[str, list[dict[str, Any]]], path: Path) -> None:
    fields = [
        "group_key",
        "group_value",
        "num_samples",
        "mean_candidate_dice",
        "std_candidate_dice",
        "mean_candidate_precision",
        "std_candidate_precision",
        "mean_candidate_recall",
        "std_candidate_recall",
        "mean_outside_fp_rate",
        "std_outside_fp_rate",
        "mean_outside_p95_pred",
        "std_outside_p95_pred",
        "mean_candidate_positive_ratio",
        "std_candidate_positive_ratio",
        "mean_candidate_num_points",
        "std_candidate_num_points",
    ]
    rows = [row for group_rows in grouped.values() for row in group_rows]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    hydra_overrides, eval_overrides = split_overrides(args.overrides)
    exp = args.exp or eval_overrides.get("exp")
    if exp is None:
        for item in hydra_overrides:
            if item.startswith("exp="):
                exp = item.split("=", 1)[1]
                break
    if exp is None:
        raise SystemExit("exp must be provided via --exp or exp=...")
    hydra_overrides = [x for x in hydra_overrides if not x.startswith("exp=")]
    cfg = compose_cfg(exp, hydra_overrides)
    ev = eval_config(args, eval_overrides)
    if ev["ckpt_path"] is None:
        raise SystemExit("ckpt_path must be provided")
    ev["debug_npz_count"] = int(args.debug_npz_count)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_net(cfg, Path(ev["ckpt_path"]), device)
    data_dir = str(cfg.data.val_dir if ev["split"] == "val" else cfg.data.test_dir)
    sample_dirs = sample_dirs_for_split(data_dir, cfg, ev["split"], ev["max_samples"])
    stats = load_sample_statistics(Path(data_dir))
    save_dir = Path(
        ev["save_dir"] or (Path(cfg.paths.output_dir) / f"candidate_eval_{ev['split']}")
    )
    print(f"[INFO] Evaluating {len(sample_dirs)} samples on {device}, save_dir={save_dir}")

    rows = []
    for i, sample_dir in enumerate(sample_dirs):
        row = evaluate_sample(sample_dir, net, cfg, ev, device, i, save_dir, stats)
        rows.append(row)
        print(
            f"{row['sample_id']}: candidate_dice={row['candidate_dice']:.4f} "
            f"precision={row['candidate_precision']:.4f} recall={row['candidate_recall']:.4f} "
            f"outside_fp_rate={row['outside_fp_rate']:.6f}"
        )

    summary = {
        "num_samples": len(rows),
        "split": ev["split"],
        "ckpt_path": str(ev["ckpt_path"]),
        "exp": exp,
        "threshold": ev["threshold"],
        "candidate_mode": ev["candidate_mode"],
        "candidate_topk_ratio": ev["candidate_topk_ratio"],
        "samples_per_candidate_cell": ev["samples_per_candidate_cell"],
        "outside_sample_num": ev["outside_sample_num"],
    }
    for key in [
        "candidate_dice",
        "candidate_precision",
        "candidate_recall",
        "outside_fp_rate",
        "outside_p95_pred",
        "candidate_num_points",
        "candidate_positive_ratio",
    ]:
        mean, std = mean_std(rows, key)
        summary[f"mean_{key}"] = mean
        summary[f"std_{key}"] = std
    write_outputs(rows, summary, save_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
