#!/usr/bin/env python
# ruff: noqa: E402, I001
"""Full-volume FMT-SimGen evaluator for query and voxel models."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.dataset.fmt_simgen_dataset import FmtSimGenProjDataset  # noqa: E402
from minr_fmt.model_factory import ModelFactory  # noqa: E402
from minr_fmt.utils.utils import get_psnr_3d, get_ssim_3d  # noqa: E402


METRIC_KEYS = [
    "dice",
    "iou",
    "precision",
    "recall",
    "nrmse",
    "psnr",
    "ssim",
    "cle",
    "ple",
    "volume_error",
    "assd",
    "hd95",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. model=gisc_fmt")
    parser.add_argument("--exp", default=None)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--chunk_size", type=int, default=65536)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--prediction_dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--min_region_size", type=int, default=10)
    parser.add_argument("--cc_connectivity", type=int, default=26)
    parser.add_argument("--voxel_spacing", nargs=3, type=float, default=[0.2, 0.2, 0.2])
    return parser.parse_args()


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


def split_hydra_args(raw: list[str]) -> tuple[str | None, list[str]]:
    exp = None
    overrides = []
    for item in raw:
        if item.startswith("exp="):
            exp = item.split("=", 1)[1]
        else:
            overrides.append(item)
    return exp, overrides


def load_net(cfg, ckpt_path: Path, device: torch.device):
    net = ModelFactory.create_model(cfg.model.name, config=cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    net_state = {
        key.removeprefix("net."): value for key, value in state.items() if key.startswith("net.")
    }
    if not net_state:
        net_state = state
    missing, unexpected = net.load_state_dict(net_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:10]}")
    if missing:
        print(f"[WARN] Missing checkpoint keys: {missing[:10]}")
    net.eval()
    return net


def is_voxel_model(cfg, net) -> bool:
    return str(getattr(cfg.model, "output_type", "")).lower() == "voxel" or str(
        getattr(net, "output_type", "")
    ).lower() == "voxel"


def sample_dirs_for_split(data_dir: str, cfg, split: str, max_samples: int | None) -> list[Path]:
    ds = FmtSimGenProjDataset(data_dir, config=cfg, split=split, is_training=False)
    dirs = list(ds.dirs)
    return dirs[:max_samples] if max_samples is not None else dirs


def load_gt(sample_dir: Path) -> np.ndarray:
    gt = np.load(sample_dir / "gt_voxels.npy").astype(np.float32)
    gt = np.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0)
    gt = np.clip(gt, 0.0, None)
    gt_max = float(gt.max())
    if gt_max > 0:
        gt = gt / gt_max
    return gt


def load_sample_statistics(data_dir: Path) -> dict[str, dict[str, str]]:
    stats_path = data_dir / "sample_statistics.csv"
    if not stats_path.exists():
        return {}
    with stats_path.open(newline="") as f:
        return {row["sample_id"]: row for row in csv.DictReader(f)}


def sample_metadata(sample_dir: Path, stats: dict[str, dict[str, str]]) -> dict[str, Any]:
    row = dict(stats.get(sample_dir.name, {}))
    meta: dict[str, Any] = {
        "num_foci": row.get("num_foci", ""),
        "depth_tier": row.get("depth_tier", ""),
        "source_type": row.get("source_type", ""),
        "shape_set": "",
    }
    tumor_path = sample_dir / "tumor_params.json"
    if tumor_path.exists():
        obj = json.loads(tumor_path.read_text())
        meta["num_foci"] = obj.get("num_foci", meta["num_foci"])
        meta["depth_tier"] = obj.get("depth_tier", meta["depth_tier"])
        meta["source_type"] = obj.get("source_type", meta["source_type"])
        shapes = [str(f.get("shape", "unknown")) for f in obj.get("foci", [])]
        if shapes:
            meta["shape_set"] = "+".join(sorted(set(shapes)))
    return meta


def pack_query_projection(projections_packed: torch.Tensor) -> torch.Tensor:
    p = projections_packed.unsqueeze(0) if projections_packed.dim() == 4 else projections_packed
    return p.permute(1, 0, 2, 3, 4).reshape(
        p.shape[0] * p.shape[1], 1, p.shape[-2], p.shape[-1]
    )


def projection_batch(projections: dict[str, torch.Tensor], device: torch.device):
    return {key: value.unsqueeze(0).to(device) for key, value in projections.items()}


def chunk_indices(shape: tuple[int, int, int], start: int, end: int) -> np.ndarray:
    linear = np.arange(start, end, dtype=np.int64)
    return np.stack(np.unravel_index(linear, shape), axis=-1).astype(np.int64)


def points_to_norm(ijk: np.ndarray, shape: tuple[int, int, int]) -> torch.Tensor:
    denom = np.maximum(np.asarray(shape, dtype=np.float32) - 1.0, 1.0)
    return torch.from_numpy(ijk.astype(np.float32) / denom).unsqueeze(0)


def predict_query_volume(
    net,
    projections_packed: torch.Tensor,
    depth_maps_tensor: torch.Tensor,
    shape: tuple[int, int, int],
    voxel_size_mm: float,
    chunk_size: int,
    device: torch.device,
) -> np.ndarray:
    total = int(np.prod(shape))
    pred = np.empty(total, dtype=np.float32)
    proj_in = pack_query_projection(projections_packed).to(device)
    depth_maps = depth_maps_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            ijk = chunk_indices(shape, start, end)
            points = points_to_norm(ijk, shape).to(device)
            points_mm = torch.from_numpy((ijk.astype(np.float32) + 0.5) * voxel_size_mm)
            points_mm = points_mm.unsqueeze(0).to(device)
            logits, _aux = net(proj_in, points, points_mm=points_mm, depth_maps=depth_maps)
            values = torch.sigmoid(logits.squeeze(0).squeeze(-1))
            if not torch.isfinite(values).all():
                raise RuntimeError("Query model prediction contains NaN/Inf")
            pred[start:end] = values.detach().cpu().numpy().astype(np.float32)
    return pred.reshape(shape)


def predict_voxel_volume(
    net,
    cfg,
    projections: dict[str, torch.Tensor],
    gt_shape: tuple[int, int, int],
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        out = net(projection_batch(projections, device), points=None)
    if isinstance(out, dict) and "pred_voxel" in out:
        pred = out["pred_voxel"]
    elif torch.is_tensor(out):
        pred = out
    else:
        raise RuntimeError(f"Unexpected voxel model output: {type(out)}")
    if pred.dim() == 5 and pred.size(1) == 1:
        pred = pred[:, 0]
    if tuple(pred.shape[1:]) != gt_shape:
        pred = F.interpolate(
            pred.unsqueeze(1),
            size=gt_shape,
            mode="trilinear",
            align_corners=False,
        )[:, 0]
    values = torch.sigmoid(pred[0])
    if not torch.isfinite(values).all():
        raise RuntimeError("Voxel model prediction contains NaN/Inf")
    return values.detach().to(dtype=torch.float32).cpu().numpy()


def component_filter(mask: np.ndarray, min_region_size: int, connectivity: int) -> np.ndarray:
    conn = {6: 1, 18: 2, 26: 3}.get(int(connectivity), int(connectivity))
    conn = max(1, min(conn, mask.ndim))
    structure = ndimage.generate_binary_structure(mask.ndim, conn)
    labeled, num = ndimage.label(mask.astype(bool), structure=structure)
    if num == 0 or min_region_size <= 1:
        return mask.astype(bool)
    sizes = ndimage.sum(mask.astype(np.uint8), labeled, index=list(range(1, num + 1)))
    keep_ids = np.nonzero(np.asarray(sizes) >= int(min_region_size))[0] + 1
    if keep_ids.size == 0:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labeled, keep_ids) & mask.astype(bool)


def assd_hd95(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float]):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return 0.0, 0.0
    if not pred.any() or not gt.any():
        return None, None
    struct = np.ones((3, 3, 3), dtype=bool)
    pred_surface = pred ^ ndimage.binary_erosion(pred, structure=struct)
    gt_surface = gt ^ ndimage.binary_erosion(gt, structure=struct)
    dt_gt = ndimage.distance_transform_edt(~gt_surface, sampling=spacing)
    dt_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    d = np.concatenate([dt_gt[pred_surface], dt_pred[gt_surface]], axis=0)
    return float(np.mean(d)), float(np.percentile(d, 95))


def volume_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    threshold: float,
    spacing: tuple[float, float, float],
    min_region_size: int,
    connectivity: int,
) -> dict[str, Any]:
    pred = np.nan_to_num(pred.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pred = np.clip(pred, 0.0, 1.0)
    gt = np.nan_to_num(gt.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    gt = np.clip(gt, 0.0, 1.0)
    pred_bin = component_filter(pred >= threshold, min_region_size, connectivity)
    gt_bin = component_filter(gt > 0.0, min_region_size, connectivity)

    tp = float(np.logical_and(pred_bin, gt_bin).sum())
    fp = float(np.logical_and(pred_bin, ~gt_bin).sum())
    fn = float(np.logical_and(~pred_bin, gt_bin).sum())
    intersection = tp
    union = float(np.logical_or(pred_bin, gt_bin).sum())
    eps = 1.0e-8
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (intersection + eps) / (union + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    gt_range = max(float(gt.max() - gt.min()), eps)
    nrmse = float(np.sqrt(np.mean((pred - gt) ** 2)) / gt_range)
    volume_error = float(abs(pred_bin.sum() - gt_bin.sum()) / max(float(gt_bin.sum()), 1.0))

    coords = np.stack(np.meshgrid(*(np.arange(s) for s in gt.shape), indexing="ij"), axis=-1)
    flat_coords = coords.reshape(-1, 3).astype(np.float64)
    pred_w = pred.reshape(-1).astype(np.float64).clip(min=0.0)
    gt_w = gt.reshape(-1).astype(np.float64).clip(min=0.0)
    pred_centroid = (pred_w @ flat_coords) / max(float(pred_w.sum()), eps)
    gt_centroid = (gt_w @ flat_coords) / max(float(gt_w.sum()), eps)
    cle = float(np.linalg.norm((pred_centroid - gt_centroid) * np.asarray(spacing)))
    pred_peak = flat_coords[int(np.argmax(pred.reshape(-1)))]
    gt_peak = flat_coords[int(np.argmax(gt.reshape(-1)))]
    ple = float(np.linalg.norm((pred_peak - gt_peak) * np.asarray(spacing)))

    gt_min, gt_max = float(gt.min()), float(gt.max())
    denom = max(gt_max - gt_min, eps)
    gt_norm = np.clip((gt - gt_min) / denom, 0.0, 1.0)
    pred_norm = np.clip((pred - gt_min) / denom, 0.0, 1.0)
    assd, hd95 = assd_hd95(pred_bin, gt_bin, spacing)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "nrmse": nrmse,
        "psnr": float(get_psnr_3d(pred_norm, gt_norm)),
        "ssim": float(get_ssim_3d(pred_norm, gt_norm)),
        "cle": cle,
        "ple": ple,
        "volume_error": volume_error,
        "assd": assd,
        "hd95": hd95,
        "pred_positive_count": int(pred_bin.sum()),
        "gt_positive_count": int(gt_bin.sum()),
    }


def mean_std(rows: list[dict[str, Any]], key: str) -> tuple[float | None, float | None]:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None, None
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"num_samples": len(rows)}
    for key in METRIC_KEYS:
        mean, std = mean_std(rows, key)
        out[f"{key}_mean"] = mean
        out[f"{key}_std"] = std
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


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_outputs(rows: list[dict[str, Any]], save_dir: Path, run_meta: dict[str, Any]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    summary.update(run_meta)
    (save_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    with (save_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key in sorted(summary):
            writer.writerow({"metric": key, "value": summary[key]})

    sample_fields = [
        "sample_id",
        "num_foci",
        "depth_tier",
        "source_type",
        "shape_set",
        *METRIC_KEYS,
        "pred_positive_count",
        "gt_positive_count",
    ]
    write_csv(save_dir / "metrics_per_sample.csv", rows, sample_fields)

    grouped_payload = {
        "num_foci": grouped(rows, "num_foci"),
        "depth_tier": grouped(rows, "depth_tier"),
    }
    (save_dir / "metrics_grouped.json").write_text(json.dumps(grouped_payload, indent=2))
    group_fields = ["group_key", "group_value", "num_samples"]
    for key in METRIC_KEYS:
        group_fields.extend([f"{key}_mean", f"{key}_std"])
    write_csv(
        save_dir / "metrics_grouped.csv",
        [r for rs in grouped_payload.values() for r in rs],
        group_fields,
    )


def main() -> None:
    args = parse_args()
    exp_from_overrides, hydra_overrides = split_hydra_args(args.overrides)
    exp = args.exp or exp_from_overrides
    if exp is None:
        raise SystemExit("exp must be provided via --exp or exp=...")

    cfg = compose_cfg(exp, hydra_overrides)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    net = load_net(cfg, Path(args.ckpt_path), device)
    voxel_model = is_voxel_model(cfg, net)
    data_dir = Path(str(cfg.data.val_dir if args.split == "val" else cfg.data.test_dir))
    sample_dirs = sample_dirs_for_split(str(data_dir), cfg, args.split, args.max_samples)
    stats = load_sample_statistics(data_dir)
    save_dir = Path(
        args.save_dir
        or (Path(str(cfg.paths.output_dir)) / f"full_volume_eval_{args.split}_{cfg.model.name}")
    )
    pred_dir = save_dir / "predictions"
    if args.save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[INFO] Evaluating {len(sample_dirs)} {args.split} samples on {device}; "
        f"model={cfg.model.name} voxel_model={voxel_model}"
    )
    loader = FmtSimGenProjDataset(str(data_dir), config=cfg, split="all", is_training=False)
    voxel_size_mm = float(getattr(cfg.data, "voxel_size_mm", 0.2))
    spacing = tuple(float(v) for v in args.voxel_spacing)
    rows = []
    for idx, sample_dir in enumerate(sample_dirs):
        gt = load_gt(sample_dir)
        gt_shape = tuple(int(v) for v in gt.shape)
        (
            projections,
            projections_packed,
            _projection_scales,
            depth_maps_tensor,
            _descatter_targets,
        ) = loader._load_projection(sample_dir)
        if voxel_model:
            pred = predict_voxel_volume(net, cfg, projections, gt_shape, device)
        else:
            pred = predict_query_volume(
                net,
                projections_packed,
                depth_maps_tensor,
                gt_shape,
                voxel_size_mm,
                args.chunk_size,
                device,
            )
        row = {
            "sample_id": sample_dir.name,
            **sample_metadata(sample_dir, stats),
            **volume_metrics(
                pred,
                gt,
                args.threshold,
                spacing,
                args.min_region_size,
                args.cc_connectivity,
            ),
        }
        rows.append(row)
        print(
            f"{idx + 1:04d}/{len(sample_dirs):04d} {sample_dir.name}: "
            f"dice={row['dice']:.4f} iou={row['iou']:.4f} nrmse={row['nrmse']:.4f}"
        )
        if args.save_predictions:
            pred_dtype = np.float16 if args.prediction_dtype == "float16" else np.float32
            pred_save = pred.astype(pred_dtype)
            np.savez_compressed(
                pred_dir / f"{sample_dir.name}.npz",
                pred=pred_save,
                gt=gt.astype(np.float16),
                threshold=np.float32(args.threshold),
                voxel_spacing=np.asarray(spacing, dtype=np.float32),
            )

    write_outputs(
        rows,
        save_dir,
        {
            "split": args.split,
            "exp": exp,
            "model": str(cfg.model.name),
            "ckpt_path": str(args.ckpt_path),
            "threshold": float(args.threshold),
            "chunk_size": int(args.chunk_size),
            "voxel_model": bool(voxel_model),
        },
    )
    print(json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()
