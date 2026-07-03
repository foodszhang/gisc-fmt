#!/usr/bin/env python
# ruff: noqa: E402, I001
"""Full-volume FMT-SimGen evaluator for query and voxel models."""

from __future__ import annotations

import argparse
import csv
import json
import time
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.dataset.fmt_simgen_dataset import FmtSimGenProjDataset  # noqa: E402
from minr_fmt.model_factory import ModelFactory  # noqa: E402
from minr_fmt.module import SSQ_CHECKPOINT_EXTENSION_PREFIXES  # noqa: E402
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
    "cnr",
    "flops_g",
    "volume_error",
    "assd",
    "hd95",
    "inference_time_ms",
    "peak_gpu_memory_mb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. model=gisc_fmt")
    parser.add_argument("--exp", default=None)
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--chunk_size", type=int, default=65536)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--sample_ids",
        nargs="+",
        default=None,
        help="Optional sample IDs to evaluate after resolving the requested split.",
    )
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--prediction_dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--view_count", type=int, default=None)
    parser.add_argument(
        "--view_mode",
        default="all",
        choices=["all", "centered", "left", "right", "random"],
    )
    parser.add_argument("--view_seed", type=int, default=20260703)
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


def load_net(cfg, ckpt_path: Path | None, device: torch.device):
    net = ModelFactory.create_model(cfg.model.name, config=cfg).to(device)
    if ckpt_path is None or str(ckpt_path).lower() in {"", "none", "null"}:
        net.eval()
        return net
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    net_state = {
        key.removeprefix("net."): value for key, value in state.items() if key.startswith("net.")
    }
    if not net_state:
        net_state = state
    strict = str(cfg.model.name).lower() == "ssq_fmt"
    if strict:
        current_state = net.state_dict()
        extension_prefixes = tuple(
            prefix.removeprefix("net.") for prefix in SSQ_CHECKPOINT_EXTENSION_PREFIXES
        )
        compatible_missing = [
            key
            for key in current_state
            if key.startswith(extension_prefixes)
            and (key not in net_state or net_state[key].shape != current_state[key].shape)
        ]
        if compatible_missing:
            net_state = dict(net_state)
            for key in compatible_missing:
                net_state[key] = current_state[key]
    missing, unexpected = net.load_state_dict(net_state, strict=strict)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:10]}")
    if missing:
        print(f"[WARN] Missing checkpoint keys: {missing[:10]}")
    net.eval()
    return net


def is_voxel_model(cfg, net) -> bool:
    return (
        str(getattr(cfg.model, "output_type", "")).lower() == "voxel"
        or str(getattr(net, "output_type", "")).lower() == "voxel"
    )


def sample_dirs_for_split(
    data_dir: str,
    cfg,
    split: str,
    max_samples: int | None,
    sample_ids: list[str] | None = None,
) -> list[Path]:
    ds = FmtSimGenProjDataset(data_dir, config=cfg, split=split, is_training=False)
    dirs = list(ds.dirs)
    if sample_ids is not None:
        requested = set(sample_ids)
        dirs = [path for path in dirs if path.name in requested]
        missing = requested - {path.name for path in dirs}
        if missing:
            raise ValueError(f"Requested samples are not in split={split}: {sorted(missing)}")
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
        "shape_class": "unknown",
        "has_sphere": 0,
        "has_ellipsoid": 0,
        "has_irregular": 0,
    }
    tumor_path = sample_dir / "tumor_params.json"
    if tumor_path.exists():
        obj = json.loads(tumor_path.read_text())
        meta["num_foci"] = obj.get("num_foci", meta["num_foci"])
        meta["depth_tier"] = obj.get("depth_tier", meta["depth_tier"])
        meta["source_type"] = obj.get("source_type", meta["source_type"])
        shapes = [str(f.get("shape", "unknown")) for f in obj.get("foci", [])]
        if shapes:
            unique_shapes = sorted(set(shapes))
            meta["shape_set"] = "+".join(unique_shapes)
            for shape in ["sphere", "ellipsoid", "irregular"]:
                meta[f"has_{shape}"] = int(shape in unique_shapes)
            if len(unique_shapes) == 1:
                meta["shape_class"] = unique_shapes[0]
            elif len(unique_shapes) == 2:
                meta["shape_class"] = "mixed_two_shape"
            else:
                meta["shape_class"] = "mixed_three_shape"
    return meta


def pack_query_projection(projections_packed: torch.Tensor) -> torch.Tensor:
    p = projections_packed.unsqueeze(0) if projections_packed.dim() == 4 else projections_packed
    return p.permute(1, 0, 2, 3, 4).reshape(p.shape[0] * p.shape[1], 1, p.shape[-2], p.shape[-1])


def projection_batch(projections: dict[str, torch.Tensor], device: torch.device):
    return {key: value.unsqueeze(0).to(device) for key, value in projections.items()}


def estimate_conv_linear_flops(net, projections, batch, device: torch.device) -> float | None:
    """Approximate Conv/Linear FLOPs for one voxel-model forward pass.

    This intentionally reports a conservative hook-based estimate. Custom attention
    matrix multiplications are not fully counted, so the value should be treated as
    approximate and used alongside measured inference time and peak memory.
    """
    flops = 0
    hooks = []

    def conv_hook(module, inputs, output):
        nonlocal flops
        if not torch.is_tensor(output):
            return
        batch_size = int(output.shape[0])
        out_channels = int(output.shape[1])
        out_spatial = int(np.prod(output.shape[2:]))
        kernel_ops = int(np.prod(module.kernel_size)) * int(module.in_channels // module.groups)
        flops += batch_size * out_channels * out_spatial * kernel_ops * 2

    def linear_hook(module, inputs, output):
        nonlocal flops
        if not torch.is_tensor(output):
            return
        out_elems = int(output.numel())
        flops += out_elems * int(module.in_features) * 2

    for module in net.modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
    try:
        with torch.no_grad():
            _ = net(projection_batch(projections, device), points=None, batch=batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return float(flops) / 1.0e9
    except Exception as exc:
        print(f"[WARN] FLOPs profiling failed: {exc}")
        return None
    finally:
        for hook in hooks:
            hook.remove()


def chunk_indices(shape: tuple[int, int, int], start: int, end: int) -> np.ndarray:
    linear = np.arange(start, end, dtype=np.int64)
    return np.stack(np.unravel_index(linear, shape), axis=-1).astype(np.int64)


def points_to_norm(ijk: np.ndarray, shape: tuple[int, int, int]) -> torch.Tensor:
    denom = np.maximum(np.asarray(shape, dtype=np.float32) - 1.0, 1.0)
    return torch.from_numpy(ijk.astype(np.float32) / denom).unsqueeze(0)


def source_hypotheses_batch(loader, cfg, sample_dir: Path, device: torch.device):
    source_cfg = getattr(cfg.data, "source_hypothesis", None)
    ssq_cfg = getattr(cfg.data, "ssq_candidates", None)
    source_enabled = source_cfg is not None and bool(getattr(source_cfg, "enabled", False))
    ssq_enabled = ssq_cfg is not None and bool(getattr(ssq_cfg, "enabled", False))
    if not source_enabled and not ssq_enabled:
        return None
    source_hyp = loader._load_source_hypotheses(sample_dir)
    out = {
        "centers": torch.from_numpy(source_hyp["centers"]).unsqueeze(0).to(device),
        "peak_scores": torch.from_numpy(source_hyp["peak_scores"]).unsqueeze(0).to(device),
        "scales": torch.from_numpy(source_hyp["scales"]).unsqueeze(0).to(device),
        "valid": torch.from_numpy(source_hyp["valid"]).unsqueeze(0).to(device),
    }
    return out


def predict_query_volume(
    net,
    cfg,
    projections_packed: torch.Tensor,
    depth_maps_tensor: torch.Tensor,
    shape: tuple[int, int, int],
    voxel_size_mm: float,
    chunk_size: int,
    device: torch.device,
    source_hypotheses: dict[str, torch.Tensor] | None = None,
    view_indices: list[int] | None = None,
) -> np.ndarray:
    total = int(np.prod(shape))
    pred = np.empty(total, dtype=np.float32)
    proj_in = pack_query_projection(projections_packed).to(device)
    surface = projections_packed.unsqueeze(0).to(device)
    depth_maps = depth_maps_tensor.unsqueeze(0).to(device)
    detector_valid_mask = torch.isfinite(depth_maps_tensor).unsqueeze(0).to(device)
    if view_indices is not None:
        keep = torch.as_tensor(view_indices, dtype=torch.long, device=device)
        drop = torch.ones(surface.shape[1], dtype=torch.bool, device=device)
        drop[keep] = False
        surface = surface.clone()
        surface[:, drop] = 0.0
        detector_valid_mask = detector_valid_mask.clone()
        detector_valid_mask[:, drop] = False
    is_ssq = str(getattr(cfg.model, "name", "")).lower() == "ssq_fmt"
    with torch.no_grad():
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            ijk = chunk_indices(shape, start, end)
            points = points_to_norm(ijk, shape).to(device)
            points_mm = torch.from_numpy((ijk.astype(np.float32) + 0.5) * voxel_size_mm)
            points_mm = points_mm.unsqueeze(0).to(device)
            if is_ssq:
                ssq_batch = {
                    "surface_measurements_packed": surface,
                    "query_coordinates_mm": points_mm,
                    "detector_valid_mask": detector_valid_mask,
                    "depth_maps": depth_maps,
                }
                if source_hypotheses is not None:
                    ssq_batch.update(
                        {
                            "candidate_centers_mm": source_hypotheses["centers"],
                            "candidate_scores": source_hypotheses["peak_scores"],
                            "candidate_support_scales_mm": source_hypotheses["scales"],
                            "candidate_scales_mm": source_hypotheses["scales"],
                            "candidate_valid_mask": source_hypotheses["valid"].bool(),
                        }
                    )
                out = net(
                    surface,
                    points_mm,
                    detector_valid_mask=detector_valid_mask,
                    depth_maps=depth_maps,
                    batch=ssq_batch,
                )
                values = out["density"].squeeze(0).squeeze(-1)
            else:
                logits, _aux = net(
                    proj_in,
                    points,
                    points_mm=points_mm,
                    depth_maps=depth_maps,
                    source_hypotheses=source_hypotheses,
                )
                values = torch.sigmoid(logits.squeeze(0).squeeze(-1))
            if not torch.isfinite(values).all():
                raise RuntimeError("Query model prediction contains NaN/Inf")
            pred[start:end] = values.detach().cpu().numpy().astype(np.float32)
    return pred.reshape(shape)


def select_view_indices(
    view_count: int, keep_count: int | None, mode: str, seed: int
) -> list[int] | None:
    if keep_count is None or keep_count >= view_count or mode == "all":
        return None
    keep_count = max(1, min(int(keep_count), view_count))
    if mode == "left":
        return list(range(keep_count))
    if mode == "right":
        return list(range(view_count - keep_count, view_count))
    if mode == "centered":
        start = max(0, (view_count - keep_count) // 2)
        return list(range(start, start + keep_count))
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(view_count, size=keep_count, replace=False))


def predict_voxel_volume(
    net,
    cfg,
    projections: dict[str, torch.Tensor],
    batch: dict[str, Any],
    gt_shape: tuple[int, int, int],
    device: torch.device,
) -> np.ndarray:
    with torch.no_grad():
        out = net(projection_batch(projections, device), points=None, batch=batch)
    if isinstance(out, dict) and "pred_voxel" in out:
        pred = out["pred_voxel"]
    elif torch.is_tensor(out):
        pred = out
    else:
        raise RuntimeError(f"Unexpected voxel model output: {type(out)}")
    if pred.dim() == 5 and pred.size(1) == 1:
        pred = pred[:, 0]
    if tuple(pred.shape[1:]) != gt_shape:
        vr = cfg.data.voxel_ranges
        x0, x1 = int(vr.x[0]), int(vr.x[1])
        y0, y1 = int(vr.y[0]), int(vr.y[1])
        z0, z1 = int(vr.z[0]), int(vr.z[1])
        roi_shape = (x1 - x0, y1 - y0, z1 - z0)
        if tuple(pred.shape[1:]) != roi_shape:
            raise ValueError(
                f"Prediction shape {tuple(pred.shape[1:])} is neither full GT {gt_shape} "
                f"nor configured ROI {roi_shape}."
            )
        full = pred.new_zeros((pred.shape[0], *gt_shape))
        full[:, x0:x1, y0:y1, z0:z1] = pred
        pred = full
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
    fg = gt_bin.astype(bool)
    bg = ~fg
    if fg.any() and bg.any():
        fg_vals = pred[fg]
        bg_vals = pred[bg]
        cnr = float(
            abs(float(fg_vals.mean()) - float(bg_vals.mean()))
            / np.sqrt(float(fg_vals.var()) + float(bg_vals.var()) + eps)
        )
    else:
        cnr = None
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
        "cnr": cnr,
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


def grouped_multi(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        parts = []
        for key in group_keys:
            value = row.get(key, "")
            parts.append(f"{key}={value if value not in {'', None} else 'unknown'}")
        buckets.setdefault(" | ".join(parts), []).append(row)
    out = []
    for value, group_rows in sorted(buckets.items()):
        summary = summarize(group_rows)
        summary.update({"group_key": " x ".join(group_keys), "group_value": value})
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
        "shape_class",
        "has_sphere",
        "has_ellipsoid",
        "has_irregular",
        *METRIC_KEYS,
        "pred_positive_count",
        "gt_positive_count",
    ]
    write_csv(save_dir / "metrics_per_sample.csv", rows, sample_fields)

    grouped_payload = {
        "num_foci": grouped(rows, "num_foci"),
        "depth_tier": grouped(rows, "depth_tier"),
        "shape_set": grouped(rows, "shape_set"),
        "shape_class": grouped(rows, "shape_class"),
        "num_foci_depth_tier": grouped_multi(rows, ["num_foci", "depth_tier"]),
        "shape_set_num_foci": grouped_multi(rows, ["shape_set", "num_foci"]),
        "shape_set_depth_tier": grouped_multi(rows, ["shape_set", "depth_tier"]),
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
    ckpt_path = None
    if args.ckpt_path and str(args.ckpt_path).lower() not in {"none", "null", ""}:
        ckpt_path = Path(args.ckpt_path)
    net = load_net(cfg, ckpt_path, device)
    voxel_model = is_voxel_model(cfg, net)
    data_dir = Path(str(cfg.data.val_dir if args.split == "val" else cfg.data.test_dir))
    sample_dirs = sample_dirs_for_split(
        str(data_dir), cfg, args.split, args.max_samples, args.sample_ids
    )
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
    flops_g = None
    selected_view_indices = select_view_indices(
        len(list(cfg.data.view_angles)),
        args.view_count,
        args.view_mode,
        args.view_seed,
    )
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
        stage1_prior, stage1_source = loader._load_stage1_prior(sample_dir, gt_shape)
        stage1_mesh, stage1_mesh_source = loader._load_stage1_mesh(sample_dir)
        measurement_b = loader._load_measurement_b(sample_dir)
        batch: dict[str, Any] = {
            "sample_id": [sample_dir.name],
            "projections": projection_batch(projections, device),
            "gt_voxels": torch.from_numpy(gt).unsqueeze(0).to(device),
            "global_voxel_shape": gt_shape,
            "feasible_voxel_shape": gt_shape,
        }
        if stage1_prior is not None:
            batch["stage1_voxel"] = torch.from_numpy(stage1_prior).unsqueeze(0).to(device)
            batch["stage1_source"] = stage1_source
        if stage1_mesh is not None:
            batch["stage1_mesh"] = torch.from_numpy(stage1_mesh).unsqueeze(0).to(device)
            batch["stage1_mesh_source"] = stage1_mesh_source
        if measurement_b is not None:
            batch["measurement_b"] = torch.from_numpy(measurement_b).unsqueeze(0).to(device)
        gt_nodes_path = sample_dir / "gt_nodes.npy"
        if gt_nodes_path.exists():
            gt_nodes = np.load(gt_nodes_path).astype(np.float32).reshape(-1)
            gt_nodes = np.nan_to_num(gt_nodes, nan=0.0, posinf=0.0, neginf=0.0)
            gt_nodes = np.clip(gt_nodes, 0.0, None)
            gt_nodes_max = float(gt_nodes.max())
            if gt_nodes_max > 0:
                gt_nodes = gt_nodes / gt_nodes_max
            batch["gt_nodes"] = torch.from_numpy(gt_nodes).unsqueeze(0).to(device)
        if voxel_model and flops_g is None:
            flops_g = estimate_conv_linear_flops(net, projections, batch, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        start_time = time.perf_counter()
        if voxel_model:
            pred = predict_voxel_volume(net, cfg, projections, batch, gt_shape, device)
        else:
            source_hypotheses = source_hypotheses_batch(loader, cfg, sample_dir, device)
            pred = predict_query_volume(
                net,
                cfg,
                projections_packed,
                depth_maps_tensor,
                gt_shape,
                voxel_size_mm,
                args.chunk_size,
                device,
                source_hypotheses=source_hypotheses,
                view_indices=selected_view_indices,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        else:
            peak_mem = None
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0
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
            "flops_g": flops_g,
            "inference_time_ms": float(inference_time_ms),
            "peak_gpu_memory_mb": float(peak_mem) if peak_mem is not None else None,
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
            "view_count": args.view_count,
            "view_mode": args.view_mode,
            "view_indices": selected_view_indices,
        },
    )
    print(json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()
