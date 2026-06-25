"""Shared SSQ measurement-derived candidate extraction and cache IO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

CANDIDATE_CACHE_VERSION = "ssq_candidate_anchors_v2"


def load_candidate_cache(
    sample_dir: Path,
    *,
    expected_version: str = CANDIDATE_CACHE_VERSION,
    filename: str = "candidate_anchors.npz",
) -> dict[str, np.ndarray]:
    path = sample_dir / "proposal" / filename
    if not path.exists():
        raise FileNotFoundError(f"SSQ candidate cache missing: {path}")
    z = np.load(path, allow_pickle=False)
    required = ("centers_mm", "scores", "raw_support_scales_mm", "valid")
    missing = [key for key in required if key not in z.files]
    if missing:
        raise ValueError(f"{path} missing candidate fields: {missing}")
    metadata = json.loads(str(z["metadata_json"])) if "metadata_json" in z.files else {}
    version = str(metadata.get("version", ""))
    if expected_version and version != expected_version:
        raise ValueError(
            f"{path} candidate version {version!r} != expected {expected_version!r}; "
            "regenerate cache"
        )
    return {
        "centers_mm": z["centers_mm"].astype(np.float32),
        "scores": z["scores"].astype(np.float32),
        "raw_support_scales_mm": z["raw_support_scales_mm"].astype(np.float32),
        "valid": z["valid"].astype(bool),
        "metadata": metadata,
    }


def _support_scale_mm(
    heatmap: np.ndarray,
    center: np.ndarray,
    cell_size_mm: np.ndarray,
    radius_mm: float,
) -> float:
    radius_cells = np.maximum(np.ceil(radius_mm / cell_size_mm).astype(np.int64), 1)
    lo = np.maximum(center.astype(np.int64) - radius_cells, 0)
    hi = np.minimum(center.astype(np.int64) + radius_cells + 1, np.asarray(heatmap.shape))
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi, strict=True))
    local = heatmap[slices]
    if local.size == 0 or float(local.sum()) <= 0.0:
        return float(np.mean(cell_size_mm))
    coords = np.stack(np.meshgrid(*[np.arange(s.start, s.stop) for s in slices], indexing="ij"), -1)
    rel_mm = (coords.reshape(-1, 3).astype(np.float32) - center[None].astype(np.float32))
    rel_mm = rel_mm * cell_size_mm[None]
    weights = local.reshape(-1).astype(np.float64)
    moment = float(np.sqrt((weights * np.sum(rel_mm * rel_mm, axis=1)).sum() / weights.sum()))
    return max(moment, float(np.mean(cell_size_mm)))


def extract_candidate_anchors(
    heatmap: np.ndarray,
    metadata: dict[str, Any],
    *,
    top_m: int,
    smoothing_sigma_mm: float | None = None,
    smoothing_sigma_cells: float | None = None,
    nms_radius_mm: float = 3.0,
    support_moment_radius_mm: float = 4.0,
    min_value_ratio: float = 0.1,
) -> dict[str, np.ndarray | dict[str, Any]]:
    heat = np.clip(np.nan_to_num(heatmap.astype(np.float32), nan=0.0), 0.0, None)
    grid_size = np.asarray(metadata.get("grid_size", heat.shape), dtype=np.int64)
    if tuple(grid_size.tolist()) != tuple(heat.shape):
        raise ValueError(f"heatmap shape {heat.shape} != metadata grid_size {tuple(grid_size)}")
    cell_size_mm = np.asarray(metadata["cell_size_mm"], dtype=np.float32)
    if smoothing_sigma_cells is None:
        if smoothing_sigma_mm is None:
            smoothing_sigma_cells = 0.0
        else:
            smoothing_sigma_cells = float(smoothing_sigma_mm) / float(np.mean(cell_size_mm))
    smooth = (
        ndimage.gaussian_filter(heat, sigma=float(smoothing_sigma_cells)).astype(np.float32)
        if float(smoothing_sigma_cells) > 0
        else heat
    )
    max_value = float(np.max(smooth)) if smooth.size else 0.0
    centers = np.zeros((top_m, 3), dtype=np.float32)
    scores = np.zeros((top_m,), dtype=np.float32)
    scales = np.zeros((top_m,), dtype=np.float32)
    valid = np.zeros((top_m,), dtype=bool)
    if max_value <= 0.0 or not np.isfinite(max_value):
        return {
            "centers_mm": centers,
            "scores": scores,
            "raw_support_scales_mm": scales,
            "valid": valid,
        }

    candidates = np.argwhere(smooth >= max_value * float(min_value_ratio))
    values = smooth[tuple(candidates.T)] if len(candidates) else np.asarray([], dtype=np.float32)
    order = np.argsort(values)[::-1]
    selected_cells: list[np.ndarray] = []
    selected_mm: list[np.ndarray] = []
    tree: cKDTree | None = None
    for idx in order:
        cell = candidates[idx].astype(np.float32)
        mm = (cell + 0.5) * cell_size_mm
        if tree is not None and tree.query(mm, k=1)[0] < float(nms_radius_mm):
            continue
        selected_cells.append(cell)
        selected_mm.append(mm)
        tree = cKDTree(np.asarray(selected_mm, dtype=np.float32))
        if len(selected_mm) >= int(top_m):
            break
    for i, (cell, mm) in enumerate(zip(selected_cells, selected_mm, strict=True)):
        centers[i] = mm
        scores[i] = float(smooth[tuple(cell.astype(np.int64))] / max_value)
        scales[i] = _support_scale_mm(smooth, cell, cell_size_mm, support_moment_radius_mm)
        valid[i] = True
    return {
        "centers_mm": centers,
        "scores": scores,
        "raw_support_scales_mm": scales,
        "valid": valid,
    }


def write_candidate_cache(
    sample_dir: Path,
    anchors: dict[str, np.ndarray | dict[str, Any]],
    metadata: dict[str, Any],
    *,
    filename: str = "candidate_anchors.npz",
) -> Path:
    out = sample_dir / "proposal" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(metadata)
    meta["version"] = CANDIDATE_CACHE_VERSION
    np.savez_compressed(
        out,
        centers_mm=np.asarray(anchors["centers_mm"], dtype=np.float32),
        scores=np.asarray(anchors["scores"], dtype=np.float32),
        raw_support_scales_mm=np.asarray(anchors["raw_support_scales_mm"], dtype=np.float32),
        valid=np.asarray(anchors["valid"], dtype=bool),
        metadata_json=json.dumps(meta, sort_keys=True),
    )
    return out
