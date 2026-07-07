#!/usr/bin/env python3
"""Paired full-volume evaluation for view-complementary SSQ-FMT checkpoints.

For each sample and model, source hypotheses are constructed once from a deterministic
proposal set. The surface encoding and candidate-construction outputs are then frozen
and reused for every full-volume query chunk. This avoids chunk-dependent hypotheses
without changing the trained model or enabling the fixed-grid/A3-v2 path.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.dataset.fmt_simgen_dataset import FmtSimGenProjDataset  # noqa: E402
from minr_fmt.module import TrainingLightningModule  # noqa: E402
from scripts.eval_components_fmt_simgen import evaluate_sample as component_metrics  # noqa: E402
from scripts.eval_full_volume_fmt_simgen import (  # noqa: E402
    load_gt,
    load_sample_statistics,
    sample_metadata,
    volume_metrics,
)

MODEL_SPECS = {
    "a2u": {
        "label": "A2-U",
        "run_dir": "outputs/view_complementary/long_1k_continue_a2u_seed42",
        "ablation": "a2",
        "separability_mode": "none",
    },
    "a3_old": {
        "label": "A3-old",
        "run_dir": "outputs/view_complementary/long_1k_continue_a3_seed42",
        "ablation": "a3",
        "separability_mode": "geometry_only",
    },
    "phsa": {
        "label": "PHSA",
        "run_dir": (
            "outputs/view_complementary/"
            "a3v2_fast_geometry_only_seed42_corrected"
        ),
        "ablation": "a3_geometry_only",
        "separability_mode": "geometry_only",
    },
}

PROPOSAL_DIAGNOSTIC_KEYS = {
    "evidence": "per_view_evidence",
    "offsets_mm": "proposal_offsets_mm",
    "proposal_points_mm": "proposal_points_mm",
    "proposal_valid_mask": "proposal_valid_mask",
}

CANDIDATE_KEYS = (
    "candidate_centers_mm",
    "candidate_covariances_mm",
    "candidate_covariance_eigenvalues",
    "covariance_lower_bound_hit",
    "covariance_upper_bound_hit",
    "candidate_scores",
    "candidate_valid_mask",
    "candidate_slot_valid_mask",
    "candidate_analysis_valid_mask",
    "candidate_existence_probability",
    "candidate_view_support",
    "proposal_assignment",
    "proposal_compatibility",
)

HIGHER_IS_BETTER = {
    "dice",
    "iou",
    "precision",
    "recall",
    "cnr",
    "component_recall",
    "component_precision",
    "small_component_recall",
    "mean_matched_iou",
}
LOWER_IS_BETTER = {
    "nrmse",
    "cle",
    "ple",
    "volume_error",
    "assd",
    "hd95",
    "merge_count",
    "split_count",
    "false_component_count",
    "missed_component_count",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--models", nargs="+", default=["a2u", "a3_old", "phsa"])
    parser.add_argument("--proposal_count", type=int, default=4096)
    parser.add_argument("--proposal_seed", type=int, default=42)
    parser.add_argument("--chunk_size", type=int, default=32768)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--min_region_size", type=int, default=10)
    parser.add_argument("--cc_connectivity", type=int, default=26)
    parser.add_argument("--centroid_threshold_vox", type=float, default=3.0)
    parser.add_argument("--iou_threshold", type=float, default=0.01)
    parser.add_argument("--small_component_max_voxels", type=int, default=64)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--equivalence_atol", type=float, default=1.0e-6)
    parser.add_argument("--a2u_run", type=Path, default=None)
    parser.add_argument("--a3_old_run", type=Path, default=None)
    parser.add_argument("--phsa_run", type=Path, default=None)
    return parser.parse_args()


def stable_sample_seed(base_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return (int(base_seed) + int.from_bytes(digest[:8], "little")) % (2**63 - 1)


def deterministic_proposal_indices(
    shape: tuple[int, int, int], count: int, base_seed: int, sample_id: str
) -> np.ndarray:
    total = int(np.prod(shape))
    actual = min(int(count), total)
    rng = np.random.default_rng(stable_sample_seed(base_seed, sample_id))
    indices = rng.choice(total, size=actual, replace=False)
    return np.sort(indices.astype(np.int64))


def linear_indices_to_points_mm(
    indices: np.ndarray, shape: tuple[int, int, int], voxel_size_mm: float
) -> torch.Tensor:
    ijk = np.stack(np.unravel_index(indices, shape), axis=-1).astype(np.float32)
    return torch.from_numpy((ijk + 0.5) * float(voxel_size_mm)).unsqueeze(0)


def chunk_points_mm(
    shape: tuple[int, int, int], start: int, end: int, voxel_size_mm: float
) -> torch.Tensor:
    indices = np.arange(start, end, dtype=np.int64)
    return linear_indices_to_points_mm(indices, shape, voxel_size_mm)


def best_checkpoint(run_dir: Path) -> Path:
    checkpoints = [p for p in (run_dir / "checkpoints").glob("*.ckpt") if p.name != "last.ckpt"]
    if not checkpoints:
        last = run_dir / "checkpoints/last.ckpt"
        if last.exists():
            return last
        raise FileNotFoundError(f"No checkpoint found under {run_dir}")

    def score(path: Path) -> tuple[float, float]:
        for pattern in (r"val_dice=([0-9.]+)", r"-([0-9]+\.[0-9]+)\.ckpt$"):
            match = re.search(pattern, path.name)
            if match:
                return float(match.group(1).rstrip(".")), path.stat().st_mtime
        return float("-inf"), path.stat().st_mtime

    return max(checkpoints, key=score)


def resolve_run_dirs(args: argparse.Namespace) -> dict[str, Path]:
    overrides = {
        "a2u": args.a2u_run,
        "a3_old": args.a3_old_run,
        "phsa": args.phsa_run,
    }
    result = {}
    for key in args.models:
        if key not in MODEL_SPECS:
            raise ValueError(f"Unknown model {key}; choose from {sorted(MODEL_SPECS)}")
        run_dir = overrides.get(key) or ROOT / MODEL_SPECS[key]["run_dir"]
        run_dir = Path(run_dir)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        if not (run_dir / "config/config.yaml").exists():
            raise FileNotFoundError(f"Missing resolved config: {run_dir / 'config/config.yaml'}")
        result[key] = run_dir
    return result


def load_model(run_dir: Path, device: torch.device):
    cfg = OmegaConf.load(run_dir / "config/config.yaml")
    cfg.data.num_workers = 0
    cfg.data.persistent_workers = False
    cfg.data.train_max_samples = None
    cfg.data.val_max_samples = None
    cfg.data.test_max_samples = None
    module = TrainingLightningModule(cfg)
    checkpoint = best_checkpoint(run_dir)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    module.load_state_dict(state, strict=True)
    module.eval().to(device)
    net = module.net
    if hasattr(net, "set_view_training_phase"):
        net.set_view_training_phase("phase_b")
    return cfg, module, checkpoint


def clone_tensor_dict(values: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.detach().clone() if torch.is_tensor(value) else value
        for key, value in values.items()
    }


@contextlib.contextmanager
def replace_forward(module: torch.nn.Module, replacement: Callable[..., Any]) -> Iterator[None]:
    original = module.forward
    module.forward = replacement  # type: ignore[method-assign]
    try:
        yield
    finally:
        module.forward = original  # type: ignore[method-assign]


def call_net(
    net,
    surface: torch.Tensor,
    points_mm: torch.Tensor,
    detector_valid_mask: torch.Tensor,
    depth_maps: torch.Tensor,
    *,
    return_diagnostics: bool,
):
    batch = {
        "surface_measurements_packed": surface,
        "query_coordinates_mm": points_mm,
        "detector_valid_mask": detector_valid_mask,
        "depth_maps": depth_maps,
    }
    return net(
        surface,
        points_mm,
        detector_valid_mask=detector_valid_mask,
        depth_maps=depth_maps,
        batch=batch,
        return_diagnostics=return_diagnostics,
    )


def build_sample_cache(
    net,
    surface: torch.Tensor,
    proposal_points_mm: torch.Tensor,
    detector_valid_mask: torch.Tensor,
    depth_maps: torch.Tensor,
    equivalence_atol: float,
) -> tuple[dict[str, Any], float]:
    y_norm, norm_scale = net.normalizer(surface, detector_valid_mask)
    candidate_y_norm, candidate_norm_scale = net.candidate_normalizer(
        surface, detector_valid_mask
    )
    features = net.surface_encoder(y_norm)

    with contextlib.ExitStack() as stack:
        stack.enter_context(replace_forward(net.normalizer, lambda *_a, **_k: (y_norm, norm_scale)))
        stack.enter_context(
            replace_forward(
                net.candidate_normalizer,
                lambda *_a, **_k: (candidate_y_norm, candidate_norm_scale),
            )
        )
        stack.enter_context(replace_forward(net.surface_encoder, lambda *_a, **_k: features))
        reference = call_net(
            net,
            surface,
            proposal_points_mm,
            detector_valid_mask,
            depth_maps,
            return_diagnostics=True,
        )

    diagnostics = reference["diagnostics"]
    proposal_cache = clone_tensor_dict(
        {
            output_key: diagnostics[diagnostic_key]
            for output_key, diagnostic_key in PROPOSAL_DIAGNOSTIC_KEYS.items()
        }
    )
    candidate_cache = clone_tensor_dict({key: diagnostics[key] for key in CANDIDATE_KEYS})
    cache = {
        "y_norm": y_norm.detach(),
        "norm_scale": norm_scale.detach(),
        "candidate_y_norm": candidate_y_norm.detach(),
        "candidate_norm_scale": candidate_norm_scale.detach(),
        "features": features.detach(),
        "proposal": proposal_cache,
        "candidates": candidate_cache,
    }

    with frozen_sample_cache(net, cache):
        cached = call_net(
            net,
            surface,
            proposal_points_mm,
            detector_valid_mask,
            depth_maps,
            return_diagnostics=False,
        )
    error = float((reference["density"] - cached["density"]).abs().max())
    if not math.isfinite(error) or error > float(equivalence_atol):
        raise RuntimeError(
            f"Cache-forward equivalence failed: max_abs={error:.3e}, atol={equivalence_atol:.3e}"
        )
    return cache, error


@contextlib.contextmanager
def frozen_sample_cache(net, cache: dict[str, Any]) -> Iterator[None]:
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            replace_forward(
                net.normalizer,
                lambda *_a, **_k: (cache["y_norm"], cache["norm_scale"]),
            )
        )
        stack.enter_context(
            replace_forward(
                net.candidate_normalizer,
                lambda *_a, **_k: (
                    cache["candidate_y_norm"],
                    cache["candidate_norm_scale"],
                ),
            )
        )
        stack.enter_context(
            replace_forward(net.surface_encoder, lambda *_a, **_k: cache["features"])
        )
        stack.enter_context(
            replace_forward(
                net.view_candidate_evidence,
                lambda *_a, **_k: cache["proposal"],
            )
        )
        stack.enter_context(
            replace_forward(
                net.diverse_candidate_constructor,
                lambda *_a, **_k: cache["candidates"],
            )
        )
        yield


def predict_full_volume(
    net,
    surface: torch.Tensor,
    detector_valid_mask: torch.Tensor,
    depth_maps: torch.Tensor,
    shape: tuple[int, int, int],
    voxel_size_mm: float,
    proposal_points_mm: torch.Tensor,
    chunk_size: int,
    equivalence_atol: float,
) -> tuple[np.ndarray, dict[str, float]]:
    cache_start = time.perf_counter()
    cache, equivalence_error = build_sample_cache(
        net,
        surface,
        proposal_points_mm,
        detector_valid_mask,
        depth_maps,
        equivalence_atol,
    )
    cache_time_ms = (time.perf_counter() - cache_start) * 1000.0
    center_reference = cache["candidates"]["candidate_centers_mm"].detach().clone()
    total = int(np.prod(shape))
    prediction = np.empty(total, dtype=np.float32)
    decode_start = time.perf_counter()
    center_drift = 0.0
    with frozen_sample_cache(net, cache), torch.inference_mode():
        for start in range(0, total, int(chunk_size)):
            end = min(start + int(chunk_size), total)
            points_mm = chunk_points_mm(shape, start, end, voxel_size_mm).to(surface.device)
            inspect = start == 0 or end == total
            output = call_net(
                net,
                surface,
                points_mm,
                detector_valid_mask,
                depth_maps,
                return_diagnostics=inspect,
            )
            values = output["density"].squeeze(0).squeeze(-1)
            if not torch.isfinite(values).all():
                raise RuntimeError("Full-volume prediction contains NaN/Inf")
            prediction[start:end] = values.float().cpu().numpy()
            if inspect:
                observed = output["diagnostics"]["candidate_centers_mm"]
                center_drift = max(
                    center_drift,
                    float((observed - center_reference).norm(dim=-1).max()),
                )
    decode_time_ms = (time.perf_counter() - decode_start) * 1000.0
    return prediction.reshape(shape), {
        "cache_equivalence_max_abs": equivalence_error,
        "candidate_center_drift_max_mm": center_drift,
        "cache_build_time_ms": cache_time_ms,
        "volume_decode_time_ms": decode_time_ms,
    }


def prediction_path(output_dir: Path, model_key: str, sample_id: str) -> Path:
    return output_dir / "predictions" / model_key / f"{sample_id}.npz"


def evaluate_prediction(
    pred: np.ndarray,
    gt: np.ndarray,
    metadata: dict[str, Any],
    args: argparse.Namespace,
    voxel_size_mm: float,
) -> dict[str, Any]:
    spacing = (float(voxel_size_mm),) * 3
    volume = volume_metrics(
        pred,
        gt,
        args.threshold,
        spacing,
        args.min_region_size,
        args.cc_connectivity,
    )
    components = component_metrics(
        pred,
        gt,
        args.threshold,
        args.min_region_size,
        args.cc_connectivity,
        args.iou_threshold,
        args.centroid_threshold_vox,
        args.small_component_max_voxels,
    )
    return {**metadata, **volume, **components}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_pairs(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], metric: str):
    by_a = {row["sample_id"]: row for row in rows_a}
    by_b = {row["sample_id"]: row for row in rows_b}
    ids = sorted(by_a.keys() & by_b.keys())
    values = []
    for sample_id in ids:
        a, b = by_a[sample_id].get(metric), by_b[sample_id].get(metric)
        if a is None or b is None:
            continue
        a, b = float(a), float(b)
        if math.isfinite(a) and math.isfinite(b):
            values.append((sample_id, a, b))
    return values


def paired_statistics(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    metric: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    pairs = finite_pairs(rows_a, rows_b, metric)
    if not pairs:
        return {"n": 0}
    a = np.asarray([item[1] for item in pairs], dtype=np.float64)
    b = np.asarray([item[2] for item in pairs], dtype=np.float64)
    delta = a - b
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(delta), size=(int(bootstrap_samples), len(delta)))
    boot = delta[index].mean(axis=1)
    if np.allclose(delta, 0.0):
        p_value = 1.0
    else:
        try:
            p_value = float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
        except ValueError:
            p_value = None
    if metric in LOWER_IS_BETTER:
        wins = a < b
    else:
        wins = a > b
    return {
        "n": len(delta),
        "mean_delta_a_minus_b": float(delta.mean()),
        "median_delta_a_minus_b": float(np.median(delta)),
        "bootstrap_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "win_rate_a_over_b": float(wins.mean()),
        "wilcoxon_p": p_value,
        "direction": "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better",
    }


def mean_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"num_samples": len(rows)}
    numeric_keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float))})
    for key in numeric_keys:
        values = np.asarray(
            [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))],
            dtype=np.float64,
        )
        if values.size:
            result[f"{key}_mean"] = float(values.mean())
            result[f"{key}_std"] = float(values.std())
    return result


def grouped_summary(rows_by_model: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group_key in ("num_foci", "depth_tier", "shape_class"):
        result[group_key] = {}
        group_values = sorted(
            {
                str(row.get(group_key, "unknown"))
                for rows in rows_by_model.values()
                for row in rows
            }
        )
        for value in group_values:
            result[group_key][value] = {
                model: mean_metrics([row for row in rows if str(row.get(group_key, "unknown")) == value])
                for model, rows in rows_by_model.items()
            }
    return result


def summarize_and_write(
    output_dir: Path,
    rows_by_model: dict[str, list[dict[str, Any]]],
    bootstrap_samples: int,
    seed: int,
) -> None:
    all_rows = [row for rows in rows_by_model.values() for row in rows]
    write_csv(output_dir / "metrics_per_sample_all.csv", all_rows)
    metrics = sorted((HIGHER_IS_BETTER | LOWER_IS_BETTER) & set(all_rows[0]))
    comparisons = (("phsa", "a2u"), ("phsa", "a3_old"), ("a2u", "a3_old"))
    paired = {
        f"{a}_vs_{b}": {
            metric: paired_statistics(
                rows_by_model[a], rows_by_model[b], metric, bootstrap_samples, seed
            )
            for metric in metrics
        }
        for a, b in comparisons
        if a in rows_by_model and b in rows_by_model
    }
    summary = {
        "models": {model: mean_metrics(rows) for model, rows in rows_by_model.items()},
        "paired": paired,
        "grouped": grouped_summary(rows_by_model),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = ["# PHSA Full-Volume Paired Evaluation", ""]
    for model, values in summary["models"].items():
        lines.append(
            f"- **{MODEL_SPECS[model]['label']}**: n={values['num_samples']}, "
            f"Dice={values.get('dice_mean', float('nan')):.6f}, "
            f"component recall={values.get('component_recall_mean', float('nan')):.6f}, "
            f"small-component recall={values.get('small_component_recall_mean', float('nan')):.6f}"
        )
    lines.extend(["", "## Paired Dice", ""])
    for comparison, values in paired.items():
        item = values.get("dice", {})
        lines.append(
            f"- **{comparison}**: mean delta={item.get('mean_delta_a_minus_b')}, "
            f"95% CI={item.get('bootstrap_ci95')}, win rate={item.get('win_rate_a_over_b')}, "
            f"Wilcoxon p={item.get('wilcoxon_p')}"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    run_dirs = resolve_run_dirs(args)

    canonical_cfg = OmegaConf.load(run_dirs[args.models[0]] / "config/config.yaml")
    data_dir = Path(str(canonical_cfg.data.val_dir if args.split == "val" else canonical_cfg.data.test_dir))
    canonical_dataset = FmtSimGenProjDataset(
        str(data_dir), config=canonical_cfg, split=args.split, is_training=False
    )
    sample_dirs = list(canonical_dataset.dirs)
    if args.max_samples is not None:
        sample_dirs = sample_dirs[: int(args.max_samples)]
    if not sample_dirs:
        raise RuntimeError("No samples selected")
    sample_ids = [path.name for path in sample_dirs]
    (args.output_dir / "sample_ids.json").write_text(json.dumps(sample_ids, indent=2))
    stats_rows = load_sample_statistics(data_dir)

    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    for model_key in args.models:
        spec = MODEL_SPECS[model_key]
        run_dir = run_dirs[model_key]
        cfg, module, checkpoint = load_model(run_dir, device)
        net = module.net
        actual_ablation = str(net.view_complementary_ablation)
        if actual_ablation != spec["ablation"]:
            raise RuntimeError(
                f"{model_key} resolved ablation={actual_ablation}, expected {spec['ablation']}"
            )
        loader = FmtSimGenProjDataset(str(data_dir), config=cfg, split="all", is_training=False)
        voxel_size_mm = float(getattr(cfg.data, "voxel_size_mm", 0.2))
        model_rows = []
        for index, sample_dir in enumerate(sample_dirs, start=1):
            pred_file = prediction_path(args.output_dir, model_key, sample_dir.name)
            pred_file.parent.mkdir(parents=True, exist_ok=True)
            gt = load_gt(sample_dir)
            cache_meta: dict[str, float] = {}
            inference_time_ms = 0.0
            peak_memory_mb = 0.0
            if pred_file.exists() and not args.overwrite:
                payload = np.load(pred_file)
                pred = payload["pred"].astype(np.float32)
                cache_meta = json.loads(str(payload["cache_meta_json"])) if "cache_meta_json" in payload else {}
                inference_time_ms = float(payload.get("inference_time_ms", 0.0))
                peak_memory_mb = float(payload.get("peak_gpu_memory_mb", 0.0))
            else:
                (
                    _projections,
                    projections_packed,
                    _projection_scales,
                    depth_maps_tensor,
                    _descatter_targets,
                ) = loader._load_projection(sample_dir)
                surface = projections_packed.unsqueeze(0).to(device)
                depth_maps = depth_maps_tensor.unsqueeze(0).to(device)
                detector_valid_mask = torch.isfinite(depth_maps_tensor).unsqueeze(0).to(device)
                proposal_indices = deterministic_proposal_indices(
                    tuple(int(v) for v in gt.shape),
                    args.proposal_count,
                    args.proposal_seed,
                    sample_dir.name,
                )
                proposal_points_mm = linear_indices_to_points_mm(
                    proposal_indices, tuple(int(v) for v in gt.shape), voxel_size_mm
                ).to(device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                with torch.inference_mode():
                    pred, cache_meta = predict_full_volume(
                        net,
                        surface,
                        detector_valid_mask,
                        depth_maps,
                        tuple(int(v) for v in gt.shape),
                        voxel_size_mm,
                        proposal_points_mm,
                        args.chunk_size,
                        args.equivalence_atol,
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0**2)
                inference_time_ms = (time.perf_counter() - started) * 1000.0
                np.savez_compressed(
                    pred_file,
                    pred=pred.astype(np.float16),
                    gt=gt.astype(np.float16),
                    proposal_indices=proposal_indices,
                    proposal_sha256=np.asarray(
                        hashlib.sha256(proposal_indices.tobytes()).hexdigest()
                    ),
                    cache_meta_json=np.asarray(json.dumps(cache_meta)),
                    inference_time_ms=np.float64(inference_time_ms),
                    peak_gpu_memory_mb=np.float64(peak_memory_mb),
                    threshold=np.float32(args.threshold),
                    voxel_size_mm=np.float32(voxel_size_mm),
                )
            row = {
                "sample_id": sample_dir.name,
                "model": model_key,
                "model_label": spec["label"],
                "checkpoint": str(checkpoint),
                "inference_time_ms": inference_time_ms,
                "peak_gpu_memory_mb": peak_memory_mb,
                **cache_meta,
                **evaluate_prediction(
                    pred,
                    gt,
                    sample_metadata(sample_dir, stats_rows),
                    args,
                    voxel_size_mm,
                ),
            }
            model_rows.append(row)
            print(
                f"[{model_key}] {index:04d}/{len(sample_dirs):04d} {sample_dir.name}: "
                f"dice={row['dice']:.4f}, comp_recall={row.get('component_recall')}"
            )
        rows_by_model[model_key] = model_rows
        del module, net
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    expected = set(sample_ids)
    for model, rows in rows_by_model.items():
        observed = {row["sample_id"] for row in rows}
        if observed != expected:
            raise RuntimeError(f"Sample alignment failed for {model}")
    summarize_and_write(
        args.output_dir,
        rows_by_model,
        args.bootstrap_samples,
        args.proposal_seed,
    )
    print(args.output_dir / "summary.md")


if __name__ == "__main__":
    main()
