#!/usr/bin/env python
"""Profile SSQ-FMT stage memory and timing for query-count engineering gates."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.dataset.fmt_simgen_dataset import FmtSimGenProjDataset  # noqa: E402
from minr_fmt.model_factory import ModelFactory  # noqa: E402
from minr_fmt.network.ssq_geometry import infer_detector_margin_map  # noqa: E402


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mem(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0
    return (
        torch.cuda.max_memory_allocated(device) / 1024**3,
        torch.cuda.max_memory_reserved(device) / 1024**3,
    )


def _time_stage(device: torch.device, name: str, rows: list[dict[str, Any]], fn):
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    out = fn()
    _sync(device)
    forward_s = time.perf_counter() - t0
    alloc, reserved = _mem(device)
    rows.append(
        {
            "stage": name,
            "forward_s": forward_s,
            "backward_s": "",
            "peak_allocated_gib": alloc,
            "peak_reserved_gib": reserved,
        }
    )
    return out


def _profile_one(cfg, batch: dict[str, Any], device: torch.device, num_queries: int):
    model = ModelFactory.create_model(cfg.model.name, config=cfg).to(device)
    model.train()
    batch = _move_batch(batch, device)
    surface = batch.get("surface_measurements_packed", batch["projections_packed"])
    query_mm = batch.get("query_coordinates_mm", batch["points_mm"])[:, :num_queries].contiguous()
    point_densities = batch["point_densities"][:, :num_queries].contiguous().unsqueeze(-1)
    prof_batch = dict(batch)
    prof_batch["query_coordinates_mm"] = query_mm
    prof_batch["points_mm"] = query_mm
    rows: list[dict[str, Any]] = []

    (y_norm, _scale), (candidate_y_norm, _candidate_scale) = _time_stage(
        device,
        "normalizer",
        rows,
        lambda: (
            model.normalizer(surface, prof_batch.get("detector_valid_mask")),
            model.candidate_normalizer(surface, prof_batch.get("detector_valid_mask")),
        ),
    )
    features = _time_stage(device, "encoder", rows, lambda: model.surface_encoder(y_norm))
    margin_map = infer_detector_margin_map(prof_batch)
    mapped = _time_stage(
        device,
        "geometry",
        rows,
        lambda: model.geometry_mapper(
            query_mm,
            depth_maps=prof_batch.get("depth_maps"),
            detector_margin_map=margin_map,
            detector_valid_mask=prof_batch.get("detector_valid_mask"),
        ),
    )
    samples = _time_stage(
        device,
        "sampler",
        rows,
        lambda: model.surface_sampler(
            features,
            y_norm,
            mapped,
            detector_valid_mask=prof_batch.get("detector_valid_mask"),
            depth_maps=prof_batch.get("depth_maps"),
        ),
    )
    candidates = _time_stage(
        device,
        "candidate_builder",
        rows,
        lambda: model.candidate_builder(candidate_y_norm, batch=prof_batch),
    )
    if candidates["candidate_centers_mm"].shape[1] > 0:
        cand_mapped = model.geometry_mapper(
            candidates["candidate_centers_mm"],
            depth_maps=prof_batch.get("depth_maps"),
            detector_valid_mask=prof_batch.get("detector_valid_mask"),
        )
        from minr_fmt.network.ssq_candidates import attach_candidate_detector_priors

        candidates = attach_candidate_detector_priors(
            candidates,
            cand_mapped,
            float(model._candidate_cfg.get("detector_scale_min_px", 0.5)),
            float(model._candidate_cfg.get("detector_scale_max_px", 20.0)),
            measurements=candidate_y_norm
            if model.measurement_consistency_enabled or model.candidate_context_enabled
            else None,
        )
    router = _time_stage(
        device,
        "routing",
        rows,
        lambda: model.candidate_router(samples, query_mm, candidates, mapped),
    )
    per_view = _time_stage(
        device,
        "view_representation",
        rows,
        lambda: model.view_encoder(samples, router, mapped),
    )
    z, _view_weights, lambda_view = _time_stage(
        device,
        "view_fusion",
        rows,
        lambda: model.view_fusion(
            per_view,
            router["r"],
            mapped["valid_mask"],
            router["branch_valid"],
        ),
    )
    candidate_context = None
    if model.candidate_context_enabled and candidates["candidate_centers_mm"].shape[1] > 0:

        def _candidate_context():
            covariance = candidates["candidate_support_covariances_mm"]
            eigen_scales = torch.linalg.eigvalsh(covariance.float()).clamp_min(1.0e-8).sqrt()
            eigen_scales = eigen_scales.to(dtype=z.dtype) / max(
                model._candidate_cfg.get("scale_max_mm", 12.0), 1.0e-6
            )
            context_trunk = torch.tensor(model.trunk_size_mm, device=z.device, dtype=z.dtype)
            center_norm = candidates["candidate_centers_mm"].to(dtype=z.dtype) / context_trunk
            center_measurements = candidates["candidate_center_measurements"].to(dtype=z.dtype)
            center_valid = candidates["candidate_detector_valid_mask"]
            valid_float = center_valid.to(dtype=z.dtype)
            valid_count = valid_float.sum(dim=1).clamp_min(1.0)
            measurement_mean = (center_measurements * valid_float).sum(dim=1) / valid_count
            centered = center_measurements - measurement_mean[:, None]
            measurement_std = (
                (centered.square() * valid_float).sum(dim=1) / valid_count
            ).sqrt()
            measurement_max = center_measurements.masked_fill(~center_valid, 0.0).amax(dim=1)
            visibility = valid_float.mean(dim=1)
            context_input = torch.cat(
                [
                    center_norm.mul(2.0).sub(1.0),
                    candidates["candidate_scores"][..., None].to(dtype=z.dtype),
                    eigen_scales,
                    measurement_mean[..., None],
                    measurement_std[..., None],
                    measurement_max[..., None],
                    visibility[..., None],
                ],
                dim=-1,
            )
            return model.candidate_context_encoder(context_input)

        candidate_context = _time_stage(
            device, "candidate_context", rows, _candidate_context
        )
        z = torch.cat([z[:, :, :1], z[:, :, 1:] + candidate_context[:, None]], dim=2)
    router["Lambda"] = lambda_view
    router["valid_view_count"] = mapped["valid_mask"].sum(dim=1)
    if candidates["candidate_centers_mm"].shape[1] > 0:
        candidate_visible = candidates["candidate_detector_valid_mask"].any(dim=1)
        router["candidate_geometric_visible"] = candidate_visible[:, None, :].expand(
            -1, query_mm.shape[1], -1
        )
    prior_pi = _time_stage(
        device,
        "assignment",
        rows,
        lambda: model.assignment_head(z, query_mm, router, candidates),
    )
    trunk = torch.tensor(model.trunk_size_mm, device=device, dtype=query_mm.dtype)
    encoded_query = model.position_encoding((query_mm / trunk.clamp_min(1e-6)).mul(2.0).sub(1.0))

    def _decode():
        d0 = model.compensation_density_decoder(z[:, :, 0], encoded_query)
        m = z.shape[2] - 1
        if m <= 0:
            branch_density = d0[:, :, None].clamp(0.0, 1.0)
        else:
            delta = query_mm[:, :, None] - candidates["candidate_centers_mm"][:, None]
            rel = torch.einsum(
                "bnmi,bmij->bnmj",
                delta,
                candidates["candidate_support_inverse_sqrt_mm"],
            )
            encoded_rel = model.position_encoding(rel.clamp(-4.0, 4.0))
            candidate_field = model.candidate_density_decoder(z[:, :, 1:], encoded_rel)
            if model.candidate_field_mode == "shared_residual":
                shared_logit = torch.logit(d0.float().clamp(1.0e-5, 1.0 - 1.0e-5))
                dm = torch.sigmoid(shared_logit[:, :, None] + candidate_field.float()).to(
                    dtype=d0.dtype
                )
            else:
                dm = candidate_field
            if candidate_context is not None and model.candidate_field_calibrator is not None:
                context_expanded = candidate_context[:, None].expand(-1, rel.shape[1], -1, -1)
                field_residual = model.candidate_field_calibrator(context_expanded, rel)
                dm_dtype = dm.dtype
                dm_logit = torch.logit(dm.float().clamp(1.0e-5, 1.0 - 1.0e-5))
                dm = torch.sigmoid(dm_logit + field_residual.float()).to(dtype=dm_dtype)
            if model.candidate_support_envelope_power > 0.0:
                support_envelope = router["p_all"][..., 1:].clamp(0.0, 1.0).pow(
                    model.candidate_support_envelope_power
                )
                dm = dm * support_envelope[..., None]
            dm = torch.where(
                candidates["candidate_valid_mask"][:, None, :, None],
                dm,
                torch.zeros_like(dm),
            )
            branch_density = torch.cat([d0[:, :, None], dm], dim=2).clamp(0.0, 1.0)
        pi = model.assignment_head.apply_occupancy_evidence(prior_pi, branch_density)
        density = (pi[..., None] * branch_density).sum(dim=2).clamp(0.0, 1.0)
        measurement_supported = router["valid_view_count"] > 0
        return density * measurement_supported[..., None].to(dtype=density.dtype)

    density = _time_stage(device, "decoder", rows, _decode)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    t0 = time.perf_counter()
    loss = torch.nn.functional.binary_cross_entropy(
        density.clamp(1e-6, 1.0 - 1e-6),
        point_densities,
    )
    loss.backward()
    _sync(device)
    alloc, reserved = _mem(device)
    rows.append(
        {
            "stage": "backward",
            "forward_s": "",
            "backward_s": time.perf_counter() - t0,
            "peak_allocated_gib": alloc,
            "peak_reserved_gib": reserved,
        }
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="fmt_simgen_v2_ssq_gate_32gb")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--queries", default="1024,2048,4096,8192")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--out", default="outputs/ssq_fmt_memory_profile")
    args, extra = parser.parse_known_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    max_queries = max(int(x) for x in args.queries.split(",") if x)
    overrides = [
        f"exp={args.exp}",
        "data.batch_size=1",
        "data.eval_batch_size=1",
        f"data.sample_num={max_queries}",
        f"data.num_queries={max_queries}",
        f"data.eval_sample_num={max_queries}",
    ] + extra
    if args.data_dir is not None:
        overrides.extend(
            [
                f"data.data_dir={args.data_dir}",
                f"data.train_dir={args.data_dir}",
                f"data.val_dir={args.data_dir}",
                f"data.test_dir={args.data_dir}",
            ]
        )
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = hydra.compose(config_name="config", overrides=overrides)
    ds = FmtSimGenProjDataset(str(cfg.data.train_dir), config=cfg, split="train", is_training=True)
    batch = next(iter(DataLoader(ds, batch_size=1, num_workers=0)))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for n in [int(x) for x in args.queries.split(",") if x]:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and args.precision == "bf16",
        ):
            rows = _profile_one(cfg, batch, device, n)
        for row in rows:
            all_rows.append({"num_queries": n, **row})
        print(json.dumps({"num_queries": n, "rows": rows}, indent=2))
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    csv_path = out_dir / "ssq_memory_profile.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    (out_dir / "ssq_memory_profile.json").write_text(json.dumps(all_rows, indent=2))
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
