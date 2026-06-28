#!/usr/bin/env python
"""Profile the formal SHQ quotient-residual path by stage."""

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
from minr_fmt.models.ssq_fmt import compose_quotient_residual_density  # noqa: E402
from minr_fmt.network.ssq_candidates import attach_candidate_detector_priors  # noqa: E402
from minr_fmt.network.ssq_geometry import infer_detector_margin_map  # noqa: E402


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0
    return (
        torch.cuda.max_memory_allocated(device) / 1024**3,
        torch.cuda.max_memory_reserved(device) / 1024**3,
    )


def _stage(device: torch.device, name: str, rows: list[dict[str, Any]], function):
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result = function()
    _sync(device)
    allocated, reserved = _memory(device)
    rows.append(
        {
            "stage": name,
            "forward_s": time.perf_counter() - start,
            "backward_s": "",
            "peak_allocated_gib": allocated,
            "peak_reserved_gib": reserved,
        }
    )
    return result


def _profile_one(cfg, raw_batch: dict[str, Any], device: torch.device, queries: int):
    model = ModelFactory.create_model(cfg.model.name, config=cfg).to(device).train()
    if model.composition_mode != "quotient_residual":
        raise ValueError("profile_ssq_memory.py requires composition.mode=quotient_residual")
    batch = _move_batch(raw_batch, device)
    surface = batch.get("surface_measurements_packed", batch["projections_packed"])
    points = batch.get("query_coordinates_mm", batch["points_mm"])[:, :queries].contiguous()
    target = batch["point_densities"][:, :queries].contiguous().unsqueeze(-1)
    batch = {**batch, "query_coordinates_mm": points, "points_mm": points}
    rows: list[dict[str, Any]] = []
    step_start = time.perf_counter()

    (normalized, _), (candidate_normalized, _) = _stage(
        device,
        "normalization",
        rows,
        lambda: (
            model.normalizer(surface, batch.get("detector_valid_mask")),
            model.candidate_normalizer(surface, batch.get("detector_valid_mask")),
        ),
    )
    features = _stage(device, "encoder", rows, lambda: model.surface_encoder(normalized))
    mapped = _stage(
        device,
        "geometry",
        rows,
        lambda: model.geometry_mapper(
            points,
            depth_maps=batch.get("depth_maps"),
            detector_margin_map=infer_detector_margin_map(batch),
            detector_valid_mask=batch.get("detector_valid_mask"),
        ),
    )
    samples = _stage(
        device,
        "surface_sampler",
        rows,
        lambda: model.surface_sampler(
            features,
            normalized,
            mapped,
            detector_valid_mask=batch.get("detector_valid_mask"),
            depth_maps=batch.get("depth_maps"),
        ),
    )

    def candidate_mapping():
        candidates = model.candidate_builder(candidate_normalized, batch=batch)
        if candidates["candidate_centers_mm"].shape[1] == 0:
            return candidates
        candidate_mapped = model.geometry_mapper(
            candidates["candidate_centers_mm"],
            depth_maps=batch.get("depth_maps"),
            detector_valid_mask=batch.get("detector_valid_mask"),
        )
        return attach_candidate_detector_priors(
            candidates,
            candidate_mapped,
            float(model._candidate_cfg.get("detector_scale_min_px", 0.5)),
            float(model._candidate_cfg.get("detector_scale_max_px", 20.0)),
        )

    candidates = _stage(device, "candidate_mapping", rows, candidate_mapping)
    candidates = {
        key: value.detach() if torch.is_tensor(value) else value
        for key, value in candidates.items()
    }
    router = _stage(
        device,
        "factorized_router",
        rows,
        lambda: model.candidate_router(samples, points, candidates, mapped),
    )
    per_view = _stage(
        device,
        "candidate_view_encoder",
        rows,
        lambda: model.view_encoder(samples, router, mapped),
    )
    geometry = model._quotient_geometry(samples, mapped)
    view_valid = mapped["valid_mask"].permute(0, 2, 1)
    support = router["r"].permute(0, 2, 1, 3).clamp(0.0, 1.0)
    candidate_valid = candidates["candidate_valid_mask"]

    def aggregate():
        shared_valid = torch.ones((points.shape[0], 1), dtype=torch.bool, device=device)
        shared = model.quotient_aggregator(
            per_view[..., :1, :], geometry, support[..., :1], view_valid, shared_valid
        )
        candidate_view_valid = view_valid[..., None]
        if candidate_valid.shape[1] > 0:
            candidate_view_valid = (
                candidate_view_valid & candidates["candidate_detector_valid_mask"][:, None]
            )
        candidate = model.quotient_aggregator(
            per_view[..., 1:, :],
            geometry,
            support[..., 1:],
            candidate_view_valid,
            candidate_valid,
        )
        return shared, candidate

    shared, candidate = _stage(device, "quotient_aggregation", rows, aggregate)
    q_shared = shared["quotient"].squeeze(2)
    trunk = torch.tensor(model.trunk_size_mm, device=device, dtype=points.dtype)
    query_encoding = model.position_encoding((points / trunk).mul(2.0).sub(1.0))
    shared_logit = _stage(
        device,
        "shared_decoder",
        rows,
        lambda: model.shared_density_logit_decoder(q_shared, query_encoding),
    )

    def candidate_decode():
        count = candidate_valid.shape[1]
        if count == 0:
            empty = q_shared.new_zeros((*q_shared.shape[:2], 0))
            return empty[..., None], empty, empty.new_zeros((*empty.shape, 3))
        relative = torch.einsum(
            "bnmi,bmij->bnmj",
            points[:, :, None] - candidates["candidate_centers_mm"][:, None],
            candidates["candidate_support_inverse_sqrt_mm"],
        )
        score = candidates["candidate_scores"].to(q_shared.dtype)
        eigenvalues = (
            torch.linalg.eigvalsh(candidates["candidate_support_covariances_mm"].float())
            .clamp_min(1.0e-8)
            .sqrt()
            .to(q_shared.dtype)
        )
        eigenvalues = eigenvalues / max(model._candidate_cfg.get("scale_max_mm", 12.0), 1.0e-6)
        context = torch.cat(
            [
                q_shared[:, :, None].expand(-1, -1, count, -1),
                candidate["quotient"] - q_shared[:, :, None],
                candidate["dispersion"][..., None],
                score[:, None, :, None].expand(-1, points.shape[1], -1, -1),
                candidate["support"][..., None],
                eigenvalues[:, None].expand(-1, points.shape[1], -1, -1),
            ],
            dim=-1,
        )
        raw = model.source_hypothesis_residual_decoder(
            context, model.position_encoding(relative.clamp(-4.0, 4.0))
        )
        return model.delta_l_max * torch.tanh(raw), candidate["dispersion"], relative

    delta_logit, dispersion, relative = _stage(
        device, "candidate_residual_decoder", rows, candidate_decode
    )
    composition = _stage(
        device,
        "composition",
        rows,
        lambda: compose_quotient_residual_density(
            shared_logit,
            delta_logit,
            relative,
            candidates["candidate_scores"].to(shared_logit.dtype),
            candidate["support"],
            dispersion,
            candidate_valid,
            model.tau_u,
        ),
    )
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    backward_start = time.perf_counter()
    final_logit = shared_logit + composition["residual_correction"]
    torch.nn.functional.binary_cross_entropy_with_logits(final_logit, target).backward()
    _sync(device)
    allocated, reserved = _memory(device)
    backward_time = time.perf_counter() - backward_start
    rows.append(
        {
            "stage": "backward",
            "forward_s": "",
            "backward_s": backward_time,
            "peak_allocated_gib": allocated,
            "peak_reserved_gib": reserved,
        }
    )
    rows.append(
        {
            "stage": "total_step",
            "forward_s": time.perf_counter() - step_start - backward_time,
            "backward_s": backward_time,
            "peak_allocated_gib": allocated,
            "peak_reserved_gib": reserved,
        }
    )
    return rows


def _add_baseline_deltas(rows: list[dict[str, Any]], baseline_path: str | None) -> None:
    if not baseline_path:
        return
    baseline_rows = json.loads(Path(baseline_path).read_text())
    baseline = {(row["num_queries"], row["stage"]): row for row in baseline_rows}
    for row in rows:
        old = baseline.get((row["num_queries"], row["stage"]))
        for key in ("forward_s", "backward_s", "peak_allocated_gib", "peak_reserved_gib"):
            current_value, old_value = row.get(key), old.get(key) if old else None
            row[f"delta_{key}"] = (
                float(current_value) - float(old_value)
                if current_value not in (None, "") and old_value not in (None, "")
                else ""
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="fmt_simgen_v2_shq_quotient_residual")
    parser.add_argument("--data-dir")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--queries", default="1024,2048,4096,8192")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--baseline-json", help="Previous profiler JSON for before/after deltas")
    parser.add_argument("--out", default="outputs/shq_memory_profile")
    args, extra = parser.parse_known_args()
    device = torch.device(args.device)
    query_counts = [int(value) for value in args.queries.split(",") if value]
    overrides = [
        f"exp={args.exp}",
        "data.batch_size=1",
        "data.eval_batch_size=1",
        f"data.sample_num={max(query_counts)}",
        f"data.num_queries={max(query_counts)}",
    ] + extra
    if args.data_dir:
        overrides.extend(
            f"data.{key}={args.data_dir}"
            for key in ("data_dir", "train_dir", "val_dir", "test_dir")
        )
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = hydra.compose(config_name="config", overrides=overrides)
    dataset = FmtSimGenProjDataset(
        str(cfg.data.train_dir), config=cfg, split="train", is_training=True
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
    all_rows: list[dict[str, Any]] = []
    for queries in query_counts:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda" and args.precision == "bf16",
        ):
            rows = _profile_one(cfg, batch, device, queries)
        all_rows.extend({"num_queries": queries, **row} for row in rows)
        print(json.dumps({"num_queries": queries, "rows": rows}, indent=2))
    _add_baseline_deltas(all_rows, args.baseline_json)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "shq_memory_profile.json"
    csv_path = output / "shq_memory_profile.csv"
    json_path.write_text(json.dumps(all_rows, indent=2))
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
