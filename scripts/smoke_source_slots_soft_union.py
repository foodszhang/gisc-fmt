#!/usr/bin/env python
"""Smoke test source-slot soft-union decoding."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import hydra
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.dataset.fmt_simgen_dataset import FmtSimGenProjDataset  # noqa: E402
from minr_fmt.model_factory import ModelFactory  # noqa: E402


def _projection_input_from_batch(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    packed = batch["projections_packed"]
    return packed.permute(1, 0, 2, 3, 4).reshape(
        packed.shape[0] * packed.shape[1], 1, packed.shape[-2], packed.shape[-1]
    )


def _source_hypotheses_from_batch(batch: dict[str, torch.Tensor]):
    if "source_hypothesis_centers" not in batch:
        return None
    return {
        "centers": batch["source_hypothesis_centers"],
        "peak_scores": batch.get("source_hypothesis_peak_scores"),
        "scales": batch.get("source_hypothesis_scales"),
        "valid": batch.get("source_hypothesis_valid"),
    }


def _synthetic_batch(
    cfg, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    batch_size = 1
    num_views = int(cfg.model.num_views)
    num_queries = 32
    image_size = 32
    proj_in = torch.randn(batch_size * num_views, 1, image_size, image_size, device=device)
    depth_maps = torch.full(
        (batch_size, num_views, image_size, image_size), 200.0, dtype=torch.float32, device=device
    )
    points = torch.rand(batch_size, num_queries, 3, device=device)
    points_mm = points * 32.0
    source_hypotheses = {
        "centers": torch.tensor(
            [[[8.0, 8.0, 8.0], [22.0, 12.0, 12.0], [0.0, 0.0, 0.0]]],
            dtype=torch.float32,
            device=device,
        ),
        "peak_scores": torch.tensor([[1.0, 0.8, 0.0]], dtype=torch.float32, device=device),
        "scales": torch.ones(batch_size, 3, dtype=torch.float32, device=device),
        "valid": torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32, device=device),
    }
    return proj_in, points, points_mm, depth_maps, source_hypotheses


def _assert_outputs(logits: torch.Tensor, aux: dict, expected_m: int | None = None) -> None:
    if logits.ndim != 3 or logits.shape[-1] != 1:
        raise AssertionError(f"density logits must have shape [B,N,1], got {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise AssertionError("density logits contain NaN or Inf")
    required = [
        "source_component_logits",
        "source_component_ownership",
        "source_component_valid",
    ]
    for key in required:
        if key not in aux:
            raise AssertionError(f"missing aux output: {key}")
    comp = aux["source_component_logits"]
    own = aux["source_component_ownership"]
    valid = aux["source_component_valid"]
    if comp.shape[:2] != logits.shape[:2] or comp.shape[-1] != 1:
        raise AssertionError(f"bad source_component_logits shape: {tuple(comp.shape)}")
    if own.shape != comp.shape[:-1]:
        raise AssertionError(f"bad ownership shape: {tuple(own.shape)}")
    if valid.shape != (logits.shape[0], comp.shape[2]):
        raise AssertionError(f"bad valid shape: {tuple(valid.shape)}")
    if expected_m is not None and comp.shape[2] != expected_m:
        raise AssertionError(f"expected M={expected_m}, got {comp.shape[2]}")
    invalid_own = own.masked_select(~valid[:, None, :].expand_as(own))
    if invalid_own.numel() and invalid_own.abs().max() > 1.0e-5:
        raise AssertionError("invalid source slots have nonzero ownership")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--split", default="train")
    args, overrides = parser.parse_known_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    device = torch.device(args.device)
    if not any(override.startswith("exp=") for override in overrides):
        overrides.append("exp=fmt_simgen_v2_source_slots_soft_union")

    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = hydra.compose(config_name="config", overrides=overrides)

    net = ModelFactory.create_model(cfg.model.name, config=cfg).to(device)
    net.eval()

    data_dir = cfg.data.get(f"{args.split}_dir", None) or cfg.data.get("data_dir", None)
    if data_dir and Path(str(data_dir)).exists():
        try:
            dataset = FmtSimGenProjDataset(
                str(data_dir), config=cfg, split=args.split, is_training=False
            )
            batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
            proj_in = _projection_input_from_batch(batch).to(device)
            points = batch["points"].to(device)
            points_mm = batch["points_mm"].to(device)
            depth_maps = batch["depth_maps"].to(device)
            source_hypotheses = {
                key: value.to(device) if value is not None else None
                for key, value in (_source_hypotheses_from_batch(batch) or {}).items()
            }
        except (FileNotFoundError, RuntimeError, KeyError) as exc:
            print(f"dataset smoke input unavailable, using synthetic batch: {exc}")
            proj_in, points, points_mm, depth_maps, source_hypotheses = _synthetic_batch(
                cfg, device
            )
    else:
        proj_in, points, points_mm, depth_maps, source_hypotheses = _synthetic_batch(cfg, device)

    with torch.no_grad():
        logits, aux = net(
            proj_in,
            points,
            points_mm=points_mm,
            depth_maps=depth_maps,
            source_hypotheses=source_hypotheses,
        )
    _assert_outputs(logits, aux, expected_m=source_hypotheses["centers"].shape[1])

    with torch.no_grad():
        fallback_logits, _fallback_aux = net(
            proj_in,
            points,
            points_mm=points_mm,
            depth_maps=depth_maps,
            source_hypotheses=None,
        )
    if fallback_logits.shape != logits.shape or not torch.isfinite(fallback_logits).all():
        raise AssertionError("no-hypothesis fallback failed")

    net.source_instance_decoder_merge = "weighted_sum"
    net.source_instance_ownership_mode = "distance"
    net.source_instance_decoder_use_all_slots = False
    net.source_instance_decoder_top_k = 2
    with torch.no_grad():
        old_logits, old_aux = net(
            proj_in,
            points,
            points_mm=points_mm,
            depth_maps=depth_maps,
            source_hypotheses=source_hypotheses,
        )
    _assert_outputs(old_logits, old_aux, expected_m=2)
    print("source-slot soft-union smoke passed")


if __name__ == "__main__":
    main()
