#!/usr/bin/env python3
"""Verify zero-gain isolated routing is numerically identical to A2-U."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from minr_fmt.datamodule import TrainingDataModule  # noqa: E402
from minr_fmt.module import TrainingLightningModule  # noqa: E402


def to_device(value):
    if torch.is_tensor(value):
        return value.cuda()
    if isinstance(value, dict):
        return {key: to_device(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg_a2 = OmegaConf.load(args.run_dir / "config/config.yaml")
    cfg_a2.data.num_workers = 0
    cfg_a2.data.val_max_samples = 1
    cfg_routing = OmegaConf.create(OmegaConf.to_container(cfg_a2, resolve=True))
    view = cfg_routing.model.ssq_fmt.view_complementary
    view.ablation = "isolated_bounded_routing"
    view.continuous_applicability = False
    view.hypothesis_grid = {"enabled": False}
    view.routing = {"enabled": True, "delta_logit_max": 1.0, "zero_init": True}
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    a2 = TrainingLightningModule(cfg_a2)
    a2.load_state_dict(state, strict=True)
    routing = TrainingLightningModule(cfg_routing)
    net_state = {
        key.removeprefix("net."): value
        for key, value in state.items()
        if key.startswith("net.")
    }
    missing, unexpected = routing.net.load_state_dict(net_state, strict=False)
    expected = ["bounded_view_routing.routing_gain_raw"]
    if missing != expected or unexpected:
        raise RuntimeError(
            f"Unexpected checkpoint delta: missing={missing}, unexpected={unexpected}"
        )
    a2.eval().cuda()
    routing.eval().cuda()
    data = TrainingDataModule(cfg_a2)
    data.setup("validate")
    batch = to_device(next(iter(data.val_dataloader())))
    with torch.inference_mode():
        pred_a2 = a2._call_ssq_model(batch, return_diagnostics=True)["density"]
        pred_routing = routing._call_ssq_model(batch, return_diagnostics=True)["density"]
    maximum = float((pred_a2 - pred_routing).abs().max())
    payload = {
        "max_abs_density_difference": maximum,
        "tolerance": 1e-6,
        "passed": maximum < 1e-6,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    if maximum >= 1e-6:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
