#!/usr/bin/env python
"""Run one GISC-FMT forward pass on an FMT-SimGen batch."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--num_queries", type=int, default=2048)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args, overrides = parser.parse_known_args()

    compose_overrides = list(overrides)
    if not compose_overrides:
        compose_overrides.append("exp=fmt_simgen_e1_data_smoke")
    if args.data_dir is not None:
        compose_overrides.extend(
            [
                f"data.data_dir={args.data_dir}",
                f"data.train_dir={args.data_dir}",
                f"data.val_dir={args.data_dir}",
                f"data.test_dir={args.data_dir}",
            ]
        )
    compose_overrides.extend(
        [
            f"data.sample_num={args.num_queries}",
            f"data.num_queries={args.num_queries}",
            "data.train_max_samples=2",
            f"trainer.accelerator={args.device}",
            "trainer.precision=32-true",
        ]
    )

    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = hydra.compose(config_name="config", overrides=compose_overrides)

    data_dir = args.data_dir or str(cfg.data.train_dir)
    if not data_dir:
        raise SystemExit("--data_dir or data.train_dir must be provided")
    ds = FmtSimGenProjDataset(data_dir, config=cfg, split="train", is_training=True)
    batch = next(iter(DataLoader(ds, batch_size=1, num_workers=0)))
    p = batch["projections_packed"]
    proj_in = p.permute(1, 0, 2, 3, 4).reshape(p.shape[0] * p.shape[1], 1, p.shape[-2], p.shape[-1])

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")
    net = ModelFactory.create_model(cfg.model.name, config=cfg).to(device)
    net.eval()
    with torch.no_grad():
        if str(cfg.model.name).lower() == "ssq_fmt":
            batch_for_model = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            out = net(
                batch.get("surface_measurements_packed", p).to(device),
                batch.get("query_coordinates_mm", batch["points_mm"]).to(device),
                detector_valid_mask=batch_for_model.get("detector_valid_mask"),
                depth_maps=batch_for_model.get("depth_maps"),
                batch=batch_for_model,
            )
            density_pred = out["density"]
            aux = out.get("aux_outputs", {})
        else:
            density_pred, aux = net(proj_in.to(device), batch["points"].to(device))

    print("batch projections_packed shape:", tuple(p.shape))
    print("batch points shape:", tuple(batch["points"].shape))
    print("density_pred shape:", tuple(density_pred.shape))
    print(
        "density_pred min/max/mean:",
        float(density_pred.min()),
        float(density_pred.max()),
        float(density_pred.mean()),
    )
    print("aux output keys:", [str(k) for k in aux.keys()])


if __name__ == "__main__":
    main()
