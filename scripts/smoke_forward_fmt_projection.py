#!/usr/bin/env python
"""Forward smoke test for FMT-SimGen physical projection sampling."""

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
from minr_fmt.utils.fmt_simgen_projection import project_points_mm_to_detector  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--num_queries", type=int, default=2048)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()

    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                "exp=fmt_simgen_e1_data_smoke",
                f"data.data_dir={args.data_dir}",
                f"data.train_dir={args.data_dir}",
                f"data.val_dir={args.data_dir}",
                f"data.test_dir={args.data_dir}",
                f"data.sample_num={args.num_queries}",
                f"data.num_queries={args.num_queries}",
                "data.train_max_samples=2",
                "model.geometry.use_fmt_simgen_projection=true",
            ],
        )

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    ds = FmtSimGenProjDataset(args.data_dir, config=cfg, split="train", is_training=True)
    batch = next(iter(DataLoader(ds, batch_size=1, num_workers=0)))
    points_mm = batch["points_mm"]
    print("points shape:", tuple(batch["points"].shape))
    print("points_mm shape:", tuple(points_mm.shape))
    print("points_mm min:", points_mm.amin(dim=(0, 1)).tolist())
    print("points_mm max:", points_mm.amax(dim=(0, 1)).tolist())

    for angle in cfg.data.view_angles:
        grid, _depth, valid, _uv_px, _uv_phys = project_points_mm_to_detector(
            points_mm,
            int(angle),
            camera_distance_mm=cfg.model.geometry.camera_distance,
            fov_mm=cfg.model.geometry.fov_mm,
            detector_resolution=tuple(cfg.model.geometry.detector_resolution),
            volume_center_world=tuple(cfg.model.geometry.volume_center_world),
        )
        print(
            f"angle={angle} grid_min={float(grid.min()):.6f} "
            f"grid_max={float(grid.max()):.6f} valid_ratio={float(valid.float().mean()):.6f}"
        )

    p = batch["projections_packed"]
    proj_in = p.permute(1, 0, 2, 3, 4).reshape(
        p.shape[0] * p.shape[1], 1, p.shape[-2], p.shape[-1]
    )
    net = ModelFactory.create_model(cfg.model.name, config=cfg).to(device)
    net.eval()
    with torch.no_grad():
        density_pred, aux = net(
            proj_in.to(device),
            batch["points"].to(device),
            points_mm=points_mm.to(device),
        )

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
