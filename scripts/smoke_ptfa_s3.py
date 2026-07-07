#!/usr/bin/env python
"""Smoke test for s3-only fixed Gaussian PTFA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.dataset.fmt_simgen_dataset import FmtSimGenProjDataset  # noqa: E402
from minr_fmt.model_factory import ModelFactory  # noqa: E402
from minr_fmt.network.ptfa import ptfa_sample_fixed_gaussian  # noqa: E402


def pack_projection_input(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    p = batch["projections_packed"]
    return p.permute(1, 0, 2, 3, 4).reshape(
        p.shape[0] * p.shape[1], 1, p.shape[-2], p.shape[-1]
    )


def nearest_sample(feature_map: torch.Tensor, center_grid: torch.Tensor) -> torch.Tensor:
    B, V, C, _H, _W = feature_map.shape
    N = center_grid.shape[1]
    feat_flat = feature_map.reshape(B * V, C, feature_map.shape[-2], feature_map.shape[-1])
    grid = center_grid.permute(0, 2, 1, 3).reshape(B * V, N, 2).unsqueeze(1)
    sampled = F.grid_sample(
        feat_flat,
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(2)
    return sampled.permute(0, 2, 1).reshape(B, V, N, C).permute(0, 2, 1, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--num_queries", type=int, default=512)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()

    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                "exp=fmt_simgen_e2_ptfa_s3_fixed",
                f"data.data_dir={args.data_dir}",
                f"data.train_dir={args.data_dir}",
                f"data.val_dir={args.data_dir}",
                f"data.test_dir={args.data_dir}",
                f"data.sample_num={args.num_queries}",
                f"data.num_queries={args.num_queries}",
                "data.train_max_samples=1",
                "data.val_max_samples=1",
                "model.geometry.use_fmt_simgen_projection=true",
            ],
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device(args.device)

    ds = FmtSimGenProjDataset(args.data_dir, config=cfg, split="train", is_training=True)
    batch = next(iter(DataLoader(ds, batch_size=1, num_workers=0)))
    proj_in = pack_projection_input(batch).to(device)
    points = batch["points"].to(device)
    points_mm = batch["points_mm"].to(device)

    net = ModelFactory.create_model(cfg.model.name, config=cfg).to(device)
    net.eval()
    with torch.no_grad():
        _s1, _s2, s3 = net._forward_shared_unet(proj_in)
        bilinear = net._vectorized_grid_sample_fmt(s3, points_mm)
        ptfa = net._ptfa_sample_fixed_gaussian(s3, points_mm)
        diff = (ptfa - bilinear).abs()

        center_grid, valid_mask = net._fmt_projection_grids(points_mm)
        feature_map = net._view_feature_dict_to_tensor(s3)
        ptfa_near = ptfa_sample_fixed_gaussian(
            feature_map,
            center_grid,
            valid_mask,
            window=5,
            sigma_px=1.0e-3,
        )
        nearest = nearest_sample(feature_map, center_grid)
        nearest = nearest * valid_mask.unsqueeze(-1).to(dtype=nearest.dtype)
        nearest_diff = (ptfa_near - nearest).abs()

        density_pred, _aux = net(proj_in, points, points_mm=points_mm)

    print("f_s3 bilinear shape:", tuple(bilinear.shape))
    print("f_s3 ptfa shape:", tuple(ptfa.shape))
    print("f_s3 max_abs_delta:", float(diff.max()))
    print("f_s3 mean_abs_delta:", float(diff.mean()))
    print("f_s3 has_nan:", bool(torch.isnan(ptfa).any()))
    print("sigma_to_zero_nearest max_abs_delta:", float(nearest_diff.max()))
    print("sigma_to_zero_nearest mean_abs_delta:", float(nearest_diff.mean()))
    print("density_pred shape:", tuple(density_pred.shape))
    print(
        "density_pred min/max/mean:",
        float(density_pred.min()),
        float(density_pred.max()),
        float(density_pred.mean()),
    )


if __name__ == "__main__":
    main()
