#!/usr/bin/env python
"""Smoke test for s3 exit-depth Gaussian PTFA."""

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
from minr_fmt.network.ptfa import (  # noqa: E402
    ptfa_sample_exit_depth_gaussian,
    ptfa_sample_fixed_gaussian,
)
from minr_fmt.utils.fmt_simgen_projection import project_points_mm_to_detector  # noqa: E402


def pack_projection_input(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    p = batch["projections_packed"]
    return p.permute(1, 0, 2, 3, 4).reshape(
        p.shape[0] * p.shape[1], 1, p.shape[-2], p.shape[-1]
    )


def print_hist(name: str, x: torch.Tensor, bins: int = 4) -> None:
    x = x.detach().float().cpu()
    hist = torch.histc(x, bins=bins, min=float(x.min()), max=float(x.max()))
    print(f"{name} hist bins={bins} range=({float(x.min()):.4f},{float(x.max()):.4f})")
    print(f"{name} hist counts:", [int(v) for v in hist.tolist()])


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
                "exp=fmt_simgen_e3_ptfa_s3_exit_depth",
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
    depth_maps = batch["depth_maps"].to(device)

    net = ModelFactory.create_model(cfg.model.name, config=cfg).to(device)
    net.eval()
    with torch.no_grad():
        _s1, _s2, s3 = net._forward_shared_unet(proj_in)
        center_grid, valid_mask = net._fmt_projection_grids(points_mm)
        feature_map = net._view_feature_dict_to_tensor(s3)

        query_depths = []
        for angle in cfg.data.view_angles:
            _grid, depth, _valid, _uv_px, _uv_phys = project_points_mm_to_detector(
                points_mm,
                int(angle),
                camera_distance_mm=cfg.model.geometry.camera_distance,
                fov_mm=cfg.model.geometry.fov_mm,
                detector_resolution=tuple(cfg.model.geometry.detector_resolution),
                volume_center_world=tuple(cfg.model.geometry.volume_center_world),
            )
            query_depths.append(depth)
        query_depth = torch.stack(query_depths, dim=2)

        e3, stats = ptfa_sample_exit_depth_gaussian(
            feature_map,
            center_grid,
            valid_mask,
            depth_maps,
            query_depth,
            sigma_min=cfg.model.ptfa.sigma_min,
            sigma_max=cfg.model.ptfa.sigma_max,
            exit_depth_max=cfg.model.ptfa.exit_depth_max_mm,
            window=cfg.model.ptfa.window,
        )
        e2 = ptfa_sample_fixed_gaussian(
            feature_map, center_grid, valid_mask, window=5, sigma_px=1.0
        )
        degenerate, _stats_degenerate = ptfa_sample_exit_depth_gaussian(
            feature_map,
            center_grid,
            valid_mask,
            depth_maps,
            query_depth,
            sigma_min=1.0,
            sigma_max=1.0,
            exit_depth_max=cfg.model.ptfa.exit_depth_max_mm,
            window=cfg.model.ptfa.window,
        )
        density_pred, _aux = net(proj_in, points, points_mm=points_mm, depth_maps=depth_maps)

    exit_depth = stats["exit_depth_mm"][valid_mask]
    sigma = stats["sigma_px"][valid_mask]
    print("depth_maps shape:", tuple(depth_maps.shape))
    print(
        "exit_depth min/mean/max:",
        float(exit_depth.min()),
        float(exit_depth.mean()),
        float(exit_depth.max()),
    )
    print("exit_depth >0 ratio:", float((exit_depth > 0).float().mean()))
    print(
        "sigma min/mean/max:",
        float(sigma.min()),
        float(sigma.mean()),
        float(sigma.max()),
    )
    for v_idx, angle in enumerate(cfg.data.view_angles):
        print_hist(f"sigma angle={angle}", stats["sigma_px"][:, :, v_idx][valid_mask[:, :, v_idx]])

    diff = (e3 - e2).abs()
    degenerate_diff = (degenerate - e2).abs()
    print("E3_vs_E2 max_abs_delta:", float(diff.max()))
    print("E3_vs_E2 mean_abs_delta:", float(diff.mean()))
    print("degenerate_fixed_sigma max_abs_delta:", float(degenerate_diff.max()))
    print("degenerate_fixed_sigma mean_abs_delta:", float(degenerate_diff.mean()))
    print("E3 has_nan:", bool(torch.isnan(e3).any()))
    print("E3 has_inf:", bool(torch.isinf(e3).any()))
    print("density_pred shape:", tuple(density_pred.shape))
    print(
        "density_pred min/max/mean:",
        float(density_pred.min()),
        float(density_pred.max()),
        float(density_pred.mean()),
    )


if __name__ == "__main__":
    main()
