#!/usr/bin/env python
"""Dump first-batch depth/source sanity for FMT-SimGen."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.dataset.fmt_simgen_dataset import FmtSimGenProjDataset  # noqa: E402
from minr_fmt.utils.fmt_simgen_projection import project_points_mm_to_detector  # noqa: E402


def sample_surface_depth(depth_maps: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    b, v, h, w = depth_maps.shape
    n = grid.shape[1]
    depth_flat = depth_maps.reshape(b * v, 1, h, w)
    grid_flat = grid.permute(0, 2, 1, 3).reshape(b * v, n, 2).unsqueeze(1)
    sampled = F.grid_sample(
        depth_flat,
        grid_flat,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).squeeze(2)
    return sampled[:, 0].reshape(b, v, n).permute(0, 2, 1)


def describe(x: np.ndarray) -> tuple[float, float, float]:
    return float(np.min(x)), float(np.median(x)), float(np.max(x))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="fmt_simgen_e4_residual_scorer_lambda005")
    parser.add_argument("--out", default="outputs/diag/depth_source_sanity.txt")
    parser.add_argument("--num_queries", type=int, default=16384)
    args = parser.parse_args()

    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                f"exp={args.exp}",
                f"data.sample_num={args.num_queries}",
                f"data.num_queries={args.num_queries}",
                "data.train_max_samples=1",
                "data.val_max_samples=1",
                "model.geometry.use_fmt_simgen_projection=true",
            ],
        )

    ds = FmtSimGenProjDataset(cfg.data.data_dir, config=cfg, split="train", is_training=True)
    batch = next(iter(DataLoader(ds, batch_size=1, num_workers=0)))
    points_mm = batch["points_mm"]
    depth_maps = batch["depth_maps"]

    grids, query_depths, valid_masks = [], [], []
    for angle in cfg.data.view_angles:
        grid, depth, valid, _uv_px, _uv_phys = project_points_mm_to_detector(
            points_mm,
            int(angle),
            camera_distance_mm=cfg.model.geometry.camera_distance,
            fov_mm=cfg.model.geometry.fov_mm,
            detector_resolution=tuple(cfg.model.geometry.detector_resolution),
            volume_center_world=tuple(cfg.model.geometry.volume_center_world),
        )
        grids.append(grid)
        query_depths.append(depth)
        valid_masks.append(valid)
    grid_all = torch.stack(grids, dim=2)
    query_depth = torch.stack(query_depths, dim=2).squeeze(-1)
    valid = torch.stack(valid_masks, dim=2)
    surface_depth = sample_surface_depth(depth_maps, grid_all)
    exit_depth = torch.clamp(query_depth - surface_depth, min=0.0, max=20.8)

    foreground = batch["point_densities"] > 0
    label_z = points_mm[..., 2][foreground]
    fg_view_mask = foreground.unsqueeze(-1).expand_as(valid) & valid
    depth_fg = exit_depth[fg_view_mask]
    label_z_view = points_mm[..., 2].unsqueeze(-1).expand_as(valid)[fg_view_mask]

    depth_np = depth_fg.detach().cpu().numpy().astype(np.float64)
    label_z_np = label_z.detach().cpu().numpy().astype(np.float64)
    label_z_view_np = label_z_view.detach().cpu().numpy().astype(np.float64)
    if len(depth_np) > 1 and np.std(depth_np) > 0 and np.std(label_z_view_np) > 0:
        corr = float(np.corrcoef(depth_np, label_z_view_np)[0, 1])
    else:
        corr = float("nan")

    d_min, d_med, d_max = describe(depth_np) if len(depth_np) else (float("nan"),) * 3
    z_min, z_med, z_max = describe(label_z_np) if len(label_z_np) else (float("nan"),) * 3
    if corr > 0.5:
        verdict = "corr > 0.5: depth input direction is consistent with label_z trend."
    elif corr < -0.3:
        verdict = "corr < -0.3: depth behaves like an inverted/exit-depth-like quantity."
    else:
        verdict = "corr near 0: depth is weakly related to label_z for foreground queries."

    text = "\n".join(
        [
            f"depth_per_query_exit_mm min/median/max: {d_min:.6f} {d_med:.6f} {d_max:.6f}",
            f"label_z_mm foreground min/median/max: {z_min:.6f} {z_med:.6f} {z_max:.6f}",
            f"foreground_queries: {int(foreground.sum())} foreground_view_pairs: {len(depth_np)}",
            f"corr(depth_per_query_exit_mm, label_z_mm) foreground_view_pairs: {corr:.6f}",
            f"interpretation: {verdict}",
        ]
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
