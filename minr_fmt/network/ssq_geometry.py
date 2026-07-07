"""Geometry mapping and detector-coordinate helpers for SSQ-FMT."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from minr_fmt.utils.fmt_simgen_projection import project_points_mm_to_detector


def grid_to_pixel(
    grid: torch.Tensor, height: int, width: int, align_corners: bool = True
) -> torch.Tensor:
    if align_corners:
        x = (grid[..., 0] + 1.0) * 0.5 * max(width - 1, 1)
        y = (grid[..., 1] + 1.0) * 0.5 * max(height - 1, 1)
    else:
        x = ((grid[..., 0] + 1.0) * width - 1.0) * 0.5
        y = ((grid[..., 1] + 1.0) * height - 1.0) * 0.5
    return torch.stack([x, y], dim=-1)


def pixel_to_grid(
    px: torch.Tensor, height: int, width: int, align_corners: bool = True
) -> torch.Tensor:
    if align_corners:
        gx = px[..., 0] / max(width - 1, 1) * 2.0 - 1.0
        gy = px[..., 1] / max(height - 1, 1) * 2.0 - 1.0
    else:
        gx = (2.0 * px[..., 0] + 1.0) / max(width, 1) - 1.0
        gy = (2.0 * px[..., 1] + 1.0) / max(height, 1) - 1.0
    return torch.stack([gx, gy], dim=-1)


def sample_scalar_map(
    image: torch.Tensor,
    grid: torch.Tensor,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """Sample [B,V,H,W] or [B,V,1,H,W] at [B,V,N,K,2] or [B,V,N,2]."""
    if image.dim() == 4:
        image = image[:, :, None]
    b, v, c, h, w = image.shape
    grid_in = grid
    squeeze_k = False
    if grid.dim() == 4:
        grid_in = grid[:, :, :, None, :]
        squeeze_k = True
    n, k = grid_in.shape[2], grid_in.shape[3]
    sampled = F.grid_sample(
        image.reshape(b * v, c, h, w),
        grid_in.reshape(b * v, n * k, 1, 2),
        mode=mode,
        padding_mode="zeros",
        align_corners=align_corners,
    )
    sampled = sampled.squeeze(-1).transpose(1, 2).reshape(b, v, n, k, c)
    if squeeze_k:
        sampled = sampled.squeeze(3)
    return sampled.squeeze(-1) if c == 1 else sampled


def sample_finite_scalar_map(
    image: torch.Tensor,
    grid: torch.Tensor,
    *,
    valid_weight_min: float = 1.0e-4,
    align_corners: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinearly sample finite scalar maps without letting inf/nan contaminate interpolation."""
    finite = torch.isfinite(image)
    clean = torch.where(finite, image, torch.zeros_like(image))
    value = sample_scalar_map(clean, grid, mode="bilinear", align_corners=align_corners)
    weight = sample_scalar_map(
        finite.to(dtype=image.dtype), grid, mode="bilinear", align_corners=align_corners
    )
    valid = weight > float(valid_weight_min)
    sampled = value / weight.clamp_min(float(valid_weight_min))
    sampled = torch.where(valid, sampled, torch.zeros_like(sampled))
    return sampled, valid


class GeometryQueryMapper(nn.Module):
    """Vectorized FMT-SimGen orthographic query-to-detector mapper."""

    def __init__(
        self,
        view_angles: list[int],
        camera_distance_mm: float = 200.0,
        fov_mm: float = 80.0,
        detector_resolution: tuple[int, int] = (256, 256),
        volume_center_world: tuple[float, float, float] = (19.0, 20.0, 10.4),
        fd_step_mm: float = 0.2,
        path_max_mm: float = 20.0,
        path_sign: str = "query_minus_surface",
        depth_valid_weight_min: float = 1.0e-4,
        align_corners: bool = True,
    ):
        super().__init__()
        self.view_angles = [int(a) for a in view_angles]
        self.camera_distance_mm = float(camera_distance_mm)
        self.fov_mm = float(fov_mm)
        self.detector_resolution = (int(detector_resolution[0]), int(detector_resolution[1]))
        self.volume_center_world = tuple(float(x) for x in volume_center_world)
        self.fd_step_mm = float(fd_step_mm)
        self.path_max_mm = float(path_max_mm)
        if path_sign not in {"query_minus_surface", "surface_minus_query"}:
            raise ValueError("path_sign must be query_minus_surface or surface_minus_query")
        self.path_sign = str(path_sign)
        self.depth_valid_weight_min = float(depth_valid_weight_min)
        self.align_corners = bool(align_corners)

    def forward(
        self,
        points_mm: torch.Tensor,
        *,
        depth_maps: torch.Tensor | None = None,
        detector_margin_map: torch.Tensor | None = None,
        detector_valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        grids, depths, masks, uv_px, uv_phys, rays = [], [], [], [], [], []
        h, w = self.detector_resolution
        px_per_mm = (w - 1) / max(self.fov_mm, 1e-6)
        pixels_per_mm = torch.tensor(px_per_mm, dtype=points_mm.dtype, device=points_mm.device)
        for angle in self.view_angles:
            grid, depth, valid, px, phys = project_points_mm_to_detector(
                points_mm,
                angle,
                camera_distance_mm=self.camera_distance_mm,
                fov_mm=self.fov_mm,
                detector_resolution=self.detector_resolution,
                volume_center_world=self.volume_center_world,
                align_corners=self.align_corners,
            )
            grids.append(grid)
            depths.append(depth)
            masks.append(valid)
            uv_px.append(px)
            uv_phys.append(phys)
            rad = math.radians(float(angle))
            ray = torch.tensor(
                [math.sin(rad), 0.0, -math.cos(rad)],
                dtype=points_mm.dtype,
                device=points_mm.device,
            )
            rays.append(ray.expand(points_mm.shape[0], points_mm.shape[1], 3))

        grid_all = torch.stack(grids, dim=1)
        depth_all = torch.stack(depths, dim=1)
        valid_all = torch.stack(masks, dim=1)
        if detector_valid_mask is not None:
            mask = (
                detector_valid_mask.squeeze(2)
                if detector_valid_mask.dim() == 5
                else detector_valid_mask
            )
            center_mask = (
                sample_scalar_map(
                    mask.float(), grid_all, mode="nearest", align_corners=self.align_corners
                )
                > 0.5
            )
            valid_all = valid_all & center_mask

        if depth_maps is not None:
            surf_depth, finite_surface = sample_finite_scalar_map(
                depth_maps,
                grid_all,
                valid_weight_min=self.depth_valid_weight_min,
                align_corners=self.align_corners,
            )
            query_depth = depth_all[..., 0]
            if self.path_sign == "surface_minus_query":
                path = surf_depth - query_depth
            else:
                path = query_depth - surf_depth
            path = torch.where(finite_surface, path, torch.zeros_like(path)).clamp_min(0.0)
            xi = (path / max(self.path_max_mm, 1e-6)).clamp(0.0, 1.0)
            surf_depth = torch.where(finite_surface, surf_depth, torch.zeros_like(surf_depth))
            valid_all = valid_all & finite_surface
        else:
            surf_depth = torch.full_like(depth_all[..., 0], float("nan"))
            path = torch.zeros_like(depth_all[..., 0])
            xi = torch.zeros_like(path)
            finite_surface = torch.zeros_like(valid_all)
            valid_all = torch.zeros_like(valid_all)

        if detector_margin_map is not None:
            margin = sample_scalar_map(
                detector_margin_map.float(),
                grid_all,
                mode="bilinear",
                align_corners=self.align_corners,
            ).clamp_min(0.0)
        else:
            margin = (1.0 - grid_all.abs()).amin(dim=-1).clamp(0.0, 1.0) * (min(h, w) - 1) * 0.5
        margin = torch.nan_to_num(margin, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "grid": grid_all,
            "depth": depth_all,
            "surface_depth": surf_depth,
            "detector_side_path_mm": path,
            "detector_side_path_proxy": xi,
            "valid_mask": valid_all,
            "finite_depth_mask": finite_surface,
            "uv_px": grid_to_pixel(grid_all, h, w, self.align_corners),
            "uv_phys": torch.stack(uv_phys, dim=1),
            "ray_directions": torch.stack(rays, dim=1),
            "view_pixels_per_mm": pixels_per_mm.expand(len(self.view_angles)),
            "boundary_distance": margin,
        }


def infer_detector_margin_map(batch: dict[str, Any] | None) -> torch.Tensor | None:
    if not batch:
        return None
    for key in ("detector_margin_map", "detector_margin_maps"):
        if key in batch:
            return batch[key]
    return None
