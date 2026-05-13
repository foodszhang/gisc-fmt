"""Torch projector matching FMT-SimGen turntable orthographic geometry."""

from __future__ import annotations

import math

import torch


def rotation_matrix_y_torch(angle_deg, device, dtype) -> torch.Tensor:
    """Return the FMT-SimGen Y-axis rotation matrix."""
    angle_rad = math.radians(float(angle_deg))
    cos_t = torch.tensor(math.cos(angle_rad), device=device, dtype=dtype)
    sin_t = torch.tensor(math.sin(angle_rad), device=device, dtype=dtype)
    zero = torch.zeros((), device=device, dtype=dtype)
    one = torch.ones((), device=device, dtype=dtype)
    return torch.stack(
        [
            torch.stack([cos_t, zero, sin_t]),
            torch.stack([zero, one, zero]),
            torch.stack([-sin_t, zero, cos_t]),
        ]
    )


def project_points_mm_to_detector(
    points_mm: torch.Tensor,
    angle_deg: float | int,
    camera_distance_mm: float = 200.0,
    fov_mm: float = 80.0,
    detector_resolution: tuple[int, int] = (256, 256),
    volume_center_world: tuple[float, float, float] = (19.0, 20.0, 10.4),
    align_corners: bool = True,
):
    """Project trunk-local mm points to FMT-SimGen detector coordinates.

    Returns:
      grid: [B,N,2] normalized grid for grid_sample
      depth: [B,N,1] camera-frame depth
      valid_mask: [B,N] bool
      uv_px: [B,N,2] continuous FMT-SimGen pixel coordinates
      uv_phys: [B,N,2] detector physical coordinates in mm
    """
    squeeze_batch = False
    if points_mm.dim() == 2:
        points_mm = points_mm.unsqueeze(0)
        squeeze_batch = True
    if points_mm.dim() != 3 or points_mm.shape[-1] != 3:
        raise ValueError(f"points_mm must be [N,3] or [B,N,3], got {tuple(points_mm.shape)}")

    device = points_mm.device
    dtype = points_mm.dtype
    center = torch.as_tensor(volume_center_world, device=device, dtype=dtype)
    p = points_mm - center

    angle_rad = math.radians(float(angle_deg))
    cos_t = torch.tensor(math.cos(angle_rad), device=device, dtype=dtype)
    sin_t = torch.tensor(math.sin(angle_rad), device=device, dtype=dtype)

    x_rot = p[..., 0] * cos_t + p[..., 2] * sin_t
    y_rot = p[..., 1]
    z_rot = -p[..., 0] * sin_t + p[..., 2] * cos_t
    depth = (float(camera_distance_mm) - z_rot).unsqueeze(-1)

    half_fov = float(fov_mm) / 2.0
    uv_phys = torch.stack([x_rot, y_rot], dim=-1)
    w_px, h_px = int(detector_resolution[0]), int(detector_resolution[1])
    u_px = (x_rot + half_fov) / float(fov_mm) * w_px
    v_px = (y_rot + half_fov) / float(fov_mm) * h_px
    uv_px = torch.stack([u_px, v_px], dim=-1)

    # DU2Vox and FMT-SimGen alignment use physical FOV normalization. This keeps the
    # rotation-center point exactly at grid=(0,0) and avoids pixel-center ambiguity.
    grid = uv_phys / half_fov
    if not align_corners:
        # Same physical NDC is also valid with align_corners=False; callers choose
        # interpolation convention. Kept explicit so the parameter is not ignored.
        grid = grid.clone()

    valid_mask = (
        (x_rot >= -half_fov)
        & (x_rot < half_fov)
        & (y_rot >= -half_fov)
        & (y_rot < half_fov)
        & (depth.squeeze(-1) > 0)
    )

    if squeeze_batch:
        return grid[0], depth[0], valid_mask[0], uv_px[0], uv_phys[0]
    return grid, depth, valid_mask, uv_px, uv_phys


def project_points_mm_all_views(
    points_mm: torch.Tensor,
    view_angles: list[int],
    camera_distance_mm: float = 200.0,
    fov_mm: float = 80.0,
    detector_resolution: tuple[int, int] = (256, 256),
    volume_center_world: tuple[float, float, float] = (19.0, 20.0, 10.4),
    align_corners: bool = True,
):
    """Project points for all views in the provided order."""
    grids, depths, masks, uv_pixels = [], [], [], []
    for angle in view_angles:
        grid, depth, valid_mask, uv_px, _uv_phys = project_points_mm_to_detector(
            points_mm,
            angle,
            camera_distance_mm=camera_distance_mm,
            fov_mm=fov_mm,
            detector_resolution=detector_resolution,
            volume_center_world=volume_center_world,
            align_corners=align_corners,
        )
        grids.append(grid)
        depths.append(depth)
        masks.append(valid_mask)
        uv_pixels.append(uv_px)

    return (
        torch.stack(grids, dim=2),
        torch.stack(depths, dim=2),
        torch.stack(masks, dim=2),
        torch.stack(uv_pixels, dim=2),
    )
