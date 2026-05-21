"""Fixed Gaussian footprint point-to-feature aggregation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCFSSigmaCalibrator(nn.Module):
    """Small zero-init calibration MLP for corrected exit-depth sigma generation."""

    def __init__(
        self,
        geom_dim: int,
        hidden_dim: int = 64,
        delta_max: float = 0.1,
        norm: str = "layernorm",
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.delta_max = float(delta_max)
        if self.delta_max < 0:
            raise ValueError(f"delta_max must be non-negative, got {delta_max}")
        layers: list[nn.Module] = []
        if norm == "layernorm":
            layers.append(nn.LayerNorm(geom_dim))
        elif norm not in {"none", ""}:
            raise ValueError(f"Unsupported PCFS norm: {norm}")
        layers.extend(
            [
                nn.Linear(geom_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            ]
        )
        self.net = nn.Sequential(*layers)
        if zero_init:
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(self, geom: torch.Tensor) -> torch.Tensor:
        """Return bounded delta_h [B,N,V] in normalized sigma-depth space."""
        delta = torch.tanh(self.net(geom)).squeeze(-1)
        return self.delta_max * delta


def ptfa_sample_fixed_gaussian(
    feature_map: torch.Tensor,
    center_grid: torch.Tensor,
    valid_mask: torch.Tensor,
    window: int,
    sigma_px: float,
) -> torch.Tensor:
    """Sample multi-view features with a fixed 2D Gaussian detector footprint.

    Args:
        feature_map: [B,V,C,H,W]
        center_grid: [B,N,V,2] normalized grid coordinates, align_corners=True
        valid_mask: [B,N,V] bool view validity
        window: odd square footprint size in feature-map pixel units
        sigma_px: Gaussian sigma in feature-map pixel units

    Returns:
        [B,N,V,C] aggregated features. Invalid views are exactly zero.
    """
    if feature_map.dim() != 5:
        raise ValueError(f"feature_map must be [B,V,C,H,W], got {tuple(feature_map.shape)}")
    if center_grid.dim() != 4 or center_grid.shape[-1] != 2:
        raise ValueError(f"center_grid must be [B,N,V,2], got {tuple(center_grid.shape)}")
    if valid_mask.dim() != 3:
        raise ValueError(f"valid_mask must be [B,N,V], got {tuple(valid_mask.shape)}")
    if window % 2 != 1 or window <= 0:
        raise ValueError(f"window must be a positive odd integer, got {window}")
    if sigma_px <= 0:
        raise ValueError(f"sigma_px must be positive, got {sigma_px}")

    B, V, C, H, W = feature_map.shape
    if center_grid.shape[0] != B or center_grid.shape[2] != V:
        raise ValueError(
            f"center_grid shape {tuple(center_grid.shape)} incompatible with feature_map "
            f"{tuple(feature_map.shape)}"
        )
    N = center_grid.shape[1]
    if valid_mask.shape != (B, N, V):
        raise ValueError(f"valid_mask shape {tuple(valid_mask.shape)} != {(B, N, V)}")

    dtype = feature_map.dtype
    device = feature_map.device
    feat_flat = feature_map.reshape(B * V, C, H, W)
    center = center_grid.permute(0, 2, 1, 3).reshape(B * V, N, 2)
    view_valid = valid_mask.permute(0, 2, 1).reshape(B * V, N).to(device=device)

    if W > 1:
        center_x = (center[..., 0] + 1.0) * (W - 1) / 2.0
    else:
        center_x = torch.zeros_like(center[..., 0])
    if H > 1:
        center_y = (center[..., 1] + 1.0) * (H - 1) / 2.0
    else:
        center_y = torch.zeros_like(center[..., 1])

    radius = window // 2
    offsets = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
    offsets_xy = torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=-1)

    base_x = torch.round(center_x)
    base_y = torch.round(center_y)
    out = torch.zeros(B * V, N, C, device=device, dtype=dtype)
    denom = torch.zeros(B * V, N, device=device, dtype=dtype)
    min_dist2 = torch.full((B * V, N), torch.inf, device=device, dtype=dtype)

    for offset_x, offset_y in offsets_xy:
        sample_x = base_x + offset_x
        sample_y = base_y + offset_y
        inside = (
            (sample_x >= 0)
            & (sample_x <= W - 1)
            & (sample_y >= 0)
            & (sample_y <= H - 1)
            & view_valid
        )
        dist2 = (sample_x - center_x).square() + (sample_y - center_y).square()
        min_dist2 = torch.minimum(min_dist2, torch.where(inside, dist2, min_dist2))

    for offset_x, offset_y in offsets_xy:
        sample_x = base_x + offset_x
        sample_y = base_y + offset_y
        inside = (
            (sample_x >= 0)
            & (sample_x <= W - 1)
            & (sample_y >= 0)
            & (sample_y <= H - 1)
            & view_valid
        )
        dist2 = (sample_x - center_x).square() + (sample_y - center_y).square()
        # Subtracting the nearest valid distance is algebraically cancelled by
        # normalization and prevents underflow for the sigma->0 nearest check.
        stable_dist2 = dist2 - min_dist2
        weight = torch.exp(-stable_dist2 / (2.0 * float(sigma_px) ** 2)).to(dtype=dtype)
        weight = weight * inside.to(dtype=dtype)

        if W > 1:
            grid_x = sample_x / (W - 1) * 2.0 - 1.0
        else:
            grid_x = torch.zeros_like(sample_x)
        if H > 1:
            grid_y = sample_y / (H - 1) * 2.0 - 1.0
        else:
            grid_y = torch.zeros_like(sample_y)
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(1)

        sampled = F.grid_sample(
            feat_flat,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(2)
        out = out + sampled.permute(0, 2, 1) * weight.unsqueeze(-1)
        denom = denom + weight

    out = torch.where(denom.unsqueeze(-1) > 0, out / denom.clamp_min(1.0e-12).unsqueeze(-1), out)
    return out.reshape(B, V, N, C).permute(0, 2, 1, 3)


def _sample_surface_depth(depth_maps: torch.Tensor, center_grid: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample per-view surface depth maps at query detector locations."""
    if depth_maps.dim() != 4:
        raise ValueError(f"depth_maps must be [B,V,H,W], got {tuple(depth_maps.shape)}")
    B, V, H, W = depth_maps.shape
    N = center_grid.shape[1]
    depth_flat = depth_maps.reshape(B * V, 1, H, W)
    grid = center_grid.permute(0, 2, 1, 3).reshape(B * V, N, 2).unsqueeze(1)
    sampled = F.grid_sample(
        depth_flat,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).squeeze(2)
    return sampled[:, 0].reshape(B, V, N).permute(0, 2, 1)


def ptfa_sample_exit_depth_gaussian(
    feature_map: torch.Tensor,
    center_grid: torch.Tensor,
    valid_mask: torch.Tensor,
    depth_maps: torch.Tensor,
    query_depth: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
    exit_depth_max: float,
    window: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sample features with per-query/view sigma from tissue-internal exit depth.

    Args:
        feature_map: [B,V,C,H,W]
        center_grid: [B,N,V,2]
        valid_mask: [B,N,V]
        depth_maps: [B,V,Hd,Wd] body-surface camera depth in mm
        query_depth: [B,N,V,1] query camera depth in mm
        sigma_min/sigma_max: sigma range in feature-map pixel units
        exit_depth_max: depth normalization clamp in mm
        window: odd square footprint size

    Returns:
        features [B,N,V,C] and stats dict with exit_depth_mm/sigma_px/surface_depth_mm.
    """
    if query_depth.dim() == 3:
        query_depth = query_depth.unsqueeze(-1)
    if query_depth.dim() != 4 or query_depth.shape[-1] != 1:
        raise ValueError(f"query_depth must be [B,N,V,1], got {tuple(query_depth.shape)}")
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError("sigma_min and sigma_max must be positive")
    if sigma_max < sigma_min:
        raise ValueError("sigma_max must be >= sigma_min")
    if exit_depth_max <= 0:
        raise ValueError("exit_depth_max must be positive")

    surface_depth = _sample_surface_depth(depth_maps.to(feature_map.device), center_grid)
    surface_depth = torch.nan_to_num(
        surface_depth, nan=torch.inf, posinf=torch.inf, neginf=-torch.inf
    )
    exit_depth = query_depth.squeeze(-1) - surface_depth
    exit_depth = torch.nan_to_num(exit_depth, nan=0.0, posinf=0.0, neginf=0.0)
    exit_depth = exit_depth.clamp(0.0, float(exit_depth_max))
    sigma_px = float(sigma_min) + (float(sigma_max) - float(sigma_min)) * (
        exit_depth / float(exit_depth_max)
    )

    features = _ptfa_sample_gaussian_with_sigma(
        feature_map, center_grid, valid_mask, window, sigma_px
    )
    stats = {
        "surface_depth_mm": surface_depth,
        "exit_depth_mm": exit_depth,
        "sigma_px": sigma_px,
    }
    return features, stats


def compute_exit_depth_sigma(
    center_grid: torch.Tensor,
    depth_maps: torch.Tensor,
    query_depth: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
    exit_depth_max: float,
    invert_depth: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute raw/corrected exit-depth proxy and per-query sigma.

    `raw_depth_like` is the standard query-depth minus sampled-surface-depth proxy. When
    `invert_depth=True`, this proxy is treated as inverted/exit-depth-like and converted with
    `depth_eff = exit_depth_max - raw_depth_like` before sigma mapping.
    """
    if query_depth.dim() == 3:
        query_depth = query_depth.unsqueeze(-1)
    if query_depth.dim() != 4 or query_depth.shape[-1] != 1:
        raise ValueError(f"query_depth must be [B,N,V,1], got {tuple(query_depth.shape)}")
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError("sigma_min and sigma_max must be positive")
    if sigma_max < sigma_min:
        raise ValueError("sigma_max must be >= sigma_min")
    if exit_depth_max <= 0:
        raise ValueError("exit_depth_max must be positive")

    surface_depth = _sample_surface_depth(depth_maps.to(query_depth.device), center_grid)
    surface_depth = torch.nan_to_num(
        surface_depth, nan=torch.inf, posinf=torch.inf, neginf=-torch.inf
    )
    raw_depth_like = query_depth.squeeze(-1) - surface_depth
    raw_depth_like = torch.nan_to_num(raw_depth_like, nan=0.0, posinf=0.0, neginf=0.0)
    raw_depth_like = raw_depth_like.clamp(0.0, float(exit_depth_max))
    if invert_depth:
        depth_eff = float(exit_depth_max) - raw_depth_like
    else:
        depth_eff = raw_depth_like
    depth_eff = depth_eff.clamp(0.0, float(exit_depth_max))
    sigma_px = float(sigma_min) + (float(sigma_max) - float(sigma_min)) * (
        depth_eff / float(exit_depth_max)
    )
    sigma_px = sigma_px.clamp(float(sigma_min), float(sigma_max))
    return {
        "surface_depth_mm": surface_depth,
        "raw_depth_like_mm": raw_depth_like,
        "depth_eff_mm": depth_eff,
        "sigma_px": sigma_px,
    }


def ptfa_sample_corrected_exit_depth_gaussian(
    feature_map: torch.Tensor,
    center_grid: torch.Tensor,
    valid_mask: torch.Tensor,
    depth_maps: torch.Tensor,
    query_depth: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
    exit_depth_max: float,
    window: int,
    invert_depth: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sample PTFA features with corrected exit-depth sigma mapping."""
    stats = compute_exit_depth_sigma(
        center_grid=center_grid,
        depth_maps=depth_maps.to(feature_map.device),
        query_depth=query_depth,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        exit_depth_max=exit_depth_max,
        invert_depth=invert_depth,
    )
    features = _ptfa_sample_gaussian_with_sigma(
        feature_map, center_grid, valid_mask, window, stats["sigma_px"]
    )
    return features, stats


def ptfa_sample_pcfs_corrected_exit_depth_gaussian(
    feature_map: torch.Tensor,
    center_grid: torch.Tensor,
    valid_mask: torch.Tensor,
    depth_maps: torch.Tensor,
    query_depth: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
    exit_depth_max: float,
    window: int,
    delta_h: torch.Tensor,
    alpha: float,
    invert_depth: bool = True,
    base_stats: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Corrected exit-depth PTFA with a bounded learned sigma calibration.

    The base path is identical to E8-cED when ``alpha == 0`` or the calibrator is
    zero-initialized: h = depth_eff / exit_depth_max and sigma is mapped linearly.
    """
    stats = (
        compute_exit_depth_sigma(
            center_grid=center_grid,
            depth_maps=depth_maps.to(feature_map.device),
            query_depth=query_depth,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            exit_depth_max=exit_depth_max,
            invert_depth=invert_depth,
        )
        if base_stats is None
        else base_stats
    )
    h0 = (stats["depth_eff_mm"] / float(exit_depth_max)).clamp(0.0, 1.0)
    if delta_h.shape != h0.shape:
        raise ValueError(f"delta_h shape {tuple(delta_h.shape)} != h0 shape {tuple(h0.shape)}")
    delta_h = torch.nan_to_num(delta_h.to(device=h0.device, dtype=h0.dtype), nan=0.0)
    h = (h0 + float(alpha) * delta_h).clamp(0.0, 1.0)
    sigma_px = float(sigma_min) + (float(sigma_max) - float(sigma_min)) * h
    sigma_px = sigma_px.clamp(float(sigma_min), float(sigma_max))

    features = _ptfa_sample_gaussian_with_sigma(
        feature_map, center_grid, valid_mask, window, sigma_px
    )
    stats = {
        **stats,
        "h0_norm": h0,
        "delta_h": delta_h,
        "pcfs_alpha": torch.as_tensor(float(alpha), device=h0.device, dtype=h0.dtype),
        "h_calibrated_norm": h,
        "sigma_px": sigma_px,
    }
    return features, stats


def _ptfa_sample_gaussian_with_sigma(
    feature_map: torch.Tensor,
    center_grid: torch.Tensor,
    valid_mask: torch.Tensor,
    window: int,
    sigma_px: torch.Tensor,
) -> torch.Tensor:
    """Fixed-window PTFA with per-query/view sigma [B,N,V]."""
    if feature_map.dim() != 5:
        raise ValueError(f"feature_map must be [B,V,C,H,W], got {tuple(feature_map.shape)}")
    if window % 2 != 1 or window <= 0:
        raise ValueError(f"window must be a positive odd integer, got {window}")

    B, V, C, H, W = feature_map.shape
    N = center_grid.shape[1]
    dtype = feature_map.dtype
    device = feature_map.device
    feat_flat = feature_map.reshape(B * V, C, H, W)
    center = center_grid.permute(0, 2, 1, 3).reshape(B * V, N, 2)
    view_valid = valid_mask.permute(0, 2, 1).reshape(B * V, N).to(device=device)
    sigma = sigma_px.permute(0, 2, 1).reshape(B * V, N).to(device=device, dtype=dtype)
    sigma = sigma.clamp_min(1.0e-6)

    if W > 1:
        center_x = (center[..., 0] + 1.0) * (W - 1) / 2.0
    else:
        center_x = torch.zeros_like(center[..., 0])
    if H > 1:
        center_y = (center[..., 1] + 1.0) * (H - 1) / 2.0
    else:
        center_y = torch.zeros_like(center[..., 1])

    radius = window // 2
    offsets = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
    offsets_xy = torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=-1)
    K = offsets_xy.shape[0]

    base_x = torch.round(center_x)
    base_y = torch.round(center_y)

    sample_x = base_x.unsqueeze(-1) + offsets_xy[:, 0].view(1, 1, K)
    sample_y = base_y.unsqueeze(-1) + offsets_xy[:, 1].view(1, 1, K)
    inside = (
        (sample_x >= 0)
        & (sample_x <= W - 1)
        & (sample_y >= 0)
        & (sample_y <= H - 1)
        & view_valid.unsqueeze(-1)
    )
    dist2 = (sample_x - center_x.unsqueeze(-1)).square() + (
        sample_y - center_y.unsqueeze(-1)
    ).square()
    scaled_dist = dist2 / (2.0 * sigma.unsqueeze(-1).square())
    masked_scaled_dist = torch.where(inside, scaled_dist, torch.full_like(scaled_dist, torch.inf))
    min_scaled_dist = masked_scaled_dist.amin(dim=-1, keepdim=True)
    min_scaled_dist = torch.where(
        torch.isfinite(min_scaled_dist), min_scaled_dist, torch.zeros_like(min_scaled_dist)
    )
    weight = torch.exp(-(scaled_dist - min_scaled_dist)).to(dtype=dtype)
    weight = weight * inside.to(dtype=dtype)

    if W > 1:
        grid_x = sample_x / (W - 1) * 2.0 - 1.0
    else:
        grid_x = torch.zeros_like(sample_x)
    if H > 1:
        grid_y = sample_y / (H - 1) * 2.0 - 1.0
    else:
        grid_y = torch.zeros_like(sample_y)
    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(B * V, 1, N * K, 2)
    sampled = F.grid_sample(
        feat_flat,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).squeeze(2)
    sampled = sampled.permute(0, 2, 1).reshape(B * V, N, K, C)
    out = (sampled * weight.unsqueeze(-1)).sum(dim=2)
    denom = weight.sum(dim=2)

    out = torch.where(denom.unsqueeze(-1) > 0, out / denom.clamp_min(1.0e-12).unsqueeze(-1), out)
    return out.reshape(B, V, N, C).permute(0, 2, 1, 3)
