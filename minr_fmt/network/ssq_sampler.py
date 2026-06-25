"""Query-dependent multiscale footprint sampling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from minr_fmt.network.ssq_diagnostics import make_residual_mlp
from minr_fmt.network.ssq_geometry import sample_finite_scalar_map, sample_scalar_map


def local_template(name: str, fallback: list[list[float]] | None = None) -> list[list[float]]:
    if fallback is not None:
        return fallback
    if name == "cross5":
        return [[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
    if name == "radial13":
        return [
            [0.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [0.0, 1.0],
            [-1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
            [-2.0, 0.0],
            [2.0, 0.0],
            [0.0, -2.0],
            [0.0, 2.0],
        ]
    if name != "grid3x3":
        raise ValueError(f"unknown local_sampling.template={name!r}")
    return [
        [-1.0, -1.0],
        [0.0, -1.0],
        [1.0, -1.0],
        [-1.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [-1.0, 1.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]


class QueryDependentSurfaceSampler(nn.Module):
    def __init__(
        self,
        offsets_px: list[list[float]],
        feature_channels: int | None = None,
        sample_feature_dim: int = 128,
        hidden_dim: int = 128,
        sigma_min_px: float = 1.0,
        sigma_max_px: float = 8.0,
        alpha_xi: float = 0.5,
        alpha_beta: float = 0.25,
        delta_h_max: float = 0.25,
        boundary_margin_radius_px: float = 8.0,
        mode: str = "query_dependent",
        fixed_sigma: float = 2.0,
        align_corners: bool = True,
    ):
        super().__init__()
        offsets = torch.tensor(offsets_px, dtype=torch.float32)
        if offsets.dim() != 2 or offsets.shape[-1] != 2:
            raise ValueError("offsets_px must be a list of [du,dv] pairs")
        self.register_buffer("offsets_px", offsets, persistent=False)
        channels = int(feature_channels if feature_channels is not None else sample_feature_dim)
        self.context_net = make_residual_mlp(channels + 3, hidden_dim, 1, blocks=1)
        self.footprint_context_net = self.context_net
        self.sample_projection = make_residual_mlp(
            channels + 1, hidden_dim, sample_feature_dim, blocks=1
        )
        self.sample_feature_dim = int(sample_feature_dim)
        self.sigma_min_px = float(sigma_min_px)
        self.sigma_max_px = float(sigma_max_px)
        self.alpha_xi = float(alpha_xi)
        self.alpha_beta = float(alpha_beta)
        self.delta_h_max = float(delta_h_max)
        self.boundary_margin_radius_px = float(boundary_margin_radius_px)
        self.mode = str(mode)
        self.fixed_sigma = float(fixed_sigma)
        self.align_corners = bool(align_corners)

    def _sample_feature_level(self, feat: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        b, v, c, h, w = feat.shape
        n, k = grid.shape[2], grid.shape[3]
        sampled = F.grid_sample(
            feat.reshape(b * v, c, h, w),
            grid.reshape(b * v, n * k, 1, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=self.align_corners,
        )
        return sampled.squeeze(-1).transpose(1, 2).reshape(b, v, n, k, c)

    def forward(
        self,
        features: torch.Tensor,
        measurements: torch.Tensor,
        mapped: dict[str, torch.Tensor],
        *,
        detector_valid_mask: torch.Tensor | None = None,
        depth_maps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if isinstance(features, dict):
            raise TypeError("QueryDependentSurfaceSampler expects a single [B,V,C,H,W] feature map")
        full = features
        b, v, _c, h, w = full.shape
        n = mapped["grid"].shape[2]
        offsets = self.offsets_px.to(device=full.device, dtype=full.dtype)
        k = offsets.shape[0]

        center_grid = mapped["grid"][:, :, :, None, :]
        sigma_context = torch.stack(
            [
                mapped["detector_side_path_proxy"],
                (mapped["boundary_distance"] / max(self.boundary_margin_radius_px, 1e-6)).clamp(
                    0.0, 1.0
                ),
            ],
            dim=-1,
        )
        center_feat = self._sample_feature_level(full, center_grid).squeeze(3)

        meas_center = sample_scalar_map(
            measurements, mapped["grid"], align_corners=self.align_corners
        )[..., None]
        ctx = torch.cat([center_feat, meas_center, sigma_context], dim=-1)
        alpha = torch.tensor(
            [self.alpha_xi, self.alpha_beta],
            device=full.device,
            dtype=full.dtype,
        ).clamp_min(0.0)
        alpha = alpha / alpha.sum().clamp_min(1e-8)
        h0 = (sigma_context * alpha).sum(dim=-1).clamp(0.0, 1.0)
        if self.mode == "fixed":
            sigma_f = torch.full_like(h0, self.fixed_sigma)
            h_sched = (
                (sigma_f - self.sigma_min_px) / max(self.sigma_max_px - self.sigma_min_px, 1e-6)
            ).clamp(0.0, 1.0)
        else:
            delta = self.delta_h_max * torch.tanh(self.context_net(ctx)).squeeze(-1)
            h_sched = (h0 + delta).clamp(0.0, 1.0)
            sigma_f = self.sigma_min_px + (self.sigma_max_px - self.sigma_min_px) * h_sched

        base_grid = mapped["grid"][:, :, :, None, :]
        dx = 2.0 / max(w - 1, 1)
        dy = 2.0 / max(h - 1, 1)
        off_grid = torch.stack(
            [
                offsets[None, None, None, :, 0] * sigma_f[..., None] * dx,
                offsets[None, None, None, :, 1] * sigma_f[..., None] * dy,
            ],
            dim=-1,
        )
        grid = base_grid + off_grid
        sample_features_raw = self._sample_feature_level(full, grid)
        in_bounds = (grid.abs() <= 1.0).all(dim=-1)
        sample_valid = mapped["valid_mask"][:, :, :, None] & in_bounds
        if detector_valid_mask is not None:
            mask = (
                detector_valid_mask.squeeze(2)
                if detector_valid_mask.dim() == 5
                else detector_valid_mask
            )
            sample_valid = sample_valid & (
                sample_scalar_map(
                    mask.float(), grid, mode="nearest", align_corners=self.align_corners
                )
                > 0.5
            )
        if depth_maps is not None:
            sampled_depth, finite_depth = sample_finite_scalar_map(
                depth_maps, grid, align_corners=self.align_corners
            )
            sample_valid = sample_valid & finite_depth
        else:
            sampled_depth = torch.zeros((b, v, n, k), device=full.device, dtype=full.dtype)

        sampled_meas = sample_scalar_map(measurements, grid, align_corners=self.align_corners)[
            ..., None
        ]
        concatenated = torch.cat([sample_features_raw, sampled_meas], dim=-1)
        sampled = self.sample_projection(concatenated)
        sampled = torch.where(sample_valid[..., None], sampled, torch.zeros_like(sampled))
        sampled_meas = torch.where(
            sample_valid[..., None], sampled_meas, torch.zeros_like(sampled_meas)
        )
        base_a = torch.exp(-0.5 * offsets.square().sum(dim=-1)).to(sampled.dtype)
        a = base_a[None, None, None, :].expand(b, v, n, k) * sample_valid.to(sampled.dtype)
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        a = torch.where(sample_valid.any(dim=-1, keepdim=True), a, torch.zeros_like(a))
        return {
            "sigma_f": sigma_f,
            "h": h_sched,
            "sample_coordinates": grid,
            "sample_coordinates_px": mapped["uv_px"][:, :, :, None, :]
            + sigma_f[..., None, None] * offsets[None, None, None],
            "sample_depth": sampled_depth,
            "sample_valid": sample_valid,
            "query_view_valid": sample_valid.any(dim=-1),
            "A": a,
            "sample_features": sampled,
            "sample_measurements": sampled_meas,
            "offsets_template": offsets,
        }
