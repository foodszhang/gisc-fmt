"""Geometry-first, measurement-conditioned view separability."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewSeparability(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 64, delta_max: float = 0.25) -> None:
        super().__init__()
        self.delta_max = float(delta_max)
        self.correction = nn.Sequential(
            nn.Linear(feature_dim * 4 + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        detector_centers_px: torch.Tensor,
        detector_scales_px: torch.Tensor,
        candidate_valid: torch.Tensor,
        mode: str = "geometry_measurement",
    ) -> dict[str, torch.Tensor]:
        """Use [B,V,M,D], [B,V,M,2], [B,V,M], and [B,M]."""
        if mode not in {"none", "geometry_only", "geometry_measurement"}:
            raise ValueError(f"unknown separability mode: {mode}")
        delta = (
            detector_centers_px[:, :, :, None]
            - detector_centers_px[:, :, None, :]
        )
        variance = (
            detector_scales_px[:, :, :, None].square()
            + detector_scales_px[:, :, None, :].square()
        )
        collision = torch.exp(-0.5 * delta.square().sum(dim=-1) / variance.clamp_min(1.0e-6))
        geometry = 1.0 - collision
        pair_valid = (
            candidate_valid[:, None, :, None] & candidate_valid[:, None, None, :]
        )
        if mode == "none":
            pair = torch.ones_like(geometry)
            correction = torch.zeros_like(geometry)
        elif mode == "geometry_only":
            pair, correction = geometry, torch.zeros_like(geometry)
        else:
            left = F.normalize(features, dim=-1)[:, :, :, None].expand(
                -1, -1, -1, features.shape[2], -1
            )
            right = F.normalize(features, dim=-1)[:, :, None, :].expand_as(left)
            normalized_variance = torch.log1p(variance).div(10.0).clamp(0.0, 1.0)
            geom = torch.cat([delta.clamp(-2.0, 2.0), normalized_variance[..., None]], dim=-1)
            inp = torch.cat([left, right, (left - right).abs(), left * right, geom], dim=-1)
            correction = self.delta_max * torch.tanh(self.correction(inp).squeeze(-1))
            correction = 0.5 * (correction + correction.transpose(-1, -2))
            pair = (geometry + correction).clamp(0.0, 1.0)
        pair = torch.where(pair_valid, pair, torch.ones_like(pair))
        m = features.shape[2]
        eye = torch.eye(m, dtype=torch.bool, device=features.device)[None, None]
        candidate = pair.masked_fill(eye, 1.0).amin(dim=-1)
        candidate = torch.where(
            candidate_valid[:, None], candidate, torch.zeros_like(candidate)
        )
        return {
            "pair_separability": pair,
            "separability": candidate,
            "geometry": geometry,
            "correction": correction,
        }
