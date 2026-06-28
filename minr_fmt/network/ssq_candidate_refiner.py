"""Measurement-only coarse candidate heatmap refinement."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = min(8, out_channels)
        self.body = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
        )
        self.skip = (
            nn.Conv3d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.body(x) + self.skip(x))


class MeasurementCandidateRefiner(nn.Module):
    """Small 3D residual U-Net over a coarse measurement backprojection volume."""

    def __init__(self, base_channels: int = 16):
        super().__init__()
        c = int(base_channels)
        self.stem = ResidualBlock3d(4, c)
        self.down1 = nn.Sequential(nn.Conv3d(c, 2 * c, 3, stride=2, padding=1), nn.SiLU())
        self.enc1 = ResidualBlock3d(2 * c, 2 * c)
        self.down2 = nn.Sequential(
            nn.Conv3d(2 * c, 4 * c, 3, stride=2, padding=1), nn.SiLU()
        )
        self.bottleneck = ResidualBlock3d(4 * c, 4 * c)
        self.dec1 = ResidualBlock3d(6 * c, 2 * c)
        self.dec0 = ResidualBlock3d(3 * c, c)
        self.head = nn.Conv3d(c, 1, 1)

    @staticmethod
    def _coordinates(x: torch.Tensor) -> torch.Tensor:
        axes = [
            torch.linspace(-1.0, 1.0, size, device=x.device, dtype=x.dtype)
            for size in x.shape[-3:]
        ]
        grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)
        return grid.unsqueeze(0).expand(x.shape[0], -1, -1, -1, -1)

    def forward(self, heatmap: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(torch.cat([heatmap, self._coordinates(heatmap)], dim=1))
        x1 = self.enc1(self.down1(x0))
        x2 = self.bottleneck(self.down2(x1))
        y1 = F.interpolate(x2, size=x1.shape[-3:], mode="trilinear", align_corners=False)
        y1 = self.dec1(torch.cat([y1, x1], dim=1))
        y0 = F.interpolate(y1, size=x0.shape[-3:], mode="trilinear", align_corners=False)
        return self.head(self.dec0(torch.cat([y0, x0], dim=1)))
