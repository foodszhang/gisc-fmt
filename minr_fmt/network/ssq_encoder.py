"""Shared multiscale detector-plane encoder for SSQ-FMT."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_ch), out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock2d(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_group_count(channels), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_group_count(channels), channels)
        self.drop = nn.Dropout2d(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.silu(self.norm1(self.conv1(x)))
        y = self.drop(y)
        y = self.norm2(self.conv2(y))
        return F.silu(x + y)


class SharedMultiscaleSurfaceEncoder(nn.Module):
    """Residual-FPN encoder applied independently to each detector view."""

    def __init__(
        self,
        in_channels: int = 1,
        stem_channels: int = 32,
        stage_channels: tuple[int, int, int, int] = (48, 96, 192, 256),
        stage_blocks: tuple[int, int, int, int] = (2, 2, 3, 3),
        pyramid_channels: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        c1, c2, c3, c4 = [int(v) for v in stage_channels]
        b1, b2, b3, b4 = [int(v) for v in stage_blocks]
        self.stem = ConvNormAct(in_channels, int(stem_channels))
        self.stage1_in = ConvNormAct(int(stem_channels), c1)
        self.stage1 = nn.Sequential(*(ResidualBlock2d(c1, dropout) for _ in range(b1)))
        self.down2 = ConvNormAct(c1, c2, stride=2)
        self.stage2 = nn.Sequential(*(ResidualBlock2d(c2, dropout) for _ in range(b2)))
        self.down3 = ConvNormAct(c2, c3, stride=2)
        self.stage3 = nn.Sequential(*(ResidualBlock2d(c3, dropout) for _ in range(b3)))
        self.down4 = ConvNormAct(c3, c4, stride=2)
        self.stage4 = nn.Sequential(*(ResidualBlock2d(c4, dropout) for _ in range(b4)))

        p = int(pyramid_channels)
        self.lat1 = nn.Conv2d(c1, p, 1)
        self.lat2 = nn.Conv2d(c2, p, 1)
        self.lat3 = nn.Conv2d(c3, p, 1)
        self.lat4 = nn.Conv2d(c4, p, 1)
        self.refine1 = ConvNormAct(p, p)
        self.refine2 = ConvNormAct(p, p)
        self.refine3 = ConvNormAct(p, p)
        self.out_channels = p

    def _encode_flat(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        s1 = self.stage1(self.stage1_in(x))
        s2 = self.stage2(self.down2(s1))
        s3 = self.stage3(self.down3(s2))
        s4 = self.stage4(self.down4(s3))
        p4 = self.lat4(s4)
        p3 = self.lat3(s3) + F.interpolate(
            p4, size=s3.shape[-2:], mode="bilinear", align_corners=False
        )
        p2 = self.lat2(s2) + F.interpolate(
            p3, size=s2.shape[-2:], mode="bilinear", align_corners=False
        )
        p1 = self.lat1(s1) + F.interpolate(
            p2, size=s1.shape[-2:], mode="bilinear", align_corners=False
        )
        return {
            "full": self.refine1(p1),
            "half": self.refine2(p2),
            "quarter": self.refine3(p3),
            "global_context": F.adaptive_avg_pool2d(p4, 1).flatten(1),
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 5:
            raise ValueError(f"expected [B,V,C,H,W], got {tuple(x.shape)}")
        b, v, c, h, w = x.shape
        out = self._encode_flat(x.reshape(b * v, c, h, w))
        result: dict[str, torch.Tensor] = {}
        for key, value in out.items():
            if key == "global_context":
                result[key] = value.reshape(b, v, -1)
            else:
                result[key] = value.reshape(b, v, value.shape[1], value.shape[2], value.shape[3])
        return result


class ShallowSurfaceEncoder(nn.Module):
    """Legacy two-conv encoder retained only for explicit ablations."""

    def __init__(self, in_channels: int = 1, channels: int = 32, out_channels: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.SiLU(),
        )
        self.out_channels = int(out_channels)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b, v, c, h, w = x.shape
        y = self.net(x.reshape(b * v, c, h, w))
        full = y.reshape(b, v, y.shape[1], y.shape[2], y.shape[3])
        return {
            "full": full,
            "half": F.avg_pool2d(full.reshape(b * v, y.shape[1], h, w), 2).reshape(
                b, v, y.shape[1], h // 2, w // 2
            ),
            "quarter": F.avg_pool2d(full.reshape(b * v, y.shape[1], h, w), 4).reshape(
                b, v, y.shape[1], h // 4, w // 4
            ),
            "global_context": full.mean(dim=(-2, -1)),
        }
