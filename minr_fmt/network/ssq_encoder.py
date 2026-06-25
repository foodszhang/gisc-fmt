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


class _DecoderRefine(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, blocks: int = 2, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [ConvNormAct(in_ch, out_ch)]
        layers.extend(ResidualBlock2d(out_ch, dropout) for _ in range(int(blocks)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedResidualUNetPyramidEncoder(nn.Module):
    """Residual U-Net with internal pyramid context fusion.

    The public contract is a single full-resolution detector feature map
    ``[B,V,output_channels,H,W]``. Full/half/quarter decoder states stay internal
    and are fused before returning.
    """

    def __init__(
        self,
        in_channels: int = 1,
        stem_channels: int = 32,
        stage_channels: tuple[int, int, int, int] = (48, 96, 192, 256),
        stage_blocks: tuple[int, int, int, int] = (2, 2, 3, 3),
        pyramid_channels: int = 64,
        output_channels: int = 96,
        dropout: float = 0.0,
        use_pyramid_fusion: bool = True,
    ):
        super().__init__()
        c1, c2, c3, c4 = [int(v) for v in stage_channels]
        b1, b2, b3, b4 = [int(v) for v in stage_blocks]
        self.use_pyramid_fusion = bool(use_pyramid_fusion)
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
        out = int(output_channels)
        self.up43 = ConvNormAct(c4, c3)
        self.dec3 = _DecoderRefine(c3 + c3, c3, blocks=2, dropout=dropout)
        self.up32 = ConvNormAct(c3, c2)
        self.dec2 = _DecoderRefine(c2 + c2, c2, blocks=2, dropout=dropout)
        self.up21 = ConvNormAct(c2, c1)
        self.dec1 = _DecoderRefine(c1 + c1, c1, blocks=2, dropout=dropout)

        if self.use_pyramid_fusion:
            self.proj1 = ConvNormAct(c1, p)
            self.proj2 = ConvNormAct(c2, p)
            self.proj3 = ConvNormAct(c3, p)
            self.fuse = nn.Sequential(ConvNormAct(p * 3, out), ConvNormAct(out, out))
        else:
            self.fuse = nn.Sequential(ConvNormAct(c1, out), ConvNormAct(out, out))
        self.out_channels = out

    def _encode_flat(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        s1 = self.stage1(self.stage1_in(x))
        s2 = self.stage2(self.down2(s1))
        s3 = self.stage3(self.down3(s2))
        s4 = self.stage4(self.down4(s3))
        up4 = F.interpolate(s4, size=s3.shape[-2:], mode="bilinear", align_corners=False)
        h3 = self.dec3(torch.cat([self.up43(up4), s3], dim=1))
        up3 = F.interpolate(h3, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        h2 = self.dec2(torch.cat([self.up32(up3), s2], dim=1))
        up2 = F.interpolate(h2, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        h1 = self.dec1(torch.cat([self.up21(up2), s1], dim=1))
        if not self.use_pyramid_fusion:
            return self.fuse(h1)
        p1 = self.proj1(h1)
        p2 = self.proj2(h2)
        p3 = self.proj3(h3)
        fused = torch.cat(
            [
                p1,
                F.interpolate(p2, size=p1.shape[-2:], mode="bilinear", align_corners=False),
                F.interpolate(p3, size=p1.shape[-2:], mode="bilinear", align_corners=False),
            ],
            dim=1,
        )
        return self.fuse(fused)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"expected [B,V,C,H,W], got {tuple(x.shape)}")
        b, v, c, h, w = x.shape
        out = self._encode_flat(x.reshape(b * v, c, h, w))
        return out.reshape(b, v, out.shape[1], out.shape[2], out.shape[3])


class SharedMultiscaleSurfaceEncoder(SharedResidualUNetPyramidEncoder):
    """Backward-compatible alias for the production residual U-Net pyramid encoder."""


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, v, c, h, w = x.shape
        y = self.net(x.reshape(b * v, c, h, w))
        return y.reshape(b, v, y.shape[1], y.shape[2], y.shape[3])
