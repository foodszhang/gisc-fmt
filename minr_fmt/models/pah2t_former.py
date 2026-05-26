"""PAH2T-Former adapted baseline for sparse-view FMT projections.

Reference: PAH2T-Former, IEEE TCI 2025, DOI 10.1109/TCI.2025.3559431.

This implementation keeps the two paper-specific modules requested for the
baseline comparison:
  - IMSM: intra spatial/channel attention + inter projection/depth attention
  - SC-PAM: paired spatial and channel attention with low-rank spatial keys

The input adaptation treats sparse-view projections as a shallow 3D tensor whose
depth axis is the projection-view dimension.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _roi_shape(config) -> tuple[int, int, int]:
    vr = config.data.voxel_ranges
    return (int(vr.x[1] - vr.x[0]), int(vr.y[1] - vr.y[0]), int(vr.z[1] - vr.z[0]))


def _section(config):
    return getattr(config.model, "pah2t_former", {})


def _projection_tensor(projections) -> torch.Tensor:
    if torch.is_tensor(projections):
        x = projections
        if x.dim() == 5 and x.size(2) == 1:
            x = x.squeeze(2)
        return x
    keys = sorted(projections.keys(), key=lambda x: float(x))
    x = torch.stack([projections[k] for k in keys], dim=1)
    if x.dim() == 5 and x.size(2) == 1:
        x = x.squeeze(2)
    return x


def _tuple3(value, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if value is None:
        return default
    if isinstance(value, Sequence):
        return tuple(int(v) for v in value)
    return (int(value), int(value), int(value))


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv3d(channels, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IMSMBlock(nn.Module):
    """Intra/inter multi-scale self-modulation block.

    IMSMintra uses global spatial pooling plus 1x1 and 3x3 depthwise-separable
    projections to form channel-wise attention. IMSMinter applies self-attention
    along the projection/depth dimension after spatial pooling.
    """

    def __init__(self, channels: int, num_heads: int = 2):
        super().__init__()
        heads = max(1, min(int(num_heads), channels))
        while channels % heads != 0 and heads > 1:
            heads -= 1
        self.channels = int(channels)
        self.num_heads = heads
        self.head_dim = channels // heads

        self.q1 = nn.Conv3d(channels, channels, 1, bias=False)
        self.k1 = nn.Conv3d(channels, channels, 1, bias=False)
        self.v1 = nn.Conv3d(channels, channels, 1, bias=False)
        self.qdw = DepthwiseSeparableConv3d(channels)
        self.kdw = DepthwiseSeparableConv3d(channels)
        self.vdw = DepthwiseSeparableConv3d(channels)
        self.intra_out = nn.Conv3d(channels, channels, 1, bias=False)

        self.inter_q = nn.Linear(channels, channels, bias=False)
        self.inter_k = nn.Linear(channels, channels, bias=False)
        self.inter_v = nn.Linear(channels, channels, bias=False)
        self.inter_out = nn.Linear(channels, channels, bias=False)
        self.norm = nn.GroupNorm(1, channels)

    def _intra(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, _h, _w = x.shape
        pooled = F.adaptive_avg_pool3d(x, (d, 1, 1))
        q = self.q1(pooled) + self.qdw(pooled)
        k = self.k1(pooled) + self.kdw(pooled)
        v = self.v1(pooled) + self.vdw(pooled)
        q = q.flatten(2).transpose(1, 2)  # [B,D,C]
        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)
        q = q.reshape(b * d, self.num_heads, self.head_dim)
        k = k.reshape(b * d, self.num_heads, self.head_dim)
        v = v.reshape(b * d, self.num_heads, self.head_dim)
        attn = torch.softmax((q * k) / math.sqrt(float(self.head_dim)), dim=-1)
        gate = (attn * v).reshape(b, d, c).transpose(1, 2).view(b, c, d, 1, 1)
        return self.intra_out(x * torch.sigmoid(gate))

    def _inter(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, _h, _w = x.shape
        tokens = x.mean(dim=(-2, -1)).transpose(1, 2)  # [B,D,C]
        q = self.inter_q(tokens)
        k = self.inter_k(tokens)
        v = self.inter_v(tokens)
        q = q.view(b, d, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, d, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, d, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(float(self.head_dim)), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, d, c)
        out = self.inter_out(out).transpose(1, 2).view(b, c, d, 1, 1)
        return x * torch.sigmoid(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self._intra(x) + self._inter(x) + x)


class SCPAMBlock(nn.Module):
    """Spatial-Channel Paired Attention Module with low-rank spatial attention."""

    def __init__(self, channels: int, num_heads: int = 2, proj_tokens: int = 64):
        super().__init__()
        heads = max(1, min(int(num_heads), channels))
        while channels % heads != 0 and heads > 1:
            heads -= 1
        self.channels = int(channels)
        self.num_heads = heads
        self.head_dim = channels // heads
        self.proj_tokens = int(proj_tokens)

        self.q_common = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_common = nn.Conv1d(channels, channels, 1, bias=False)
        self.v_spatial = nn.Conv1d(channels, channels, 1, bias=False)
        self.v_channel = nn.Conv1d(channels, channels, 1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv3d(channels * 2, channels, 1, bias=False),
            nn.GroupNorm(1, channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.SiLU(inplace=True),
        )

    def _spatial_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        b, c, n = q.shape
        p = min(self.proj_tokens, n)
        qh = q.transpose(1, 2).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        kp = F.adaptive_avg_pool1d(k, p)
        vp = F.adaptive_avg_pool1d(v, p)
        kh = kp.transpose(1, 2).reshape(b, p, self.num_heads, self.head_dim).transpose(1, 2)
        vh = vp.transpose(1, 2).reshape(b, p, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax((qh @ kh.transpose(-2, -1)) / math.sqrt(float(self.head_dim)), dim=-1)
        out = (attn @ vh).transpose(1, 2).reshape(b, n, c).transpose(1, 2)
        return out

    def _channel_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_c = q.mean(dim=-1)  # [B,C]
        k_c = k.mean(dim=-1)
        v_c = v.mean(dim=-1)
        attn = torch.softmax(q_c.unsqueeze(2) * k_c.unsqueeze(1) / math.sqrt(q.shape[-1]), dim=-1)
        gate = torch.sigmoid(attn @ v_c.unsqueeze(-1)).squeeze(-1).unsqueeze(-1)
        return v * gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        flat = x.reshape(b, c, -1)
        q = self.q_common(flat)
        k = self.k_common(flat)
        vs = self.v_spatial(flat)
        vc = self.v_channel(flat)
        spatial = self._spatial_attention(q, k, vs).view(b, c, d, h, w)
        channel = self._channel_attention(q, k, vc).view(b, c, d, h, w)
        return x + self.fuse(torch.cat([spatial, channel], dim=1))


class PAH2TStage(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        heads: int,
        proj_tokens: int,
        use_scpam: bool = True,
        checkpointing: bool = False,
    ):
        super().__init__()
        self.imsm = IMSMBlock(channels, heads)
        self.scpam = SCPAMBlock(channels, heads, proj_tokens) if use_scpam else nn.Identity()
        self.local = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.SiLU(inplace=True),
        )
        self.checkpointing = bool(checkpointing)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.imsm(x)
        x = self.scpam(x)
        return self.local(x) + x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.checkpointing and self.training:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


class PAH2TFormer(nn.Module):
    output_type = "voxel"

    def __init__(self, config):
        super().__init__()
        self.config = config
        params = _section(config)
        base = int(getattr(params, "base_channels", 8))
        heads = int(getattr(params, "num_heads", 2))
        proj_tokens = int(getattr(params, "proj_tokens", 64))
        use_checkpointing = bool(getattr(params, "gradient_checkpointing", True))
        self.projection_size = tuple(int(v) for v in getattr(params, "projection_size", [64, 64]))
        self.output_mode = str(getattr(params, "output_mode", "full"))
        self.roi_shape = _roi_shape(config)
        self.full_shape = _tuple3(
            getattr(params, "full_output_shape", None),
            tuple(int(v) for v in getattr(config.model.geometry, "global_voxel_shape", self.roi_shape)),
        )
        self.output_shape = _tuple3(getattr(params, "output_shape", None), self.roi_shape)
        self.paste_roi_to_full = bool(getattr(params, "paste_roi_to_full", self.output_mode == "roi"))
        self.loss_type = str(getattr(params, "loss_type", "mse"))
        self.dice_weight = float(getattr(params, "dice_weight", 0.5))

        self.stem = nn.Sequential(
            nn.Conv3d(1, base, 3, padding=1, bias=False),
            nn.GroupNorm(1, base),
            nn.SiLU(inplace=True),
        )
        self.enc1 = PAH2TStage(
            base, heads=heads, proj_tokens=proj_tokens, checkpointing=use_checkpointing
        )
        self.down1 = nn.Conv3d(base, base * 2, 3, stride=2, padding=1, bias=False)
        self.enc2 = PAH2TStage(
            base * 2, heads=heads, proj_tokens=proj_tokens, checkpointing=use_checkpointing
        )
        self.down2 = nn.Conv3d(base * 2, base * 4, 3, stride=2, padding=1, bias=False)
        self.enc3 = PAH2TStage(
            base * 4, heads=heads, proj_tokens=proj_tokens, checkpointing=use_checkpointing
        )
        self.down3 = nn.Conv3d(base * 4, base * 4, 3, stride=2, padding=1, bias=False)
        self.enc4 = PAH2TStage(
            base * 4, heads=heads, proj_tokens=proj_tokens, checkpointing=use_checkpointing
        )

        self.dec3_pre = IMSMBlock(base * 4, heads)
        self.dec3_attn = SCPAMBlock(base * 4, heads, proj_tokens)
        self.dec3 = nn.Conv3d(base * 8, base * 4, 3, padding=1, bias=False)
        self.dec2_pre = IMSMBlock(base * 4, heads)
        self.dec2_attn = SCPAMBlock(base * 4, heads, proj_tokens)
        self.dec2 = nn.Conv3d(base * 6, base * 2, 3, padding=1, bias=False)
        self.dec1_pre = IMSMBlock(base * 2, heads)
        self.dec1 = nn.Conv3d(base * 3, base, 3, padding=1, bias=False)
        self.norm3 = nn.Sequential(nn.GroupNorm(1, base * 4), nn.SiLU(inplace=True))
        self.norm2 = nn.Sequential(nn.GroupNorm(1, base * 2), nn.SiLU(inplace=True))
        self.norm1 = nn.Sequential(nn.GroupNorm(1, base), nn.SiLU(inplace=True))
        self.out = nn.Conv3d(base, 1, 1)

    def _prepare_input(self, projections) -> torch.Tensor:
        x = _projection_tensor(projections).float()  # [B,V,H,W]
        x = F.interpolate(x, size=self.projection_size, mode="bilinear", align_corners=False)
        return x.unsqueeze(1)  # [B,1,V,H,W]

    @staticmethod
    def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] == ref.shape[2:]:
            return x
        return F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)

    def _paste_roi(self, pred: torch.Tensor, batch: dict | None) -> torch.Tensor:
        if not self.paste_roi_to_full or pred.shape[2:] == self.full_shape:
            return pred
        full = pred.new_zeros((pred.shape[0], 1, *self.full_shape))
        vr = self.config.data.voxel_ranges
        x0, x1 = int(vr.x[0]), int(vr.x[1])
        y0, y1 = int(vr.y[0]), int(vr.y[1])
        z0, z1 = int(vr.z[0]), int(vr.z[1])
        if pred.shape[2:] != (x1 - x0, y1 - y0, z1 - z0):
            raise ValueError(
                f"PAH2T ROI pred shape {tuple(pred.shape[2:])} does not match configured ROI "
                f"{(x1 - x0, y1 - y0, z1 - z0)}"
            )
        full[:, :, x0:x1, y0:y1, z0:z1] = pred
        return full

    def forward(self, projections, *args, **kwargs):
        x = self.stem(self._prepare_input(projections))
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))

        d3 = self._resize_like(self.dec3_attn(self.dec3_pre(e4)), e3)
        d3 = self.norm3(self.dec3(torch.cat([d3, e3], dim=1)))
        d2 = self._resize_like(self.dec2_attn(self.dec2_pre(d3)), e2)
        d2 = self.norm2(self.dec2(torch.cat([d2, e2], dim=1)))
        d1 = self._resize_like(self.dec1_pre(d2), e1)
        d1 = self.norm1(self.dec1(torch.cat([d1, e1], dim=1)))

        logits = self.out(d1)
        logits = F.interpolate(
            logits,
            size=self.output_shape,
            mode="trilinear",
            align_corners=False,
        )
        logits = self._paste_roi(logits, kwargs.get("batch"))
        return {
            "pred_voxel": logits,
            "aux_outputs": {
                "voxel_loss_type": self.loss_type,
                "dice_weight": self.dice_weight,
                "output_space": self.output_mode,
                "alignment_mode": "paste_pred" if self.paste_roi_to_full else self.output_mode,
            },
        }
