"""Voxel-domain adapted baselines for sparse-view FMT comparisons."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _roi_shape(config) -> tuple[int, int, int]:
    vr = config.data.voxel_ranges
    return (
        int(vr.x[1] - vr.x[0]),
        int(vr.y[1] - vr.y[0]),
        int(vr.z[1] - vr.z[0]),
    )


def _model_section(config, key: str):
    return getattr(config.model, key, {})


def _projection_tensor(projections) -> torch.Tensor:
    if torch.is_tensor(projections):
        if projections.dim() == 5:
            return projections.squeeze(2)
        return projections
    keys = sorted(projections.keys(), key=lambda x: float(x))
    x = torch.stack([projections[k] for k in keys], dim=1)
    if x.dim() == 5 and x.size(2) == 1:
        x = x.squeeze(2)
    return x


class VoxelOutputAlignmentMixin:
    def _init_output_alignment(self, config, params):
        self.output_mode = str(getattr(params, "output_mode", "roi"))
        self.paste_roi_to_full = bool(
            getattr(params, "paste_roi_to_full", self.output_mode == "roi")
        )
        self.full_shape = tuple(
            int(v)
            for v in getattr(
                params,
                "full_output_shape",
                getattr(config.model.geometry, "global_voxel_shape", self.roi_shape),
            )
        )

    def _paste_roi(self, pred: torch.Tensor) -> torch.Tensor:
        if not self.paste_roi_to_full or pred.shape[2:] == self.full_shape:
            return pred
        full = pred.new_zeros((pred.shape[0], 1, *self.full_shape))
        vr = self.config.data.voxel_ranges
        x0, x1 = int(vr.x[0]), int(vr.x[1])
        y0, y1 = int(vr.y[0]), int(vr.y[1])
        z0, z1 = int(vr.z[0]), int(vr.z[1])
        expected = (x1 - x0, y1 - y0, z1 - z0)
        if pred.shape[2:] != expected:
            raise ValueError(f"ROI pred shape {tuple(pred.shape[2:])} != configured ROI {expected}")
        full[:, :, x0:x1, y0:y1, z0:z1] = pred
        return full

    def _match_roi_shape(self, pred: torch.Tensor) -> torch.Tensor:
        if pred.shape[2:] == self.roi_shape:
            return pred
        return F.interpolate(pred, size=self.roi_shape, mode="trilinear", align_corners=False)


def _identity_affine(device, dtype, batch_size: int) -> torch.Tensor:
    theta = torch.tensor(
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        device=device,
        dtype=dtype,
    )
    return theta.view(1, 3, 4).expand(batch_size, -1, -1).contiguous()


class ConvBlock3d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(1, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(1, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class VNet3D(nn.Module):
    """V-Net style encoder-decoder with 3 pooling and 3 transposed upsampling layers."""

    def __init__(self, in_ch: int, base_ch: int, out_ch: int = 1):
        super().__init__()
        self.enc1 = ConvBlock3d(in_ch, base_ch)
        self.down1 = nn.MaxPool3d(2, ceil_mode=True)
        self.enc2 = ConvBlock3d(base_ch, base_ch * 2)
        self.down2 = nn.MaxPool3d(2, ceil_mode=True)
        self.enc3 = ConvBlock3d(base_ch * 2, base_ch * 4)
        self.down3 = nn.MaxPool3d(2, ceil_mode=True)
        self.mid = ConvBlock3d(base_ch * 4, base_ch * 8)
        self.up3 = nn.ConvTranspose3d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock3d(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock3d(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock3d(base_ch * 2, base_ch)
        self.out = nn.Conv3d(base_ch, out_ch, 1)

    def _match(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] == ref.shape[2:]:
            return x
        return F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)

    def forward(self, x, return_features: bool = False):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        m = self.mid(self.down3(e3))
        d3 = self._match(self.up3(m), e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self._match(self.up2(d3), e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self._match(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        out = self.out(d1)
        if return_features:
            return out, [e1, e2, e3, m, d3, d2, d1]
        return out


class ProjectionEncoder2D(nn.Module):
    def __init__(self, in_ch: int, base_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.SiLU(inplace=True),
        )
        self.out_channels = base_ch * 2

    def forward(self, x):
        return self.net(x)


class ProjectionToVoxelNet(nn.Module):
    """Simple generic projection-to-volume backbone kept for non-priority baselines."""

    output_type = "voxel"

    def __init__(self, config, section: str):
        super().__init__()
        params = _model_section(config, section)
        self.config = config
        self.roi_shape = _roi_shape(config)
        self.num_views = int(getattr(config.model, "num_views", len(config.data.view_angles)))
        base = int(getattr(params, "base_channels", 16))
        latent = int(getattr(params, "latent_channels", base * 4))
        self.encoder = ProjectionEncoder2D(self.num_views, base)
        self.fc = nn.Linear(self.encoder.out_channels, latent)
        self.decoder = VNet3D(latent, base)

    def projection_features(self, projections):
        return self.encoder(_projection_tensor(projections))

    def coarse_volume(self, projections):
        feat = self.projection_features(projections)
        pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        seed = self.fc(pooled).view(pooled.shape[0], -1, 1, 1, 1)
        return seed.expand(-1, -1, *self.roi_shape)

    def forward(self, projections, *args, **kwargs):
        pred = self.decoder(self.coarse_volume(projections))
        return {"pred_voxel": pred, "aux_outputs": {}}


class CNN3DBaseline(VoxelOutputAlignmentMixin, nn.Module):
    """Light 3D-CNN baseline using boundary-shell projection embedding."""

    output_type = "voxel"

    def __init__(self, config):
        super().__init__()
        params = _model_section(config, "cnn3d_baseline")
        self.config = config
        self.roi_shape = _roi_shape(config)
        base = int(getattr(params, "base_channels", 8))
        self._init_output_alignment(config, params)
        self.builder = SurfaceVolumeBuilder(self.roi_shape, config.data.view_angles)
        self.net = VNet3D(1, base, 1)

    def forward(self, projections, *args, **kwargs):
        logits = self._match_roi_shape(self.net(self.builder(projections)))
        logits = self._paste_roi(logits)
        return {
            "pred_voxel": logits,
            "aux_outputs": {
                "output_space": self.output_mode,
                "alignment_mode": "paste_pred" if self.paste_roi_to_full else self.output_mode,
            },
        }


class TransformerBottleneck3D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, max_tokens: int = 512):
        super().__init__()
        heads = max(1, min(num_heads, channels))
        while channels % heads != 0 and heads > 1:
            heads -= 1
        self.max_tokens = int(max_tokens)
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        pooled_size = None
        if d * h * w > self.max_tokens:
            scale = (self.max_tokens / float(d * h * w)) ** (1.0 / 3.0)
            pooled_size = (
                max(1, int(round(d * scale))),
                max(1, int(round(h * scale))),
                max(1, int(round(w * scale))),
            )
            y = F.adaptive_avg_pool3d(x, pooled_size)
        else:
            y = x
        tokens = y.flatten(2).transpose(1, 2)
        attn_in = self.norm(tokens)
        tokens = tokens + self.attn(attn_in, attn_in, attn_in, need_weights=False)[0]
        tokens = tokens + self.ffn(tokens)
        y = tokens.transpose(1, 2).view(b, c, *y.shape[2:])
        if pooled_size is not None:
            y = F.interpolate(y, size=(d, h, w), mode="trilinear", align_corners=False)
        return x + y


class TransUNet3DBaseline(VoxelOutputAlignmentMixin, nn.Module):
    """3D-TransUNet style baseline, intentionally separate from PAH2T-Former."""

    output_type = "voxel"

    def __init__(self, config):
        super().__init__()
        params = _model_section(config, "transunet3d_baseline")
        self.config = config
        self.roi_shape = _roi_shape(config)
        base = int(getattr(params, "base_channels", 8))
        heads = int(getattr(params, "num_heads", 4))
        max_tokens = int(getattr(params, "max_tokens", 512))
        self._init_output_alignment(config, params)
        self.builder = SurfaceVolumeBuilder(self.roi_shape, config.data.view_angles)
        self.enc1 = ConvBlock3d(1, base)
        self.down1 = nn.MaxPool3d(2, ceil_mode=True)
        self.enc2 = ConvBlock3d(base, base * 2)
        self.down2 = nn.MaxPool3d(2, ceil_mode=True)
        self.enc3 = ConvBlock3d(base * 2, base * 4)
        self.trans = TransformerBottleneck3D(base * 4, heads, max_tokens)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock3d(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock3d(base * 2, base)
        self.out = nn.Conv3d(base, 1, 1)

    @staticmethod
    def _match(x, ref):
        if x.shape[2:] == ref.shape[2:]:
            return x
        return F.interpolate(x, size=ref.shape[2:], mode="trilinear", align_corners=False)

    def forward(self, projections, *args, **kwargs):
        x = self.builder(projections)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.trans(self.enc3(self.down2(e2)))
        d2 = self._match(self.up2(e3), e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self._match(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        logits = self._match_roi_shape(self.out(d1))
        logits = self._paste_roi(logits)
        return {
            "pred_voxel": logits,
            "aux_outputs": {
                "output_space": self.output_mode,
                "alignment_mode": "paste_pred" if self.paste_roi_to_full else self.output_mode,
            },
        }


class SurfaceVolumeBuilder(nn.Module):
    """Embed sparse-view projections onto a 3D cube boundary shell.

    Projections are resized and splatted onto cube faces according to view angle.
    Intermediate oblique views contribute to adjacent x/y faces using cosine weights.
    No global pooling or latent expansion is used.
    """

    def __init__(self, roi_shape: tuple[int, int, int], view_angles):
        super().__init__()
        self.roi_shape = tuple(int(v) for v in roi_shape)
        self.view_angles = [int(v) for v in view_angles]

    @staticmethod
    def _norm_projection(x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        amin = x.amin(dim=(-2, -1), keepdim=True)
        amax = x.amax(dim=(-2, -1), keepdim=True)
        return ((x - amin) / (amax - amin).clamp_min(1e-6)).clamp(0.0, 1.0)

    def _add_y_face(self, vol, weight, image, y_index: int):
        x, _y, z = self.roi_shape
        face = F.interpolate(image, size=(x, z), mode="bilinear", align_corners=False)
        vol[:, :, :, y_index, :] += weight * face

    def _add_x_face(self, vol, weight, image, x_index: int):
        _x, y, z = self.roi_shape
        face = F.interpolate(image, size=(y, z), mode="bilinear", align_corners=False)
        vol[:, :, x_index, :, :] += weight * face

    def forward(self, projections) -> torch.Tensor:
        p = self._norm_projection(_projection_tensor(projections))
        b = p.shape[0]
        x, y, z = self.roi_shape
        vol = p.new_zeros((b, 1, x, y, z))
        weight = p.new_zeros((b, 1, x, y, z))
        for idx, angle in enumerate(self.view_angles):
            image = p[:, idx : idx + 1]
            rad = math.radians(float(angle))
            wx = abs(math.sin(rad))
            wy = abs(math.cos(rad))
            if wx == 0 and wy == 0:
                wy = 1.0
            norm = max(wx + wy, 1e-6)
            wx, wy = wx / norm, wy / norm
            if wy > 0:
                yi = 0 if math.cos(rad) >= 0 else y - 1
                self._add_y_face(vol, wy, image, yi)
                self._add_y_face(weight, wy, torch.ones_like(image), yi)
            if wx > 0:
                xi = 0 if math.sin(rad) < 0 else x - 1
                self._add_x_face(vol, wx, image, xi)
                self._add_x_face(weight, wx, torch.ones_like(image), xi)
        return vol / weight.clamp_min(1e-6)


class TemplateFeatureExtractor(nn.Module):
    """Radiomics-like features for nearest-template selection."""

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        b = volume.shape[0]
        flat = volume.reshape(b, -1)
        prob = flat.clamp_min(0.0)
        total = prob.sum(dim=1, keepdim=True).clamp_min(1e-6)
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, volume.shape[2], device=volume.device, dtype=volume.dtype),
                torch.linspace(0, 1, volume.shape[3], device=volume.device, dtype=volume.dtype),
                torch.linspace(0, 1, volume.shape[4], device=volume.device, dtype=volume.dtype),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 3)
        center = prob @ coords / total
        centered = coords.unsqueeze(0) - center.unsqueeze(1)
        spread = (prob.unsqueeze(-1) * centered.square()).sum(dim=1) / total
        hist = torch.stack(
            [
                flat.mean(dim=1),
                flat.std(dim=1, unbiased=False),
                flat.amax(dim=1),
                torch.quantile(flat, 0.95, dim=1),
                (flat.square()).mean(dim=1),
                (flat > 0).float().mean(dim=1),
            ],
            dim=1,
        )
        return torch.cat([hist, center, spread], dim=1)


class AffineSTN3D(nn.Module):
    """4 conv blocks + 4 max-pooling layers + 3-layer MLP affine regressor."""

    def __init__(self, in_ch: int, base_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock3d(in_ch, base_ch),
            nn.MaxPool3d(2, ceil_mode=True),
            ConvBlock3d(base_ch, base_ch * 2),
            nn.MaxPool3d(2, ceil_mode=True),
            ConvBlock3d(base_ch * 2, base_ch * 4),
            nn.MaxPool3d(2, ceil_mode=True),
            ConvBlock3d(base_ch * 4, base_ch * 8),
            nn.MaxPool3d(2, ceil_mode=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(base_ch * 8, base_ch * 8),
            nn.SiLU(inplace=True),
            nn.Linear(base_ch * 8, base_ch * 4),
            nn.SiLU(inplace=True),
            nn.Linear(base_ch * 4, 12),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        self.mlp[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float32)
        )

    def forward(self, x):
        return self.mlp(self.conv(x)).view(-1, 3, 4).to(dtype=x.dtype)


class TemplateSTNVNetBase(VoxelOutputAlignmentMixin, nn.Module):
    """PGDPNN/FMT-ReconNet style template selection, affine STN, and V-Net refinement."""

    output_type = "voxel"

    def __init__(self, config, section: str):
        super().__init__()
        params = _model_section(config, section)
        self.config = config
        self.section = section
        self.roi_shape = _roi_shape(config)
        self.num_views = int(getattr(config.model, "num_views", len(config.data.view_angles)))
        self.builder = SurfaceVolumeBuilder(self.roi_shape, config.data.view_angles)
        self.feature_extractor = TemplateFeatureExtractor()
        base = int(getattr(params, "base_channels", 12))
        self.stn = AffineSTN3D(2, base)
        self.vnet = VNet3D(2, base)
        self.alpha = float(getattr(params, "alpha", 1.0))
        self.stage_a_epochs = int(getattr(params, "stage_a_epochs", 10))
        self.stage_b_epochs = int(getattr(params, "stage_b_epochs", 20))
        self.current_epoch = 0
        self.template_path = str(getattr(params, "template_path", "") or "")
        self.allow_template_fallback = bool(getattr(params, "allow_template_fallback", False))
        self.require_template_shape_match = bool(getattr(params, "require_template_shape_match", True))
        self.template_features_norm = None
        self._init_output_alignment(config, params)
        self._load_templates()
        self._apply_stage_freeze()

    def _expected_full_shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in getattr(self.config.model.geometry, "global_voxel_shape", self.roi_shape))

    def _full_volume_shape(self) -> tuple[int, int, int]:
        return tuple(reversed(self._expected_full_shape()))

    def _load_templates(self):
        if not self.template_path:
            if not self.allow_template_fallback:
                raise ValueError(
                    f"{self.section} requires a real template_path. "
                    "Run scripts/build_pgdpnn_templates.py first, or set allow_template_fallback=true for smoke tests only."
                )
            self.register_buffer("surface_templates", torch.empty(0), persistent=False)
            self.register_buffer("source_templates", torch.empty(0), persistent=False)
            self.register_buffer("template_features", torch.empty(0), persistent=False)
            self.template_case_ids = []
            return
        path = Path(self.template_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{self.section}.template_path does not exist: {path}")
        z = np.load(path, allow_pickle=True)
        expected_xyz = self._expected_full_shape()
        expected_zyx = tuple(reversed(expected_xyz))
        surface_templates = np.asarray(z["surface_templates"], dtype=np.float32)
        source_templates = np.asarray(z["source_templates"], dtype=np.float32)
        if surface_templates.shape[-3:] != source_templates.shape[-3:]:
            raise ValueError(
                f"{self.section} template surface/source shape mismatch: "
                f"template_path={path}, surface_shape={tuple(int(v) for v in surface_templates.shape)}, "
                f"source_shape={tuple(int(v) for v in source_templates.shape)}"
            )
        template_spatial_shape = tuple(int(v) for v in source_templates.shape[-3:])
        if self.require_template_shape_match:
            if template_spatial_shape not in {expected_xyz, expected_zyx}:
                raise ValueError(
                    f"{self.section} template shape mismatch: "
                    f"template_path={path}, template_shape={tuple(int(v) for v in source_templates.shape)}, "
                    f"expected_xyz={expected_xyz}, expected_zyx={expected_zyx}, "
                    f"global_voxel_shape={expected_xyz}"
                )
        if template_spatial_shape == expected_zyx:
            if source_templates.ndim == 5:
                source_templates = source_templates.transpose(0, 1, 4, 3, 2)
                surface_templates = surface_templates.transpose(0, 1, 4, 3, 2)
            else:
                source_templates = source_templates.transpose(0, 3, 2, 1)
                surface_templates = surface_templates.transpose(0, 3, 2, 1)
        self.template_shape = tuple(int(v) for v in source_templates.shape)
        self.register_buffer(
            "surface_templates",
            torch.from_numpy(surface_templates),
            persistent=False,
        )
        self.register_buffer(
            "source_templates",
            torch.from_numpy(source_templates),
            persistent=False,
        )
        self.register_buffer(
            "template_features",
            torch.from_numpy(z["template_features"].astype(np.float32)),
            persistent=False,
        )
        self.template_case_ids = [str(v) for v in z["template_case_ids"].tolist()]

    def set_training_epoch(self, epoch: int):
        self.current_epoch = int(epoch)
        self._apply_stage_freeze()

    def _apply_stage_freeze(self):
        if self.current_epoch < self.stage_a_epochs:
            stn_train, vnet_train = True, False
        elif self.current_epoch < self.stage_a_epochs + self.stage_b_epochs:
            stn_train, vnet_train = False, True
        else:
            stn_train, vnet_train = True, True
        for param in self.stn.parameters():
            param.requires_grad = stn_train
        for param in self.vnet.parameters():
            param.requires_grad = vnet_train

    def _fallback_templates(self, x_tr: torch.Tensor):
        b = x_tr.shape[0]
        ids = torch.zeros(b, dtype=torch.long, device=x_tr.device)
        return x_tr.detach(), torch.zeros_like(x_tr), ids

    def _select_templates(self, x_tr: torch.Tensor):
        if self.surface_templates.numel() == 0:
            return self._fallback_templates(x_tr)
        features = self.feature_extractor(x_tr)
        template_features = self.template_features.to(device=x_tr.device, dtype=x_tr.dtype)
        mean = template_features.mean(dim=0, keepdim=True)
        std = template_features.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        dist = torch.cdist((features - mean) / std, (template_features - mean) / std)
        ids = dist.argmin(dim=1)
        x_tp = self.surface_templates.to(device=x_tr.device, dtype=x_tr.dtype)[ids]
        y_tp = self.source_templates.to(device=x_tr.device, dtype=x_tr.dtype)[ids]
        return x_tp, y_tp, ids

    @staticmethod
    def _warp(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)

    def forward(self, projections, *args, **kwargs):
        x_tr = self.builder(projections)
        x_tp, y_tp, template_ids = self._select_templates(x_tr)
        theta = self.stn(torch.cat([x_tr, x_tp], dim=1))
        x_tp_def = self._warp(x_tp, theta)
        y_tp_def = self._warp(y_tp, theta)
        pred = self.vnet(torch.cat([x_tr, y_tp_def], dim=1))
        pred = self._match_roi_shape(pred)
        pred = self._paste_roi(pred)
        ident = _identity_affine(theta.device, theta.dtype, theta.shape[0])
        stn_align = F.mse_loss(x_tp_def.clamp(0.0, 1.0), x_tr.clamp(0.0, 1.0))
        stn_l2 = (theta - ident).square().mean()
        aux = {
            "stn_loss": stn_align + 0.01 * stn_l2,
            "voxel_loss_weight": self.alpha,
            "selected_template_id": template_ids.detach(),
            "theta": theta.detach(),
            "template_path": self.template_path,
            "template_shape": tuple(self.template_shape),
            "expected_xyz": self._expected_full_shape(),
            "expected_zyx": self._full_volume_shape(),
            "debug": {
                "x_tr": x_tr.detach(),
                "x_tp": x_tp.detach(),
                "x_tp_def": x_tp_def.detach(),
                "y_tp": y_tp.detach(),
                "y_tp_def": y_tp_def.detach(),
                "pred_voxel": pred.detach(),
            },
        }
        return {"pred_voxel": pred, "aux_outputs": aux}


class FMTReconNetAdapted(TemplateSTNVNetBase):
    def __init__(self, config):
        super().__init__(config, "fmt_reconnet")


class PGDPNNAdapted(TemplateSTNVNetBase):
    def __init__(self, config):
        super().__init__(config, "pgdpnn")


class ProjectionRestorationNet(nn.Module):
    def __init__(self, views: int, base: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(views, base, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base, views, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class SliceIRadonNet(nn.Module):
    """Slice-wise profile-to-transverse reconstruction network."""

    def __init__(self, views: int, input_h: int, roi_shape: tuple[int, int, int], hidden: int):
        super().__init__()
        x, y, _z = roi_shape
        self.roi_shape = roi_shape
        self.views = int(views)
        self.input_h = int(input_h)
        self.fc1 = nn.Linear(self.views * self.input_h, hidden)
        self.fc2 = nn.Linear(hidden, x * y)
        self.refine = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, restored: torch.Tensor) -> torch.Tensor:
        b, v, h, _w = restored.shape
        x, y, z = self.roi_shape
        restored_z = F.interpolate(
            restored,
            size=(self.input_h, z),
            mode="bilinear",
            align_corners=False,
        )
        profiles = restored_z.permute(0, 3, 1, 2).contiguous().reshape(b * z, v * self.input_h)
        slices = self.fc2(F.silu(self.fc1(profiles))).view(b * z, 1, x, y)
        slices = self.refine(slices)
        return slices.view(b, z, 1, x, y).permute(0, 2, 3, 4, 1).contiguous()


class TwoStageDeepFMTAdapted(VoxelOutputAlignmentMixin, nn.Module):
    """Restoration-Net followed by slice-wise iRadon-Net, without latent expand."""

    output_type = "voxel"

    def __init__(self, config):
        super().__init__()
        params = _model_section(config, "two_stage_deepfmt")
        self.config = config
        self.roi_shape = _roi_shape(config)
        self.num_views = int(getattr(config.model, "num_views", len(config.data.view_angles)))
        base = int(getattr(params, "base_channels", 16))
        hidden = int(getattr(params, "iradon_hidden_dim", 512))
        input_h = int(getattr(params, "profile_bins", self.roi_shape[0]))
        self.lambda_proj = float(getattr(params, "lambda_proj", 0.0))
        self._init_output_alignment(config, params)
        self.restorer = ProjectionRestorationNet(self.num_views, base)
        self.iradon = SliceIRadonNet(self.num_views, input_h, self.roi_shape, hidden)

    def forward(self, projections, *args, **kwargs):
        x = _projection_tensor(projections)
        restored = self.restorer(x)
        pred = self._paste_roi(self.iradon(restored))
        proj_loss = pred.new_zeros(())
        if self.lambda_proj > 0:
            proj_loss = self.lambda_proj * F.l1_loss(torch.sigmoid(restored), x.clamp(0.0, 1.0))
        return {
            "pred_voxel": pred,
            "aux_outputs": {
                "output_space": self.output_mode,
                "alignment_mode": "paste_pred" if self.paste_roi_to_full else self.output_mode,
                "restored_projections": restored,
                "projection_loss": proj_loss,
            },
        }


class AttentionFusion(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(channels, channels), nn.Tanh(), nn.Linear(channels, 1))

    def forward(self, view_features: torch.Tensor):
        pooled = view_features.mean(dim=(-2, -1))
        weights = torch.softmax(self.score(pooled).squeeze(-1), dim=1)
        fused = (view_features * weights[:, :, None, None, None]).sum(dim=1)
        return fused, weights


class PatchDiscriminator2D(nn.Module):
    def __init__(self, in_ch: int = 1, base_ch: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 2, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class MAPPGANAdapted(ProjectionToVoxelNet):
    """MAP-PGAN-adapted generator with multi-branch view encoders and attention fusion."""

    def __init__(self, config):
        super().__init__(config, "map_pgan")
        params = _model_section(config, "map_pgan")
        base = int(getattr(params, "base_channels", 16))
        latent = int(getattr(params, "latent_channels", base * 4))
        self.branches = nn.ModuleList([ProjectionEncoder2D(1, base) for _ in range(self.num_views)])
        self.fusion = AttentionFusion(self.branches[0].out_channels)
        self.fc = nn.Linear(self.branches[0].out_channels, latent)
        self.discriminator = PatchDiscriminator2D(1, base)
        self.use_gan = bool(getattr(params, "use_gan", False))

    def projection_features(self, projections):
        x = _projection_tensor(projections)
        feats = [branch(x[:, i : i + 1]) for i, branch in enumerate(self.branches)]
        fused, weights = self.fusion(torch.stack(feats, dim=1))
        self.last_attention_weights = weights.detach()
        return fused


class D2RecSTAdapted(ProjectionToVoxelNet):
    """Dual-domain adapted baseline with decoder feature consistency outputs."""

    def __init__(self, config):
        super().__init__(config, "d2_recst")

    def forward(self, projections, *args, **kwargs):
        pred, features = self.decoder(self.coarse_volume(projections), return_features=True)
        perceptual = sum(f.abs().mean() * 0.0 for f in features)
        return {
            "pred_voxel": pred,
            "aux_outputs": {"features": features, "perceptual_loss": perceptual},
        }


class DSPGNAdapted(ProjectionToVoxelNet):
    """Graph morphology prior baseline with lightweight kNN message passing."""

    def __init__(self, config):
        super().__init__(config, "dspgn")
        params = _model_section(config, "dspgn")
        latent = self.fc.out_features
        self.num_nodes = int(getattr(params, "num_nodes", 64))
        self.k = int(getattr(params, "k_neighbors", 6))
        self.node_mlp = nn.Sequential(
            nn.Linear(latent + 3, latent), nn.SiLU(), nn.Linear(latent, latent)
        )
        self.msg_mlp = nn.Sequential(
            nn.Linear(latent * 2 + 3, latent), nn.SiLU(), nn.Linear(latent, latent)
        )

    def forward(self, projections, *args, **kwargs):
        vol = self.coarse_volume(projections)
        b, c, x, y, z = vol.shape
        n = min(self.num_nodes, x * y * z)
        side = math.ceil(n ** (1 / 3))
        coords_1d = torch.linspace(0, 1, side, device=vol.device, dtype=vol.dtype)
        grid = torch.stack(torch.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij"), dim=-1)
        pos = grid.reshape(-1, 3)[:n]
        sample_grid = pos.mul(2).sub(1).view(1, n, 1, 1, 3).expand(b, -1, -1, -1, -1)
        node = F.grid_sample(vol, sample_grid, align_corners=False).view(b, c, n).transpose(1, 2)
        node = self.node_mlp(torch.cat([node, pos.view(1, n, 3).expand(b, -1, -1)], dim=-1))

        dist = torch.cdist(pos, pos)
        knn = dist.topk(k=min(self.k + 1, n), largest=False).indices[:, 1:]
        neigh = node[:, knn]
        center = node[:, :, None, :].expand_as(neigh)
        rel = (pos[knn] - pos[:, None]).view(1, n, -1, 3).expand(b, -1, -1, -1)
        msg = self.msg_mlp(torch.cat([center, neigh - center, rel], dim=-1)).mean(dim=2)
        node = node + msg

        coarse = vol + node.mean(dim=1).view(b, c, 1, 1, 1)
        pred = self.decoder(coarse)
        return {"pred_voxel": pred, "aux_outputs": {"graph_nodes": node}}


class Stage1VolumeMixin:
    stage1_key_candidates = ("stage1_voxel", "fem_prior", "coarse_prior", "stage1_recon")

    def _stage1_from_kwargs(self, kwargs) -> torch.Tensor | None:
        batch = kwargs.get("batch")
        if isinstance(batch, dict):
            for key in self.stage1_key_candidates:
                value = batch.get(key)
                if torch.is_tensor(value):
                    return value
        return None


class FEM2VoxUNet(Stage1VolumeMixin, ProjectionToVoxelNet):
    """FEM-prior + 3D U-Net baseline when a real prior volume is present."""

    def __init__(self, config):
        if str(getattr(config.model, "name", "")) == "stage1_unet":
            section = "stage1_unet"
        else:
            section = "fem2vox_unet"
        super().__init__(config, section)
        params = _model_section(config, section)
        base = int(getattr(params, "base_channels", 16))
        self.unet = VNet3D(1, base)
        self.allow_projection_fallback = bool(getattr(params, "allow_projection_fallback", False))

    def forward(self, projections, *args, **kwargs):
        stage1 = self._stage1_from_kwargs(kwargs)
        if stage1 is None:
            if not self.allow_projection_fallback:
                raise ValueError(
                    "fem2vox_unet requires a real FEM prior in the batch; "
                    "set allow_projection_fallback=true only for smoke tests."
                )
            return super().forward(projections, *args, **kwargs)
        if stage1.dim() == 4:
            stage1 = stage1.unsqueeze(1)
        pred = self.unet(stage1.float())
        return {"pred_voxel": pred, "aux_outputs": {"prior_source": "fem_prior"}}


class Stage1InterpolationBaseline(Stage1VolumeMixin, nn.Module):
    output_type = "voxel"

    def __init__(self, config):
        super().__init__()
        self.roi_shape = _roi_shape(config)

    def forward(self, projections, *args, **kwargs):
        stage1 = self._stage1_from_kwargs(kwargs)
        if stage1 is None:
            raise ValueError("fem_to_voxel requires a real FEM prior in the batch")
        if stage1.dim() == 4:
            stage1 = stage1.unsqueeze(1)
        pred = F.interpolate(
            stage1.float(),
            size=self.roi_shape,
            mode="trilinear",
            align_corners=False,
        )
        return {"pred_voxel": pred, "aux_outputs": {"prior_source": "fem_to_voxel"}}


class GenericVoxelBaseline(ProjectionToVoxelNet):
    def __init__(self, config):
        super().__init__(config, getattr(config.model, "name"))
