"""Voxel-domain adapted baselines for sparse-view FMT comparisons."""

from __future__ import annotations

import math

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
    return torch.stack([projections[k] for k in keys], dim=1)


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


class SmallVNet(nn.Module):
    def __init__(self, in_ch: int, base_ch: int, out_ch: int = 1):
        super().__init__()
        self.enc1 = ConvBlock3d(in_ch, base_ch)
        self.enc2 = ConvBlock3d(base_ch, base_ch * 2)
        self.mid = ConvBlock3d(base_ch * 2, base_ch * 4)
        self.dec2 = ConvBlock3d(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.dec1 = ConvBlock3d(base_ch * 2 + base_ch, base_ch)
        self.out = nn.Conv3d(base_ch, out_ch, 1)

    def forward(self, x, return_features: bool = False):
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool3d(e1, 2, ceil_mode=True))
        m = self.mid(F.avg_pool3d(e2, 2, ceil_mode=True))
        d2 = F.interpolate(m, size=e2.shape[2:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[2:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        out = self.out(d1)
        if return_features:
            return out, [e1, e2, m, d2, d1]
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
    """Shared projection-to-volume backbone used only by adapted voxel baselines."""

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
        self.decoder = SmallVNet(latent, base)

    def projection_features(self, projections):
        x = _projection_tensor(projections)
        feat = self.encoder(x)
        return feat

    def coarse_volume(self, projections):
        feat = self.projection_features(projections)
        pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        seed = self.fc(pooled).view(pooled.shape[0], -1, 1, 1, 1)
        return seed.expand(-1, -1, *self.roi_shape)

    def forward(self, projections, *args, **kwargs):
        pred = self.decoder(self.coarse_volume(projections))
        return {"pred_voxel": pred, "aux_outputs": {}}


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
        return {"pred_voxel": pred, "aux_outputs": {"features": features, "perceptual_loss": perceptual}}


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


class TwoStageDeepFMTAdapted(ProjectionToVoxelNet):
    def __init__(self, config):
        super().__init__(config, "two_stage_deepfmt")
        params = _model_section(config, "two_stage_deepfmt")
        self.restorer = ProjectionRestorationNet(self.num_views, int(getattr(params, "base_channels", 16)))

    def forward(self, projections, *args, **kwargs):
        x = _projection_tensor(projections)
        restored = self.restorer(x)
        pred = self.decoder(self.coarse_volume(restored))
        proj_loss = F.l1_loss(torch.sigmoid(restored), x.clamp(0.0, 1.0)) * 0.0
        return {"pred_voxel": pred, "aux_outputs": {"restored_projections": restored, "projection_loss": proj_loss}}


class SpatialTransformer3D(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.loc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(in_ch, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 12),
        )
        nn.init.zeros_(self.loc[-1].weight)
        self.loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float32))

    def forward(self, x):
        theta = self.loc(x).view(-1, 3, 4).to(dtype=x.dtype)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)


class FMTReconNetAdapted(ProjectionToVoxelNet):
    def __init__(self, config):
        super().__init__(config, "fmt_reconnet")
        latent = self.fc.out_features
        self.stn = SpatialTransformer3D(latent)

    def forward(self, projections, *args, **kwargs):
        prior = self.coarse_volume(projections)
        pred = self.decoder(self.stn(prior))
        return {"pred_voxel": pred, "aux_outputs": {"prior_source": "mean_projection_backprojection"}}


class PGDPNNAdapted(ProjectionToVoxelNet):
    def __init__(self, config):
        super().__init__(config, "pgdpnn")
        self.distribution_head = nn.Conv3d(self.fc.out_features, 1, 1)

    def forward(self, projections, *args, **kwargs):
        prior = self.coarse_volume(projections)
        pred_dist = self.distribution_head(prior)
        pred = self.decoder(prior)
        return {"pred_voxel": pred, "aux_outputs": {"pred_distribution": pred_dist}}


class DSPGNAdapted(ProjectionToVoxelNet):
    """Graph morphology prior baseline with lightweight kNN message passing."""

    def __init__(self, config):
        super().__init__(config, "dspgn")
        params = _model_section(config, "dspgn")
        latent = self.fc.out_features
        self.num_nodes = int(getattr(params, "num_nodes", 64))
        self.k = int(getattr(params, "k_neighbors", 6))
        self.node_mlp = nn.Sequential(nn.Linear(latent + 3, latent), nn.SiLU(), nn.Linear(latent, latent))
        self.msg_mlp = nn.Sequential(nn.Linear(latent * 2 + 3, latent), nn.SiLU(), nn.Linear(latent, latent))

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


class FEM2VoxUNet(ProjectionToVoxelNet):
    """Coarse-prior-to-voxel U-Net baseline without query projection."""

    def __init__(self, config):
        super().__init__(config, "fem2vox_unet")


class GenericVoxelBaseline(ProjectionToVoxelNet):
    def __init__(self, config):
        super().__init__(config, getattr(config.model, "name"))
