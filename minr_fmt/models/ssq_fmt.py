"""SSQ-FMT final-method point model.

This module intentionally does not subclass the legacy ``PointDensityNet`` path. It keeps
surface-measurement normalization, geometry-guided local sampling, measurement-derived
candidate anchors, candidate routing, candidate-specific view fusion, and probability-domain
density decoding in one explicit contract.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from minr_fmt.utils.fmt_simgen_projection import project_points_mm_to_detector


def _to_container(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)
    if isinstance(cfg, dict):
        return cfg
    return {}


def _cfg_get(cfg: dict[str, Any], path: str, default: Any) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    blocks: list[nn.Module] = []
    dim = in_dim
    for _ in range(max(1, layers - 1)):
        blocks.extend([nn.Linear(dim, hidden_dim), nn.SiLU()])
        dim = hidden_dim
    blocks.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*blocks)


class SurfaceMeasurementNormalizer(nn.Module):
    """Joint per-sample valid-pixel percentile normalization for surface measurements."""

    def __init__(self, percentile: float = 99.9, eps: float = 1e-6):
        super().__init__()
        self.percentile = float(percentile)
        self.eps = float(eps)

    def forward(
        self, measurements: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if measurements.dim() != 5:
            raise ValueError(
                f"surface measurements must be [B,V,C,H,W], got {tuple(measurements.shape)}"
            )
        y = torch.nan_to_num(measurements.float(), nan=0.0, posinf=0.0, neginf=0.0)
        B = y.shape[0]
        if valid_mask is None:
            mask = torch.isfinite(measurements).all(dim=2) & (y.abs().sum(dim=2) > 0.0)
        else:
            mask = valid_mask
            if mask.dim() == 5:
                mask = mask.squeeze(2)
            mask = mask.to(device=y.device, dtype=torch.bool)
        scale = torch.empty((B,), dtype=y.dtype, device=y.device)
        flat_y = y.abs().amax(dim=2).reshape(B, -1)
        flat_m = mask.reshape(B, -1)
        q = max(0.0, min(1.0, self.percentile / 100.0))
        for b in range(B):
            vals = flat_y[b][flat_m[b]]
            if vals.numel() == 0:
                scale[b] = torch.ones((), dtype=y.dtype, device=y.device)
                continue
            s = torch.quantile(vals, q).clamp_min(self.eps)
            scale[b] = s
        y_norm = (y / scale[:, None, None, None, None]).clamp(0.0, 1.0)
        y_norm = torch.where(mask[:, :, None], y_norm, torch.zeros_like(y_norm))
        return y_norm, scale


class SharedSurfaceEncoder(nn.Module):
    """Shared 2D encoder applied to every view."""

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
        B, V, C, H, W = x.shape
        feat = self.net(x.reshape(B * V, C, H, W))
        Cf, Hf, Wf = feat.shape[1:]
        return feat.reshape(B, V, Cf, Hf, Wf)


class GeometryQueryMapper(nn.Module):
    """Vectorized FMT-SimGen orthographic query-to-detector mapper."""

    def __init__(
        self,
        view_angles: list[int],
        camera_distance_mm: float = 200.0,
        fov_mm: float = 80.0,
        detector_resolution: tuple[int, int] = (256, 256),
        volume_center_world: tuple[float, float, float] = (19.0, 20.0, 10.4),
        fd_step_mm: float = 0.2,
    ):
        super().__init__()
        self.view_angles = [int(a) for a in view_angles]
        self.camera_distance_mm = float(camera_distance_mm)
        self.fov_mm = float(fov_mm)
        self.detector_resolution = (int(detector_resolution[0]), int(detector_resolution[1]))
        self.volume_center_world = tuple(float(x) for x in volume_center_world)
        self.fd_step_mm = float(fd_step_mm)

    def forward(self, points_mm: torch.Tensor) -> dict[str, torch.Tensor]:
        grids, depths, masks, uv_px, uv_phys, rays, jac = [], [], [], [], [], [], []
        for angle in self.view_angles:
            grid, depth, valid, px, phys = project_points_mm_to_detector(
                points_mm,
                angle,
                camera_distance_mm=self.camera_distance_mm,
                fov_mm=self.fov_mm,
                detector_resolution=self.detector_resolution,
                volume_center_world=self.volume_center_world,
            )
            grids.append(grid)
            depths.append(depth)
            masks.append(valid)
            uv_px.append(px)
            uv_phys.append(phys)
            rad = math.radians(float(angle))
            ray = torch.tensor(
                [math.sin(rad), 0.0, -math.cos(rad)],
                dtype=points_mm.dtype,
                device=points_mm.device,
            )
            rays.append(ray.expand(points_mm.shape[0], points_mm.shape[1], 3))
            # Orthographic projection has constant local detector scale. Kept as a tensor
            # so future exact mesh intersections or non-orthographic geometries can replace it.
            jac.append(torch.ones_like(depth[..., 0]).clamp_min(self.fd_step_mm * 0.0 + 1.0))
        return {
            "grid": torch.stack(grids, dim=1),
            "depth": torch.stack(depths, dim=1),
            "valid_mask": torch.stack(masks, dim=1),
            "uv_px": torch.stack(uv_px, dim=1),
            "uv_phys": torch.stack(uv_phys, dim=1),
            "ray_directions": torch.stack(rays, dim=1),
            "jacobian_scale": torch.stack(jac, dim=1),
        }


class QueryDependentSurfaceSampler(nn.Module):
    """Sample K local bilinear feature footprints around each projected query."""

    def __init__(self, offsets_px: list[list[float]], align_corners: bool = True):
        super().__init__()
        offsets = torch.tensor(offsets_px, dtype=torch.float32)
        if offsets.dim() != 2 or offsets.shape[-1] != 2:
            raise ValueError("offsets_px must be a list of [du,dv] pairs")
        self.register_buffer("offsets_px", offsets, persistent=False)
        self.align_corners = bool(align_corners)

    def forward(
        self, features: torch.Tensor, mapped: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, V, C, H, W = features.shape
        N = mapped["grid"].shape[2]
        offsets = self.offsets_px.to(device=features.device, dtype=features.dtype)
        if self.align_corners:
            dx = 2.0 / max(W - 1, 1)
            dy = 2.0 / max(H - 1, 1)
        else:
            dx = 2.0 / max(W, 1)
            dy = 2.0 / max(H, 1)
        offsets_grid = torch.stack([offsets[:, 0] * dx, offsets[:, 1] * dy], dim=-1)
        grid = mapped["grid"][:, :, :, None, :] + offsets_grid[None, None, None, :, :]
        sample_valid = mapped["valid_mask"][:, :, :, None] & (grid.abs() <= 1.0).all(dim=-1)
        feat_flat = features.reshape(B * V, C, H, W)
        grid_flat = grid.reshape(B * V, N * offsets.shape[0], 1, 2)
        sampled = F.grid_sample(
            feat_flat,
            grid_flat,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=self.align_corners,
        )
        sampled = sampled.squeeze(-1).transpose(1, 2).reshape(B, V, N, offsets.shape[0], C)
        sampled = torch.where(sample_valid[..., None], sampled, torch.zeros_like(sampled))
        return sampled, sample_valid


class MeasurementDerivedCandidateBuilder(nn.Module):
    """Build padded candidate anchors from normalized measurements or cached score fields."""

    def __init__(
        self,
        mmax: int = 5,
        threshold: float = 0.05,
        scale_min_mm: float = 1.0,
        scale_max_mm: float = 12.0,
        trunk_size_mm: tuple[float, float, float] = (38.0, 40.0, 20.8),
    ):
        super().__init__()
        self.mmax = int(mmax)
        self.threshold = float(threshold)
        self.scale_min_mm = float(scale_min_mm)
        self.scale_max_mm = float(scale_max_mm)
        self.trunk_size_mm = tuple(float(x) for x in trunk_size_mm)

    @torch.no_grad()
    def forward(
        self, measurements: torch.Tensor, batch: dict[str, Any] | None = None
    ) -> dict[str, torch.Tensor]:
        B, V, _C, H, W = measurements.shape
        device = measurements.device
        dtype = measurements.dtype
        centers = torch.zeros((B, self.mmax, 3), device=device, dtype=dtype)
        scores = torch.zeros((B, self.mmax), device=device, dtype=dtype)
        scales = torch.full((B, self.mmax), self.scale_min_mm, device=device, dtype=dtype)
        valid = torch.zeros((B, self.mmax), device=device, dtype=torch.bool)

        if batch is not None and "candidate_centers_mm" in batch:
            src_centers = batch["candidate_centers_mm"].to(device=device, dtype=dtype)
            src_scores = batch.get("candidate_scores")
            src_scales = batch.get("candidate_scales_mm")
            src_valid = batch.get("candidate_valid_mask")
            M = min(self.mmax, src_centers.shape[1])
            centers[:, :M] = src_centers[:, :M]
            if src_scores is not None:
                scores[:, :M] = src_scores[:, :M].to(device=device, dtype=dtype)
            else:
                scores[:, :M] = 1.0
            if src_scales is not None:
                raw_scale = src_scales[:, :M].to(device=device, dtype=dtype)
                scales[:, :M] = raw_scale.clamp(self.scale_min_mm, self.scale_max_mm)
            if src_valid is not None:
                valid[:, :M] = src_valid[:, :M].to(device=device, dtype=torch.bool)
            else:
                valid[:, :M] = scores[:, :M] > self.threshold
            return {
                "candidate_centers_mm": centers,
                "candidate_scores": scores.clamp(0.0, 1.0),
                "candidate_scales_mm": scales.clamp(self.scale_min_mm, self.scale_max_mm),
                "candidate_valid_mask": valid,
            }

        field = measurements.mean(dim=(1, 2))
        flat = field.reshape(B, -1)
        topk = min(self.mmax, flat.shape[1])
        vals, idx = torch.topk(flat, k=topk, dim=1)
        ys = torch.div(idx, W, rounding_mode="floor")
        xs = idx % W
        maxv = vals[:, :1].clamp_min(1e-8)
        score = (vals / maxv).clamp(0.0, 1.0)
        sx, sy, sz = self.trunk_size_mm
        centers[:, :topk, 0] = (xs.to(dtype) + 0.5) / float(W) * sx
        centers[:, :topk, 1] = (ys.to(dtype) + 0.5) / float(H) * sy
        centers[:, :topk, 2] = 0.5 * sz
        scores[:, :topk] = score
        valid[:, :topk] = vals > self.threshold
        return {
            "candidate_centers_mm": centers,
            "candidate_scores": scores,
            "candidate_scales_mm": scales,
            "candidate_valid_mask": valid,
        }


class CandidateSurfaceRouter(nn.Module):
    """Route each local surface sample over compensation plus candidate branches."""

    def __init__(self, feature_dim: int, hidden_dim: int = 64, temperature: float = 1.0):
        super().__init__()
        self.routing_net = _mlp(feature_dim + 5, hidden_dim, 1, layers=3)
        self.evidence_net = _mlp(feature_dim + 3, hidden_dim, 1, layers=2)
        self.temperature = float(temperature)

    def forward(
        self,
        sample_features: torch.Tensor,
        sample_valid: torch.Tensor,
        points_mm: torch.Tensor,
        candidates: dict[str, torch.Tensor],
        mapped: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        B, V, N, K, C = sample_features.shape
        centers = candidates["candidate_centers_mm"]
        scores = candidates["candidate_scores"]
        scales = candidates["candidate_scales_mm"].clamp_min(1e-6)
        cand_valid = candidates["candidate_valid_mask"]
        M = centers.shape[1]
        Mb = M + 1
        if M > 0:
            diff = points_mm[:, :, None, :] - centers[:, None, :, :]
            dist = torch.linalg.norm(diff, dim=-1)
            p_m = torch.exp(-dist / scales[:, None, :].clamp_min(1e-6)) * scores[:, None, :]
            p_m = torch.where(cand_valid[:, None, :], p_m, torch.zeros_like(p_m))
            p0 = (1.0 - p_m.amax(dim=-1, keepdim=True)).clamp(0.0, 1.0)
        else:
            p_m = torch.zeros((B, N, 0), device=points_mm.device, dtype=points_mm.dtype)
            p0 = torch.ones((B, N, 1), device=points_mm.device, dtype=points_mm.dtype)
        p_all = torch.cat([p0, p_m], dim=-1)

        cand_map = []
        for m in range(M):
            cand_proj = []
            for angle in range(V):
                # Candidate detector priors use the same projection geometry as query sampling.
                cand_proj.append(centers[:, m])
            cand_map.append(torch.stack(cand_proj, dim=1))
        if M > 0:
            cand_xyz = torch.stack(cand_map, dim=2)
            d3 = torch.linalg.norm(points_mm[:, None, :, None, :] - cand_xyz[:, :, None], dim=-1)
            K_m = (
                torch.exp(-d3 / scales[:, None, None, :].clamp_min(1e-6)) * scores[:, None, None, :]
            )
            K_m = torch.where(cand_valid[:, None, None, :], K_m, torch.zeros_like(K_m))
        else:
            K_m = torch.zeros((B, V, N, 0), device=points_mm.device, dtype=points_mm.dtype)
        if M > 0:
            K0 = (1.0 - K_m.amax(dim=-1, keepdim=True)).clamp(1e-6, 1.0)
        else:
            K0 = torch.ones((B, V, N, 1), device=points_mm.device, dtype=points_mm.dtype)
        K_all = torch.cat([K0, K_m], dim=-1)

        branch_valid = torch.cat(
            [torch.ones((B, 1), device=points_mm.device, dtype=torch.bool), cand_valid], dim=1
        )
        feat = sample_features[:, :, :, :, None, :].expand(B, V, N, K, Mb, C)
        p_feat = p_all[:, None, :, None, :, None].expand(B, V, N, K, Mb, 1)
        k_feat = K_all[:, :, :, None, :, None].expand(B, V, N, K, Mb, 1)
        q = points_mm[:, None, :, None, None, :].expand(B, V, N, K, Mb, 3)
        logits = self.routing_net(torch.cat([feat, p_feat, k_feat, q], dim=-1)).squeeze(-1)
        valid = sample_valid[..., None] & branch_valid[:, None, None, None, :]
        logits = logits.masked_fill(~valid, -1.0e4)
        zeta = torch.softmax(logits / max(self.temperature, 1e-6), dim=-1)
        zeta = torch.where(valid, zeta, torch.zeros_like(zeta))
        no_route = zeta.sum(dim=-1, keepdim=True) <= 0.0
        comp = torch.zeros_like(zeta)
        comp[..., 0] = sample_valid.float()
        zeta = torch.where(no_route, comp, zeta)

        mass = zeta * sample_valid[..., None].float()
        denom = mass.sum(dim=(1, 3), keepdim=False).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        a = mass.sum(dim=(1, 3)) / denom
        no_samples = sample_valid.float().sum(dim=(1, 3)) <= 0.0
        a = torch.where(no_samples[..., None], torch.zeros_like(a), a)
        a[..., 0] = torch.where(no_samples, torch.ones_like(a[..., 0]), a[..., 0])

        ev_in = torch.cat(
            [
                sample_features,
                points_mm[:, None, :, None, :].expand(B, V, N, K, 3),
            ],
            dim=-1,
        )
        e_sample = torch.sigmoid(self.evidence_net(ev_in)).squeeze(-1) * sample_valid.float()
        e = (e_sample[..., None] * zeta).sum(dim=3)
        zsum = zeta.sum(dim=3).clamp_min(1e-8)
        nu = (e / zsum).clamp(0.0, 1.0)
        r = (a[:, None] * nu).clamp(0.0, 1.0)
        return {
            "zeta": zeta,
            "a": a,
            "e": e.clamp(0.0, 1.0),
            "nu": nu,
            "r": r,
            "p_all": p_all,
            "branch_valid": branch_valid,
        }


class CandidateViewEncoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.representation_net = _mlp(feature_dim + 3, hidden_dim, hidden_dim, layers=2)

    def forward(
        self,
        sample_features: torch.Tensor,
        zeta: torch.Tensor,
        sample_valid: torch.Tensor,
        points_mm: torch.Tensor,
    ) -> torch.Tensor:
        B, V, N, K, C = sample_features.shape
        Mb = zeta.shape[-1]
        weighted = (sample_features[..., None, :] * zeta[..., None]).sum(dim=3)
        denom = zeta.sum(dim=3).clamp_min(1e-8)[..., None]
        pooled = weighted / denom
        q = points_mm[:, None, :, None, :].expand(B, V, N, Mb, 3)
        rep = self.representation_net(torch.cat([pooled, q], dim=-1))
        return rep.permute(0, 2, 1, 3, 4).contiguous()


class CandidateSpecificViewFusion(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.cross_view_fusion = _mlp(feature_dim + 1, hidden_dim, 1, layers=2)
        self.transform = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.SiLU())

    def forward(
        self,
        per_view: torch.Tensor,
        view_support: torch.Tensor,
        view_valid: torch.Tensor,
        branch_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, V, Mb, C = per_view.shape
        h = self.transform(per_view)
        support = view_support.permute(0, 2, 1, 3).contiguous()
        logits = self.cross_view_fusion(torch.cat([h, support[..., None]], dim=-1)).squeeze(-1)
        valid = view_valid.permute(0, 2, 1)[:, :, :, None] & branch_valid[:, None, None, :]
        logits = logits.masked_fill(~valid, -1.0e4)
        weights = torch.softmax(logits, dim=2)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        denom = weights.sum(dim=2, keepdim=True).clamp_min(1e-8)
        weights = weights / denom
        no_views = valid.sum(dim=2) == 0
        z = (weights[..., None] * h).sum(dim=2)
        z = torch.where(no_views[..., None], torch.zeros_like(z), z)
        Lambda = weights.sum(dim=2).clamp(0.0, 1.0)
        return z, weights, Lambda


class CandidateAssignmentHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.compensation_assignment_head = _mlp(feature_dim + 4, hidden_dim, 1, layers=2)
        self.candidate_assignment_head = _mlp(feature_dim + 4, hidden_dim, 1, layers=2)

    def forward(
        self,
        z: torch.Tensor,
        points_mm: torch.Tensor,
        router: dict[str, torch.Tensor],
        candidates: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        B, N, Mb, C = z.shape
        M = Mb - 1
        default_active = router["branch_valid"][:, None, :].expand(B, N, Mb).clone()
        active = router.get("active_branch_mask", default_active).clone()
        active[:, :, 0] = True
        q = points_mm[:, :, None, :].expand(B, N, Mb, 3)
        a = router["a"][:, :, :, None]
        comp_logits = self.compensation_assignment_head(
            torch.cat([z[:, :, :1], q[:, :, :1], a[:, :, :1]], dim=-1)
        )
        if M == 0:
            return torch.ones((B, N, 1), dtype=z.dtype, device=z.device)
        cand_logits = self.candidate_assignment_head(
            torch.cat([z[:, :, 1:], q[:, :, 1:], a[:, :, 1:]], dim=-1)
        ).squeeze(-1)
        cand_valid = active[:, :, 1:]
        cand_logits = cand_logits.masked_fill(~cand_valid, -1.0e4)
        cand_pi = torch.softmax(cand_logits, dim=-1)
        cand_pi = torch.where(cand_valid, cand_pi, torch.zeros_like(cand_pi))
        has_cand = cand_valid.any(dim=-1, keepdim=True)
        pi0 = torch.sigmoid(comp_logits.squeeze(-1))
        pi0 = torch.where(has_cand, pi0, torch.ones_like(pi0))
        cand_pi = torch.where(has_cand, (1.0 - pi0) * cand_pi, torch.zeros_like(cand_pi))
        pi = torch.cat([pi0, cand_pi], dim=-1)
        return pi / pi.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class CompensationDensityDecoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.net = _mlp(feature_dim + 3, hidden_dim, 1, layers=3)

    def forward(self, z0: torch.Tensor, points_mm: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([z0, points_mm], dim=-1)))


class CandidateDensityDecoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.net = _mlp(feature_dim + 4, hidden_dim, 1, layers=3)

    def forward(
        self, z: torch.Tensor, points_mm: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        B, N, M, C = z.shape
        q = points_mm[:, :, None, :].expand(B, N, M, 3)
        s = scores[:, None, :, None].expand(B, N, M, 1)
        return torch.sigmoid(self.net(torch.cat([z, q, s], dim=-1)))


class MorphologySDFHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.net = _mlp(feature_dim + 3, hidden_dim, 1, layers=2)

    def forward(self, z: torch.Tensor, points_mm: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, points_mm], dim=-1))


class SSQFMT(nn.Module):
    """Final SSQ-FMT query model.

    Forward input:
      surface_measurements: [B,V,C_y,H,W]
      query_coordinates_mm: [B,Nq,3]

    Output:
      {"density": [B,Nq,1], "aux_outputs": dict, "diagnostics": optional dict}
    """

    output_type = "point_probability"

    def __init__(self, config: Any):
        super().__init__()
        root = _to_container(config)
        model_cfg = root.get("model", root)
        ssq = model_cfg.get("ssq_fmt", {})
        data = root.get("data", {})
        view_angles = [int(v) for v in data.get("view_angles", [-90, -60, -30, 0, 30, 60, 90])]
        in_channels = int(model_cfg.get("in_channels", ssq.get("in_channels", 1)))
        encoder_ch = int(_cfg_get(ssq, "encoder.channels", 32))
        feature_dim = int(_cfg_get(ssq, "encoder.feature_dim", 48))
        hidden_dim = int(_cfg_get(ssq, "hidden_dim", 64))
        self.normalizer = SurfaceMeasurementNormalizer(
            percentile=float(_cfg_get(ssq, "normalization.percentile", 99.9)),
            eps=float(_cfg_get(ssq, "normalization.eps", 1e-6)),
        )
        geom = ssq.get("geometry", {})
        detector = geom.get(
            "detector_resolution", model_cfg.get("geometry", {}).get("detector_size", [256, 256])
        )
        self.geometry_mapper = GeometryQueryMapper(
            view_angles=view_angles,
            camera_distance_mm=float(
                geom.get(
                    "camera_distance_mm",
                    model_cfg.get("geometry", {}).get("camera_distance", 200.0),
                )
            ),
            fov_mm=float(geom.get("fov_mm", 80.0)),
            detector_resolution=(int(detector[0]), int(detector[1])),
            volume_center_world=tuple(geom.get("volume_center_world", [19.0, 20.0, 10.4])),
            fd_step_mm=float(geom.get("jacobian_fd_step_mm", 0.2)),
        )
        self.surface_encoder = SharedSurfaceEncoder(in_channels, encoder_ch, feature_dim)
        offsets = _cfg_get(
            ssq,
            "local_sampling.offsets_px",
            [[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]],
        )
        self.surface_sampler = QueryDependentSurfaceSampler(offsets)
        cand = ssq.get("candidates", {})
        self.candidate_builder = MeasurementDerivedCandidateBuilder(
            mmax=int(cand.get("mmax", 5)),
            threshold=float(cand.get("threshold", 0.05)),
            scale_min_mm=float(cand.get("scale_min_mm", 1.0)),
            scale_max_mm=float(cand.get("scale_max_mm", 12.0)),
            trunk_size_mm=tuple(cand.get("trunk_size_mm", [38.0, 40.0, 20.8])),
        )
        self.candidate_router = CandidateSurfaceRouter(
            feature_dim,
            hidden_dim=hidden_dim,
            temperature=float(_cfg_get(ssq, "routing.temperature", 1.0)),
        )
        self.view_encoder = CandidateViewEncoder(feature_dim, hidden_dim)
        self.view_fusion = CandidateSpecificViewFusion(hidden_dim, hidden_dim)
        self.assignment_head = CandidateAssignmentHead(hidden_dim, hidden_dim)
        self.compensation_density_decoder = CompensationDensityDecoder(hidden_dim, hidden_dim)
        self.candidate_density_decoder = CandidateDensityDecoder(hidden_dim, hidden_dim)
        self.morphology_sdf_head = MorphologySDFHead(hidden_dim, hidden_dim)
        self.lambda_sdf = float(
            _cfg_get(root, "loss.lambda_sdf", _cfg_get(ssq, "sdf.lambda_sdf", 0.0))
        )
        self.candidate_prior_weight = float(_cfg_get(ssq, "candidates.prior_density_weight", 0.0))
        self.use_query_density_backbone = bool(
            _cfg_get(ssq, "query_density_backbone.enabled", False)
        )
        self.query_density_backbone_normalization = str(
            _cfg_get(ssq, "query_density_backbone.normalization", "sample_percentile")
        )
        self.query_density_backbone = None
        if self.use_query_density_backbone:
            from .gisc_multisource import GISCFMT

            self.query_density_backbone = GISCFMT(config=config)

    def forward(
        self,
        surface_measurements: torch.Tensor,
        query_coordinates_mm: torch.Tensor,
        *,
        detector_valid_mask: torch.Tensor | None = None,
        depth_maps: torch.Tensor | None = None,
        batch: dict[str, Any] | None = None,
        return_diagnostics: bool = False,
        **_: Any,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        y_norm, norm_scale = self.normalizer(surface_measurements, detector_valid_mask)
        features = self.surface_encoder(y_norm)
        mapped = self.geometry_mapper(query_coordinates_mm)
        if detector_valid_mask is not None:
            mask = detector_valid_mask
            if mask.dim() == 5:
                mask = mask.squeeze(2)
            view_has_valid = (
                mask.to(device=query_coordinates_mm.device, dtype=torch.bool).flatten(2).any(dim=-1)
            )
            mapped["valid_mask"] = mapped["valid_mask"] & view_has_valid[:, :, None]
        sample_features, sample_valid = self.surface_sampler(features, mapped)
        # Depth maps are the currently available detector-side sample-surface proxy for
        # normalized path length. Exact mesh intersections can replace this hook later.
        candidates = self.candidate_builder(y_norm, batch=batch)
        router = self.candidate_router(
            sample_features, sample_valid, query_coordinates_mm, candidates, mapped
        )
        per_view = self.view_encoder(
            sample_features, router["zeta"], sample_valid, query_coordinates_mm
        )
        z, view_weights, Lambda = self.view_fusion(
            per_view,
            router["r"],
            mapped["valid_mask"],
            router["branch_valid"],
        )
        router["active_branch_mask"] = (Lambda > 0.0) & router["branch_valid"][:, None, :]
        pi = self.assignment_head(z, query_coordinates_mm, router, candidates)
        d0 = self.compensation_density_decoder(z[:, :, 0], query_coordinates_mm)
        M = z.shape[2] - 1
        if M > 0:
            dm = self.candidate_density_decoder(
                z[:, :, 1:], query_coordinates_mm, candidates["candidate_scores"]
            )
            dm = torch.where(
                candidates["candidate_valid_mask"][:, None, :, None],
                dm,
                torch.zeros_like(dm),
            )
            branch_density = torch.cat([d0[:, :, None], dm], dim=2).clamp(0.0, 1.0)
        else:
            branch_density = d0[:, :, None].clamp(0.0, 1.0)
        branch_contributions = pi[..., None] * branch_density
        lightweight_density = branch_contributions.sum(dim=2).clamp(0.0, 1.0)
        backbone_logits = None
        if self.query_density_backbone is not None and batch is not None and "points" in batch:
            if self.query_density_backbone_normalization == "per_view_max":
                denom = surface_measurements.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
                backbone_surface = (surface_measurements / denom).clamp(0.0, 1.0)
            else:
                backbone_surface = y_norm
            B, V = backbone_surface.shape[:2]
            legacy_surface = backbone_surface.permute(1, 0, 2, 3, 4).reshape(
                B * V,
                backbone_surface.shape[2],
                backbone_surface.shape[-2],
                backbone_surface.shape[-1],
            )
            backbone_out = self.query_density_backbone(
                legacy_surface,
                batch["points"].to(
                    device=query_coordinates_mm.device, dtype=query_coordinates_mm.dtype
                ),
                points_mm=query_coordinates_mm,
                depth_maps=depth_maps,
            )
            if isinstance(backbone_out, tuple):
                backbone_logits = backbone_out[0]
            else:
                backbone_logits = backbone_out
            decoded_density = torch.sigmoid(backbone_logits).clamp(0.0, 1.0)
        else:
            decoded_density = lightweight_density
        if M > 0 and self.candidate_prior_weight > 0.0:
            candidate_prior = router["p_all"][..., 1:].amax(dim=-1, keepdim=True)
            candidate_prior = (candidate_prior * self.candidate_prior_weight).clamp(0.0, 1.0)
            density = 1.0 - (1.0 - decoded_density) * (1.0 - candidate_prior)
        else:
            candidate_prior = torch.zeros_like(decoded_density)
            density = decoded_density
        density = density.clamp(0.0, 1.0)
        aux_outputs: dict[str, torch.Tensor] = {
            "pi": pi,
            "branch_density": branch_density,
            "branch_contributions": branch_contributions,
            "candidate_prior_density": candidate_prior,
            "lightweight_density": lightweight_density,
        }
        if backbone_logits is not None:
            aux_outputs["density_logits"] = backbone_logits
        if self.training and self.lambda_sdf > 0.0:
            z_mix = (pi[..., None] * z).sum(dim=2)
            aux_outputs["sdf"] = self.morphology_sdf_head(z_mix, query_coordinates_mm)
        diagnostics = {
            "normalization_scale": norm_scale,
            "candidate_centers_mm": candidates["candidate_centers_mm"],
            "candidate_scores": candidates["candidate_scores"],
            "candidate_scales_mm": candidates["candidate_scales_mm"],
            "candidate_valid_mask": candidates["candidate_valid_mask"],
            "zeta": router["zeta"],
            "a": router["a"],
            "e": router["e"],
            "nu": router["nu"],
            "r": router["r"],
            "view_weights": view_weights,
            "Lambda": Lambda,
            "pi": pi,
            "branch_density": branch_density,
            "branch_contributions": branch_contributions,
            "candidate_prior_density": candidate_prior,
            "decoded_density": decoded_density,
            "lightweight_density": lightweight_density,
            "sample_valid": sample_valid,
        }
        out: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "density": density,
            "aux_outputs": aux_outputs,
        }
        if return_diagnostics:
            out["diagnostics"] = diagnostics
        return out
