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


class FourierPositionEncoding(nn.Module):
    """Sin/cos positional encoding for query and candidate-relative coordinates."""

    def __init__(self, num_frequencies: int = 8):
        super().__init__()
        freqs = 2.0 ** torch.arange(int(num_frequencies), dtype=torch.float32)
        self.register_buffer("freqs", freqs, persistent=False)
        self.out_dim = 3 + 2 * 3 * int(num_frequencies)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        freqs = self.freqs.to(device=coords.device, dtype=coords.dtype)
        angles = coords[..., None, :] * freqs.view(*([1] * (coords.dim() - 1)), -1, 1) * math.pi
        sincos = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-2)
        return torch.cat([coords, sincos.flatten(start_dim=-2)], dim=-1)


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
        px_per_mm = (self.detector_resolution[0] - 1) / max(self.fov_mm, 1e-6)
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
            # Orthographic projection has analytic constant detector scale in px/mm. The
            # normalized tensor preserves the interface for future non-orthographic geometry.
            jac.append(torch.full_like(depth[..., 0], px_per_mm))
        grid_all = torch.stack(grids, dim=1)
        depth_all = torch.stack(depths, dim=1)
        valid_all = torch.stack(masks, dim=1)
        jac_all = torch.stack(jac, dim=1)
        boundary = (1.0 - grid_all.abs()).amin(dim=-1).clamp(0.0, 1.0)
        finite_depth = torch.nan_to_num(depth_all[..., 0], nan=0.0, posinf=0.0, neginf=0.0)
        detector_side_path_proxy = (finite_depth.abs() / max(self.camera_distance_mm, 1e-6)).clamp(
            0.0, 1.0
        )
        jac_norm = (jac_all / max(px_per_mm, 1e-6)).clamp(0.0, 1.0)
        return {
            "grid": grid_all,
            "depth": depth_all,
            "valid_mask": valid_all,
            "uv_px": torch.stack(uv_px, dim=1),
            "uv_phys": torch.stack(uv_phys, dim=1),
            "ray_directions": torch.stack(rays, dim=1),
            "jacobian_scale_px_per_mm": jac_all,
            "jacobian_scale": jac_norm,
            "boundary_distance": boundary,
            "detector_side_path_proxy": detector_side_path_proxy,
        }


class QueryDependentSurfaceSampler(nn.Module):
    """Sample K local bilinear feature footprints with query-dependent Gaussian scale."""

    def __init__(
        self,
        offsets_px: list[list[float]],
        feature_dim: int,
        hidden_dim: int = 64,
        sigma_min_px: float = 0.6,
        sigma_max_px: float = 4.0,
        alpha_xi: float = 0.35,
        alpha_beta: float = 0.35,
        alpha_kappa: float = 0.30,
        delta_h_max: float = 0.25,
        mode: str = "query_dependent",
        fixed_sigma: float = 2.0,
        align_corners: bool = True,
    ):
        super().__init__()
        offsets = torch.tensor(offsets_px, dtype=torch.float32)
        if offsets.dim() != 2 or offsets.shape[-1] != 2:
            raise ValueError("offsets_px must be a list of [du,dv] pairs")
        self.register_buffer("offsets_px", offsets, persistent=False)
        self.footprint_context_net = _mlp(feature_dim + 4, hidden_dim, 1, layers=2)
        self.sigma_min_px = float(sigma_min_px)
        self.sigma_max_px = float(sigma_max_px)
        self.alpha_xi = float(alpha_xi)
        self.alpha_beta = float(alpha_beta)
        self.alpha_kappa = float(alpha_kappa)
        self.delta_h_max = float(delta_h_max)
        self.mode = str(mode)
        self.fixed_sigma = float(fixed_sigma)
        self.align_corners = bool(align_corners)

    def forward(
        self,
        features: torch.Tensor,
        measurements: torch.Tensor,
        mapped: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        B, V, C, H, W = features.shape
        N = mapped["grid"].shape[2]
        offsets = self.offsets_px.to(device=features.device, dtype=features.dtype)
        K = offsets.shape[0]
        if self.align_corners:
            dx = 2.0 / max(W - 1, 1)
            dy = 2.0 / max(H - 1, 1)
        else:
            dx = 2.0 / max(W, 1)
            dy = 2.0 / max(H, 1)

        center_grid_flat = mapped["grid"].reshape(B * V, N, 1, 2)
        feat_flat = features.reshape(B * V, C, H, W)
        center_feat = F.grid_sample(
            feat_flat,
            center_grid_flat,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=self.align_corners,
        )
        center_feat = center_feat.squeeze(-1).transpose(1, 2).reshape(B, V, N, C)
        meas_flat = measurements.reshape(B * V, measurements.shape[2], H, W)
        center_meas = F.grid_sample(
            meas_flat,
            center_grid_flat,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=self.align_corners,
        )
        center_meas = center_meas.squeeze(-1).transpose(1, 2).reshape(B, V, N, -1)[..., :1]

        geom_context = torch.stack(
            [
                mapped["detector_side_path_proxy"],
                mapped["boundary_distance"],
                mapped["jacobian_scale"],
            ],
            dim=-1,
        )
        h0 = (
            self.alpha_xi * geom_context[..., 0]
            + self.alpha_beta * geom_context[..., 1]
            + self.alpha_kappa * geom_context[..., 2]
        ).clamp(0.0, 1.0)
        if self.mode == "fixed":
            sigma_f = torch.full_like(h0, self.fixed_sigma)
            h = ((sigma_f - self.sigma_min_px) / max(self.sigma_max_px - self.sigma_min_px, 1e-6))
            h = h.clamp(0.0, 1.0)
        else:
            ctx = torch.cat([center_feat, center_meas, geom_context], dim=-1)
            delta_h = self.delta_h_max * torch.tanh(self.footprint_context_net(ctx)).squeeze(-1)
            h = (h0 + delta_h).clamp(0.0, 1.0)
            sigma_f = self.sigma_min_px + (self.sigma_max_px - self.sigma_min_px) * h

        offsets_px = sigma_f[..., None, None] * offsets[None, None, None, :, :]
        offsets_grid = torch.stack([offsets_px[..., 0] * dx, offsets_px[..., 1] * dy], dim=-1)
        grid = mapped["grid"][:, :, :, None, :] + offsets_grid
        sample_valid = mapped["valid_mask"][:, :, :, None] & (grid.abs() <= 1.0).all(dim=-1)
        grid_flat = grid.reshape(B * V, N * K, 1, 2)
        sampled = F.grid_sample(
            feat_flat,
            grid_flat,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=self.align_corners,
        )
        sampled = sampled.squeeze(-1).transpose(1, 2).reshape(B, V, N, K, C)
        sampled = torch.where(sample_valid[..., None], sampled, torch.zeros_like(sampled))
        sampled_meas = F.grid_sample(
            meas_flat,
            grid_flat,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=self.align_corners,
        )
        sampled_meas = sampled_meas.squeeze(-1).transpose(1, 2).reshape(B, V, N, K, -1)[..., :1]
        sampled_meas = torch.where(
            sample_valid[..., None], sampled_meas, torch.zeros_like(sampled_meas)
        )
        base_A = torch.exp(-0.5 * offsets.square().sum(dim=-1)).to(sampled.dtype)
        A = base_A[None, None, None, :].expand(B, V, N, K) * sample_valid.to(sampled.dtype)
        A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        A = torch.where(sample_valid.any(dim=-1, keepdim=True), A, torch.zeros_like(A))
        return {
            "sigma_f": sigma_f,
            "h": h,
            "sample_coordinates": grid,
            "sample_coordinates_px": mapped["uv_px"][:, :, :, None, :] + offsets_px,
            "sample_valid": sample_valid,
            "A": A,
            "sample_features": sampled,
            "sample_measurements": sampled_meas,
            "offsets_template": offsets,
        }


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

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        temperature: float = 1.0,
        k_min: float = 1e-6,
        mode: str = "pre_aggregation",
        reliability_mode: str = "evidence",
    ):
        super().__init__()
        self.routing_net = _mlp(feature_dim + 11, hidden_dim, 1, layers=3)
        self.evidence_net = _mlp(feature_dim + 4, hidden_dim, 1, layers=2)
        self.temperature = float(temperature)
        self.k_min = float(k_min)
        self.mode = str(mode)
        self.reliability_mode = str(reliability_mode)

    def forward(
        self,
        samples: dict[str, torch.Tensor],
        points_mm: torch.Tensor,
        candidates: dict[str, torch.Tensor],
        mapped: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        sample_features = samples["sample_features"]
        sample_valid = samples["sample_valid"]
        A = samples["A"]
        sample_measurements = samples["sample_measurements"]
        B, V, N, K, C = sample_features.shape
        centers = candidates["candidate_centers_mm"]
        scales = candidates["candidate_scales_mm"].clamp_min(1e-6)
        cand_valid = candidates["candidate_valid_mask"]
        M = centers.shape[1]
        Mb = M + 1
        if M > 0:
            diff = points_mm[:, :, None, :] - centers[:, None, :, :]
            dist2 = diff.square().sum(dim=-1)
            p_m = torch.exp(-dist2 / (2.0 * scales[:, None, :].square().clamp_min(1e-12)))
            p_m = torch.where(cand_valid[:, None, :], p_m, torch.zeros_like(p_m))
            p0 = (1.0 - p_m.amax(dim=-1, keepdim=True)).clamp(0.0, 1.0)
        else:
            p_m = torch.zeros((B, N, 0), device=points_mm.device, dtype=points_mm.dtype)
            p0 = torch.ones((B, N, 1), device=points_mm.device, dtype=points_mm.dtype)
        p_all = torch.cat([p0, p_m], dim=-1)

        if M > 0:
            cand_uv = candidates["candidate_uv_px"]
            cand_visible = candidates["candidate_detector_valid_mask"] & cand_valid[:, None, :]
            sample_uv = samples["sample_coordinates_px"]
            duv = sample_uv[:, :, :, :, None, :] - cand_uv[:, :, None, None, :, :]
            detector_scale = candidates["candidate_detector_scales_px"].clamp_min(1e-6)
            K_m = torch.exp(
                -duv.square().sum(dim=-1)
                / (2.0 * detector_scale[:, :, None, None, :].square().clamp_min(1e-12))
            )
            K_m = torch.where(cand_visible[:, :, None, None, :], K_m, torch.zeros_like(K_m))
        else:
            K_m = torch.zeros((B, V, N, K, 0), device=points_mm.device, dtype=points_mm.dtype)
        if M > 0:
            K0 = (1.0 - K_m.amax(dim=-1, keepdim=True)).clamp(1e-6, 1.0)
        else:
            K0 = torch.ones((B, V, N, K, 1), device=points_mm.device, dtype=points_mm.dtype)
        K_all = torch.cat([K0, K_m], dim=-1)
        K_flat = K_all.clamp(self.k_min, 1.0)

        branch_valid = torch.cat(
            [torch.ones((B, 1), device=points_mm.device, dtype=torch.bool), cand_valid], dim=1
        )
        if self.mode == "post_aggregation":
            shared = (A[..., None] * sample_features).sum(dim=3, keepdim=True)
            sample_features_for_route = shared.expand_as(sample_features)
        else:
            sample_features_for_route = sample_features
        route_feat = sample_features_for_route[:, :, :, :, None, :].expand(B, V, N, K, Mb, C)
        p_feat = p_all[:, None, :, None, :, None].expand(B, V, N, K, Mb, 1)
        k_feat = K_flat[..., None]
        q = points_mm[:, None, :, None, None, :].expand(B, V, N, K, Mb, 3)
        if M > 0:
            rel = torch.zeros((B, N, Mb, 3), device=points_mm.device, dtype=points_mm.dtype)
            rel[:, :, 1:] = (
                points_mm[:, :, None, :] - centers[:, None, :, :]
            ) / scales[:, None, :, None].clamp_min(1e-6)
        else:
            rel = torch.zeros((B, N, 1, 3), device=points_mm.device, dtype=points_mm.dtype)
        rel_feat = rel[:, None, :, None, :, :].expand(B, V, N, K, Mb, 3)
        geom = torch.stack(
            [
                mapped["detector_side_path_proxy"],
                mapped["boundary_distance"],
                mapped["jacobian_scale"],
            ],
            dim=-1,
        )[:, :, :, None, None, :].expand(B, V, N, K, Mb, 3)
        logits = self.routing_net(
            torch.cat([route_feat, rel_feat, geom, p_feat, k_feat, q], dim=-1)
        ).squeeze(-1)
        logits = self.temperature * torch.log(K_flat) + torch.tanh(logits)
        valid = sample_valid[..., None] & branch_valid[:, None, None, None, :]
        logits = logits.masked_fill(~valid, -1.0e4)
        zeta = torch.softmax(logits, dim=-1)
        zeta = torch.where(valid, zeta, torch.zeros_like(zeta))
        no_route = zeta.sum(dim=-1, keepdim=True) <= 0.0
        comp = torch.zeros_like(zeta)
        comp[..., 0] = sample_valid.float()
        zeta = torch.where(no_route, comp, zeta)

        weighted_route = A[..., None] * zeta
        a = weighted_route.sum(dim=3).clamp(0.0, 1.0)
        no_samples = sample_valid.float().sum(dim=3) <= 0.0
        a = torch.where(no_samples[..., None], torch.zeros_like(a), a)
        a[..., 0] = torch.where(no_samples, torch.ones_like(a[..., 0]), a[..., 0])

        ev_in = torch.cat(
            [
                sample_features,
                sample_measurements,
                geom[:, :, :, :, 0, :],
            ],
            dim=-1,
        )
        e_sample = torch.sigmoid(self.evidence_net(ev_in)).squeeze(-1) * sample_valid.float()
        e = (A[..., None] * zeta * e_sample[..., None]).sum(dim=3).clamp(0.0, 1.0)
        nu = (e / a.clamp_min(1e-8)).clamp(0.0, 1.0)
        if self.reliability_mode == "assignment_only":
            r = a
        else:
            r = (a * nu).clamp(0.0, 1.0)
        return {
            "zeta": zeta,
            "a": a,
            "e": e.clamp(0.0, 1.0),
            "e_sample": e_sample.clamp(0.0, 1.0),
            "nu": nu,
            "r": r,
            "p_all": p_all,
            "K_all": K_all,
            "branch_valid": branch_valid,
            "relative_query": rel,
        }


class CandidateViewEncoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.representation_net = _mlp(feature_dim + 10, hidden_dim, hidden_dim, layers=2)

    def forward(
        self,
        samples: dict[str, torch.Tensor],
        router: dict[str, torch.Tensor],
        mapped: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        sample_features = samples["sample_features"]
        zeta = router["zeta"]
        A = samples["A"]
        B, V, N, K, C = sample_features.shape
        Mb = zeta.shape[-1]
        offsets = samples["offsets_template"].to(
            device=sample_features.device, dtype=sample_features.dtype
        )
        offsets_norm = offsets / offsets.abs().amax().clamp_min(1.0)
        offset_feat = offsets_norm[None, None, None, :, None, :].expand(B, V, N, K, Mb, 2)
        local = torch.cat(
            [sample_features[:, :, :, :, None, :].expand(B, V, N, K, Mb, C), offset_feat],
            dim=-1,
        )
        weighted = (A[..., None, None] * zeta[..., None] * local).sum(dim=3)
        denom = router["a"].clamp_min(1e-8)[..., None]
        pooled = weighted / denom
        geom = torch.stack(
            [
                mapped["detector_side_path_proxy"],
                mapped["boundary_distance"],
                mapped["jacobian_scale"],
            ],
            dim=-1,
        )[:, :, :, None, :].expand(B, V, N, Mb, 3)
        stats = torch.stack([router["a"], router["nu"]], dim=-1)
        rel = router["relative_query"][:, None].expand(B, V, N, Mb, 3)
        rep = self.representation_net(torch.cat([pooled, stats, rel, geom], dim=-1))
        return rep.permute(0, 2, 1, 3, 4).contiguous()


class CandidateSpecificViewFusion(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, mode: str = "candidate_specific"):
        super().__init__()
        self.cross_view_fusion = _mlp(feature_dim * 2, hidden_dim, 1, layers=2)
        self.transform = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.SiLU())
        self.mode = str(mode)

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
        valid = view_valid.permute(0, 2, 1)[:, :, :, None] & branch_valid[:, None, None, :]
        support_valid = torch.where(valid, support, torch.zeros_like(support))
        sum_support = support_valid.sum(dim=2, keepdim=True)
        sum_h = (support_valid[..., None] * h).sum(dim=2, keepdim=True)
        loo_denom = (sum_support - support_valid).clamp_min(1e-8)
        loo = (sum_h - support_valid[..., None] * h) / loo_denom[..., None]
        has_loo = (sum_support - support_valid)[..., None] > 0.0
        loo = torch.where(has_loo, loo, torch.zeros_like(loo))
        logits = torch.log(support.clamp_min(1e-8)) + torch.tanh(
            self.cross_view_fusion(torch.cat([h, loo], dim=-1)).squeeze(-1)
        )
        logits = logits.masked_fill(~valid, -1.0e4)
        weights = torch.softmax(logits, dim=2)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        denom = weights.sum(dim=2, keepdim=True).clamp_min(1e-8)
        weights = weights / denom
        no_views = valid.sum(dim=2) == 0
        z = (weights[..., None] * h).sum(dim=2)
        z = torch.where(no_views[..., None], torch.zeros_like(z), z)
        valid_view_count = view_valid.permute(0, 2, 1).sum(dim=2, keepdim=True).clamp_min(1)
        Lambda = (support_valid.sum(dim=2) / valid_view_count).clamp(0.0, 1.0)
        if self.mode == "shared":
            branch_support = Lambda * branch_valid[:, None, :].to(dtype=Lambda.dtype)
            shared_denom = branch_support.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            z_shared = (branch_support[..., None] * z).sum(dim=2, keepdim=True) / shared_denom[
                ..., None
            ]
            z = z_shared.expand_as(z)
        return z, weights, Lambda


class CandidateAssignmentHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        p_min: float = 1e-6,
        tau_cand: float = 0.1,
        tau0: float = 1.0,
        tau_p: float = 1.0,
    ):
        super().__init__()
        self.compensation_assignment_head = _mlp(feature_dim + 2, hidden_dim, 1, layers=2)
        self.candidate_assignment_head = _mlp(feature_dim + 4, hidden_dim, 1, layers=2)
        self.p_min = float(p_min)
        self.tau_cand = float(tau_cand)
        self.tau0 = float(tau0)
        self.tau_p = float(tau_p)

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
        p_all = router["p_all"]
        Lambda = router["Lambda"]
        no_valid_views = router["valid_view_count"] <= 0
        p0_flat = p_all[..., :1].clamp(self.p_min, 1.0 - self.p_min)
        comp_logits = self.compensation_assignment_head(
            torch.cat([z[:, :, :1], 1.0 - p_all[..., :1, None], Lambda[:, :, :1, None]], dim=-1)
        )
        comp_prior = torch.logit(p0_flat)
        if M == 0:
            return torch.ones((B, N, 1), dtype=z.dtype, device=z.device)
        centers = candidates["candidate_centers_mm"]
        scales = candidates["candidate_scales_mm"].clamp_min(1e-6)
        rel = (points_mm[:, :, None, :] - centers[:, None, :, :]) / scales[:, None, :, None]
        cand_active = active[:, :, 1:] & (Lambda[:, :, 1:] > self.tau_cand)
        cand_logits = self.candidate_assignment_head(
            torch.cat([z[:, :, 1:], rel, Lambda[:, :, 1:, None]], dim=-1)
        ).squeeze(-1)
        cand_score = self.tau_p * torch.log(p_all[..., 1:].clamp(self.p_min, 1.0)) + torch.tanh(
            cand_logits
        )
        cand_score = cand_score.masked_fill(~cand_active, -1.0e4)
        cand_pi = torch.softmax(cand_score, dim=-1)
        cand_pi = torch.where(cand_active, cand_pi, torch.zeros_like(cand_pi))
        has_cand = cand_active.any(dim=-1, keepdim=True) & ~no_valid_views[..., None]
        pi0 = torch.sigmoid(self.tau0 * comp_prior + torch.tanh(comp_logits.squeeze(-1)))
        pi0 = torch.where(has_cand, pi0, torch.ones_like(pi0))
        cand_pi = torch.where(has_cand, (1.0 - pi0) * cand_pi, torch.zeros_like(cand_pi))
        pi = torch.cat([pi0, cand_pi], dim=-1)
        return pi / pi.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class CompensationDensityDecoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, pos_dim: int):
        super().__init__()
        self.net = _mlp(feature_dim + pos_dim, hidden_dim, 1, layers=3)

    def forward(self, z0: torch.Tensor, encoded_points: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([z0, encoded_points], dim=-1)))


class CandidateDensityDecoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, pos_dim: int):
        super().__init__()
        self.net = _mlp(feature_dim + pos_dim + 1, hidden_dim, 1, layers=3)

    def forward(
        self,
        z: torch.Tensor,
        encoded_relative: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        B, N, M, C = z.shape
        scale_feat = scales[:, None, :, None].expand(B, N, M, 1)
        return torch.sigmoid(self.net(torch.cat([z, encoded_relative, scale_feat], dim=-1)))


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
        self.trunk_size_mm = tuple(
            float(x) for x in _cfg_get(ssq, "candidates.trunk_size_mm", [38.0, 40.0, 20.8])
        )
        self.position_encoding = FourierPositionEncoding(
            num_frequencies=int(_cfg_get(ssq, "fourier.num_frequencies", 8))
        )
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
        footprint_cfg = ssq.get("footprint", {})
        self.surface_sampler = QueryDependentSurfaceSampler(
            offsets,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            sigma_min_px=float(
                footprint_cfg.get("scale_min_px", footprint_cfg.get("scale_min_mm", 0.6))
            ),
            sigma_max_px=float(
                footprint_cfg.get("scale_max_px", footprint_cfg.get("scale_max_mm", 4.0))
            ),
            alpha_xi=float(footprint_cfg.get("alpha_xi", 0.35)),
            alpha_beta=float(footprint_cfg.get("alpha_beta", 0.35)),
            alpha_kappa=float(footprint_cfg.get("alpha_kappa", 0.30)),
            delta_h_max=float(footprint_cfg.get("delta_h_max", 0.25)),
            mode=str(footprint_cfg.get("mode", "query_dependent")),
            fixed_sigma=float(footprint_cfg.get("fixed_sigma", 2.0)),
        )
        cand = ssq.get("candidates", {})
        self._candidate_cfg = dict(cand)
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
            k_min=float(_cfg_get(ssq, "routing.k_min", 1e-6)),
            mode=str(_cfg_get(ssq, "routing.mode", "pre_aggregation")),
            reliability_mode=str(_cfg_get(ssq, "reliability.mode", "evidence")),
        )
        self.view_encoder = CandidateViewEncoder(feature_dim, hidden_dim)
        self.view_fusion = CandidateSpecificViewFusion(
            hidden_dim, hidden_dim, mode=str(_cfg_get(ssq, "fusion.mode", "candidate_specific"))
        )
        self.assignment_head = CandidateAssignmentHead(
            hidden_dim,
            hidden_dim,
            p_min=float(_cfg_get(ssq, "routing.p_min", 1e-6)),
            tau_cand=float(_cfg_get(ssq, "routing.tau_cand", 0.1)),
            tau0=float(_cfg_get(ssq, "routing.tau0", 1.0)),
            tau_p=float(_cfg_get(ssq, "routing.tau_p", 1.0)),
        )
        self.compensation_density_decoder = CompensationDensityDecoder(
            hidden_dim, hidden_dim, self.position_encoding.out_dim
        )
        self.candidate_density_decoder = CandidateDensityDecoder(
            hidden_dim, hidden_dim, self.position_encoding.out_dim
        )
        self.morphology_sdf_head = MorphologySDFHead(hidden_dim, hidden_dim)
        self.lambda_sdf = float(
            _cfg_get(root, "loss.lambda_sdf", _cfg_get(ssq, "sdf.lambda_sdf", 0.0))
        )
        self.candidate_prior_weight = float(_cfg_get(ssq, "candidates.prior_density_weight", 0.0))
        self.density_output_mode = str(
            _cfg_get(ssq, "density_output_mode", "candidate_scalar_composition")
        )
        allowed_modes = {"candidate_scalar_composition", "e15_backbone_baseline"}
        if self.density_output_mode not in allowed_modes:
            raise ValueError(
                f"Unsupported ssq_fmt.density_output_mode={self.density_output_mode!r}; "
                f"allowed={sorted(allowed_modes)}"
            )
        self.use_query_density_backbone = bool(
            _cfg_get(ssq, "query_density_backbone.enabled", False)
        )
        self.query_density_backbone_normalization = str(
            _cfg_get(ssq, "query_density_backbone.normalization", "sample_percentile")
        )
        self.query_density_backbone = None
        if self.use_query_density_backbone and self.density_output_mode == "e15_backbone_baseline":
            from .gisc_multisource import GISCFMT

            self.query_density_backbone = GISCFMT(config=config)
        elif self.use_query_density_backbone:
            print(
                "[SSQ-FMT] query_density_backbone.enabled is ignored for "
                "candidate_scalar_composition; final density uses sum_m pi_m d_m."
            )

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
            mask = mask.to(device=query_coordinates_mm.device, dtype=torch.float32)
            Bm, Vm, Hm, Wm = mask.shape
            center_mask = F.grid_sample(
                mask.reshape(Bm * Vm, 1, Hm, Wm),
                mapped["grid"].reshape(Bm * Vm, -1, 1, 2),
                mode="nearest",
                padding_mode="zeros",
                align_corners=True,
            )
            center_mask = center_mask.squeeze(1).squeeze(-1).reshape(Bm, Vm, -1) > 0.5
            mapped["valid_mask"] = mapped["valid_mask"] & center_mask
        samples = self.surface_sampler(features, y_norm, mapped)
        # Depth maps are the currently available detector-side sample-surface proxy for
        # normalized path length. Exact mesh intersections can replace this hook later.
        candidates = self.candidate_builder(y_norm, batch=batch)
        if candidates["candidate_centers_mm"].shape[1] > 0:
            cand_mapped = self.geometry_mapper(candidates["candidate_centers_mm"])
            candidates["candidate_uv_px"] = cand_mapped["uv_px"]
            candidates["candidate_detector_valid_mask"] = cand_mapped["valid_mask"]
            scale_px = candidates["candidate_scales_mm"][:, None, :] * cand_mapped[
                "jacobian_scale_px_per_mm"
            ]
            cand_cfg = _to_container(getattr(self, "_candidate_cfg", {}))
            s_min = float(cand_cfg.get("detector_scale_min_px", 0.5))
            s_max = float(cand_cfg.get("detector_scale_max_px", 12.0))
            candidates["candidate_detector_scales_px"] = scale_px.clamp(s_min, s_max)
        else:
            B = query_coordinates_mm.shape[0]
            V = mapped["grid"].shape[1]
            candidates["candidate_uv_px"] = torch.zeros(
                (B, V, 0, 2), device=query_coordinates_mm.device, dtype=query_coordinates_mm.dtype
            )
            candidates["candidate_detector_valid_mask"] = torch.zeros(
                (B, V, 0), device=query_coordinates_mm.device, dtype=torch.bool
            )
            candidates["candidate_detector_scales_px"] = torch.zeros(
                (B, V, 0), device=query_coordinates_mm.device, dtype=query_coordinates_mm.dtype
            )
        router = self.candidate_router(
            samples, query_coordinates_mm, candidates, mapped
        )
        per_view = self.view_encoder(samples, router, mapped)
        z, view_weights, Lambda = self.view_fusion(
            per_view,
            router["r"],
            mapped["valid_mask"],
            router["branch_valid"],
        )
        valid_view_count = mapped["valid_mask"].sum(dim=1)
        router["Lambda"] = Lambda
        router["valid_view_count"] = valid_view_count
        router["active_branch_mask"] = (Lambda > self.assignment_head.tau_cand) & router[
            "branch_valid"
        ][:, None, :]
        pi = self.assignment_head(z, query_coordinates_mm, router, candidates)
        trunk = torch.tensor(
            self.trunk_size_mm, device=query_coordinates_mm.device, dtype=query_coordinates_mm.dtype
        )
        query_norm = (query_coordinates_mm / trunk.clamp_min(1e-6)).mul(2.0).sub(1.0)
        encoded_query = self.position_encoding(query_norm)
        d0 = self.compensation_density_decoder(z[:, :, 0], encoded_query)
        M = z.shape[2] - 1
        if M > 0:
            rel = (
                query_coordinates_mm[:, :, None, :] - candidates["candidate_centers_mm"][:, None]
            ) / candidates["candidate_scales_mm"][:, None, :, None].clamp_min(1e-6)
            encoded_rel = self.position_encoding(rel.clamp(-4.0, 4.0))
            dm = self.candidate_density_decoder(
                z[:, :, 1:],
                encoded_rel,
                candidates["candidate_scales_mm"],
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
        if (
            self.density_output_mode == "e15_backbone_baseline"
            and self.query_density_backbone is not None
            and batch is not None
            and "points" in batch
        ):
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
        if (
            self.density_output_mode == "e15_backbone_baseline"
            and M > 0
            and self.candidate_prior_weight > 0.0
        ):
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
            "density_output_mode": torch.tensor(
                0 if self.density_output_mode == "candidate_scalar_composition" else 1,
                device=density.device,
            ),
        }
        if backbone_logits is not None:
            aux_outputs["density_logits"] = backbone_logits
        if self.training and self.lambda_sdf > 0.0:
            raise RuntimeError(
                "SSQ-FMT SDF is disabled in this final-method pass; set lambda_sdf=0."
            )
        diagnostics = {
            "normalization_scale": norm_scale,
            "sigma_f": samples["sigma_f"],
            "sample_coordinates": samples["sample_coordinates"],
            "sample_coordinates_px": samples["sample_coordinates_px"],
            "sample_valid": samples["sample_valid"],
            "A": samples["A"],
            "sample_features": samples["sample_features"],
            "sample_measurements": samples["sample_measurements"],
            "candidate_centers_mm": candidates["candidate_centers_mm"],
            "candidate_scores": candidates["candidate_scores"],
            "candidate_scales_mm": candidates["candidate_scales_mm"],
            "candidate_valid_mask": candidates["candidate_valid_mask"],
            "K_all": router["K_all"],
            "zeta": router["zeta"],
            "a": router["a"],
            "e": router["e"],
            "e_sample": router["e_sample"],
            "nu": router["nu"],
            "r": router["r"],
            "view_weights": view_weights,
            "Lambda": Lambda,
            "active_branch_mask": router["active_branch_mask"],
            "pi": pi,
            "branch_density": branch_density,
            "branch_contributions": branch_contributions,
            "candidate_prior_density": candidate_prior,
            "decoded_density": decoded_density,
            "lightweight_density": lightweight_density,
            "density": density,
        }
        out: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "density": density,
            "aux_outputs": aux_outputs,
        }
        if return_diagnostics:
            out["diagnostics"] = diagnostics
        return out
