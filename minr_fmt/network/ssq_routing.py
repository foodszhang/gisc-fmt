"""Pre-aggregation candidate-normalized routing and evidence scoring."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from minr_fmt.network.ssq_diagnostics import (
    make_residual_mlp,
    masked_softmax,
    zero_init_last_linear,
)


class CandidateSurfaceRouter(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        tau_K: float = 1.0,
        delta_q: float = 1.0,
        k_min: float = 1e-6,
        mode: str = "pre_aggregation",
        reliability_mode: str = "evidence",
        query_chunk_size: int = 4096,
    ):
        super().__init__()
        self.routing_net = make_residual_mlp(feature_dim + 10, hidden_dim, 1, blocks=2)
        self.evidence_net = make_residual_mlp(feature_dim + 3, hidden_dim, 1, blocks=1)
        zero_init_last_linear(self.routing_net)
        zero_init_last_linear(self.evidence_net)
        self.tau_K = float(tau_K)
        self.delta_q = float(delta_q)
        self.k_min = float(k_min)
        self.mode = str(mode)
        self.reliability_mode = str(reliability_mode)
        self.query_chunk_size = int(query_chunk_size)

    def forward(
        self,
        samples: dict[str, torch.Tensor],
        points_mm: torch.Tensor,
        candidates: dict[str, torch.Tensor],
        mapped: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        n = points_mm.shape[1]
        chunk = self.query_chunk_size
        if chunk > 0 and n > chunk:
            parts = []
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                parts.append(
                    self._forward_impl(
                        _slice_samples(samples, start, end),
                        points_mm[:, start:end],
                        candidates,
                        _slice_mapped(mapped, start, end),
                    )
                )
            return _cat_router_parts(parts)
        return self._forward_impl(samples, points_mm, candidates, mapped)

    def _forward_impl(
        self,
        samples: dict[str, torch.Tensor],
        points_mm: torch.Tensor,
        candidates: dict[str, torch.Tensor],
        mapped: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        sample_features = samples["sample_features"]
        sample_valid = samples["sample_valid"]
        a_kernel = samples["A"]
        sample_measurements = samples["sample_measurements"]
        b, v, n, k, c = sample_features.shape
        centers = candidates["candidate_centers_mm"]
        scales = candidates["candidate_support_scales_mm"].clamp_min(1e-6)
        cand_valid = candidates["candidate_valid_mask"]
        m = centers.shape[1]
        mb = m + 1

        if m > 0:
            rel_query = torch.zeros((b, n, mb, 3), device=points_mm.device, dtype=points_mm.dtype)
            rel_query[:, :, 1:] = (points_mm[:, :, None] - centers[:, None]) / scales[
                :, None, :, None
            ]
            dist2 = (points_mm[:, :, None] - centers[:, None]).square().sum(dim=-1)
            p_m = torch.exp(-dist2 / (2.0 * scales[:, None].square().clamp_min(1e-12)))
            p_m = torch.where(cand_valid[:, None], p_m, torch.zeros_like(p_m))
            p0 = (1.0 - p_m.amax(dim=-1, keepdim=True)).clamp(0.0, 1.0)
            cand_uv = candidates["candidate_uv_px"]
            cand_visible = candidates["candidate_detector_valid_mask"] & cand_valid[:, None]
            duv = samples["sample_coordinates_px"][:, :, :, :, None] - cand_uv[:, :, None, None]
            s_px = candidates["candidate_detector_support_scales_px"].clamp_min(1e-6)
            k_m = torch.exp(-duv.square().sum(dim=-1) / (2.0 * s_px[:, :, None, None].square()))
            k_m = torch.where(cand_visible[:, :, None, None], k_m, torch.zeros_like(k_m))
        else:
            rel_query = torch.zeros((b, n, 1, 3), device=points_mm.device, dtype=points_mm.dtype)
            p_m = torch.zeros((b, n, 0), device=points_mm.device, dtype=points_mm.dtype)
            p0 = torch.ones((b, n, 1), device=points_mm.device, dtype=points_mm.dtype)
            k_m = torch.zeros((b, v, n, k, 0), device=points_mm.device, dtype=points_mm.dtype)

        p_all = torch.cat([p0, p_m], dim=-1)
        if m > 0:
            k0 = (1.0 - k_m.amax(dim=-1, keepdim=True)).clamp(self.k_min, 1.0)
        else:
            k0 = torch.ones((b, v, n, k, 1), device=points_mm.device, dtype=points_mm.dtype)
        k_all = torch.cat([k0, k_m], dim=-1).clamp(self.k_min, 1.0)
        branch_valid = torch.cat(
            [torch.ones((b, 1), device=points_mm.device, dtype=torch.bool), cand_valid], dim=1
        )
        route_features = sample_features
        if self.mode == "post_aggregation":
            shared = (a_kernel[..., None] * sample_features).sum(dim=3, keepdim=True)
            route_features = shared.expand_as(sample_features)
        geom = torch.stack(
            [
                mapped["detector_side_path_proxy"],
                mapped["boundary_distance"],
            ],
            dim=-1,
        )
        route_feat = route_features[:, :, :, :, None].expand(b, v, n, k, mb, c)
        rel_feat = rel_query[:, None, :, None].expand(b, v, n, k, mb, 3)
        geom_feat = geom[:, :, :, None, None].expand(b, v, n, k, mb, 2)
        p_feat = p_all[:, None, :, None, :, None].expand(b, v, n, k, mb, 1)
        q_feat = points_mm[:, None, :, None, None].expand(b, v, n, k, mb, 3)
        correction_in = torch.cat(
            [route_feat, rel_feat, geom_feat, p_feat, k_all[..., None], q_feat], dim=-1
        )
        if self.training and correction_in.requires_grad:
            correction_raw = checkpoint(self.routing_net, correction_in, use_reentrant=False)
        else:
            correction_raw = self.routing_net(correction_in)
        correction = torch.tanh(correction_raw).squeeze(-1)
        logits = self.tau_K * torch.log(k_all) + self.delta_q * correction
        valid = sample_valid[..., None] & branch_valid[:, None, None, None]
        zeta = masked_softmax(logits, valid, dim=-1, fallback_index=0)
        zeta = torch.where(sample_valid[..., None], zeta, torch.zeros_like(zeta))

        weighted_route = a_kernel[..., None] * zeta
        a = weighted_route.sum(dim=3).clamp(0.0, 1.0)
        no_samples = ~sample_valid.any(dim=3)
        a = torch.where(no_samples[..., None], torch.zeros_like(a), a)
        a[..., 0] = torch.where(no_samples, torch.ones_like(a[..., 0]), a[..., 0])
        ev_in = torch.cat(
            [sample_features, sample_measurements, geom[:, :, :, None].expand(b, v, n, k, 2)],
            dim=-1,
        )
        if self.training and ev_in.requires_grad:
            ev_raw = checkpoint(self.evidence_net, ev_in, use_reentrant=False)
        else:
            ev_raw = self.evidence_net(ev_in)
        e_sample = torch.sigmoid(ev_raw).squeeze(-1) * sample_valid.float()
        e = (a_kernel[..., None] * zeta * e_sample[..., None]).sum(dim=3).clamp(0.0, 1.0)
        nu = (e / a.clamp_min(1e-8)).clamp(0.0, 1.0)
        r = a if self.reliability_mode == "assignment_only" else (a * nu).clamp(0.0, 1.0)
        return {
            "zeta": zeta,
            "a": a,
            "e": e,
            "e_sample": e_sample.clamp(0.0, 1.0),
            "nu": nu,
            "r": r,
            "p_all": p_all,
            "K_all": k_all,
            "branch_valid": branch_valid,
            "relative_query": rel_query,
        }


def _slice_samples(
    samples: dict[str, torch.Tensor], start: int, end: int
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in samples.items():
        if torch.is_tensor(value) and value.dim() >= 3 and value.shape[2] >= end:
            out[key] = value[:, :, start:end]
        else:
            out[key] = value
    return out


def _slice_mapped(mapped: dict[str, torch.Tensor], start: int, end: int) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in mapped.items():
        if not torch.is_tensor(value):
            out[key] = value
        elif value.dim() >= 3 and value.shape[2] >= end:
            out[key] = value[:, :, start:end]
        elif value.dim() >= 2 and value.shape[1] >= end:
            out[key] = value[:, start:end]
        else:
            out[key] = value
    return out


def _cat_router_parts(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    query_dim_by_key = {
        "zeta": 2,
        "a": 2,
        "e": 2,
        "e_sample": 2,
        "nu": 2,
        "r": 2,
        "p_all": 1,
        "K_all": 2,
        "relative_query": 1,
    }
    for key, value in parts[0].items():
        dim = query_dim_by_key.get(key)
        out[key] = torch.cat([part[key] for part in parts], dim=dim) if dim is not None else value
    return out
