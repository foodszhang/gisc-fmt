"""Pre-aggregation candidate-normalized routing and evidence scoring."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from minr_fmt.network.ssq_diagnostics import (
    masked_softmax,
    zero_init_last_linear,
)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.net(self.norm(inputs))


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
        checkpoint_routing: bool = False,
        measurement_consistency_temperature: float = 0.15,
        compensation_routing_mode: str = "joint",
        sample_embedding_dim: int = 64,
        candidate_embedding_dim: int = 16,
        query_coordinate_scale_mm: float = 40.0,
    ):
        super().__init__()
        del hidden_dim
        sample_dim = int(sample_embedding_dim)
        candidate_dim = int(candidate_embedding_dim)
        self.sample_encoder = nn.Sequential(
            nn.Linear(feature_dim + 5, sample_dim),
            nn.SiLU(),
            ResidualMLPBlock(sample_dim),
            nn.LayerNorm(sample_dim),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(5, candidate_dim),
            nn.SiLU(),
            nn.Linear(candidate_dim, candidate_dim),
            nn.SiLU(),
        )
        self.candidate_projection = nn.Linear(candidate_dim, sample_dim)
        self.routing_head = nn.Sequential(
            nn.Linear(sample_dim * 2 + candidate_dim, 64), nn.SiLU(), nn.Linear(64, 1)
        )
        self.evidence_head = nn.Sequential(nn.Linear(sample_dim, 32), nn.SiLU(), nn.Linear(32, 1))
        zero_init_last_linear(self.routing_head)
        zero_init_last_linear(self.evidence_head)
        self.tau_K = float(tau_K)
        self.delta_q = float(delta_q)
        self.k_min = float(k_min)
        self.mode = str(mode)
        self.reliability_mode = str(reliability_mode)
        self.query_chunk_size = int(query_chunk_size)
        self.checkpoint_routing = bool(checkpoint_routing)
        self.measurement_consistency_temperature = float(measurement_consistency_temperature)
        self.query_coordinate_scale_mm = float(query_coordinate_scale_mm)
        if compensation_routing_mode not in {"joint", "independent", "proposal_only"}:
            raise ValueError(
                "compensation_routing_mode must be 'joint', 'independent', or 'proposal_only'"
            )
        self.compensation_routing_mode = str(compensation_routing_mode)

    @property
    def routing_net(self) -> nn.Module:
        """Compatibility alias for legacy diagnostics."""
        return self.routing_head

    @property
    def evidence_net(self) -> nn.Module:
        """Compatibility alias for legacy diagnostics."""
        return self.evidence_head

    def forward(
        self,
        samples: dict[str, torch.Tensor],
        points_mm: torch.Tensor,
        candidates: dict[str, torch.Tensor],
        mapped: dict[str, torch.Tensor],
        return_diagnostics: bool = False,
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
                        return_diagnostics=return_diagnostics,
                    )
                )
            return _cat_router_parts(parts)
        return self._forward_impl(
            samples,
            points_mm,
            candidates,
            mapped,
            return_diagnostics=return_diagnostics,
        )

    def _forward_impl(
        self,
        samples: dict[str, torch.Tensor],
        points_mm: torch.Tensor,
        candidates: dict[str, torch.Tensor],
        mapped: dict[str, torch.Tensor],
        return_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        sample_features = samples["sample_features"]
        sample_valid = samples["sample_valid"]
        a_kernel = samples["A"]
        sample_measurements = samples["sample_measurements"]
        b, v, n, k, c = sample_features.shape
        centers = candidates["candidate_centers_mm"]
        inverse_sqrt = candidates["candidate_support_inverse_sqrt_mm"]
        cand_valid = candidates["candidate_valid_mask"]
        m = centers.shape[1]
        mb = m + 1

        if m > 0:
            rel_query = torch.zeros((b, n, mb, 3), device=points_mm.device, dtype=points_mm.dtype)
            delta = points_mm[:, :, None] - centers[:, None]
            relative = torch.einsum("bnmi,bmij->bnmj", delta, inverse_sqrt)
            rel_query[:, :, 1:] = relative
            p_m = torch.exp(-0.5 * relative.square().sum(dim=-1))
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
        k_all = torch.cat([k0, k_m], dim=-1).clamp_max(1.0)
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
        geom_feat = geom[:, :, :, None].expand(b, v, n, k, 2)
        q_feat = (points_mm / max(self.query_coordinate_scale_mm, 1.0e-6)).clamp(-1.0, 1.0)
        q_feat = q_feat[:, None, :, None].expand(b, v, n, k, 3)
        sample_input = torch.cat([route_features, geom_feat, q_feat], dim=-1)
        if self.checkpoint_routing and self.training and sample_input.requires_grad:
            sample_embedding = checkpoint(self.sample_encoder, sample_input, use_reentrant=False)
        else:
            sample_embedding = self.sample_encoder(sample_input)

        correction = sample_embedding.new_zeros((b, v, n, k, mb))
        candidate_logits = sample_embedding.new_zeros((b, v, n, k, m))
        if m > 0:
            rel_feat = relative[:, None, :, None].expand(b, v, n, k, m, 3)
            p_feat = p_m[:, None, :, None, :, None].expand(b, v, n, k, m, 1)
            candidate_input = torch.cat([rel_feat, p_feat, k_m[..., None]], dim=-1)
            candidate_embedding = self.candidate_encoder(candidate_input)
            sample_expanded = sample_embedding.unsqueeze(-2).expand(b, v, n, k, m, -1)
            projected = self.candidate_projection(candidate_embedding)
            routing_input = torch.cat(
                [sample_expanded, candidate_embedding, sample_expanded * projected], dim=-1
            )
            candidate_logits = self.routing_head(routing_input).squeeze(-1)
            correction[..., 1:] = torch.tanh(candidate_logits)
        logits = self.tau_K * torch.log(k_all.clamp_min(self.k_min)) + self.delta_q * correction
        candidate_view_valid = (
            cand_visible[:, :, None, None, :] if m > 0 else sample_valid[..., None, :0]
        )
        valid = sample_valid[..., None] & branch_valid[:, None, None, None]
        if m > 0:
            valid[..., 1:] = valid[..., 1:] & candidate_view_valid
        if self.compensation_routing_mode in {"independent", "proposal_only"} and m > 0:
            compensation_zeta = sample_valid[..., None].to(dtype=logits.dtype)
            candidate_zeta = masked_softmax(logits[..., 1:], valid[..., 1:], dim=-1)
            zeta = torch.cat([compensation_zeta, candidate_zeta], dim=-1)
        else:
            zeta = masked_softmax(logits, valid, dim=-1, fallback_index=0)
        zeta = torch.where(sample_valid[..., None], zeta, torch.zeros_like(zeta))

        weighted_route = a_kernel[..., None] * zeta
        a = weighted_route.sum(dim=3).clamp(0.0, 1.0)
        no_samples = ~sample_valid.any(dim=3)
        a = torch.where(no_samples[..., None], torch.zeros_like(a), a)
        a[..., 0] = torch.where(no_samples, torch.ones_like(a[..., 0]), a[..., 0])
        ev_raw = self.evidence_head(sample_embedding)
        e_sample = torch.sigmoid(ev_raw).squeeze(-1) * sample_valid.float()
        e = (a_kernel[..., None] * zeta * e_sample[..., None]).sum(dim=3).clamp(0.0, 1.0)
        nu = (e / a.clamp_min(1e-8)).clamp(0.0, 1.0)
        r = a if self.reliability_mode == "assignment_only" else (a * nu).clamp(0.0, 1.0)
        measurement_consistency = torch.ones_like(p_m)
        center_measurements = candidates.get("candidate_center_measurements")
        if m > 0 and torch.is_tensor(center_measurements):
            query_measurements = (a_kernel * sample_measurements.squeeze(-1)).sum(dim=3)
            difference = (query_measurements[..., None] - center_measurements[:, :, None]).abs()
            similarity = torch.exp(
                -difference / max(self.measurement_consistency_temperature, 1.0e-6)
            )
            pair_valid = mapped["valid_mask"][..., None] & cand_visible[:, :, None]
            pair_weight = pair_valid.to(dtype=similarity.dtype)
            measurement_consistency = (similarity * pair_weight).sum(dim=1)
            measurement_consistency = measurement_consistency / pair_weight.sum(dim=1).clamp_min(
                1.0
            )
        out = {
            "zeta": zeta,
            "a": a,
            "e": e,
            "nu": nu,
            "r": r,
            "p_all": p_all,
            "branch_valid": branch_valid,
            "relative_query": rel_query,
            "measurement_consistency": measurement_consistency,
        }
        if return_diagnostics:
            out["e_sample"] = e_sample.clamp(0.0, 1.0)
            out["K_all"] = k_all
        return out


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
        "relative_query": 1,
        "measurement_consistency": 1,
        "K_all": 2,
    }
    for key, value in parts[0].items():
        dim = query_dim_by_key.get(key)
        out[key] = torch.cat([part[key] for part in parts], dim=dim) if dim is not None else value
    return out
