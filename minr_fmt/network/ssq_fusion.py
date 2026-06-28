"""Per-view source-hypothesis representations, quotient fusion, and legacy assignment."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from minr_fmt.network.ssq_diagnostics import (
    make_residual_mlp,
    masked_softmax,
    zero_init_last_linear,
)


class CandidateViewEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        blocks: int = 3,
        dropout: float = 0.0,
        query_chunk_size: int = 4096,
        checkpoint_representation: bool = False,
    ):
        super().__init__()
        self.representation_net = make_residual_mlp(
            feature_dim + 9,
            hidden_dim,
            hidden_dim,
            blocks=blocks,
            dropout=dropout,
        )
        self.query_chunk_size = int(query_chunk_size)
        self.checkpoint_representation = bool(checkpoint_representation)

    def forward(
        self,
        samples: dict[str, torch.Tensor],
        router: dict[str, torch.Tensor],
        mapped: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        n = samples["sample_features"].shape[2]
        chunk = self.query_chunk_size
        if chunk > 0 and n > chunk:
            parts = []
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                parts.append(
                    self._forward_impl(
                        _slice_samples(samples, start, end),
                        _slice_router(router, start, end),
                        _slice_mapped(mapped, start, end),
                    )
                )
            return torch.cat(parts, dim=1)
        return self._forward_impl(samples, router, mapped)

    def _forward_impl(
        self,
        samples: dict[str, torch.Tensor],
        router: dict[str, torch.Tensor],
        mapped: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        f = samples["sample_features"]
        zeta = router["zeta"]
        a_kernel = samples["A"]
        b, v, n, k, c = f.shape
        mb = zeta.shape[-1]
        offsets = samples["offsets_template"].to(device=f.device, dtype=f.dtype)
        offsets = offsets / offsets.abs().amax().clamp_min(1.0)
        assignment = a_kernel[..., None] * zeta
        pooled_features = torch.einsum("bvnkm,bvnkc->bvnmc", assignment, f)
        pooled_offsets = torch.einsum("bvnkm,kd->bvnmd", assignment, offsets)
        pooled = torch.cat([pooled_features, pooled_offsets], dim=-1)
        pooled = pooled / router["a"].clamp_min(1e-8)[..., None]
        geom = torch.stack(
            [
                mapped["detector_side_path_proxy"],
                mapped["boundary_distance"],
            ],
            dim=-1,
        )[:, :, :, None].expand(b, v, n, mb, 2)
        stats = torch.stack([router["a"], router["nu"]], dim=-1)
        rel = router["relative_query"][:, None].expand(b, v, n, mb, 3)
        rep_in = torch.cat([pooled, stats, rel, geom], dim=-1)
        if self.checkpoint_representation and self.training and rep_in.requires_grad:
            rep = checkpoint(self.representation_net, rep_in, use_reentrant=False)
        else:
            rep = self.representation_net(rep_in)
        return rep.permute(0, 2, 1, 3, 4).contiguous()


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


def _slice_router(router: dict[str, torch.Tensor], start: int, end: int) -> dict[str, torch.Tensor]:
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
    for key, value in router.items():
        dim = query_dim_by_key.get(key)
        if torch.is_tensor(value) and dim is not None and value.shape[dim] >= end:
            slices = [slice(None)] * value.dim()
            slices[dim] = slice(start, end)
            out[key] = value[tuple(slices)]
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


class CandidateSpecificViewFusion(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        mode: str = "candidate_specific",
        query_chunk_size: int = 4096,
        checkpoint_fusion: bool = False,
    ):
        super().__init__()
        self.transform = make_residual_mlp(
            feature_dim, hidden_dim, feature_dim, blocks=2, final_activation=True
        )
        self.cross_view_fusion = make_residual_mlp(feature_dim * 4, hidden_dim, 1, blocks=1)
        zero_init_last_linear(self.cross_view_fusion)
        self.mode = str(mode)
        self.query_chunk_size = int(query_chunk_size)
        self.checkpoint_fusion = bool(checkpoint_fusion)

    def forward(
        self,
        per_view: torch.Tensor,
        view_support: torch.Tensor,
        view_valid: torch.Tensor,
        branch_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = per_view.shape[1]
        chunk = self.query_chunk_size
        if chunk > 0 and n > chunk:
            z_parts = []
            weight_parts = []
            lam_parts = []
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                z_i, w_i, lam_i = self._forward_impl(
                    per_view[:, start:end],
                    view_support[:, :, start:end],
                    view_valid[:, :, start:end],
                    branch_valid,
                )
                z_parts.append(z_i)
                weight_parts.append(w_i)
                lam_parts.append(lam_i)
            return (
                torch.cat(z_parts, dim=1),
                torch.cat(weight_parts, dim=1),
                torch.cat(lam_parts, dim=1),
            )
        return self._forward_impl(per_view, view_support, view_valid, branch_valid)

    def _forward_impl(
        self,
        per_view: torch.Tensor,
        view_support: torch.Tensor,
        view_valid: torch.Tensor,
        branch_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, v, mb, c = per_view.shape
        if self.checkpoint_fusion and self.training and per_view.requires_grad:
            h = checkpoint(self.transform, per_view, use_reentrant=False)
        else:
            h = self.transform(per_view)
        support = view_support.permute(0, 2, 1, 3).contiguous()
        valid = view_valid.permute(0, 2, 1)[:, :, :, None] & branch_valid[:, None, None]
        support_valid = torch.where(valid, support, torch.zeros_like(support))
        sum_support = support_valid.sum(dim=2, keepdim=True)
        sum_h = (support_valid[..., None] * h).sum(dim=2, keepdim=True)
        loo_denom = (sum_support - support_valid).clamp_min(1e-8)
        loo = (sum_h - support_valid[..., None] * h) / loo_denom[..., None]
        loo = torch.where(
            (sum_support - support_valid)[..., None] > 0.0, loo, torch.zeros_like(loo)
        )
        correction_in = torch.cat([h, loo, h - loo, h * loo], dim=-1)
        if self.checkpoint_fusion and self.training and correction_in.requires_grad:
            correction = checkpoint(self.cross_view_fusion, correction_in, use_reentrant=False)
        else:
            correction = self.cross_view_fusion(correction_in)
        logits = torch.log(support.clamp_min(1e-8)) + torch.tanh(correction.squeeze(-1))
        weights = masked_softmax(logits, valid, dim=2)
        z = (weights[..., None] * h).sum(dim=2)
        no_views = ~valid.any(dim=2)
        z = torch.where(no_views[..., None], torch.zeros_like(z), z)
        valid_view_count = view_valid.permute(0, 2, 1).sum(dim=2, keepdim=True).clamp_min(1)
        lam = (support_valid.sum(dim=2) / valid_view_count).clamp(0.0, 1.0)
        if self.mode == "shared":
            branch_support = lam * branch_valid[:, None].to(dtype=lam.dtype)
            denom = branch_support.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            z_shared = (branch_support[..., None] * z).sum(dim=2, keepdim=True) / denom[..., None]
            z = z_shared.expand_as(z)
        return z, weights, lam


class SourceHypothesisQuotientAggregator(nn.Module):
    """Physics-anchored reliability quotient over the view equivalence class."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        temperature: float = 1.0,
        delta_r_max: float = 0.25,
        a_beta: float = 1.0,
        a_xi: float = 1.0,
        a_sigma: float = 0.5,
        a_center: float = 0.25,
        eps: float = 1.0e-8,
    ):
        super().__init__()
        self.reliability_residual = make_residual_mlp(
            feature_dim + 4, hidden_dim, 1, blocks=1
        )
        zero_init_last_linear(self.reliability_residual)
        self.temperature = float(temperature)
        self.delta_r_max = float(delta_r_max)
        self.a_beta = float(a_beta)
        self.a_xi = float(a_xi)
        self.a_sigma = float(a_sigma)
        self.a_center = float(a_center)
        self.eps = float(eps)

    def forward(
        self,
        per_view_evidence: torch.Tensor,
        geometry_features: torch.Tensor,
        routed_support: torch.Tensor,
        view_valid: torch.Tensor,
        proposal_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Aggregate ``[B,N,V,M,C]`` evidence into a quotient representative."""
        if per_view_evidence.dim() != 5:
            raise ValueError("per_view_evidence must have shape [B,N,V,M,C]")
        if geometry_features.shape[-1] != 4:
            raise ValueError("geometry_features must contain [beta, xi, sigma, center_distance]")
        beta, xi, sigma, center = geometry_features.unbind(dim=-1)
        log_r_phy = (
            torch.log(routed_support.clamp_min(self.eps))
            + self.a_beta * beta[..., None]
            - self.a_xi * xi[..., None]
            - self.a_sigma * sigma[..., None]
            - self.a_center * center[..., None]
        )
        geom = geometry_features[..., None, :].expand(*per_view_evidence.shape[:-1], 4)
        learned = self.delta_r_max * torch.tanh(
            self.reliability_residual(torch.cat([per_view_evidence, geom], dim=-1)).squeeze(-1)
        )
        valid = view_valid[..., None] & proposal_valid[:, None, None, :]
        weights = masked_softmax(
            (log_r_phy + learned) / max(self.temperature, self.eps), valid, dim=2
        )
        quotient = (weights[..., None] * per_view_evidence).sum(dim=2)
        dispersion = (
            weights
            * (per_view_evidence - quotient[:, :, None]).square().mean(dim=-1)
        ).sum(dim=2)
        support = (weights * routed_support).sum(dim=2)
        has_view = valid.any(dim=2)
        quotient = torch.where(has_view[..., None], quotient, torch.zeros_like(quotient))
        dispersion = torch.where(has_view, dispersion, torch.zeros_like(dispersion))
        support = torch.where(has_view, support, torch.zeros_like(support))
        return {
            "quotient": quotient,
            "dispersion": dispersion,
            "view_weights": weights,
            "support": support.clamp(0.0, 1.0),
            "valid": has_view,
            "log_reliability_physics": log_r_phy,
            "reliability_residual": learned,
        }


class CandidateAssignmentHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        p_min: float = 1e-4,
        support_center: float = 0.10,
        support_temperature: float = 0.10,
        support_logit_weight: float = 1.0,
        tau0: float = 0.25,
        tau_p: float = 1.0,
        measurement_consistency_logit_weight: float = 0.0,
        active_candidate_top_k: int = 0,
        occupancy_evidence_weight: float = 0.0,
        occupancy_evidence_floor: float = 0.05,
    ):
        super().__init__()
        self.compensation_assignment_head = make_residual_mlp(
            feature_dim + 2, hidden_dim, 1, blocks=1
        )
        self.candidate_assignment_head = make_residual_mlp(feature_dim + 4, hidden_dim, 1, blocks=1)
        zero_init_last_linear(self.compensation_assignment_head)
        zero_init_last_linear(self.candidate_assignment_head)
        self.p_min = float(p_min)
        self.support_center = float(support_center)
        self.support_temperature = float(support_temperature)
        self.support_logit_weight = float(support_logit_weight)
        self.tau0 = float(tau0)
        self.tau_p = float(tau_p)
        self.measurement_consistency_logit_weight = float(
            measurement_consistency_logit_weight
        )
        self.active_candidate_top_k = int(active_candidate_top_k)
        self.occupancy_evidence_weight = float(occupancy_evidence_weight)
        self.occupancy_evidence_floor = float(occupancy_evidence_floor)

    def apply_occupancy_evidence(
        self, prior_pi: torch.Tensor, branch_density: torch.Tensor
    ) -> torch.Tensor:
        """Convert geometric assignment into a density-evidence candidate posterior."""
        if self.occupancy_evidence_weight <= 0.0:
            return prior_pi
        evidence = branch_density.squeeze(-1).clamp(self.occupancy_evidence_floor, 1.0)
        log_posterior = prior_pi.clamp_min(1.0e-8).log()
        log_posterior = log_posterior + self.occupancy_evidence_weight * evidence.log()
        return torch.softmax(log_posterior, dim=-1)

    def forward(
        self,
        z: torch.Tensor,
        points_mm: torch.Tensor,
        router: dict[str, torch.Tensor],
        candidates: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        b, n, mb, _c = z.shape
        m = mb - 1
        branch_exists = router["branch_valid"][:, None].expand(b, n, mb).clone()
        branch_exists[:, :, 0] = True
        p_all = router["p_all"]
        lam = router["Lambda"]
        no_valid_views = router["valid_view_count"] <= 0
        p0 = p_all[..., :1].clamp(self.p_min, 1.0 - self.p_min)
        comp_delta = torch.tanh(
            self.compensation_assignment_head(
                torch.cat([z[:, :, :1], 1.0 - p0[..., None], lam[:, :, :1, None]], dim=-1)
            )
        ).squeeze(-1)
        pi0 = torch.sigmoid(self.tau0 * torch.logit(p0) + comp_delta)
        if m == 0:
            return torch.ones((b, n, 1), dtype=z.dtype, device=z.device)
        delta = points_mm[:, :, None] - candidates["candidate_centers_mm"][:, None]
        rel = torch.einsum(
            "bnmi,bmij->bnmj", delta, candidates["candidate_support_inverse_sqrt_mm"]
        )
        cand_visible = router.get("candidate_geometric_visible")
        if cand_visible is None:
            cand_visible = lam[:, :, 1:] > 0.0
        cand_active = branch_exists[:, :, 1:] & cand_visible
        support_gate = torch.sigmoid(
            (lam[:, :, 1:] - self.support_center) / max(self.support_temperature, 1e-6)
        )
        if 0 < self.active_candidate_top_k < m:
            activation_score = p_all[..., 1:] * support_gate
            top_indices = activation_score.topk(self.active_candidate_top_k, dim=-1).indices
            top_mask = torch.zeros_like(cand_active).scatter(
                dim=-1, index=top_indices, value=True
            )
            cand_active = cand_active & top_mask
        cand_delta = torch.tanh(
            self.candidate_assignment_head(
                torch.cat([z[:, :, 1:], rel, lam[:, :, 1:, None]], dim=-1)
            )
        ).squeeze(-1)
        cand_logits = (
            self.tau_p * torch.log(p_all[..., 1:].clamp(self.p_min, 1.0))
            + self.support_logit_weight * torch.log(support_gate.clamp_min(self.p_min))
            + cand_delta
        )
        if self.measurement_consistency_logit_weight != 0.0:
            consistency = router["measurement_consistency"].clamp(self.p_min, 1.0)
            cand_logits = cand_logits + self.measurement_consistency_logit_weight * torch.log(
                consistency
            )
        cand_pi = masked_softmax(cand_logits, cand_active, dim=-1)
        has_cand = cand_active.any(dim=-1, keepdim=True) & ~no_valid_views[..., None]
        pi0 = torch.where(has_cand, pi0, torch.ones_like(pi0))
        cand_pi = torch.where(has_cand, (1.0 - pi0) * cand_pi, torch.zeros_like(cand_pi))
        pi = torch.cat([pi0, cand_pi], dim=-1)
        return pi / pi.sum(dim=-1, keepdim=True).clamp_min(1e-8)
