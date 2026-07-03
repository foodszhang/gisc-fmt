"""Bounded hypothesis-conditioned view routing for the A3-v2 feasibility path."""

from __future__ import annotations

import torch
import torch.nn as nn


class BoundedHypothesisViewRouting(nn.Module):
    """Add a bounded, centered residual to valid-view uniform logits."""

    def __init__(self, delta_logit_max: float = 1.0, zero_init: bool = True) -> None:
        super().__init__()
        initial = 0.0 if zero_init else 0.1
        self.delta_logit_max = float(delta_logit_max)
        self.routing_gain_raw = nn.Parameter(torch.tensor(initial))
        self.gain_predictor = nn.Sequential(
            nn.Linear(5, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.gain_predictor[-1].weight)
        nn.init.zeros_(self.gain_predictor[-1].bias)

    def forward(
        self,
        points_mm: torch.Tensor,
        centers_mm: torch.Tensor,
        covariance: torch.Tensor,
        existence_probability: torch.Tensor,
        slot_valid: torch.Tensor,
        candidate_view_support: torch.Tensor,
        geometry_separability: torch.Tensor,
        view_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return [B,V,N] weights and continuous hypothesis diagnostics."""
        eps = 1.0e-8
        delta = points_mm[:, :, None] - centers_mm[:, None]
        variance = torch.diagonal(covariance.float(), dim1=-2, dim2=-1).clamp_min(1.0e-4)
        mahal = (delta.float().square() / variance[:, None]).sum(dim=-1)
        applicability = torch.exp(-0.5 * mahal)
        applicability = applicability * existence_probability[:, None].float()
        applicability = applicability * slot_valid[:, None].to(applicability.dtype)
        applicability = torch.nan_to_num(applicability, nan=0.0, posinf=0.0, neginf=0.0)
        applicability_sum = applicability.sum(dim=-1)
        hypothesis_gate = -torch.expm1(-applicability_sum)

        # Competition is strictly within each view; absolute support is never compared
        # across views. A view with fewer than two valid slots has no relative term.
        support = candidate_view_support.transpose(1, 2).float().clamp_min(eps)
        valid_vm = slot_valid[:, None].expand_as(support)
        count = valid_vm.sum(dim=-1, keepdim=True)
        log_support = support.log()
        mean_log_support = (log_support * valid_vm).sum(dim=-1, keepdim=True) / count.clamp_min(1)
        relative_support = (log_support - mean_log_support) * valid_vm
        geometry = geometry_separability.float()
        mean_geometry = (geometry * valid_vm).sum(dim=-1, keepdim=True) / count.clamp_min(1)
        centered_geometry = (geometry - mean_geometry) * valid_vm
        relative = relative_support + centered_geometry
        relative = torch.where(count > 1, relative, torch.zeros_like(relative))

        routing = torch.einsum("bnm,bvm->bnv", applicability, relative)
        routing = routing / applicability_sum[..., None].clamp_min(eps)
        valid_bnv = view_valid.transpose(1, 2)
        valid_float = valid_bnv.to(routing.dtype)
        valid_count = valid_float.sum(dim=-1, keepdim=True)
        routing_mean = (routing * valid_float).sum(dim=-1, keepdim=True) / valid_count.clamp_min(
            1.0
        )
        centered = (routing - routing_mean) * valid_float
        centered_abs_mean = (centered.abs() * valid_float).sum(dim=-1, keepdim=True)
        centered_abs_mean = centered_abs_mean / valid_count.clamp_min(1.0)
        centered_rms = (
            (centered.square() * valid_float).sum(dim=-1, keepdim=True)
            / valid_count.clamp_min(1.0)
        ).clamp_min(1.0e-8).sqrt()
        centered_max = centered.masked_fill(~valid_bnv, -1.0e4).amax(dim=-1, keepdim=True)
        centered_min = centered.masked_fill(~valid_bnv, 1.0e4).amin(dim=-1, keepdim=True)
        has_valid = valid_count > 0
        centered_max = torch.where(has_valid, centered_max, torch.zeros_like(centered_max))
        centered_min = torch.where(has_valid, centered_min, torch.zeros_like(centered_min))
        gain_features = torch.cat(
            [
                hypothesis_gate[..., None],
                centered_abs_mean,
                centered_rms,
                centered_max,
                centered_min,
            ],
            dim=-1,
        ).detach()
        conditional_gain_raw = self.gain_predictor(gain_features)
        scale = self.delta_logit_max * torch.tanh(
            self.routing_gain_raw + conditional_gain_raw
        )
        residual = hypothesis_gate[..., None] * scale * torch.tanh(centered)
        unit_residual = hypothesis_gate[..., None] * torch.tanh(centered)
        logits = residual.masked_fill(~valid_bnv, -torch.inf)
        weights = torch.softmax(logits, dim=-1)
        weights = torch.where(valid_bnv, weights, torch.zeros_like(weights))
        weights = torch.where(
            valid_count > 0,
            weights / weights.sum(dim=-1, keepdim=True).clamp_min(eps),
            torch.zeros_like(weights),
        )
        return {
            "view_weights": weights.transpose(1, 2),
            "applicability": applicability,
            "hypothesis_gate": hypothesis_gate,
            "routing_residual": residual,
            "routing_unit_residual": unit_residual,
            "routing_scale": scale,
            "conditional_gain_raw": conditional_gain_raw,
            "gain_features": gain_features,
            "relative_evidence": relative,
            # Query-level proxy for ambiguity caused by overlapping candidate
            # projections.  Low separability means high projected overlap.
            "projected_overlap": (
                applicability
                / applicability_sum[..., None].clamp_min(eps)
                * (1.0 - geometry_separability.float()).mean(dim=1)[:, None]
            ).sum(dim=-1),
        }
