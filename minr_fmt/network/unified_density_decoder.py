"""Unified continuous density decoder; candidate context is an internal feature."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class UnifiedDensityDecoder(nn.Module):
    def __init__(
        self,
        representation_dim: int,
        position_dim: int,
        hidden_dim: int = 96,
        fusion_mode: str = "additive",
    ) -> None:
        super().__init__()
        if fusion_mode not in {"additive", "joint_nonresidual"}:
            raise ValueError(f"unknown unified decoder fusion mode: {fusion_mode}")
        self.fusion_mode = fusion_mode
        context_in = representation_dim + 3 + 3 + 1
        self.candidate_context = nn.Sequential(
            nn.Linear(context_in, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, representation_dim)
        )
        self.control_context = nn.Sequential(
            nn.Linear(representation_dim + position_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, representation_dim),
        )
        self.shared_norm = nn.LayerNorm(representation_dim)
        self.candidate_norm = nn.LayerNorm(representation_dim)
        self.shared_input = nn.Linear(representation_dim + position_dim, hidden_dim)
        self.candidate_input = nn.Linear(representation_dim, hidden_dim, bias=False)
        # The hypothesis-conditioned latent interface must initially preserve the
        # Phase-A function exactly. A zero matrix still receives gradients once a
        # non-zero candidate context is presented in Phase B.
        nn.init.zeros_(self.candidate_input.weight)
        if fusion_mode == "additive":
            self.head = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.head = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
        self.candidate_residual_head = None
        self.candidate_branch_head = None
        if fusion_mode == "joint_nonresidual":
            self.candidate_residual_head = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.zeros_(self.candidate_residual_head[-1].weight)
            nn.init.zeros_(self.candidate_residual_head[-1].bias)
            self.candidate_branch_head = nn.Sequential(
                nn.LayerNorm(representation_dim * 2 + 7),
                nn.Linear(representation_dim * 2 + 7, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            nn.init.normal_(self.candidate_branch_head[-1].weight, std=1.0e-2)
            nn.init.constant_(self.candidate_branch_head[-1].bias, -4.595)
        self.support_head = copy.deepcopy(self.head)
        self.intensity_head = copy.deepcopy(self.head)
        self.initialize_factorized_from_density_head()

    def initialize_factorized_from_density_head(self) -> None:
        self.intensity_head.load_state_dict(self.head.state_dict())
        for parameter in self.support_head.parameters():
            nn.init.zeros_(parameter)
        last = self.support_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, 8.0)

    def forward(
        self,
        shared: torch.Tensor,
        candidate: torch.Tensor,
        points_mm: torch.Tensor,
        encoded_points: torch.Tensor,
        centers_mm: torch.Tensor,
        covariance: torch.Tensor,
        scores: torch.Tensor,
        valid: torch.Tensor,
        ablation: str = "full",
        context_scale: float = 1.0,
        continuous_applicability: bool = False,
    ) -> dict[str, torch.Tensor]:
        b, n, _ = points_mm.shape
        m = centers_mm.shape[1]
        context = shared.new_zeros((b, n, shared.shape[-1]))
        alpha = shared.new_zeros((b, n, m))
        applicability = shared.new_zeros((b, n, m))
        encoded = shared.new_zeros((b, n, m, shared.shape[-1]))
        branch_features = shared.new_zeros((b, n, m, shared.shape[-1] * 2 + 7))
        hypothesis_gate = shared.new_zeros((b, n, 1))
        if ablation == "shared_capacity":
            context = self.control_context(torch.cat([shared, encoded_points], dim=-1))
        elif ablation not in {"shared_only", "a0"} and m:
            delta = points_mm[:, :, None] - centers_mm[:, None]
            variance = torch.diagonal(
                covariance.float(), dim1=-2, dim2=-1
            ).clamp_min(1.0e-4)
            mahal = (
                delta.float().square() / variance[:, None]
            ).sum(dim=-1)
            applicability = torch.exp(-0.5 * mahal) * scores[:, None]
            applicability = applicability * valid[:, None].to(applicability.dtype)
            alpha = applicability / applicability.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            hypothesis_gate = -torch.expm1(-applicability.sum(dim=-1, keepdim=True))
            scales = variance.sqrt().to(delta.dtype)
            candidate_input = torch.cat(
                [
                    candidate[:, None].expand(-1, n, -1, -1),
                    delta,
                    scales[:, None].expand(-1, n, -1, -1),
                    scores[:, None, :, None].expand(-1, n, -1, -1),
                ],
                dim=-1,
            )
            encoded = self.candidate_context(candidate_input)
            context = (alpha[..., None] * encoded).sum(dim=2)
            relative = (delta / scales[:, None].clamp_min(1.0e-4)).clamp(-4.0, 4.0)
            scale_feature = torch.log1p(scales)[:, None].expand(-1, n, -1, -1)
            branch_features = torch.cat(
                [
                    shared[:, :, None].expand(-1, -1, m, -1),
                    candidate[:, None].expand(-1, n, -1, -1),
                    relative,
                    scale_feature,
                    scores[:, None, :, None].expand(-1, n, -1, -1),
                ],
                dim=-1,
            )
        shared_hidden = self.shared_input(
            torch.cat([self.shared_norm(shared), encoded_points], dim=-1)
        )
        candidate_hidden = self.candidate_input(self.candidate_norm(context))
        # LayerNorm largely cancels a pre-normalization amplitude gate.  Apply
        # continuous existence/applicability after normalization instead.
        if continuous_applicability:
            candidate_hidden = candidate_hidden * hypothesis_gate.to(candidate_hidden.dtype)
        if self.fusion_mode == "joint_nonresidual":
            shared_pre_activation = torch.cat(
                [shared_hidden, torch.zeros_like(candidate_hidden)], dim=-1
            )
            pre_activation = torch.cat(
                [shared_hidden, float(context_scale) * candidate_hidden], dim=-1
            )
            shared_density = torch.sigmoid(self.head(shared_pre_activation))
            density = torch.sigmoid(self.head(pre_activation))
            candidate_delta_logit = torch.logit(
                density.float().clamp(1.0e-6, 1.0 - 1.0e-6)
            ) - torch.logit(shared_density.float().clamp(1.0e-6, 1.0 - 1.0e-6))
            candidate_branch_density = density.new_zeros((b, n, m, 1))
        else:
            pre_activation = shared_hidden + float(context_scale) * candidate_hidden
            candidate_delta_logit = self.head(pre_activation) * 0.0
            density = torch.sigmoid(self.head(pre_activation))
            shared_density = density
            candidate_branch_density = density.new_zeros((b, n, m, 1))
        support_logits = self.support_head(pre_activation)
        intensity_logits = self.intensity_head(pre_activation)
        support = torch.sigmoid(support_logits)
        intensity = torch.sigmoid(intensity_logits)
        shared_prior = (1.0 - hypothesis_gate).clamp(0.0, 1.0)
        all_prior = torch.cat([shared_prior, applicability], dim=-1)
        all_prior = all_prior / all_prior.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        return {
            "density": density,
            "factorized_density": support * intensity,
            "support": support,
            "intensity": intensity,
            "support_logits": support_logits,
            "intensity_logits": intensity_logits,
            "shared_density": shared_density,
            "branch_density": torch.cat(
                [shared_density[:, :, None], candidate_branch_density], dim=2
            ),
            "p_all": all_prior,
            "pi": all_prior,
            "candidate_context": context,
            "candidate_branch_features": branch_features,
            "candidate_applicability": applicability,
            "alpha": alpha,
            "decoder_pre_activation": pre_activation,
            "candidate_delta_logit": candidate_delta_logit,
            "hypothesis_gate": hypothesis_gate,
        }
