"""Unified continuous density decoder; candidate context is an internal feature."""

from __future__ import annotations

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
        if fusion_mode == "additive":
            nn.init.normal_(self.candidate_input.weight, std=1.0e-3)
        else:
            nn.init.xavier_uniform_(self.candidate_input.weight)
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
        hypothesis_gate = shared.new_ones((b, n, 1))
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
        shared_hidden = self.shared_input(
            torch.cat([self.shared_norm(shared), encoded_points], dim=-1)
        )
        candidate_hidden = self.candidate_input(self.candidate_norm(context))
        # LayerNorm largely cancels a pre-normalization amplitude gate.  Apply
        # continuous existence/applicability after normalization instead.
        if continuous_applicability:
            candidate_hidden = candidate_hidden * hypothesis_gate.to(candidate_hidden.dtype)
        if self.fusion_mode == "joint_nonresidual":
            pre_activation = torch.cat(
                [shared_hidden, float(context_scale) * candidate_hidden], dim=-1
            )
        else:
            pre_activation = shared_hidden + float(context_scale) * candidate_hidden
        density = torch.sigmoid(self.head(pre_activation))
        return {
            "density": density,
            "candidate_context": context,
            "alpha": alpha,
            "decoder_pre_activation": pre_activation,
        }
