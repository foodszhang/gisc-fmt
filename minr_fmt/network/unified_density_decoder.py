"""Unified continuous density decoder; candidate context is an internal feature."""

from __future__ import annotations

import torch
import torch.nn as nn


class UnifiedDensityDecoder(nn.Module):
    def __init__(self, representation_dim: int, position_dim: int, hidden_dim: int = 96) -> None:
        super().__init__()
        context_in = representation_dim + 3 + 3 + 1
        self.candidate_context = nn.Sequential(
            nn.Linear(context_in, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, representation_dim)
        )
        self.control_context = nn.Sequential(
            nn.Linear(representation_dim + position_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, representation_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(representation_dim * 2 + position_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
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
    ) -> dict[str, torch.Tensor]:
        b, n, _ = points_mm.shape
        m = centers_mm.shape[1]
        context = shared.new_zeros((b, n, shared.shape[-1]))
        alpha = shared.new_zeros((b, n, m))
        if ablation == "shared_capacity":
            context = self.control_context(torch.cat([shared, encoded_points], dim=-1))
        elif ablation not in {"shared_only", "a0"} and m:
            delta = points_mm[:, :, None] - centers_mm[:, None]
            inverse = torch.linalg.inv(covariance.float()).to(delta.dtype)
            mahal = torch.einsum("bnmi,bmij,bnmj->bnm", delta, inverse, delta)
            applicability = torch.exp(-0.5 * mahal) * scores[:, None]
            applicability = applicability * valid[:, None].to(applicability.dtype)
            alpha = applicability / applicability.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            eigen = (
                torch.linalg.eigvalsh(covariance.float())
                .clamp_min(1.0e-8)
                .sqrt()
                .to(delta.dtype)
            )
            candidate_input = torch.cat(
                [
                    candidate[:, None].expand(-1, n, -1, -1),
                    delta,
                    eigen[:, None].expand(-1, n, -1, -1),
                    scores[:, None, :, None].expand(-1, n, -1, -1),
                ],
                dim=-1,
            )
            encoded = self.candidate_context(candidate_input)
            context = (alpha[..., None] * encoded).sum(dim=2)
        density = torch.sigmoid(self.decoder(torch.cat([shared, context, encoded_points], dim=-1)))
        return {"density": density, "candidate_context": context, "alpha": alpha}
