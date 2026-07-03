"""Complementary cross-view aggregation retaining ambiguous shared evidence."""

from __future__ import annotations

import torch
import torch.nn as nn


class ComplementaryAggregation(nn.Module):
    def __init__(self, feature_dim: int, output_dim: int, epsilon_s: float = 0.1) -> None:
        super().__init__()
        self.epsilon_s = float(epsilon_s)
        self.shared_projection = nn.Linear(feature_dim, output_dim)
        self.candidate_projection = nn.Linear(feature_dim, output_dim)

    def forward(
        self,
        shared_per_view: torch.Tensor,
        candidate_per_view: torch.Tensor,
        view_valid: torch.Tensor,
        candidate_support: torch.Tensor,
        separability: torch.Tensor,
        candidate_valid: torch.Tensor,
        candidate_view_valid: torch.Tensor | None = None,
        uniform_views: bool = False,
        geometry_only: bool = False,
        aggregation_strategy: str | None = None,
    ) -> dict[str, torch.Tensor]:
        shared_weight = view_valid.to(shared_per_view.dtype)
        shared_weight = shared_weight / shared_weight.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        shared = self.shared_projection((shared_per_view * shared_weight[..., None]).sum(dim=1))
        if candidate_view_valid is None:
            candidate_view_valid = candidate_valid[:, None].expand_as(candidate_support)
        valid_float = candidate_view_valid.to(candidate_support.dtype)
        strategy = aggregation_strategy
        if strategy is None or strategy == "legacy":
            strategy = "uniform" if uniform_views else "geometry" if geometry_only else "full"
        if strategy in {"uniform", "oracle"}:
            reliability = valid_float
        elif strategy == "support":
            reliability = candidate_support * valid_float
        elif strategy == "geometry":
            reliability = (self.epsilon_s + separability) * valid_float
        elif strategy == "full":
            reliability = candidate_support * (self.epsilon_s + separability) * valid_float
        elif strategy == "shuffled":
            keys = torch.rand_like(separability).masked_fill(~candidate_view_valid, 2.0)
            permutation = keys.argsort(dim=1)
            shuffled = separability.gather(1, permutation)
            reliability = (self.epsilon_s + shuffled) * valid_float
        else:
            raise ValueError(f"unknown aggregation strategy: {strategy}")
        weight = reliability / reliability.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        weight = weight * candidate_valid[:, None].to(weight.dtype)
        candidate = self.candidate_projection(
            (candidate_per_view * weight[..., None]).sum(dim=1)
        )
        candidate = torch.where(candidate_valid[..., None], candidate, torch.zeros_like(candidate))
        return {"shared": shared, "candidate": candidate, "view_weights": weight}
