"""Query-level feature refinement blocks."""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureRefinement(nn.Module):
    """Zero-initialized residual refinement for fused query features.

    The initial output is exactly `f_base`, because the last layer of `delta`
    is zero-initialized.
    """

    def __init__(
        self,
        feature_dim: int,
        geom_dim: int = 5,
        hidden_dim: int = 128,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if geom_dim <= 0:
            raise ValueError(f"geom_dim must be positive, got {geom_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.feature_dim = int(feature_dim)
        self.geom_dim = int(geom_dim)
        in_dim = feature_dim * 3 + geom_dim
        self.delta = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        if zero_init:
            nn.init.zeros_(self.delta[-1].weight)
            nn.init.zeros_(self.delta[-1].bias)

    def forward(
        self,
        f_base: torch.Tensor,
        f_ptfa: torch.Tensor,
        geom: torch.Tensor,
    ) -> torch.Tensor:
        """Refine fused query features.

        Args:
            f_base: [B,N,C] legacy fused feature.
            f_ptfa: [B,N,C] high-resolution PTFA evidence.
            geom: [B,N,G] query geometry summary.

        Returns:
            f_refined: [B,N,C]
        """
        if f_base.shape != f_ptfa.shape:
            raise ValueError(
                f"f_base and f_ptfa must have same shape, got "
                f"{tuple(f_base.shape)} vs {tuple(f_ptfa.shape)}"
            )
        if geom.shape[:2] != f_base.shape[:2]:
            raise ValueError(
                f"geom must share [B,N] with f_base, got "
                f"{tuple(geom.shape)} vs {tuple(f_base.shape)}"
            )
        if f_base.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected feature_dim={self.feature_dim}, got {f_base.shape[-1]}"
            )
        if geom.shape[-1] != self.geom_dim:
            raise ValueError(f"expected geom_dim={self.geom_dim}, got {geom.shape[-1]}")

        x = torch.cat([f_base, f_ptfa, f_ptfa - f_base, geom.to(f_base.dtype)], dim=-1)
        delta = self.delta(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
        return f_base + delta
