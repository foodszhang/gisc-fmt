"""Numerical helpers and diagnostics utilities for SSQ-FMT."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def zero_init_last_linear(module: nn.Module) -> None:
    """Zero-initialize the last Linear layer in a head-like module."""
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Linear):
            nn.init.zeros_(child.weight)
            nn.init.zeros_(child.bias)
            return


def initialize_probability_head(module: nn.Module, positive_ratio: float = 0.03) -> None:
    """Initialize a sigmoid density head to a sparse foreground prior."""
    positive_ratio = float(min(max(positive_ratio, 0.01), 0.20))
    bias = math.log(positive_ratio / (1.0 - positive_ratio))
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Linear):
            nn.init.zeros_(child.weight)
            nn.init.constant_(child.bias, bias)
            return


def validate_finite_tensor(name: str, value: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN/Inf")
    return value


def masked_softmax(
    logits: torch.Tensor,
    mask: torch.Tensor,
    dim: int,
    fallback_index: int | None = None,
) -> torch.Tensor:
    mask = mask.to(device=logits.device, dtype=torch.bool)
    masked = logits.masked_fill(~mask, -1.0e4)
    out = torch.softmax(masked, dim=dim)
    out = torch.where(mask, out, torch.zeros_like(out))
    denom = out.sum(dim=dim, keepdim=True)
    has_any = mask.any(dim=dim, keepdim=True)
    out = torch.where(has_any, out / denom.clamp_min(1e-8), torch.zeros_like(out))
    if fallback_index is not None:
        fallback = torch.zeros_like(out)
        fallback.select(dim, fallback_index).fill_(1.0)
        out = torch.where(has_any, out, fallback)
    return out


def safe_weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    dim: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    num = (values * weights.unsqueeze(-1)).sum(dim=dim)
    den = weights.sum(dim=dim, keepdim=True).clamp_min(eps)
    return num / den


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * expansion)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def make_residual_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    blocks: int = 2,
    expansion: int = 2,
    dropout: float = 0.0,
    final_activation: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
    layers.extend(
        ResidualMLPBlock(hidden_dim, expansion=expansion, dropout=dropout)
        for _ in range(int(blocks))
    )
    layers.extend([nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, out_dim)])
    if final_activation:
        layers.append(nn.SiLU())
    return nn.Sequential(*layers)
