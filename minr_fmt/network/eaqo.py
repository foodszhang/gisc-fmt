"""EAQO utilities for SSQ-FMT."""

from __future__ import annotations

import torch


def prediction_ambiguity(density: torch.Tensor) -> torch.Tensor:
    """Return 4p(1-p) for probability-domain query density."""
    p = density.squeeze(-1).float().clamp(0.0, 1.0)
    return (4.0 * p * (1.0 - p)).clamp(0.0, 1.0)
