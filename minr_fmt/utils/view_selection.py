"""Utilities for sparse-view ablations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def select_views_by_angles(
    projections: torch.Tensor | dict[str, Any],
    view_angles: Sequence[int | float],
    selected_angles: Sequence[int | float],
) -> tuple[torch.Tensor | dict[str, Any], list[int]]:
    """Select projections and view angles by exact angle values.

    Tensor projections may be shaped [V,...] or [B,V,...]. Dict projections are keyed by
    stringified angle values.
    """
    all_angles = [int(v) for v in view_angles]
    selected = [int(v) for v in selected_angles]
    missing = sorted(set(selected) - set(all_angles))
    if missing:
        raise ValueError(
            f"selected angle(s) {missing} not present in view_angles={all_angles}"
        )
    indices = [all_angles.index(angle) for angle in selected]

    if isinstance(projections, dict):
        return {str(angle): projections[str(angle)] for angle in selected}, selected

    if not torch.is_tensor(projections):
        raise TypeError(f"projections must be a Tensor or dict, got {type(projections)}")

    index_tensor = torch.as_tensor(indices, device=projections.device, dtype=torch.long)
    if projections.dim() >= 5:
        return projections.index_select(1, index_tensor), selected
    return projections.index_select(0, index_tensor), selected
