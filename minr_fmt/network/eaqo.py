"""EAQO utilities for SSQ-FMT."""

from __future__ import annotations

import warnings

import torch


def prediction_ambiguity(density: torch.Tensor) -> torch.Tensor:
    """Return 4p(1-p) for probability-domain query density."""
    p = density.squeeze(-1).float().clamp(0.0, 1.0)
    return (4.0 * p * (1.0 - p)).clamp(0.0, 1.0)


def view_evidence_ambiguity(
    aux_outputs: dict,
    reference: torch.Tensor,
    *,
    eps: float = 1.0e-6,
    require: bool = False,
) -> torch.Tensor:
    """Compute query-level inter-view evidence conflict from Phase A evidence."""
    out_shape = reference.shape[:2]
    if not isinstance(aux_outputs, dict):
        if require:
            raise RuntimeError("EAQO requires per-view evidence, but aux_outputs is not a dict")
        warnings.warn("EAQO per-view evidence unavailable; using zero A_view", stacklevel=2)
        return reference.new_zeros(out_shape)
    evidence = aux_outputs.get("per_view_evidence")
    valid = aux_outputs.get("query_view_valid")
    if not torch.is_tensor(evidence) or evidence.dim() != 3:
        if require:
            raise RuntimeError("EAQO require_view_evidence=true but per_view_evidence is missing")
        warnings.warn("EAQO per-view evidence unavailable; using zero A_view", stacklevel=2)
        return reference.new_zeros(out_shape)
    if evidence.shape[0] != out_shape[0] or evidence.shape[2] != out_shape[1]:
        if require:
            raise RuntimeError(
                "EAQO per_view_evidence shape must be [B,V,N] and match density queries"
            )
        warnings.warn("EAQO per-view evidence shape mismatch; using zero A_view", stacklevel=2)
        return reference.new_zeros(out_shape)
    strength = evidence.to(device=reference.device, dtype=torch.float32).abs()
    if torch.is_tensor(valid) and valid.shape == strength.shape:
        view_valid = valid.to(device=reference.device).bool()
    else:
        view_valid = torch.ones_like(strength, dtype=torch.bool)
    valid_f = view_valid.to(dtype=strength.dtype)
    count = valid_f.sum(dim=1)
    denom = count.clamp_min(1.0)
    mean = (strength * valid_f).sum(dim=1) / denom
    var = ((strength - mean[:, None]).square() * valid_f).sum(dim=1) / denom
    score = var / mean.abs().clamp_min(eps)
    score = torch.where(count >= 2.0, score, torch.zeros_like(score))
    return torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


def geometry_ambiguity(batch: dict, reference: torch.Tensor) -> torch.Tensor:
    """Weight foreground boundary-near queries when distance targets are available."""
    out = reference.new_zeros(reference.shape[:2])
    if not isinstance(batch, dict):
        return out
    distance = batch.get("distance_target")
    fg_mask = batch.get("center_distance_fg_mask")
    if not torch.is_tensor(distance):
        return out
    distance = distance.to(device=reference.device, dtype=torch.float32)
    if distance.dim() == 3 and distance.shape[-1] == 1:
        distance = distance.squeeze(-1)
    if distance.shape != out.shape:
        return out
    if torch.is_tensor(fg_mask):
        fg = fg_mask.to(device=reference.device, dtype=torch.float32)
        if fg.dim() == 3 and fg.shape[-1] == 1:
            fg = fg.squeeze(-1)
        if fg.shape == out.shape:
            return (1.0 - distance.clamp(0.0, 1.0)) * (fg > 0.5).to(dtype=distance.dtype)
    return out


def component_ambiguity(batch: dict, reference: torch.Tensor) -> torch.Tensor:
    """Assign higher ambiguity to foreground queries from smaller components."""
    out = reference.new_zeros(reference.shape[:2])
    if not isinstance(batch, dict):
        return out
    component_ids = batch.get("query_component_ids")
    valid_mask = batch.get("gt_component_valid_mask")
    if not torch.is_tensor(component_ids):
        return out
    component_ids = component_ids.to(device=reference.device)
    if component_ids.shape != out.shape:
        return out
    if torch.is_tensor(valid_mask):
        valid_mask = valid_mask.to(device=reference.device).bool()
    for b in range(component_ids.shape[0]):
        ids = component_ids[b]
        fg_ids = ids[ids > 0]
        if fg_ids.numel() == 0:
            continue
        unique_ids, counts = torch.unique(fg_ids, return_counts=True)
        scores = counts.float().reciprocal()
        if scores.numel() > 1:
            scores = (scores - scores.min()) / (scores.max() - scores.min()).clamp_min(1.0e-6)
        else:
            scores = torch.ones_like(scores)
        for component_id, score in zip(unique_ids, scores, strict=False):
            idx = int(component_id.item()) - 1
            if valid_mask is not None and (
                idx < 0 or idx >= valid_mask.shape[1] or not valid_mask[b, idx]
            ):
                continue
            out[b] = torch.where(ids == component_id, score.to(dtype=out.dtype), out[b])
    return out


def normalize_score(
    score: torch.Tensor,
    valid: torch.Tensor,
    mode: str,
    eps: float,
) -> torch.Tensor:
    """Normalize ambiguity scores without changing invalid queries."""
    if mode in {"none", "identity", "null"}:
        return torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if mode != "batch_minmax":
        raise ValueError(f"Unsupported EAQO normalize_score={mode!r}")
    valid = valid.bool()
    out = torch.zeros_like(score)
    for b in range(score.shape[0]):
        vals = score[b][valid[b]]
        if vals.numel():
            out[b] = (score[b] - vals.min()) / (vals.max() - vals.min()).clamp_min(eps)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
