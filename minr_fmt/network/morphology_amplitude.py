"""Morphology-amplitude factorization utilities for the audited Phase-A path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from minr_fmt.network.ssq_decoder import SharedDensityLogitDecoder


@dataclass(frozen=True)
class MorphologyAmplitudeLossConfig:
    support_threshold: float = 0.05
    support_weight: float = 0.10
    support_dice_weight: float = 1.0
    amplitude_weight: float = 0.25
    component_weight: float = 0.0
    eps: float = 1.0e-6


class MorphologyAmplitudeObjective:
    """Compute auxiliary terms without creating a second density field."""

    def __init__(self, config: MorphologyAmplitudeLossConfig) -> None:
        if not 0.0 <= config.support_threshold < 1.0:
            raise ValueError("support_threshold must lie in [0, 1)")
        for name in ("support_weight", "support_dice_weight", "amplitude_weight", "component_weight"):
            if getattr(config, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        self.config = config

    @staticmethod
    def _as_scalar_field(x: torch.Tensor, name: str) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(-1)
        if x.dim() == 3 and x.shape[-1] == 1:
            return x
        raise ValueError(f"{name} must have shape [B,N] or [B,N,1], got {tuple(x.shape)}")

    def _component_balanced_l1(
        self,
        density: torch.Tensor,
        target: torch.Tensor,
        component_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if component_ids is None or self.config.component_weight <= 0.0:
            return density.sum() * 0.0
        ids = component_ids.to(device=density.device)
        if ids.dim() == 3 and ids.shape[-1] == 1:
            ids = ids.squeeze(-1)
        if ids.shape != density.squeeze(-1).shape:
            raise ValueError(
                "component ids must match query layout: "
                f"ids={tuple(ids.shape)} density={tuple(density.shape)}"
            )
        error = (density - target).abs().squeeze(-1)
        per_component: list[torch.Tensor] = []
        for batch_index in range(ids.shape[0]):
            valid_ids = torch.unique(ids[batch_index][ids[batch_index] >= 0])
            for component_id in valid_ids:
                mask = ids[batch_index] == component_id
                if mask.any():
                    per_component.append(error[batch_index][mask].mean())
        return torch.stack(per_component).mean() if per_component else density.sum() * 0.0

    def __call__(
        self,
        *,
        support_probability: torch.Tensor,
        amplitude: torch.Tensor,
        density: torch.Tensor,
        target_density: torch.Tensor,
        component_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        support_probability = self._as_scalar_field(
            support_probability, "support_probability"
        ).clamp(self.config.eps, 1.0 - self.config.eps)
        amplitude = self._as_scalar_field(amplitude, "amplitude").clamp(0.0, 1.0)
        density = self._as_scalar_field(density, "density").clamp(0.0, 1.0)
        target = self._as_scalar_field(target_density, "target_density").to(
            device=density.device, dtype=density.dtype
        ).clamp(0.0, 1.0)
        if not (support_probability.shape == amplitude.shape == density.shape == target.shape):
            raise ValueError("factorized predictions and targets must have identical shapes")

        support_target = (target > self.config.support_threshold).to(density.dtype)
        support_bce = F.binary_cross_entropy(support_probability, support_target)
        intersection = (support_probability * support_target).sum(dim=1)
        denominator = support_probability.sum(dim=1) + support_target.sum(dim=1)
        support_dice = 1.0 - (
            (2.0 * intersection + self.config.eps)
            / (denominator + self.config.eps)
        ).mean()
        support_loss = support_bce + self.config.support_dice_weight * support_dice

        positive = support_target
        amplitude_loss = (
            ((amplitude - target).abs() * positive).sum()
            / positive.sum().clamp_min(1.0)
        )
        component_loss = self._component_balanced_l1(
            density, target, component_ids
        )
        total = (
            self.config.support_weight * support_loss
            + self.config.amplitude_weight * amplitude_loss
            + self.config.component_weight * component_loss
        )
        return {
            "total": total,
            "support_bce": support_bce,
            "support_dice_loss": support_dice,
            "support_loss": support_loss,
            "amplitude_loss": amplitude_loss,
            "component_balanced_density_loss": component_loss,
            "support_target_ratio": support_target.mean(),
            "support_prediction_mean": support_probability.mean(),
            "amplitude_prediction_mean": amplitude.mean(),
        }


def _build_support_decoder(
    amplitude_decoder: SharedDensityLogitDecoder,
    support_init_logit: float,
) -> SharedDensityLogitDecoder:
    """Create a support decoder with the same input contract as the old Phase-A head."""

    feature_dim = int(amplitude_decoder.feat[0].in_features)
    position_dim = int(amplitude_decoder.coord[0].in_features)
    hidden_dim = int(amplitude_decoder.coord[0].out_features)
    support_decoder = SharedDensityLogitDecoder(
        feature_dim,
        position_dim,
        hidden_dim,
        positive_ratio=0.5,
        query_chunk_size=int(amplitude_decoder.query_chunk_size),
        checkpoint_decoder=bool(amplitude_decoder.checkpoint_decoder),
    )
    nn.init.zeros_(support_decoder.fusion[-1].weight)
    nn.init.constant_(support_decoder.fusion[-1].bias, float(support_init_logit))
    reference_parameter = next(amplitude_decoder.parameters())
    return support_decoder.to(
        device=reference_parameter.device,
        dtype=reference_parameter.dtype,
    )


def _legacy_phase_a_forward(
    self,
    query_per_view: torch.Tensor,
    query_view_valid: torch.Tensor,
    points_mm: torch.Tensor,
    norm_scale: torch.Tensor,
    return_diagnostics: bool,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    """Run the checkpoint-compatible Phase-A scalar path and optional factorization."""

    shared, shared_weights = self.complementary_aggregation.aggregate_shared(
        query_per_view,
        query_view_valid,
    )
    encoded_points = self._encoded_query_points(points_mm)
    amplitude_logits = self.shared_density_logit_decoder(shared, encoded_points)
    amplitude = torch.sigmoid(amplitude_logits)

    support_decoder = getattr(self, "factorized_support_decoder", None)
    if support_decoder is None:
        support_probability = torch.ones_like(amplitude)
    else:
        support_probability = torch.sigmoid(support_decoder(shared, encoded_points))
    compose_density = bool(getattr(self, "factorized_compose_density", False))
    density = support_probability * amplitude if compose_density else amplitude

    self.last_factorized_outputs = {
        "density": density,
        "support_probability": support_probability,
        "amplitude": amplitude,
    }
    empty_context = shared.new_zeros(shared.shape)
    aux_outputs: dict[str, torch.Tensor] = {
        "shared_density": density,
        "support_probability": support_probability,
        "amplitude": amplitude,
        "amplitude_logits": amplitude_logits,
        "candidate_context": empty_context,
        "candidate_context_scale": density.new_zeros(()),
        "decoder_pre_activation": amplitude_logits,
        "measurement_supported": query_view_valid.any(dim=1),
        "shared_view_weights": shared_weights,
        "view_complementary_mode": torch.ones((), device=points_mm.device),
        "clean_shared_ablation": torch.ones((), device=points_mm.device),
        "legacy_phase_a_decoder": torch.ones((), device=points_mm.device),
    }
    out: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
        "density": density,
        "aux_outputs": aux_outputs,
    }
    if return_diagnostics:
        diagnostics = dict(aux_outputs)
        diagnostics["normalization_scale"] = norm_scale
        out["diagnostics"] = diagnostics
    return out


def activate_legacy_phase_a_decoder(net: nn.Module, factor_cfg: Any) -> None:
    """Restore the decoder path used by the existing high-Dice Phase-A checkpoint.

    The current refactor contains both the historical ``shared_density_logit_decoder``
    and a newer ``unified_density_decoder``. The old checkpoint predates the latter.
    This activation deliberately routes predictions through the historical decoder;
    no parameter mapping into the newer decoder is performed.
    """

    if not hasattr(net, "shared_density_logit_decoder"):
        raise RuntimeError("model has no checkpoint-compatible shared density decoder")
    if not hasattr(net, "_forward_clean_shared_ablation"):
        raise RuntimeError("model has no clean shared Phase-A path to replace")

    enabled = bool(factor_cfg.get("enabled", False))
    net.factorized_compose_density = bool(
        enabled and factor_cfg.get("compose_density", True)
    )
    net.last_factorized_outputs = {}
    if enabled:
        net.factorized_support_decoder = _build_support_decoder(
            net.shared_density_logit_decoder,
            support_init_logit=float(factor_cfg.get("support_init_logit", 8.0)),
        )
    else:
        net.factorized_support_decoder = None
    net._forward_clean_shared_ablation = MethodType(_legacy_phase_a_forward, net)


def load_legacy_phase_a_checkpoint(
    net: nn.Module,
    checkpoint_path: str | Path,
    *,
    allowed_missing_prefixes: tuple[str, ...] = ("unified_density_decoder.",),
) -> dict[str, Any]:
    """Load an old Phase-A checkpoint while rejecting active-path mismatches.

    ``strict=False`` is not used as a blanket escape hatch. Missing tensors are only
    accepted for explicitly listed, inactive modules introduced after the checkpoint.
    Every tensor belonging to the old active Phase-A path must match by name and shape.
    """

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    source = {
        key.removeprefix("net."): value
        for key, value in state.items()
        if key.startswith("net.")
    }
    if not source:
        source = dict(state)
    current = net.state_dict()
    matched: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for key, value in source.items():
        if key not in current:
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(current[key].shape):
            shape_mismatches.append((key, tuple(value.shape), tuple(current[key].shape)))
            continue
        matched[key] = value

    missing = sorted(set(current) - set(matched))
    illegal_missing = [
        key for key in missing if not key.startswith(allowed_missing_prefixes)
    ]
    required_prefixes = (
        "surface_encoder.",
        "surface_sampler.",
        "complementary_aggregation.",
        "shared_density_logit_decoder.",
    )
    absent_required = [
        prefix for prefix in required_prefixes if not any(key.startswith(prefix) for key in matched)
    ]
    if unexpected or shape_mismatches or illegal_missing or absent_required:
        raise RuntimeError(
            "Phase-A checkpoint is incompatible with the historical active path: "
            f"unexpected={unexpected[:20]} shape_mismatches={shape_mismatches[:20]} "
            f"illegal_missing={illegal_missing[:20]} absent_required={absent_required}"
        )
    missing_after, unexpected_after = net.load_state_dict(matched, strict=False)
    if unexpected_after or sorted(missing_after) != missing:
        raise RuntimeError(
            "controlled Phase-A checkpoint load returned inconsistent keys: "
            f"missing={missing_after[:20]} unexpected={unexpected_after[:20]}"
        )
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "matched_keys": len(matched),
        "allowed_missing_keys": missing,
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
    }
