"""Losses for morphology-amplitude factorized continuous reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


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
