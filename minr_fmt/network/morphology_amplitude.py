"""Morphology-amplitude factorization on the checkpoint-compatible Phase-A path."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn as nn
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
    """Auxiliary supervision for one factorized fluorescence density field."""

    def __init__(self, config: MorphologyAmplitudeLossConfig) -> None:
        if not 0.0 <= config.support_threshold < 1.0:
            raise ValueError("support_threshold must lie in [0, 1)")
        for name in (
            "support_weight",
            "support_dice_weight",
            "amplitude_weight",
            "component_weight",
        ):
            if getattr(config, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        self.config = config

    @staticmethod
    def _as_scalar_field(x: torch.Tensor, name: str) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(-1)
        if x.dim() == 3 and x.shape[-1] == 1:
            return x
        raise ValueError(
            f"{name} must have shape [B,N] or [B,N,1], got {tuple(x.shape)}"
        )

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
        return (
            torch.stack(per_component).mean()
            if per_component
            else density.sum() * 0.0
        )

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
            device=density.device,
            dtype=density.dtype,
        ).clamp(0.0, 1.0)
        if not (
            support_probability.shape
            == amplitude.shape
            == density.shape
            == target.shape
        ):
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
            density,
            target,
            component_ids,
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


def _last_linear(module: nn.Module) -> nn.Linear:
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Linear):
            return child
    raise TypeError("decoder head contains no Linear layer")


def _build_support_head(
    amplitude_head: nn.Module,
    support_init_logit: float,
) -> nn.Module:
    """Clone the trained latent transform and initialize a neutral support output."""

    support_head = copy.deepcopy(amplitude_head)
    output = _last_linear(support_head)
    nn.init.zeros_(output.weight)
    nn.init.constant_(output.bias, float(support_init_logit))
    return support_head


def attach_factorized_unified_output(net: nn.Module, factor_cfg: Any) -> None:
    """Attach support prediction after loading the original Phase-A checkpoint.

    The original ``UnifiedDensityDecoder.head`` remains the amplitude predictor. The
    adapter consumes the same decoder pre-activation and is restricted to the exact
    Phase-A shared-only call: ``ablation='shared_only', context_scale=0``. Any attempt
    to use it in Phase B/C or candidate-conditioned decoding fails immediately.
    """

    decoder = getattr(net, "unified_density_decoder", None)
    if decoder is None:
        raise RuntimeError("SSQ-FMT model has no UnifiedDensityDecoder")
    if str(getattr(decoder, "fusion_mode", "")) != "joint_nonresidual":
        raise ValueError("factorized Phase-A experiment requires joint_nonresidual fusion")
    if getattr(decoder, "_morphology_amplitude_attached", False):
        return

    enabled = bool(factor_cfg.get("enabled", False))
    decoder.factorized_output_enabled = enabled
    decoder.factorized_compose_density = bool(
        enabled and factor_cfg.get("compose_density", True)
    )
    decoder.last_factorized_outputs = {}
    if enabled:
        decoder.factorized_support_head = _build_support_head(
            decoder.head,
            support_init_logit=float(factor_cfg.get("support_init_logit", 8.0)),
        )
    else:
        decoder.factorized_support_head = None

    original_forward = decoder.forward

    def factorized_forward(self, *args, **kwargs):
        out = original_forward(*args, **kwargs)
        amplitude = out["density"]
        if not self.factorized_output_enabled:
            self.last_factorized_outputs = {
                "density": amplitude,
                "support_probability": torch.ones_like(amplitude),
                "amplitude": amplitude,
            }
            return out

        ablation = kwargs.get("ablation", args[8] if len(args) > 8 else "full")
        context_scale = kwargs.get(
            "context_scale",
            args[9] if len(args) > 9 else 1.0,
        )
        if str(ablation) not in {"shared_only", "a0"} or abs(float(context_scale)) > 1.0e-12:
            raise RuntimeError(
                "morphology-amplitude factorization is restricted to the reproduced "
                "Phase-A shared-only path and cannot be used in Phase B/C"
            )

        pre_activation = out["decoder_pre_activation"]
        support_probability = torch.sigmoid(
            self.factorized_support_head(pre_activation)
        )
        density = (
            support_probability * amplitude
            if self.factorized_compose_density
            else amplitude
        )
        updated = dict(out)
        updated["density"] = density
        updated["support_probability"] = support_probability
        updated["amplitude"] = amplitude
        if "shared_density" in updated:
            updated["shared_density"] = density
        if "branch_density" in updated and updated["branch_density"].shape[2] > 0:
            branch_density = updated["branch_density"].clone()
            branch_density[:, :, 0] = density
            updated["branch_density"] = branch_density
        self.last_factorized_outputs = {
            "density": density,
            "support_probability": support_probability,
            "amplitude": amplitude,
            "decoder_pre_activation": pre_activation,
        }
        return updated

    decoder.forward = MethodType(factorized_forward, decoder)
    decoder._morphology_amplitude_attached = True


_HISTORICAL_EXTENSION_PREFIXES = (
    "unified_density_decoder.candidate_residual_head.",
    "unified_density_decoder.candidate_branch_head.",
)


def _net_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = checkpoint.get("state_dict", checkpoint)
    net_state = {
        key.removeprefix("net."): value
        for key, value in state.items()
        if key.startswith("net.")
    }
    return net_state or dict(state)


def load_historical_phase_a_checkpoint(
    net: nn.Module,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Load the reproduced 0.7507 checkpoint into the current refactor safely.

    Missing tensors are accepted only for two candidate-only heads introduced after
    the historical Phase-A run. All historical tensors, including the complete
    UnifiedDensityDecoder shared path, must match by name and shape.
    """

    checkpoint = torch.load(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    source = _net_state(checkpoint)
    current = net.state_dict()
    unexpected = sorted(key for key in source if key not in current)
    mismatched = sorted(
        (
            key,
            tuple(source[key].shape),
            tuple(current[key].shape),
        )
        for key in source
        if key in current and source[key].shape != current[key].shape
    )
    if unexpected or mismatched:
        raise RuntimeError(
            "historical Phase-A checkpoint has incompatible source tensors: "
            f"unexpected={unexpected[:20]} mismatched={mismatched[:20]}"
        )

    matching = {key: value for key, value in source.items() if key in current}
    missing = sorted(set(current) - set(matching))
    illegal_missing = [
        key
        for key in missing
        if not key.startswith(_HISTORICAL_EXTENSION_PREFIXES)
    ]
    required_prefixes = (
        "surface_encoder.",
        "surface_sampler.",
        "complementary_aggregation.",
        "unified_density_decoder.shared_norm.",
        "unified_density_decoder.shared_input.",
        "unified_density_decoder.head.",
    )
    absent_required = [
        prefix
        for prefix in required_prefixes
        if not any(key.startswith(prefix) for key in matching)
    ]
    if illegal_missing or absent_required:
        raise RuntimeError(
            "historical Phase-A checkpoint does not cover the active shared path: "
            f"illegal_missing={illegal_missing[:20]} absent_required={absent_required}"
        )

    missing_after, unexpected_after = net.load_state_dict(matching, strict=False)
    if unexpected_after or sorted(missing_after) != missing:
        raise RuntimeError(
            "controlled historical checkpoint load returned inconsistent keys: "
            f"missing={missing_after[:20]} unexpected={unexpected_after[:20]}"
        )
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "matched_keys": len(matching),
        "allowed_current_extensions": missing,
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
    }


def load_factorized_or_historical_checkpoint(
    net: nn.Module,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Strictly load a factorized checkpoint, otherwise use historical compatibility."""

    checkpoint = torch.load(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    source = _net_state(checkpoint)
    factorized_prefix = "unified_density_decoder.factorized_support_head."
    if any(key.startswith(factorized_prefix) for key in source):
        missing, unexpected = net.load_state_dict(source, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"strict factorized load failed: missing={missing[:20]} "
                f"unexpected={unexpected[:20]}"
            )
        return {
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "strict_factorized": True,
            "matched_keys": len(source),
            "epoch": checkpoint.get("epoch"),
            "global_step": checkpoint.get("global_step"),
        }
    return load_historical_phase_a_checkpoint(net, checkpoint_path)
