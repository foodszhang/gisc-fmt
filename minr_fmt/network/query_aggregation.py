"""Query-level multi-view aggregation modules."""

from __future__ import annotations

import torch
import torch.nn as nn


class GeometryViewGate(nn.Module):
    """Geometry-only masked view gate for query evidence aggregation.

    The gate intentionally does not consume feature evidence. With zero-initialized final
    projection, valid views start as a uniform mean and invalid views are excluded exactly.
    """

    def __init__(
        self,
        geom_dim: int,
        hidden_dim: int = 64,
        temperature: float = 1.0,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if geom_dim <= 0:
            raise ValueError(f"geom_dim must be positive, got {geom_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        self.temperature = float(temperature)
        self.net = nn.Sequential(
            nn.Linear(geom_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        f: torch.Tensor,
        geom: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate per-view query features.

        Args:
            f: [B,N,V,C] per-view features.
            geom: [B,N,V,G] geometry-only features.
            valid: [B,N,V] boolean view mask.

        Returns:
            fused: [B,N,C]
            weights: [B,N,V]
        """
        if f.dim() != 4:
            raise ValueError(f"f must be [B,N,V,C], got {tuple(f.shape)}")
        if geom.dim() != 4:
            raise ValueError(f"geom must be [B,N,V,G], got {tuple(geom.shape)}")
        if valid.dim() != 3:
            raise ValueError(f"valid must be [B,N,V], got {tuple(valid.shape)}")
        if f.shape[:3] != geom.shape[:3] or f.shape[:3] != valid.shape:
            raise ValueError(
                f"incompatible shapes: f={tuple(f.shape)}, geom={tuple(geom.shape)}, "
                f"valid={tuple(valid.shape)}"
            )

        valid_bool = valid.to(dtype=torch.bool, device=f.device)
        logits = self.net(geom.to(device=f.device, dtype=f.dtype)).squeeze(-1)
        logits = logits / self.temperature
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        masked_logits = logits.masked_fill(~valid_bool, -torch.inf)
        has_valid = valid_bool.any(dim=-1, keepdim=True)
        safe_logits = torch.where(has_valid, masked_logits, torch.zeros_like(masked_logits))
        weights = torch.softmax(safe_logits, dim=-1)
        weights = torch.where(has_valid, weights, torch.zeros_like(weights))
        weights = weights * valid_bool.to(dtype=weights.dtype)

        fused = (f * weights.unsqueeze(-1)).sum(dim=2)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        return fused, weights


class ReliabilityViewGate(nn.Module):
    """Geometry-only reliability gate for per-view PTFA evidence.

    This is intentionally not feature attention: logits depend only on transport
    reliability features. With a zero-initialized last layer, the initial output
    is exactly a valid-view uniform mean.
    """

    def __init__(
        self,
        geom_dim: int,
        hidden_dim: int = 64,
        temperature: float = 1.5,
        zero_init: bool = True,
        norm: str = "none",
        residual_mix_enabled: bool = False,
        residual_mix_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        if geom_dim <= 0:
            raise ValueError(f"geom_dim must be positive, got {geom_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if norm not in {"none", "layernorm"}:
            raise ValueError(f"norm must be 'none' or 'layernorm', got {norm}")
        if not 0.0 <= residual_mix_gamma <= 1.0:
            raise ValueError(
                f"residual_mix_gamma must be in [0,1], got {residual_mix_gamma}"
            )

        self.geom_dim = int(geom_dim)
        self.hidden_dim = int(hidden_dim)
        self.temperature = float(temperature)
        self.norm_name = str(norm)
        self.geom_norm = nn.LayerNorm(self.geom_dim) if self.norm_name == "layernorm" else None
        self.residual_mix_enabled = bool(residual_mix_enabled)
        self.residual_mix_gamma = float(residual_mix_gamma)
        self.net = nn.Sequential(
            nn.Linear(self.geom_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        f_view: torch.Tensor,
        geom_view: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate per-view PTFA evidence.

        Args:
            f_view: [B,N,V,C] per-view evidence.
            geom_view: [B,N,V,G] reliability features.
            valid: [B,N,V] boolean view mask.

        Returns:
            f_agg: [B,N,C]
            weights: [B,N,V]
        """
        if f_view.dim() != 4:
            raise ValueError(f"f_view must be [B,N,V,C], got {tuple(f_view.shape)}")
        if geom_view.dim() != 4:
            raise ValueError(f"geom_view must be [B,N,V,G], got {tuple(geom_view.shape)}")
        if valid.dim() != 3:
            raise ValueError(f"valid must be [B,N,V], got {tuple(valid.shape)}")
        if f_view.shape[:3] != geom_view.shape[:3] or f_view.shape[:3] != valid.shape:
            raise ValueError(
                f"incompatible shapes: f_view={tuple(f_view.shape)}, "
                f"geom_view={tuple(geom_view.shape)}, valid={tuple(valid.shape)}"
            )

        valid_bool = valid.to(device=f_view.device, dtype=torch.bool)
        geom = geom_view.to(device=f_view.device, dtype=f_view.dtype)
        geom = torch.nan_to_num(geom, nan=0.0, posinf=0.0, neginf=0.0)
        if self.geom_norm is not None:
            geom = self.geom_norm(geom)
        logits = self.net(geom).squeeze(-1)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        logits = logits / self.temperature

        has_valid = valid_bool.any(dim=-1, keepdim=True)
        valid_float = valid_bool.to(dtype=f_view.dtype)
        valid_count = valid_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
        uniform = torch.where(has_valid, valid_float / valid_count, torch.zeros_like(valid_float))

        masked_logits = logits.masked_fill(~valid_bool, -1.0e4)
        safe_logits = torch.where(has_valid, masked_logits, torch.zeros_like(masked_logits))
        learned = torch.softmax(safe_logits, dim=-1)
        learned = torch.where(has_valid, learned, torch.zeros_like(learned))
        learned = learned * valid_float
        learned = torch.nan_to_num(learned, nan=0.0, posinf=0.0, neginf=0.0)

        if self.residual_mix_enabled:
            gamma = self.residual_mix_gamma
            weights = (1.0 - gamma) * uniform + gamma * learned
        else:
            weights = learned
        weights = weights * valid_float
        weights = torch.where(has_valid, weights, torch.zeros_like(weights))
        weight_sum = weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        weights = torch.where(has_valid, weights / weight_sum, weights)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

        f_agg = (f_view * weights.unsqueeze(-1)).sum(dim=2)
        f_agg = torch.nan_to_num(f_agg, nan=0.0, posinf=0.0, neginf=0.0)
        return f_agg, weights


class ConsensusResidualGate(nn.Module):
    """Small residual calibration around valid-view uniform consensus.

    Unlike ReliabilityViewGate, this module never replaces the uniform mean with a learned
    weighted sum. It only learns per-view confidence for residual evidence:
    f_out = f_mean + gamma * mean_valid(sigmoid(MLP(x_v)) * (f_v - f_mean)).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        gamma: float = 0.1,
        norm: str = "layernorm",
        final_bias: float = -2.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if norm not in {"none", "layernorm"}:
            raise ValueError(f"norm must be 'none' or 'layernorm', got {norm}")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.gamma = float(gamma)
        self.norm_name = str(norm)
        self.input_norm = nn.LayerNorm(self.input_dim) if self.norm_name == "layernorm" else None
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, float(final_bias))

    @staticmethod
    def _valid_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weights = valid.unsqueeze(-1).to(dtype=x.dtype, device=x.device)
        denom = weights.sum(dim=2).clamp_min(1.0)
        return (x * weights).sum(dim=2) / denom

    def forward(
        self,
        f_view: torch.Tensor,
        gate_input: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Apply residual evidence calibration.

        Args:
            f_view: [B,N,V,C] per-view evidence.
            gate_input: [B,N,V,D] compact geom + evidence stats.
            valid: [B,N,V] boolean view mask.

        Returns:
            f_out: [B,N,C]
            stats: tensors for confidence/residual diagnostics.
        """
        if f_view.dim() != 4:
            raise ValueError(f"f_view must be [B,N,V,C], got {tuple(f_view.shape)}")
        if gate_input.dim() != 4 or gate_input.shape[-1] != self.input_dim:
            raise ValueError(
                f"gate_input must be [B,N,V,{self.input_dim}], got {tuple(gate_input.shape)}"
            )
        if valid.dim() != 3 or valid.shape != f_view.shape[:3]:
            raise ValueError(f"valid shape {tuple(valid.shape)} incompatible with {tuple(f_view.shape)}")
        if gate_input.shape[:3] != f_view.shape[:3]:
            raise ValueError(
                f"gate_input shape {tuple(gate_input.shape)} incompatible with "
                f"f_view {tuple(f_view.shape)}"
            )

        valid_bool = valid.to(device=f_view.device, dtype=torch.bool)
        valid_float = valid_bool.to(dtype=f_view.dtype)
        x = gate_input.to(device=f_view.device, dtype=f_view.dtype)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if self.input_norm is not None:
            x = self.input_norm(x)

        logits = self.net(x).squeeze(-1)
        confidence = torch.sigmoid(torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0))
        confidence = confidence * valid_float

        f_mean = self._valid_mean(f_view, valid_bool)
        residual = f_view - f_mean.unsqueeze(2)
        residual_update = self._valid_mean(confidence.unsqueeze(-1) * residual, valid_bool)
        f_out = f_mean + self.gamma * residual_update
        f_out = torch.nan_to_num(f_out, nan=0.0, posinf=0.0, neginf=0.0)

        valid_conf = confidence[valid_bool]
        if valid_conf.numel() == 0:
            valid_conf = confidence.new_zeros(1)
        stats = {
            "confidence": confidence,
            "f_mean": f_mean,
            "residual_update": residual_update,
            "confidence_mean": valid_conf.detach().mean(),
            "confidence_std": valid_conf.detach().std(unbiased=False),
            "confidence_min": valid_conf.detach().min(),
            "confidence_max": valid_conf.detach().max(),
            "ptfa_mean_norm": f_mean.detach().norm(dim=-1).mean(),
            "ptfa_residual_update_norm": residual_update.detach().norm(dim=-1).mean(),
            "ptfa_delta_norm": (self.gamma * residual_update.detach()).norm(dim=-1).mean(),
            "ptfa_delta_abs_mean": (f_out.detach() - f_mean.detach()).abs().mean(),
        }
        return f_out, stats


class MeanPriorResidualGate(nn.Module):
    """Mean-anchored residual view correction.

    This gate never replaces the valid-view mean. It only applies a bounded
    residual correction around that mean:
    f_out = f_mean + alpha * mean_valid(scale_v * (f_v - f_mean)).
    With zero-initialized final projection, scale_v starts at 0 and alpha=0 at
    epoch 0, so the initial path is exactly the E8-cED valid-view mean.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        scale_max: float = 0.5,
        norm: str = "layernorm",
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if scale_max < 0:
            raise ValueError(f"scale_max must be non-negative, got {scale_max}")
        if norm not in {"none", "layernorm"}:
            raise ValueError(f"norm must be 'none' or 'layernorm', got {norm}")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.scale_max = float(scale_max)
        self.norm_name = str(norm)
        self.input_norm = nn.LayerNorm(self.input_dim) if self.norm_name == "layernorm" else None
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def _valid_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weights = valid.unsqueeze(-1).to(dtype=x.dtype, device=x.device)
        denom = weights.sum(dim=2).clamp_min(1.0)
        return (x * weights).sum(dim=2) / denom

    def forward(
        self,
        f_view: torch.Tensor,
        gate_input: torch.Tensor,
        valid: torch.Tensor,
        alpha: float | torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Apply bounded residual correction around valid-view consensus.

        Args:
            f_view: [B,N,V,C] per-view evidence.
            gate_input: [B,N,V,D] compact geom + evidence stats.
            valid: [B,N,V] boolean view mask.
            alpha: scalar residual strength after warmup.

        Returns:
            f_out: [B,N,C]
            stats: tensors for scale/residual diagnostics and anchor loss.
        """
        if f_view.dim() != 4:
            raise ValueError(f"f_view must be [B,N,V,C], got {tuple(f_view.shape)}")
        if gate_input.dim() != 4 or gate_input.shape[-1] != self.input_dim:
            raise ValueError(
                f"gate_input must be [B,N,V,{self.input_dim}], got {tuple(gate_input.shape)}"
            )
        if valid.dim() != 3 or valid.shape != f_view.shape[:3]:
            raise ValueError(f"valid shape {tuple(valid.shape)} incompatible with {tuple(f_view.shape)}")
        if gate_input.shape[:3] != f_view.shape[:3]:
            raise ValueError(
                f"gate_input shape {tuple(gate_input.shape)} incompatible with "
                f"f_view {tuple(f_view.shape)}"
            )

        valid_bool = valid.to(device=f_view.device, dtype=torch.bool)
        valid_float = valid_bool.to(dtype=f_view.dtype)
        x = gate_input.to(device=f_view.device, dtype=f_view.dtype)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if self.input_norm is not None:
            x = self.input_norm(x)

        raw = self.net(x).squeeze(-1)
        residual_scale = self.scale_max * torch.tanh(
            torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        )
        residual_scale = residual_scale * valid_float

        f_mean = self._valid_mean(f_view, valid_bool)
        residual = f_view - f_mean.unsqueeze(2)
        delta = self._valid_mean(residual_scale.unsqueeze(-1) * residual, valid_bool)

        alpha_t = torch.as_tensor(alpha, device=f_view.device, dtype=f_view.dtype)
        f_out = f_mean + alpha_t * delta
        f_out = torch.nan_to_num(f_out, nan=0.0, posinf=0.0, neginf=0.0)
        anchor_loss = (f_out - f_mean.detach()).square().mean()

        valid_scale = residual_scale[valid_bool]
        if valid_scale.numel() == 0:
            valid_scale = residual_scale.new_zeros(1)

        stats = {
            "residual_scale": residual_scale,
            "f_mean": f_mean,
            "delta": delta,
            "alpha": alpha_t.detach(),
            "anchor_loss": anchor_loss,
            "residual_scale_mean": valid_scale.detach().mean(),
            "residual_scale_std": valid_scale.detach().std(unbiased=False),
            "residual_scale_min": valid_scale.detach().min(),
            "residual_scale_max": valid_scale.detach().max(),
            "ptfa_mean_norm": f_mean.detach().norm(dim=-1).mean(),
            "ptfa_delta_norm": delta.detach().norm(dim=-1).mean(),
            "ptfa_alpha_delta_norm": (alpha_t.detach() * delta.detach()).norm(dim=-1).mean(),
            "ptfa_delta_abs_mean": (f_out.detach() - f_mean.detach()).abs().mean(),
        }
        return f_out, stats


class TransportConsensusAdapter(nn.Module):
    """Zero-initialized adapter after valid-view transport consensus.

    This module keeps valid-view mean as the anchor and consumes per-channel
    disagreement statistics across views. It does not learn view weights.
    """

    def __init__(
        self,
        feature_dim: int,
        geom_dim: int = 5,
        hidden_dim: int = 128,
        epsilon: float = 0.1,
        include_dev_norm: bool = False,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if geom_dim <= 0:
            raise ValueError(f"geom_dim must be positive, got {geom_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {epsilon}")

        self.feature_dim = int(feature_dim)
        self.geom_dim = int(geom_dim)
        self.hidden_dim = int(hidden_dim)
        self.epsilon = float(epsilon)
        self.include_dev_norm = bool(include_dev_norm)
        stats_dim = self.feature_dim * 3 if self.include_dev_norm else self.feature_dim * 2
        in_dim = self.feature_dim + stats_dim + self.geom_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.feature_dim),
        )
        if zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def _valid_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weights = valid.unsqueeze(-1).to(dtype=x.dtype, device=x.device)
        denom = weights.sum(dim=2).clamp_min(1.0)
        return (x * weights).sum(dim=2) / denom

    def forward(
        self,
        f_view: torch.Tensor,
        valid: torch.Tensor,
        geom_summary: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Adapt valid-view mean using view disagreement statistics.

        Args:
            f_view: [B,N,V,C] per-view PTFA evidence.
            valid: [B,N,V] boolean view mask.
            geom_summary: [B,N,G] query-level geometry summary.

        Returns:
            f_out: [B,N,C]
            stats: diagnostic tensors.
        """
        if f_view.dim() != 4:
            raise ValueError(f"f_view must be [B,N,V,C], got {tuple(f_view.shape)}")
        if valid.dim() != 3 or valid.shape != f_view.shape[:3]:
            raise ValueError(f"valid shape {tuple(valid.shape)} incompatible with {tuple(f_view.shape)}")
        if geom_summary.dim() != 3 or geom_summary.shape[:2] != f_view.shape[:2]:
            raise ValueError(
                f"geom_summary shape {tuple(geom_summary.shape)} incompatible with "
                f"f_view {tuple(f_view.shape)}"
            )
        if f_view.shape[-1] != self.feature_dim:
            raise ValueError(f"expected feature_dim={self.feature_dim}, got {f_view.shape[-1]}")
        if geom_summary.shape[-1] != self.geom_dim:
            raise ValueError(f"expected geom_dim={self.geom_dim}, got {geom_summary.shape[-1]}")

        valid_bool = valid.to(device=f_view.device, dtype=torch.bool)
        f_mean = self._valid_mean(f_view, valid_bool)
        f_dev = f_view - f_mean.unsqueeze(2)
        dev_abs_mean = self._valid_mean(f_dev.abs(), valid_bool)
        dev_sq_mean = self._valid_mean(f_dev.square(), valid_bool)
        parts = [f_mean, dev_abs_mean, dev_sq_mean]
        if self.include_dev_norm:
            dev_norm = f_dev.norm(dim=-1, keepdim=True)
            dev_norm_mean = self._valid_mean(dev_norm, valid_bool)
            parts.append(dev_norm_mean.expand_as(f_mean))
        parts.append(geom_summary.to(device=f_view.device, dtype=f_view.dtype))
        adapter_input = torch.cat(parts, dim=-1)
        adapter_input = torch.nan_to_num(adapter_input, nan=0.0, posinf=0.0, neginf=0.0)
        delta = self.net(adapter_input)
        f_out = f_mean + self.epsilon * delta
        f_out = torch.nan_to_num(f_out, nan=0.0, posinf=0.0, neginf=0.0)
        stats = {
            "f_mean": f_mean,
            "delta": delta,
            "epsilon": torch.as_tensor(self.epsilon, device=f_view.device, dtype=f_view.dtype),
            "dev_abs_mean": dev_abs_mean,
            "dev_sq_mean": dev_sq_mean,
            "ptfa_mean_norm": f_mean.detach().norm(dim=-1).mean(),
            "dev_abs_mean_norm": dev_abs_mean.detach().norm(dim=-1).mean(),
            "dev_sq_mean_norm": dev_sq_mean.detach().norm(dim=-1).mean(),
            "adapter_delta_norm": delta.detach().norm(dim=-1).mean(),
            "adapter_epsilon_delta_norm": (self.epsilon * delta.detach()).norm(dim=-1).mean(),
            "adapter_delta_abs_mean": (f_out.detach() - f_mean.detach()).abs().mean(),
        }
        return f_out, stats


class CanonicalReliabilityAggregator(nn.Module):
    """Interpretable canonical-view aggregation for quotient transport evidence.

    The canonical feature is a reliability-weighted representative of all valid
    per-view features for the same query. Reliability is computed from physical
    geometry only, not from feature content:

      high boundary margin, low corrected depth, and low sigma are preferred.

    A zero-initialized residual adapter can then make a small correction around
    the canonical feature without changing initial behavior.
    """

    def __init__(
        self,
        feature_dim: int,
        geom_dim: int = 8,
        hidden_dim: int = 128,
        temperature: float = 1.0,
        residual_epsilon: float = 0.1,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if geom_dim != 8:
            raise ValueError(f"CanonicalReliabilityAggregator expects compact geom_dim=8, got {geom_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if residual_epsilon < 0:
            raise ValueError(f"residual_epsilon must be non-negative, got {residual_epsilon}")

        self.feature_dim = int(feature_dim)
        self.geom_dim = int(geom_dim)
        self.hidden_dim = int(hidden_dim)
        self.temperature = float(temperature)
        self.residual_epsilon = float(residual_epsilon)
        self.residual = nn.Sequential(
            nn.Linear(self.feature_dim + self.geom_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.feature_dim),
        )
        if zero_init:
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

    @staticmethod
    def _safe_softmax(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        valid_bool = valid.to(dtype=torch.bool, device=logits.device)
        has_valid = valid_bool.any(dim=-1, keepdim=True)
        masked = logits.masked_fill(~valid_bool, -1.0e4)
        safe = torch.where(has_valid, masked, torch.zeros_like(masked))
        weights = torch.softmax(safe, dim=-1)
        weights = torch.where(has_valid, weights, torch.zeros_like(weights))
        weights = weights * valid_bool.to(dtype=weights.dtype)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        weights = torch.where(has_valid, weights / denom, weights)
        return torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(
        self,
        f_view: torch.Tensor,
        geom_view: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Build a canonical feature from valid per-view evidence.

        Args:
            f_view: [B,N,V,C] per-view PTFA evidence.
            geom_view: [B,N,V,8] compact geometry features:
                grid_x, grid_y, boundary_margin, depth_eff_norm, sigma_norm,
                sin(angle), cos(angle), valid_float.
            valid: [B,N,V] boolean view mask.

        Returns:
            f_out: [B,N,C] canonical feature plus residual correction.
            weights: [B,N,V] interpretable reliability weights.
            stats: diagnostic tensors.
        """
        if f_view.dim() != 4:
            raise ValueError(f"f_view must be [B,N,V,C], got {tuple(f_view.shape)}")
        if geom_view.dim() != 4 or geom_view.shape[-1] != self.geom_dim:
            raise ValueError(
                f"geom_view must be [B,N,V,{self.geom_dim}], got {tuple(geom_view.shape)}"
            )
        if valid.dim() != 3 or valid.shape != f_view.shape[:3]:
            raise ValueError(f"valid shape {tuple(valid.shape)} incompatible with {tuple(f_view.shape)}")
        if geom_view.shape[:3] != f_view.shape[:3]:
            raise ValueError(
                f"geom_view shape {tuple(geom_view.shape)} incompatible with "
                f"f_view {tuple(f_view.shape)}"
            )
        if f_view.shape[-1] != self.feature_dim:
            raise ValueError(f"expected feature_dim={self.feature_dim}, got {f_view.shape[-1]}")

        geom = geom_view.to(device=f_view.device, dtype=f_view.dtype)
        geom = torch.nan_to_num(geom, nan=0.0, posinf=0.0, neginf=0.0)
        grid_x = geom[..., 0]
        grid_y = geom[..., 1]
        boundary_margin = geom[..., 2].clamp(0.0, 1.0)
        depth_eff_norm = geom[..., 3].clamp(0.0, 1.0)
        sigma_norm = geom[..., 4].clamp(0.0, 1.0)
        center_distance = torch.sqrt((grid_x.square() + grid_y.square()).clamp_min(0.0))

        reliability = (
            boundary_margin
            + (1.0 - depth_eff_norm)
            + (1.0 - sigma_norm)
            - 0.25 * center_distance
        )
        weights = self._safe_softmax(reliability / self.temperature, valid)
        f_canonical = (f_view * weights.unsqueeze(-1)).sum(dim=2)
        geom_canonical = (geom * weights.unsqueeze(-1)).sum(dim=2)
        adapter_input = torch.cat([f_canonical, geom_canonical], dim=-1)
        adapter_input = torch.nan_to_num(adapter_input, nan=0.0, posinf=0.0, neginf=0.0)
        delta = self.residual(adapter_input)
        f_out = f_canonical + self.residual_epsilon * delta
        f_out = torch.nan_to_num(f_out, nan=0.0, posinf=0.0, neginf=0.0)

        valid_bool = valid.to(dtype=torch.bool, device=f_view.device)
        valid_weights = weights[valid_bool]
        if valid_weights.numel() == 0:
            valid_weights = weights.new_zeros(1)
        entropy = -(weights * torch.log(weights.clamp_min(1.0e-12))).sum(dim=-1)
        stats = {
            "weights": weights,
            "f_canonical": f_canonical,
            "delta": delta,
            "reliability": reliability,
            "weight_mean": valid_weights.detach().mean(),
            "weight_std": valid_weights.detach().std(unbiased=False),
            "weight_min": valid_weights.detach().min(),
            "weight_max": valid_weights.detach().max(),
            "weight_entropy": entropy.detach().mean(),
            "canonical_norm": f_canonical.detach().norm(dim=-1).mean(),
            "adapter_delta_norm": delta.detach().norm(dim=-1).mean(),
            "adapter_epsilon_delta_norm": (self.residual_epsilon * delta.detach()).norm(dim=-1).mean(),
        }
        return f_out, weights, stats
