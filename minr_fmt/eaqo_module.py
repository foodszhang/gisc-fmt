"""EAQO training module for SSQ-FMT."""

from __future__ import annotations

import torch

from .module import TrainingLightningModule
from .network.eaqo import (
    component_ambiguity,
    geometry_ambiguity,
    normalize_score,
    prediction_ambiguity,
    view_evidence_ambiguity,
)


class EAQOTrainingLightningModule(TrainingLightningModule):
    """Training-only EAQO wrapper around the existing SSQ forward path."""

    def __init__(self, cfg):
        super().__init__(cfg)
        model_cfg = cfg.get("model", {})
        model_name = str(model_cfg.get("name", "")).lower()
        if model_name != "ssq_fmt":
            raise RuntimeError("EAQOTrainingLightningModule is only valid for model=ssq_fmt")

    def _call_ssq_model(self, batch, return_diagnostics: bool = False):
        cfg = getattr(getattr(self.cfg.model, "ssq_fmt", None), "eaqo", None)
        enabled = bool(getattr(cfg, "enabled", False)) and self.training
        if enabled and return_diagnostics:
            raise RuntimeError("EAQO must not force return_diagnostics=True in Phase A training")
        out = super()._call_ssq_model(batch, return_diagnostics=return_diagnostics)
        if not enabled:
            return out
        density = out["density"]
        aux = out.get("aux_outputs", {})
        if not isinstance(aux, dict):
            aux = {}
        support = aux.get("measurement_supported")
        if not torch.is_tensor(support):
            support = torch.ones(density.shape[:2], dtype=torch.bool, device=density.device)
        if support.dim() == 3 and support.shape[-1] == 1:
            support = support.squeeze(-1)
        support_binary = support.to(device=density.device).bool()
        score_pred = prediction_ambiguity(density)
        eps = float(getattr(cfg, "eps", 1.0e-6))
        score_view = view_evidence_ambiguity(
            aux,
            density,
            eps=eps,
            require=bool(getattr(cfg, "require_view_evidence", False)),
        )
        score_geo = geometry_ambiguity(batch, density)
        score_comp = component_ambiguity(batch, density)
        score = (
            float(getattr(cfg, "alpha_pred", 0.0)) * score_pred
            + float(getattr(cfg, "beta_view", 1.0)) * score_view
            + float(getattr(cfg, "gamma_geo", 0.0)) * score_geo
            + float(getattr(cfg, "delta_comp", 0.0)) * score_comp
        )
        score = normalize_score(
            score,
            support_binary,
            str(getattr(cfg, "normalize_score", "batch_minmax")),
            eps,
        )
        if bool(getattr(cfg, "detach_score", True)):
            score = score.detach()
            if score.requires_grad:
                raise RuntimeError("EAQO detach_score=true but eaqo_score still requires grad")
        weight = (1.0 + float(getattr(cfg, "lambda", 0.25)) * score).clamp(
            float(getattr(cfg, "weight_clip_min", 1.0)),
            float(getattr(cfg, "weight_clip_max", 3.0)),
        )
        if weight.shape != density.squeeze(-1).shape:
            raise RuntimeError(
                "EAQO loss weight shape must match density.squeeze(-1): "
                f"{tuple(weight.shape)} vs {tuple(density.squeeze(-1).shape)}"
            )
        aux["measurement_supported_binary"] = support_binary.detach()
        aux["measurement_supported"] = support_binary
        aux["query_loss_weight"] = weight.to(dtype=density.dtype)
        aux["eaqo_score"] = score
        aux["eaqo_loss_weight"] = weight
        valid = support_binary
        denom = valid.float().sum().clamp_min(1.0)
        target = batch.get("point_densities")
        proxy_loss = None
        if torch.is_tensor(target) and target.shape == density.squeeze(-1).shape:
            target = target.to(device=density.device, dtype=density.dtype)
            proxy_loss = (density.squeeze(-1).detach() - target).abs()
        high_ratio, high_loss, low_loss = self._eaqo_high_low_stats(
            score, valid, target, proxy_loss
        )
        stats = {
            "score_mean": (score * valid.float()).sum() / denom,
            "score_std": score[valid].std(unbiased=False) if valid.any() else score.new_zeros(()),
            "weight_mean": (weight * valid.float()).sum() / denom,
            "weight_max": weight[valid].max() if valid.any() else weight.new_zeros(()),
            "pred_ambiguity_mean": (score_pred * valid.float()).sum() / denom,
            "view_ambiguity_mean": (score_view * valid.float()).sum() / denom,
            "geo_ambiguity_mean": (score_geo * valid.float()).sum() / denom,
            "comp_ambiguity_mean": (score_comp * valid.float()).sum() / denom,
            "high_ambiguity_query_foreground_ratio": high_ratio,
            "high_ambiguity_query_average_loss": high_loss,
            "low_ambiguity_query_average_loss": low_loss,
        }
        for key, value in stats.items():
            aux[f"eaqo/{key}"] = value.detach()
            self.log(
                f"train_eaqo/{key}",
                value.detach(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        out["aux_outputs"] = aux
        return out

    @staticmethod
    def _eaqo_high_low_stats(score, valid, target, proxy_loss):
        zero = score.new_zeros(())
        if not torch.is_tensor(proxy_loss) or not torch.is_tensor(target) or not valid.any():
            return zero, zero, zero
        high_fg = []
        high_losses = []
        low_losses = []
        for b in range(score.shape[0]):
            mask = valid[b]
            vals = score[b][mask]
            if vals.numel() == 0:
                continue
            k = max(1, int(vals.numel() * 0.25))
            valid_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
            high_local = torch.topk(vals, k=k, largest=True).indices
            low_local = torch.topk(vals, k=k, largest=False).indices
            high_idx = valid_idx[high_local]
            low_idx = valid_idx[low_local]
            high_fg.append((target[b, high_idx] > 0.0).float().mean())
            high_losses.append(proxy_loss[b, high_idx].mean())
            low_losses.append(proxy_loss[b, low_idx].mean())
        if not high_losses:
            return zero, zero, zero
        return (
            torch.stack(high_fg).mean(),
            torch.stack(high_losses).mean(),
            torch.stack(low_losses).mean(),
        )
