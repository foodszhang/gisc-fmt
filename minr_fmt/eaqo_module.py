"""EAQO training module for SSQ-FMT."""

from __future__ import annotations

import torch

from .module import TrainingLightningModule
from .network.eaqo import prediction_ambiguity


class EAQOTrainingLightningModule(TrainingLightningModule):
    """Training-only EAQO wrapper around the existing SSQ forward path."""

    def _call_ssq_model(self, batch, return_diagnostics: bool = False):
        cfg = getattr(getattr(self.cfg.model, "ssq_fmt", None), "eaqo", None)
        enabled = bool(getattr(cfg, "enabled", False)) and self.training
        # Do not force return_diagnostics here.  In Phase A, SSQFMT uses the
        # formal shared path only when return_diagnostics is false.
        out = super()._call_ssq_model(batch, return_diagnostics=return_diagnostics)
        if not enabled:
            return out
        density = out["density"]
        aux = out.get("aux_outputs", {})
        diag = out.get("diagnostics", aux)
        support = aux.get("measurement_supported")
        if not torch.is_tensor(support):
            support = torch.ones(density.shape[:2], dtype=torch.bool, device=density.device)
        if support.dim() == 3 and support.shape[-1] == 1:
            support = support.squeeze(-1)
        support = support.to(device=density.device)
        score_pred = prediction_ambiguity(density)
        score_view = self._eaqo_view_ambiguity(diag, density)
        score = float(getattr(cfg, "alpha_pred", 0.0)) * score_pred + float(getattr(cfg, "beta_view", 1.0)) * score_view
        score = self._eaqo_normalize(score, support, float(getattr(cfg, "eps", 1.0e-6)))
        if bool(getattr(cfg, "detach_score", True)):
            score = score.detach()
        weight = (1.0 + float(getattr(cfg, "lambda", 0.25)) * score).clamp(
            float(getattr(cfg, "weight_clip_min", 1.0)),
            float(getattr(cfg, "weight_clip_max", 3.0)),
        )
        aux["measurement_supported_binary"] = support.detach()
        aux["measurement_supported"] = support.to(density.dtype) * weight.to(density.dtype)
        aux["eaqo_score"] = score
        aux["eaqo_loss_weight"] = weight
        valid = support.bool()
        denom = valid.float().sum().clamp_min(1.0)
        stats = {
            "score_mean": (score * valid.float()).sum() / denom,
            "weight_mean": (weight * valid.float()).sum() / denom,
            "weight_max": weight[valid].max() if valid.any() else weight.new_zeros(()),
            "pred_ambiguity_mean": (score_pred * valid.float()).sum() / denom,
            "view_ambiguity_mean": (score_view * valid.float()).sum() / denom,
        }
        for key, value in stats.items():
            aux[f"eaqo/{key}"] = value.detach()
            self.log(f"train_eaqo/{key}", value.detach(), on_step=False, on_epoch=True, sync_dist=True)
        out["aux_outputs"] = aux
        return out

    @staticmethod
    def _eaqo_view_ambiguity(diag, density):
        if not isinstance(diag, dict):
            return density.new_zeros(density.shape[:2])
        evidence = diag.get("per_view_evidence")
        valid = diag.get("query_view_valid")
        if torch.is_tensor(evidence) and evidence.dim() == 3:
            strength = evidence.float().abs()
        else:
            feat = diag.get("sample_features")
            a = diag.get("A")
            if not (torch.is_tensor(feat) and torch.is_tensor(a) and feat.dim() == 5 and a.dim() == 4):
                return density.new_zeros(density.shape[:2])
            strength = (feat.float() * a.float()[..., None]).sum(dim=3).norm(dim=-1)
        if not torch.is_tensor(valid) or valid.shape != strength.shape:
            valid = torch.ones_like(strength, dtype=torch.bool)
        vf = valid.to(strength.device).float()
        count = vf.sum(dim=1).clamp_min(1.0)
        mean = (strength * vf).sum(dim=1) / count
        var = ((strength - mean[:, None]).square() * vf).sum(dim=1) / count
        score = var / mean.abs().clamp_min(1.0e-6)
        score = torch.where(vf.sum(dim=1) >= 2.0, score, torch.zeros_like(score))
        return torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _eaqo_normalize(score, support, eps):
        valid = support.bool()
        out = torch.zeros_like(score)
        for b in range(score.shape[0]):
            vals = score[b][valid[b]]
            if vals.numel():
                out[b] = (score[b] - vals.min()) / (vals.max() - vals.min()).clamp_min(eps)
        return out.clamp(0.0, 1.0)
