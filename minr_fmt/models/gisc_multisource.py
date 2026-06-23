"""Multi-source extensions for GISC-FMT.

This module keeps the existing PTFA/PCFS/scorer trunk unchanged and replaces only
source-conditioned query construction when requested by configuration.
"""

from __future__ import annotations

import math

import torch

from ..config_extractor import ConfigExtractor
from .minr_fmt import PointDensityNet


class MultiSourceGISCFMT(PointDensityNet):
    """GISC-FMT with permutation-invariant all-candidate source conditioning.

    The legacy top-2 cue remains available for backward-compatible experiments.
    Set ``model.source_instance_cue.aggregation=all_candidates_null`` to enable
    the multi-source cue defined here.
    """

    def __init__(self, config):
        super().__init__(config)
        model_cfg = ConfigExtractor._model_cfg(config)
        cue_cfg = model_cfg.get("source_instance_cue", {}) or {}
        self.source_cue_aggregation = str(cue_cfg.get("aggregation", "legacy_top2"))
        self.source_cue_null_logit = float(cue_cfg.get("null_logit", 0.0))
        self.source_cue_score_weight = float(cue_cfg.get("score_weight", 2.0))
        self.source_cue_min_scale_mm = float(cue_cfg.get("min_scale_mm", 1.0))
        self.source_cue_max_scale_mm = float(cue_cfg.get("max_scale_mm", 20.0))
        if self.source_cue_aggregation not in {"legacy_top2", "all_candidates_null"}:
            raise ValueError(
                "model.source_instance_cue.aggregation must be 'legacy_top2' or "
                f"'all_candidates_null', got {self.source_cue_aggregation!r}"
            )
        if self.source_cue_min_scale_mm <= 0:
            raise ValueError("model.source_instance_cue.min_scale_mm must be positive")
        if self.source_cue_max_scale_mm < self.source_cue_min_scale_mm:
            raise ValueError(
                "model.source_instance_cue.max_scale_mm must be >= min_scale_mm"
            )

    def _source_instance_cue(
        self,
        points_mm: torch.Tensor,
        source_hypotheses: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if self.source_cue_aggregation != "all_candidates_null":
            return super()._source_instance_cue(points_mm, source_hypotheses)

        batch_size, num_queries, _ = points_mm.shape
        if not self.source_instance_cue_enabled:
            return points_mm.new_zeros((batch_size, num_queries, 0))
        if source_hypotheses is None:
            self.last_source_cue_stats = {
                "num_detected_peaks": points_mm.new_zeros(()),
                "candidate_support_mean": points_mm.new_zeros(()),
                "null_ownership_mean": points_mm.new_ones(()),
                "ownership_entropy_mean": points_mm.new_zeros(()),
                "cue_abs_mean": points_mm.new_zeros(()),
            }
            self.last_source_nearest_dist = None
            return points_mm.new_zeros(
                (batch_size, num_queries, self.source_instance_cue_dim)
            )

        centers = source_hypotheses["centers"].to(
            device=points_mm.device, dtype=points_mm.dtype
        )
        scores = source_hypotheses["peak_scores"].to(
            device=points_mm.device, dtype=points_mm.dtype
        )
        scales = source_hypotheses["scales"].to(
            device=points_mm.device, dtype=points_mm.dtype
        )
        valid = source_hypotheses["valid"].to(device=points_mm.device) > 0.5

        if centers.ndim != 3 or centers.shape[-1] != 3:
            raise ValueError(
                "source_hypotheses['centers'] must have shape [B,M,3], got "
                f"{tuple(centers.shape)}"
            )
        if centers.shape[0] != batch_size:
            raise ValueError(
                "source hypothesis batch size differs from query batch size: "
                f"{centers.shape[0]} != {batch_size}"
            )

        fallback_scale = max(float(self.source_instance_tau_s_mm), self.source_cue_min_scale_mm)
        candidate_scale = torch.where(
            (scales > 0) & valid,
            scales,
            torch.full_like(scales, fallback_scale),
        ).clamp(self.source_cue_min_scale_mm, self.source_cue_max_scale_mm)

        rel_xyz = points_mm[:, :, None, :] - centers[:, None, :, :]
        dist_mm = torch.linalg.norm(rel_xyz, dim=-1)
        scaled_rel = rel_xyz / candidate_scale[:, None, :, None].clamp_min(1.0e-6)
        scaled_dist = dist_mm / candidate_scale[:, None, :].clamp_min(1.0e-6)

        candidate_logits = (
            self.source_cue_score_weight * scores[:, None, :]
            - 0.5 * scaled_dist.square()
        )
        candidate_logits = torch.nan_to_num(
            candidate_logits, nan=-1.0e9, posinf=30.0, neginf=-1.0e9
        ).clamp(-30.0, 30.0)
        candidate_logits = candidate_logits.masked_fill(~valid[:, None, :], -1.0e9)
        null_logits = candidate_logits.new_full(
            (batch_size, num_queries, 1), self.source_cue_null_logit
        )
        all_logits = torch.cat([null_logits, candidate_logits], dim=-1)
        all_prob = torch.softmax(all_logits, dim=-1)
        null_prob = all_prob[..., 0:1]
        ownership = all_prob[..., 1:]
        support = 1.0 - null_prob

        weighted_rel = (ownership[..., None] * scaled_rel).sum(dim=2)
        weighted_dist = (ownership * scaled_dist).sum(dim=2, keepdim=True)
        weighted_score = (ownership * scores[:, None, :]).sum(dim=2, keepdim=True)
        scale_norm = candidate_scale / max(fallback_scale, 1.0e-6)
        weighted_scale = (ownership * scale_norm[:, None, :]).sum(dim=2, keepdim=True)

        entropy = -(all_prob * torch.log(all_prob.clamp_min(1.0e-8))).sum(
            dim=-1, keepdim=True
        )
        valid_count = valid.float().sum(dim=-1, keepdim=True)
        entropy_denom = (valid_count + 1.0).clamp_min(2.0).log()[:, None, :]
        entropy = entropy / entropy_denom

        conditional_ownership = ownership / support.clamp_min(1.0e-8)
        concentration = support * conditional_ownership.square().sum(
            dim=-1, keepdim=True
        )
        valid_ratio = valid.float().mean(dim=-1, keepdim=True)[:, None, :].expand(
            -1, num_queries, -1
        )

        if self.source_instance_cue_dim == 10:
            cue = torch.cat(
                [
                    support,
                    weighted_rel,
                    weighted_dist,
                    weighted_score,
                    weighted_scale,
                    entropy,
                    concentration,
                    valid_ratio,
                ],
                dim=-1,
            )
        else:
            cue = torch.cat(
                [
                    support,
                    weighted_rel,
                    weighted_dist,
                    weighted_score,
                    entropy,
                    valid_ratio,
                ],
                dim=-1,
            )

        cue = torch.nan_to_num(cue, nan=0.0, posinf=0.0, neginf=0.0)
        masked_dist = dist_mm.masked_fill(~valid[:, None, :], math.inf)
        nearest_dist = masked_dist.amin(dim=-1)
        has_candidate = valid.any(dim=-1)[:, None]
        nearest_dist = torch.where(
            has_candidate,
            nearest_dist / max(fallback_scale, 1.0e-6),
            torch.zeros_like(nearest_dist),
        )
        self.last_source_nearest_dist = nearest_dist.detach()
        self.last_source_cue_stats = {
            "num_detected_peaks": valid.float().sum(dim=1).mean().detach(),
            "candidate_support_mean": support.detach().mean(),
            "null_ownership_mean": null_prob.detach().mean(),
            "ownership_entropy_mean": entropy.detach().mean(),
            "ownership_concentration_mean": concentration.detach().mean(),
            "nearest_peak_dist_mean": nearest_dist.detach().mean(),
            "cue_abs_mean": cue.detach().abs().mean(),
        }
        return cue


GISCFMT = MultiSourceGISCFMT
