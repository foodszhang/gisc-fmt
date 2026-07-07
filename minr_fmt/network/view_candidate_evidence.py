"""Independent per-view evidence for 3-D source proposals."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewCandidateEvidence(nn.Module):
    """Predict proposal evidence, descriptors and offsets without mixing views."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 96,
        descriptor_dim: int = 32,
        delta_max_mm: float = 3.0,
        topk_per_view: int = 8,
        nms_radius_mm: float = 3.0,
        exact_nms_threshold: int = 4096,
        pre_nms_topk: int = 2048,
        pre_nms_factor: int = 64,
        max_nms_candidates: int = 4096,
    ) -> None:
        super().__init__()
        self.delta_max_mm = float(delta_max_mm)
        self.topk_per_view = int(topk_per_view)
        self.nms_radius_mm = float(nms_radius_mm)
        self.exact_nms_threshold = int(exact_nms_threshold)
        self.pre_nms_topk = int(pre_nms_topk)
        self.pre_nms_factor = int(pre_nms_factor)
        self.max_nms_candidates = int(max_nms_candidates)
        self.position = nn.Sequential(nn.Linear(3, hidden_dim), nn.SiLU())
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.evidence = nn.Linear(hidden_dim, 1)
        self.descriptor = nn.Linear(hidden_dim, descriptor_dim)
        self.offset = nn.Linear(hidden_dim, 3)

    @staticmethod
    def _local_maxima_exact(
        scores: torch.Tensor,
        points: torch.Tensor,
        radius: float,
    ) -> torch.Tensor:
        distance = torch.cdist(points.float(), points.float()).to(scores.dtype)
        neighbor = distance <= radius
        # A deterministic index tie-break prevents a plateau from producing duplicates.
        index = torch.arange(scores.shape[-1], device=scores.device)
        better = scores[..., None, :] > scores[..., :, None]
        tied_later = (scores[..., None, :] == scores[..., :, None]) & (
            index[None, None, None, :] < index[None, None, :, None]
        )
        return ~((better | tied_later) & neighbor[:, None]).any(dim=-1)

    @classmethod
    def _local_maxima_topk_prefilter(
        cls,
        scores: torch.Tensor,
        points: torch.Tensor,
        radius: float,
        valid_mask: torch.Tensor,
        *,
        topk_per_view: int,
        pre_nms_topk: int,
        pre_nms_factor: int,
        max_nms_candidates: int,
    ) -> torch.Tensor:
        """Approximate local NMS with bounded pairwise memory for sampled large-N queries."""
        b, v, n = scores.shape
        local = torch.zeros_like(valid_mask, dtype=torch.bool)
        candidate_budget = max(int(topk_per_view) * int(pre_nms_factor), int(pre_nms_topk))
        candidate_budget = min(max(candidate_budget, int(topk_per_view)), int(max_nms_candidates))
        candidate_budget = max(candidate_budget, 1)
        masked_scores = scores.masked_fill(~valid_mask, -1.0)
        for batch_index in range(b):
            for view_index in range(v):
                valid_count = int(valid_mask[batch_index, view_index].sum().item())
                if valid_count <= 0:
                    continue
                k = min(n, valid_count, candidate_budget)
                _, selected = masked_scores[batch_index, view_index].topk(k, dim=-1)
                selected_scores = scores[batch_index, view_index, selected][None, None]
                selected_points = points[batch_index, selected][None]
                selected_local = cls._local_maxima_exact(
                    selected_scores,
                    selected_points,
                    radius,
                ).reshape(-1)
                local[batch_index, view_index, selected] = selected_local
        return local

    @staticmethod
    def _grid_local_maxima(scores: torch.Tensor, grid_shape: tuple[int, int, int]) -> torch.Tensor:
        """O(N) local NMS for a structured fixed hypothesis grid."""
        b, v, n = scores.shape
        if n != grid_shape[0] * grid_shape[1] * grid_shape[2]:
            raise ValueError(f"grid shape {grid_shape} does not match {n} points")
        # CUDA max_pool3d is not implemented consistently for bf16.  NMS is a
        # ranking operation, so perform it in fp32 and return a boolean mask.
        volume = scores.float().reshape(b * v, 1, *grid_shape)
        pooled = F.max_pool3d(volume, kernel_size=3, stride=1, padding=1)
        # Stable index tie break for flat maxima.
        maxima = volume == pooled
        flat = maxima.reshape(b, v, n)
        index = torch.arange(n, device=scores.device)
        adjusted = scores.float() - index.float()[None, None] * torch.finfo(torch.float32).eps
        adjusted_volume = adjusted.reshape(b * v, 1, *grid_shape)
        adjusted_pool = F.max_pool3d(adjusted_volume, 3, stride=1, padding=1)
        return flat & (adjusted_volume == adjusted_pool).reshape(b, v, n)

    def forward(
        self,
        query_features: torch.Tensor,
        geometry_features: torch.Tensor,
        points_mm: torch.Tensor,
        valid_mask: torch.Tensor,
        grid_shape: tuple[int, int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Inputs are [B,V,N,D], [B,V,N,4], [B,N,3], and [B,V,N]."""
        b, v, n, _ = query_features.shape
        pos = self.position(points_mm)[:, None].expand(-1, v, -1, -1)
        hidden = self.trunk(torch.cat([query_features, geometry_features, pos], dim=-1))
        evidence = torch.sigmoid(self.evidence(hidden).squeeze(-1))
        evidence = torch.where(valid_mask, evidence, torch.zeros_like(evidence))
        descriptor = F.normalize(self.descriptor(hidden), dim=-1)
        offset = self.delta_max_mm * torch.tanh(self.offset(hidden))
        proposals = points_mm[:, None] + offset

        if grid_shape is not None:
            local = self._grid_local_maxima(evidence, grid_shape)
        elif n <= self.exact_nms_threshold:
            local = self._local_maxima_exact(evidence, points_mm, self.nms_radius_mm)
        else:
            local = self._local_maxima_topk_prefilter(
                evidence,
                points_mm,
                self.nms_radius_mm,
                valid_mask,
                topk_per_view=self.topk_per_view,
                pre_nms_topk=self.pre_nms_topk,
                pre_nms_factor=self.pre_nms_factor,
                max_nms_candidates=self.max_nms_candidates,
            )
        local = local & valid_mask
        masked = evidence.masked_fill(~local, -1.0)
        k = min(self.topk_per_view, n)
        score, index = masked.topk(k, dim=-1)
        selected_valid = score >= 0.0
        gather3 = index[..., None].expand(-1, -1, -1, 3)
        gatherd = index[..., None].expand(-1, -1, -1, descriptor.shape[-1])
        return {
            "evidence": evidence,
            "descriptors": descriptor,
            "offsets_mm": offset,
            "proposal_points_mm": proposals.gather(2, gather3),
            "proposal_scores": score.clamp_min(0.0),
            "proposal_descriptors": descriptor.gather(2, gatherd),
            "proposal_valid_mask": selected_valid,
            "proposal_query_indices": index,
        }
