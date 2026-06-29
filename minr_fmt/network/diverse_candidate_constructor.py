"""Permutation-equivariant association of independent view proposals."""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiverseCandidateConstructor(nn.Module):
    def __init__(
        self,
        mmax: int = 5,
        score_threshold: float = 0.05,
        sigma_nms_mm: float = 4.0,
        sigma_assoc_mm: float = 5.0,
        sigma_min_mm: float = 1.0,
        sigma_max_mm: float = 8.0,
        beta_descriptor: float = 0.5,
        beta_evidence: float = 0.25,
        refinement_steps: int = 2,
    ) -> None:
        super().__init__()
        self.mmax = int(mmax)
        self.score_threshold = float(score_threshold)
        self.sigma_nms_mm = float(sigma_nms_mm)
        self.sigma_assoc_mm = float(sigma_assoc_mm)
        self.sigma_min_mm = float(sigma_min_mm)
        self.sigma_max_mm = float(sigma_max_mm)
        self.beta_descriptor = nn.Parameter(torch.tensor(float(beta_descriptor)))
        self.beta_evidence = nn.Parameter(torch.tensor(float(beta_evidence)))
        self.refinement_steps = int(refinement_steps)

    def _initialize(
        self, points: torch.Tensor, scores: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, p, _ = points.shape
        anchors = points.new_zeros((b, self.mmax, 3))
        anchor_valid = torch.zeros((b, self.mmax), dtype=torch.bool, device=points.device)
        diverse = scores.masked_fill(~valid, -1.0)
        for m in range(self.mmax):
            value, index = diverse.max(dim=1)
            chosen = value > self.score_threshold
            center = points.gather(1, index[:, None, None].expand(-1, 1, 3)).squeeze(1)
            anchors[:, m] = center
            anchor_valid[:, m] = chosen
            distance2 = (points - center[:, None]).square().sum(dim=-1)
            suppression = 1.0 - torch.exp(
                -distance2 / (2.0 * self.sigma_nms_mm**2)
            )
            diverse = diverse * torch.where(chosen[:, None], suppression, torch.ones_like(diverse))
            diverse = diverse.masked_fill(~valid, -1.0)
        return anchors, anchor_valid

    def forward(self, proposals: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        points_v = proposals["proposal_points_mm"]
        scores_v = proposals["proposal_scores"]
        descriptors_v = proposals["proposal_descriptors"]
        valid_v = proposals["proposal_valid_mask"]
        b, v, k, d = descriptors_v.shape
        points = points_v.reshape(b, v * k, 3)
        scores = scores_v.reshape(b, v * k)
        descriptors = descriptors_v.reshape(b, v * k, d)
        valid = valid_v.reshape(b, v * k)
        anchors, anchor_valid = self._initialize(points, scores, valid)
        anchor_desc = descriptors.new_zeros((b, self.mmax, d))
        assignment = scores.new_zeros((b, self.mmax, v * k))
        covariance = torch.eye(3, device=points.device, dtype=points.dtype)[None, None].repeat(
            b, self.mmax, 1, 1
        ) * self.sigma_min_mm**2
        for _ in range(self.refinement_steps):
            distance2 = (anchors[:, :, None] - points[:, None]).square().sum(dim=-1)
            cosine = torch.einsum(
                "bmd,bpd->bmp", F.normalize(anchor_desc, dim=-1), descriptors
            )
            compatibility = (
                -distance2 / (2.0 * self.sigma_assoc_mm**2)
                + self.beta_descriptor * cosine
                + self.beta_evidence * scores.clamp_min(1.0e-8).log()[:, None]
            )
            compatibility = compatibility.masked_fill(~anchor_valid[:, :, None], -1.0e4)
            assignment = torch.softmax(compatibility, dim=1)
            assignment = assignment * valid[:, None].to(assignment.dtype)
            weight = assignment * scores[:, None]
            denom = weight.sum(dim=-1).clamp_min(1.0e-8)
            updated = torch.einsum("bmp,bpd->bmd", weight, points) / denom[..., None]
            anchors = torch.where(anchor_valid[..., None], updated, anchors)
            delta = points[:, None] - anchors[:, :, None]
            covariance = torch.einsum("bmp,bmpi,bmpj->bmij", weight, delta, delta)
            covariance = covariance / denom[..., None, None]
            eye = torch.eye(3, device=points.device, dtype=points.dtype)
            covariance = covariance + self.sigma_min_mm**2 * eye
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance.float())
            lower = self.sigma_min_mm**2
            upper = self.sigma_max_mm**2
            clamped = eigenvalues.clamp(lower, upper).to(covariance.dtype)
            covariance = torch.einsum(
                "bmik,bmk,bmjk->bmij",
                eigenvectors.to(covariance.dtype),
                clamped,
                eigenvectors.to(covariance.dtype),
            )
            anchor_desc = torch.einsum("bmp,bpd->bmd", weight, descriptors) / denom[..., None]
            anchor_desc = F.normalize(anchor_desc, dim=-1)
        assignment_v = assignment.reshape(b, self.mmax, v, k)
        support = (assignment_v * scores_v[:, None]).sum(dim=-1)
        confidence = 1.0 - (1.0 - support.clamp(0.0, 1.0)).prod(dim=-1)
        confidence = confidence * anchor_valid.to(confidence.dtype)
        covariance_eigenvalues = torch.linalg.eigvalsh(covariance.float()).to(points.dtype)
        return {
            "candidate_centers_mm": anchors,
            "candidate_covariances_mm": covariance,
            "candidate_scores": confidence,
            "candidate_valid_mask": anchor_valid,
            "candidate_view_support": support,
            "proposal_assignment": assignment_v,
            "candidate_descriptors": anchor_desc,
            "candidate_covariance_eigenvalues": covariance_eigenvalues,
            "covariance_lower_bound_hit": covariance_eigenvalues <= self.sigma_min_mm**2 * 1.001,
            "covariance_upper_bound_hit": covariance_eigenvalues >= self.sigma_max_mm**2 * 0.999,
        }

    @staticmethod
    def supervision_losses(
        candidates: dict[str, torch.Tensor],
        gt_centers_mm: torch.Tensor,
        gt_valid_mask: torch.Tensor,
        gt_covariances_mm: torch.Tensor | None = None,
        duplicate_distance_mm: float = 4.0,
    ) -> dict[str, torch.Tensor]:
        """Hungarian-style set supervision used only when GT is explicitly supplied."""
        centers = candidates["candidate_centers_mm"]
        scores = candidates["candidate_scores"].clamp(1.0e-6, 1.0 - 1.0e-6)
        valid = candidates["candidate_valid_mask"]
        center_loss = centers.sum() * 0.0
        existence_loss = centers.sum() * 0.0
        coverage_loss = centers.sum() * 0.0
        covariance_loss = centers.sum() * 0.0
        match_count = 0
        existence_targets = torch.zeros_like(scores)
        for batch_index in range(centers.shape[0]):
            candidate_index = torch.where(valid[batch_index])[0]
            component_index = torch.where(gt_valid_mask[batch_index])[0]
            count = min(candidate_index.numel(), component_index.numel())
            if count:
                cost = torch.cdist(
                    centers[batch_index, candidate_index].detach().float(),
                    gt_centers_mm[batch_index, component_index].detach().float(),
                    p=1,
                )
                permutations = torch.tensor(
                    list(itertools.permutations(range(candidate_index.numel()), count)),
                    device=centers.device,
                )
                columns = torch.arange(count, device=centers.device)
                selected = permutations[cost[permutations, columns].sum(dim=1).argmin()]
                matched_candidates = candidate_index[selected]
                matched_components = component_index[:count]
                center_loss = center_loss + torch.nn.functional.l1_loss(
                    centers[batch_index, matched_candidates],
                    gt_centers_mm[batch_index, matched_components].to(centers.dtype),
                    reduction="mean",
                )
                if gt_covariances_mm is not None:
                    predicted_eigen = torch.linalg.eigvalsh(
                        candidates["candidate_covariances_mm"][
                            batch_index, matched_candidates
                        ].float()
                    ).clamp_min(1.0e-8)
                    target_eigen = torch.linalg.eigvalsh(
                        gt_covariances_mm[batch_index, matched_components].float()
                    ).clamp_min(1.0e-8)
                    covariance_loss = covariance_loss + torch.nn.functional.l1_loss(
                        predicted_eigen.log(), target_eigen.log(), reduction="sum"
                    )
                existence_targets[batch_index, matched_candidates] = 1.0
                match_count += count
            if component_index.numel():
                distance2 = torch.cdist(
                    gt_centers_mm[batch_index, component_index].float(),
                    centers[batch_index].float(),
                ).square()
                distance2 = distance2.masked_fill(~valid[batch_index][None], float("inf"))
                finite = distance2.amin(dim=-1)
                finite = torch.where(torch.isfinite(finite), finite, torch.full_like(finite, 1e3))
                coverage_loss = coverage_loss + finite.mean() / max(
                    duplicate_distance_mm**2, 1.0e-6
                )
        score_float = scores.float().clamp(1.0e-6, 1.0 - 1.0e-6)
        target_float = existence_targets.float()
        existence_loss = -(
            target_float * score_float.log()
            + (1.0 - target_float) * (1.0 - score_float).log()
        ).mean()
        center_loss = center_loss / max(match_count, 1)
        covariance_loss = covariance_loss / max(match_count, 1)
        coverage_loss = coverage_loss / max(centers.shape[0], 1)
        delta = centers[:, :, None] - centers[:, None, :]
        overlap = torch.exp(
            -0.5 * delta.square().sum(dim=-1) / max(duplicate_distance_mm**2, 1.0e-6)
        )
        pair_valid = valid[:, :, None] & valid[:, None, :]
        upper = torch.triu(torch.ones_like(pair_valid), diagonal=1).bool() & pair_valid
        duplicate_loss = (
            (overlap * scores[:, :, None] * scores[:, None])[upper].mean()
            if upper.any()
            else centers.sum() * 0.0
        )
        return {
            "candidate_center_loss": center_loss,
            "candidate_covariance_loss": covariance_loss,
            "candidate_existence_loss": existence_loss,
            "candidate_coverage_loss": coverage_loss,
            "candidate_duplicate_loss": duplicate_loss,
        }
