"""Measurement-derived candidate anchors and detector-plane priors."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from minr_fmt.network.ssq_geometry import sample_scalar_map


class MeasurementDerivedCandidateBuilder(nn.Module):
    def __init__(
        self,
        mmax: int = 5,
        threshold: float = 0.05,
        scale_min_mm: float = 1.0,
        scale_max_mm: float = 12.0,
        trunk_size_mm: tuple[float, float, float] = (38.0, 40.0, 20.8),
    ):
        super().__init__()
        self.mmax = int(mmax)
        self.threshold = float(threshold)
        self.scale_min_mm = float(scale_min_mm)
        self.scale_max_mm = float(scale_max_mm)
        self.trunk_size_mm = tuple(float(x) for x in trunk_size_mm)

    @torch.no_grad()
    def forward(
        self, measurements: torch.Tensor, batch: dict[str, Any] | None = None
    ) -> dict[str, torch.Tensor]:
        b, _v, _c, _h, _w = measurements.shape
        device, dtype = measurements.device, measurements.dtype
        centers = torch.zeros((b, self.mmax, 3), device=device, dtype=dtype)
        scores = torch.zeros((b, self.mmax), device=device, dtype=dtype)
        raw_scales = torch.full((b, self.mmax), self.scale_min_mm, device=device, dtype=dtype)
        valid = torch.zeros((b, self.mmax), device=device, dtype=torch.bool)

        if batch is None or "candidate_centers_mm" not in batch:
            raise FileNotFoundError(
                "SSQ candidate anchors are required. Generate proposal/candidate_anchors.npz "
                "and load it through the dataset; online detector top-k fallback is disabled."
            )
        src_centers = batch["candidate_centers_mm"].to(device=device, dtype=dtype)
        m = min(self.mmax, src_centers.shape[1])
        centers[:, :m] = src_centers[:, :m]
        if "candidate_scores" in batch:
            scores[:, :m] = batch["candidate_scores"][:, :m].to(device=device, dtype=dtype)
        else:
            scores[:, :m] = 1.0
        if "candidate_support_scales_mm" in batch:
            raw_scales[:, :m] = batch["candidate_support_scales_mm"][:, :m].to(
                device=device, dtype=dtype
            )
        elif "candidate_scales_mm" in batch:
            raw_scales[:, :m] = batch["candidate_scales_mm"][:, :m].to(device=device, dtype=dtype)
        if "candidate_valid_mask" in batch:
            valid[:, :m] = batch["candidate_valid_mask"][:, :m].to(
                device=device, dtype=torch.bool
            )
        else:
            valid[:, :m] = scores[:, :m] > self.threshold

        clipped = raw_scales.clamp(self.scale_min_mm, self.scale_max_mm)
        covariance = torch.diag_embed(clipped.square()[..., None].expand(-1, -1, 3))
        if batch is not None and "candidate_support_covariances_mm" in batch:
            covariance[:, :m] = batch["candidate_support_covariances_mm"][:, :m].to(
                device=device, dtype=dtype
            )
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance.float())
        eigenvalues = eigenvalues.clamp(self.scale_min_mm**2, self.scale_max_mm**2)
        eigenvectors = eigenvectors.to(dtype=dtype)
        eigenvalues = eigenvalues.to(dtype=dtype)
        covariance = eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-1, -2)
        inverse_sqrt = (
            eigenvectors
            @ torch.diag_embed(eigenvalues.rsqrt())
            @ eigenvectors.transpose(-1, -2)
        )
        return {
            "candidate_centers_mm": centers,
            "candidate_scores": scores.clamp(0.0, 1.0),
            "candidate_raw_support_scales_mm": raw_scales,
            "candidate_support_scales_mm": clipped,
            "candidate_raw_scales_mm": raw_scales,
            "candidate_scales_mm": clipped,
            "candidate_support_covariances_mm": covariance,
            "candidate_support_inverse_sqrt_mm": inverse_sqrt,
            "candidate_valid_mask": valid,
            "candidate_support_scale_low_clip": raw_scales < self.scale_min_mm,
            "candidate_support_scale_high_clip": raw_scales > self.scale_max_mm,
            "candidate_scale_low_clip": raw_scales < self.scale_min_mm,
            "candidate_scale_high_clip": raw_scales > self.scale_max_mm,
        }


def attach_candidate_detector_priors(
    candidates: dict[str, torch.Tensor],
    cand_mapped: dict[str, torch.Tensor],
    detector_scale_min_px: float,
    detector_scale_max_px: float,
    measurements: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if candidates["candidate_centers_mm"].shape[1] == 0:
        return candidates
    candidates["candidate_uv_px"] = cand_mapped["uv_px"]
    candidates["candidate_detector_valid_mask"] = cand_mapped["valid_mask"]
    pixels_per_mm = cand_mapped["view_pixels_per_mm"].to(
        device=candidates["candidate_support_scales_mm"].device,
        dtype=candidates["candidate_support_scales_mm"].dtype,
    )
    covariance = candidates["candidate_support_covariances_mm"]
    rays = cand_mapped["ray_directions"]
    ray_variance = torch.einsum("bvmi,bmij,bvmj->bvm", rays, covariance, rays)
    plane_variance = (covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1)[:, None] - ray_variance)
    scale_mm = (0.5 * plane_variance.clamp_min(1.0e-8)).sqrt()
    scale_px = scale_mm * pixels_per_mm[None, :, None]
    clipped = scale_px.clamp(float(detector_scale_min_px), float(detector_scale_max_px))
    candidates["candidate_detector_support_scales_px"] = clipped
    if measurements is not None:
        center_measurements = sample_scalar_map(measurements, cand_mapped["grid"])
        center_measurements = torch.where(
            cand_mapped["valid_mask"],
            center_measurements.clamp_min(0.0),
            torch.zeros_like(center_measurements),
        )
        candidates["candidate_center_measurements"] = center_measurements
    return candidates
