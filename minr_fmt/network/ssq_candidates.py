"""Measurement-derived candidate anchors and detector-plane priors."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


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
        return {
            "candidate_centers_mm": centers,
            "candidate_scores": scores.clamp(0.0, 1.0),
            "candidate_raw_support_scales_mm": raw_scales,
            "candidate_support_scales_mm": clipped,
            "candidate_raw_scales_mm": raw_scales,
            "candidate_scales_mm": clipped,
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
) -> dict[str, torch.Tensor]:
    if candidates["candidate_centers_mm"].shape[1] == 0:
        return candidates
    candidates["candidate_uv_px"] = cand_mapped["uv_px"]
    candidates["candidate_detector_valid_mask"] = cand_mapped["valid_mask"]
    pixels_per_mm = cand_mapped["view_pixels_per_mm"].to(
        device=candidates["candidate_support_scales_mm"].device,
        dtype=candidates["candidate_support_scales_mm"].dtype,
    )
    scale_px = candidates["candidate_support_scales_mm"][:, None, :] * pixels_per_mm[None, :, None]
    clipped = scale_px.clamp(float(detector_scale_min_px), float(detector_scale_max_px))
    candidates["candidate_detector_support_scales_px"] = clipped
    return candidates
