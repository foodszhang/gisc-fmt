#!/usr/bin/env python3
"""Low-memory entrypoint for the PHSA paired full-volume evaluator."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_view_complementary_full_volume_paired as base  # noqa: E402


def predict_full_volume_low_memory(
    net,
    surface: torch.Tensor,
    detector_valid_mask: torch.Tensor,
    depth_maps: torch.Tensor,
    shape: tuple[int, int, int],
    voxel_size_mm: float,
    proposal_points_mm: torch.Tensor,
    chunk_size: int,
    equivalence_atol: float,
) -> tuple[np.ndarray, dict[str, float]]:
    cache_start = time.perf_counter()
    cache, equivalence_error = base.build_sample_cache(
        net,
        surface,
        proposal_points_mm,
        detector_valid_mask,
        depth_maps,
        equivalence_atol,
    )
    cache_time_ms = (time.perf_counter() - cache_start) * 1000.0
    total = int(np.prod(shape))
    prediction = np.empty(total, dtype=np.float32)
    decode_start = time.perf_counter()
    with base.frozen_sample_cache(net, cache), torch.inference_mode():
        for start in range(0, total, int(chunk_size)):
            end = min(start + int(chunk_size), total)
            points_mm = base.chunk_points_mm(shape, start, end, voxel_size_mm).to(surface.device)
            output = base.call_net(
                net,
                surface,
                points_mm,
                detector_valid_mask,
                depth_maps,
                return_diagnostics=False,
            )
            values = output["density"].squeeze(0).squeeze(-1)
            if not torch.isfinite(values).all():
                raise RuntimeError("Full-volume prediction contains NaN/Inf")
            prediction[start:end] = values.float().cpu().numpy()
    decode_time_ms = (time.perf_counter() - decode_start) * 1000.0
    return prediction.reshape(shape), {
        "cache_equivalence_max_abs": equivalence_error,
        "candidate_center_drift_max_mm": 0.0,
        "cache_build_time_ms": cache_time_ms,
        "volume_decode_time_ms": decode_time_ms,
    }


base.predict_full_volume = predict_full_volume_low_memory

if __name__ == "__main__":
    base.main()
