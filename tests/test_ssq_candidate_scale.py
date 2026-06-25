import torch

from minr_fmt.network.ssq_candidates import MeasurementDerivedCandidateBuilder


def test_candidate_scale_clipping_and_flags():
    builder = MeasurementDerivedCandidateBuilder(mmax=3, scale_min_mm=1.0, scale_max_mm=12.0)
    y = torch.ones(1, 2, 1, 8, 8)
    batch = {
        "candidate_centers_mm": torch.zeros(1, 3, 3),
        "candidate_scores": torch.ones(1, 3),
        "candidate_scales_mm": torch.tensor([[0.2, 4.0, 20.0]]),
        "candidate_valid_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    out = builder(y, batch)
    assert torch.allclose(out["candidate_scales_mm"], torch.tensor([[1.0, 4.0, 12.0]]))
    assert out["candidate_scale_low_clip"][0, 0]
    assert out["candidate_scale_high_clip"][0, 2]
