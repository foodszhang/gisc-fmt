import numpy as np
import torch

from minr_fmt.network.ssq_candidates import MeasurementDerivedCandidateBuilder
from minr_fmt.utils.ssq_candidate_extraction import candidate_support_covariances_mm


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


def test_anisotropic_heatmap_moment_recovers_long_axis():
    axes = [np.arange(9) for _ in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    center = np.asarray([4.0, 4.0, 4.0], dtype=np.float32)
    relative = grid.astype(np.float32) - center
    heatmap = np.exp(
        -0.5
        * (
            relative[..., 0] ** 2 / 2.5**2
            + relative[..., 1] ** 2 / 0.8**2
            + relative[..., 2] ** 2 / 0.8**2
        )
    ).astype(np.float32)
    covariance = candidate_support_covariances_mm(
        heatmap,
        centers_mm=np.asarray([[4.5, 4.5, 4.5]], dtype=np.float32),
        valid=np.asarray([True]),
        cell_size_mm=np.ones(3, dtype=np.float32),
        radius_mm=4.0,
        scale_min_mm=0.25,
        scale_max_mm=8.0,
    )[0]
    eigenvalues = np.linalg.eigvalsh(covariance)
    assert np.all(eigenvalues > 0.0)
    assert covariance[0, 0] > 2.0 * covariance[1, 1]
    assert covariance[0, 0] > 2.0 * covariance[2, 2]


def test_builder_whitens_candidate_coordinates_with_covariance():
    builder = MeasurementDerivedCandidateBuilder(mmax=1, scale_min_mm=0.5, scale_max_mm=12.0)
    measurements = torch.ones(1, 2, 1, 8, 8)
    covariance = torch.diag(torch.tensor([9.0, 1.0, 1.0])).reshape(1, 1, 3, 3)
    batch = {
        "candidate_centers_mm": torch.zeros(1, 1, 3),
        "candidate_scores": torch.ones(1, 1),
        "candidate_scales_mm": torch.ones(1, 1),
        "candidate_support_covariances_mm": covariance,
        "candidate_valid_mask": torch.ones(1, 1, dtype=torch.bool),
    }
    out = builder(measurements, batch)
    inverse_sqrt = out["candidate_support_inverse_sqrt_mm"][0, 0]
    long_axis_distance = torch.tensor([3.0, 0.0, 0.0]) @ inverse_sqrt
    short_axis_distance = torch.tensor([0.0, 3.0, 0.0]) @ inverse_sqrt
    assert torch.allclose(long_axis_distance.norm(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(short_axis_distance.norm(), torch.tensor(3.0), atol=1e-5)
