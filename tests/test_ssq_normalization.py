import torch

from minr_fmt.models.ssq_fmt import SurfaceMeasurementNormalizer


def test_joint_percentile_over_all_valid_views_pixels():
    y = torch.tensor([[[[[1.0, 2.0], [3.0, 100.0]]], [[[4.0, 5.0], [6.0, 7.0]]]]])
    mask = torch.ones(1, 2, 2, 2, dtype=torch.bool)
    norm, scale = SurfaceMeasurementNormalizer(percentile=50.0, eps=1e-6)(y, mask)

    assert torch.allclose(scale, torch.tensor([4.5]))
    assert torch.allclose(norm[0, 0, 0, 0, 0], torch.tensor(1.0 / 4.5))
    assert norm.max() <= 1.0


def test_invalid_detector_pixels_are_excluded():
    y = torch.tensor([[[[[1.0, 1000.0]]]]])
    mask = torch.tensor([[[[True, False]]]])
    norm, scale = SurfaceMeasurementNormalizer(percentile=99.9, eps=1e-6)(y, mask)

    assert torch.allclose(scale, torch.tensor([1.0]))
    assert torch.allclose(norm[0, 0, 0, 0, 1], torch.tensor(0.0))


def test_empty_valid_and_tiny_alpha_are_finite():
    y = torch.zeros(2, 3, 1, 2, 2)
    mask = torch.zeros(2, 3, 2, 2, dtype=torch.bool)
    norm, scale = SurfaceMeasurementNormalizer(percentile=99.9, eps=1e-6)(y, mask)

    assert torch.isfinite(norm).all()
    assert torch.isfinite(scale).all()
    assert torch.all(norm == 0.0)
