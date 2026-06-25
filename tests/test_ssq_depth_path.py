import torch

from minr_fmt.models.ssq_fmt import SSQFMT
from tests.test_ssq_math_invariants import make_cfg


def test_depth_path_uses_sampled_surface_depth():
    model = SSQFMT(make_cfg(mmax=1))
    points = torch.tensor([[[1.0, 1.0, 1.0], [1.0, 1.0, 0.5]]])
    base = model.geometry_mapper(points)
    depth = base["depth"][..., 0].min().floor()
    depth_maps = torch.full((1, 2, 16, 16), depth.item())
    mapped = model.geometry_mapper(points, depth_maps=depth_maps)
    assert torch.all(mapped["detector_side_path_proxy"] >= 0)
    assert torch.any(mapped["detector_side_path_proxy"] > 0)
    assert (
        mapped["detector_side_path_proxy"][0, :, 1].mean()
        >= mapped["detector_side_path_proxy"][0, :, 0].mean()
    )
    assert (mapped["detector_side_path_proxy"] < 1.0).any()


def test_missing_depth_invalidates_view():
    model = SSQFMT(make_cfg(mmax=1))
    points = torch.rand(1, 3, 3) * 2
    depth_maps = torch.full((1, 2, 16, 16), float("inf"))
    mapped = model.geometry_mapper(points, depth_maps=depth_maps)
    assert not mapped["valid_mask"].any()
