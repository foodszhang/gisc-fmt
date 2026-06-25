import torch

from minr_fmt.models.ssq_fmt import SSQFMT
from minr_fmt.network.ssq_geometry import grid_to_pixel, pixel_to_grid
from tests.test_ssq_math_invariants import make_cfg


def test_grid_pixel_convention_align_corners_true():
    grid = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
    px = grid_to_pixel(grid, height=16, width=16, align_corners=True)
    assert torch.allclose(px[0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(px[1], torch.tensor([7.5, 7.5]))
    assert torch.allclose(px[2], torch.tensor([15.0, 15.0]))
    assert torch.allclose(pixel_to_grid(px, 16, 16, True), grid)


def test_query_candidate_same_point_detector_distance_zero():
    model = SSQFMT(make_cfg(mmax=1))
    point = torch.tensor([[[1.0, 1.0, 1.0]]])
    mapped = model.geometry_mapper(point)
    cand = model.geometry_mapper(point)
    assert torch.allclose(mapped["uv_px"], cand["uv_px"], atol=1e-5)
