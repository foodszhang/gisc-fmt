import torch

from minr_fmt.models.ssq_fmt import SSQFMT
from tests.test_ssq_math_invariants import make_batch, make_cfg


def test_no_valid_views_density_and_contributions_are_zero():
    model = SSQFMT(make_cfg(mmax=2))
    batch = make_batch(mmax=2)
    batch["detector_valid_mask"][:] = False
    batch["depth_maps"][:] = float("inf")
    out = model(
        batch["surface_measurements_packed"],
        batch["query_coordinates_mm"],
        batch=batch,
        return_diagnostics=True,
    )
    assert torch.all(out["density"] == 0)
    assert torch.all(out["diagnostics"]["branch_contributions"] == 0)
    assert not out["diagnostics"]["measurement_supported"].any()
