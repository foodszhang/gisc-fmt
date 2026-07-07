import torch

from minr_fmt.models.ssq_fmt import SSQFMT
from tests.test_ssq_math_invariants import make_batch, make_cfg


def _forward(batch, mmax):
    model = SSQFMT(make_cfg(mmax=mmax))
    return model(
        batch["surface_measurements_packed"],
        batch["query_coordinates_mm"],
        detector_valid_mask=batch.get("detector_valid_mask"),
        batch=batch,
        return_diagnostics=True,
    )


def test_m_zero_and_only_compensation_branch():
    batch = make_batch(mmax=0)
    out = _forward(batch, mmax=0)
    assert out["density"].shape == (2, 5, 1)
    assert out["diagnostics"]["pi"].shape[-1] == 1
    assert torch.allclose(out["diagnostics"]["pi"], torch.ones_like(out["diagnostics"]["pi"]))


def test_one_valid_view_has_unit_weight():
    batch = make_batch(mmax=2)
    batch["detector_valid_mask"][:, 1] = False
    out = _forward(batch, mmax=2)
    weights = out["diagnostics"]["view_weights"]
    assert torch.allclose(
        weights[:, :, 0][weights[:, :, 0] > 0],
        torch.ones_like(weights[:, :, 0][weights[:, :, 0] > 0]),
    )
    assert torch.all(weights[:, :, 1] == 0.0)


def test_no_valid_views_and_all_footprint_samples_out_of_bounds():
    batch = make_batch(mmax=2)
    batch["query_coordinates_mm"] = torch.full_like(batch["query_coordinates_mm"], 1000.0)
    batch["detector_valid_mask"][:] = False
    out = _forward(batch, mmax=2)
    d = out["diagnostics"]
    assert torch.all(d["Lambda"] == 0.0)
    assert torch.isfinite(out["density"]).all()
    assert torch.allclose(d["pi"][..., 0], torch.ones_like(d["pi"][..., 0]))


def test_zero_candidate_raw_scale_and_overlapping_candidates_are_finite():
    batch = make_batch(mmax=3)
    batch["candidate_scales_mm"][:] = 0.0
    batch["candidate_centers_mm"][:, 1:] = batch["candidate_centers_mm"][:, :1]
    out = _forward(batch, mmax=3)
    assert torch.isfinite(out["density"]).all()
    assert torch.isfinite(out["diagnostics"]["pi"]).all()


def test_all_candidates_invisible_and_one_candidate_with_zero_assignment_mass():
    valid = torch.zeros(2, 3, dtype=torch.bool)
    batch = make_batch(mmax=3, candidate_valid=valid)
    out = _forward(batch, mmax=3)
    d = out["diagnostics"]
    assert torch.allclose(d["pi"][..., 0], torch.ones_like(d["pi"][..., 0]))
    assert torch.all(d["pi"][..., 1:] == 0.0)


def test_different_candidate_counts_per_batch_sample():
    valid = torch.tensor([[True, False, False], [True, True, False]])
    batch = make_batch(mmax=3, candidate_valid=valid)
    out = _forward(batch, mmax=3)
    d = out["diagnostics"]
    assert torch.all(d["pi"][0, :, 2:] == 0.0)
    assert torch.all(d["pi"][1, :, 3:] == 0.0)
