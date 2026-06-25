from minr_fmt.models.ssq_fmt import SSQFMT
from tests.test_ssq_math_invariants import make_batch, make_cfg


def test_orthographic_geometry_has_no_query_local_kappa_diagnostics():
    model = SSQFMT(make_cfg(mmax=1))
    batch = make_batch(mmax=1)
    out = model(
        batch["surface_measurements_packed"],
        batch["query_coordinates_mm"],
        batch=batch,
        return_diagnostics=True,
    )
    assert "jacobian_scale" not in out["diagnostics"]
    assert "kappa" not in out["diagnostics"]
    mapped = model.geometry_mapper(batch["query_coordinates_mm"], depth_maps=batch["depth_maps"])
    assert mapped["view_pixels_per_mm"].ndim == 1
