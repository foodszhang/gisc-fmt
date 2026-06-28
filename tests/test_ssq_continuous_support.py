from copy import deepcopy

import torch

from minr_fmt.models.ssq_fmt import SSQFMT
from tests.test_ssq_gradients import _has_grad
from tests.test_ssq_math_invariants import make_batch, make_cfg


def test_low_lambda_candidate_keeps_assignment_and_decoder_gradients():
    cfg = make_cfg(mmax=2)
    cfg.model.ssq_fmt.routing = {
        "support_mode": "continuous",
        "support_center": 0.20,
        "support_temperature": 0.05,
        "support_logit_weight": 1.0,
        "p_min": 1e-4,
    }
    model = SSQFMT(cfg)
    batch = make_batch(mmax=2)
    batch["candidate_centers_mm"][:] = batch["query_coordinates_mm"][:, :1]
    out = model(
        batch["surface_measurements_packed"],
        batch["query_coordinates_mm"],
        batch=batch,
        return_diagnostics=True,
    )
    weights = torch.arange(
        1,
        out["diagnostics"]["pi"].shape[-1],
        device=out["density"].device,
        dtype=out["density"].dtype,
    )
    loss = (
        out["density"].sum()
        + (out["diagnostics"]["pi"][..., 1:] * weights).sum()
        + out["diagnostics"]["support_gate"].sum()
    )
    loss.backward()
    assert torch.isfinite(out["diagnostics"]["support_gate"]).all()
    assert (out["diagnostics"]["pi"][..., 1:] > 0).any()
    assert _has_grad(model.assignment_head.candidate_assignment_head)
    assert _has_grad(model.candidate_density_decoder)


def test_independent_compensation_is_invariant_to_candidate_centers():
    cfg = make_cfg(mmax=2)
    cfg.model.ssq_fmt.routing = {"compensation_routing_mode": "independent"}
    model = SSQFMT(cfg).eval()
    batch_a = make_batch(mmax=2)
    batch_b = deepcopy(batch_a)
    batch_b["candidate_centers_mm"] = batch_a["candidate_centers_mm"] + 8.0
    with torch.no_grad():
        out_a = model(
            batch_a["surface_measurements_packed"],
            batch_a["query_coordinates_mm"],
            batch=batch_a,
            return_diagnostics=True,
        )
        out_b = model(
            batch_b["surface_measurements_packed"],
            batch_b["query_coordinates_mm"],
            batch=batch_b,
            return_diagnostics=True,
        )
    d0_a = out_a["diagnostics"]["branch_density"][:, :, 0]
    d0_b = out_b["diagnostics"]["branch_density"][:, :, 0]
    assert torch.allclose(d0_a, d0_b, atol=1.0e-6, rtol=1.0e-6)
