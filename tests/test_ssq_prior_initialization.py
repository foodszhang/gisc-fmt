import torch

from minr_fmt.models.ssq_fmt import SSQFMT
from tests.test_ssq_math_invariants import make_batch, make_cfg


def _last_linear(module):
    for child in reversed(list(module.modules())):
        if isinstance(child, torch.nn.Linear):
            return child
    raise AssertionError("no Linear found")


def test_correction_heads_zero_initialized_and_evidence_half():
    model = SSQFMT(make_cfg(mmax=2))
    footprint_head = _last_linear(model.surface_sampler.footprint_context_net)
    assert torch.isfinite(footprint_head.weight).all()
    assert footprint_head.weight.abs().sum() > 0
    assert torch.all(_last_linear(model.candidate_router.routing_net).weight == 0)
    assert torch.all(_last_linear(model.view_fusion.cross_view_fusion).weight == 0)
    assert torch.all(_last_linear(model.assignment_head.compensation_assignment_head).weight == 0)
    assert torch.all(_last_linear(model.assignment_head.candidate_assignment_head).weight == 0)
    density_head = _last_linear(model.candidate_density_decoder.fusion)
    assert density_head.weight.abs().sum() > 0
    assert density_head.weight.abs().max() < 1.0e-1
    batch = make_batch(mmax=2)
    out = model(
        batch["surface_measurements_packed"],
        batch["query_coordinates_mm"],
        batch=batch,
        return_diagnostics=True,
    )
    e = out["diagnostics"]["e_sample"][out["diagnostics"]["sample_valid"]]
    assert torch.allclose(e.mean(), torch.tensor(0.5), atol=1e-2)
