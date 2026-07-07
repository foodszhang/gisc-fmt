import torch

from minr_fmt.models.ssq_fmt import SSQFMT
from tests.test_ssq_math_invariants import make_batch, make_cfg


def _has_grad(module):
    return any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in module.parameters()
    )


def test_density_loss_reaches_rebuilt_modules():
    model = SSQFMT(make_cfg(mmax=2))
    batch = make_batch(mmax=2)
    out = model(batch["surface_measurements_packed"], batch["query_coordinates_mm"], batch=batch)
    loss = torch.nn.functional.binary_cross_entropy(out["density"], torch.rand_like(out["density"]))
    loss.backward()
    assert _has_grad(model.surface_encoder)
    assert _has_grad(model.candidate_router.routing_net)
    assert _has_grad(model.candidate_router.evidence_net)
    assert _has_grad(model.view_encoder.representation_net)
    assert _has_grad(model.view_fusion.cross_view_fusion)
    assert _has_grad(model.assignment_head.compensation_assignment_head)
    assert _has_grad(model.assignment_head.candidate_assignment_head)
    assert _has_grad(model.compensation_density_decoder)
    assert _has_grad(model.candidate_density_decoder)
