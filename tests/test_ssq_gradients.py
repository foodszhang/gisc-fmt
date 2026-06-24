import torch

from minr_fmt.loss import MorphologyAwareDensityLoss
from minr_fmt.models.ssq_fmt import SSQFMT
from tests.test_ssq_math_invariants import make_batch, make_cfg


def _has_grad(module):
    return any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in module.parameters()
    )


def test_ssq_forward_backward_reaches_core_modules():
    cfg = make_cfg(mmax=2, lambda_sdf=0.5)
    model = SSQFMT(cfg)
    model.train()
    batch = make_batch(mmax=2)
    gt = torch.zeros(2, 4, 4, 4)
    gt[:, 1:3, 1:3, 1:3] = 1.0
    points_ijk = (batch["query_coordinates_mm"] / 2.0 * 3.0).clamp(0, 3)
    out = model(
        batch["surface_measurements_packed"],
        batch["query_coordinates_mm"],
        batch=batch,
        return_diagnostics=True,
    )
    target = torch.rand_like(out["density"])
    loss = MorphologyAwareDensityLoss(lambda_sdf=0.5)(
        out["density"],
        target,
        out["aux_outputs"],
        gt_voxels=gt,
        points_ijk=points_ijk,
    )["total_loss"]
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
    assert _has_grad(model.morphology_sdf_head)
