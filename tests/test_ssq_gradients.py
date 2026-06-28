import torch

from minr_fmt.loss import MorphologyAwareDensityLoss
from minr_fmt.models.ssq_fmt import SSQFMT, SurfaceMeasurementNormalizer
from minr_fmt.network.ssq_decoder import CandidateDensityResidualDecoder
from minr_fmt.network.ssq_fusion import CandidateAssignmentHead
from tests.test_ssq_math_invariants import make_batch, make_cfg


def _has_grad(module):
    return any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in module.parameters()
    )


def test_per_view_max_normalization_preserves_weak_view_dynamic_range():
    measurements = torch.tensor(
        [[[[[0.0, 1.0], [0.5, 0.0]]], [[[0.0, 100.0], [50.0, 0.0]]]]]
    )
    valid = measurements.squeeze(2) > 0.0
    normalizer = SurfaceMeasurementNormalizer(mode="per_view_max")
    normalized, scale = normalizer(measurements, valid)
    assert torch.allclose(scale, torch.tensor([[1.0, 100.0]]))
    assert torch.allclose(normalized[:, 0], normalized[:, 1])


def test_occupancy_evidence_moves_assignment_toward_supported_branch():
    head = CandidateAssignmentHead(
        feature_dim=4,
        occupancy_evidence_weight=1.0,
        occupancy_evidence_floor=0.01,
    )
    prior = torch.tensor([[[0.5, 0.5]]], requires_grad=True)
    density = torch.tensor([[[[0.1], [0.9]]]], requires_grad=True)
    posterior = head.apply_occupancy_evidence(prior, density)
    assert posterior[0, 0, 1] > posterior[0, 0, 0]
    posterior[..., 1].sum().backward()
    assert density.grad is not None and density.grad.abs().sum() > 0.0


def test_candidate_residual_decoder_starts_as_identity_correction():
    decoder = CandidateDensityResidualDecoder(
        feature_dim=8,
        pos_dim=6,
        hidden_dim=16,
        query_chunk_size=4,
    )
    features = torch.randn(2, 5, 3, 8, requires_grad=True)
    positions = torch.randn(2, 5, 3, 6)
    residual = decoder(features, positions)
    assert torch.count_nonzero(residual) == 0
    residual.sum().backward()
    assert decoder.fusion[-1].weight.grad is not None


def test_ssq_forward_backward_reaches_core_modules():
    cfg = make_cfg(mmax=2, lambda_sdf=0.0)
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
    loss = MorphologyAwareDensityLoss(lambda_sdf=0.0)(
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


def test_candidate_branch_aux_loss_reaches_candidate_decoder():
    cfg = make_cfg(mmax=2, lambda_sdf=0.0)
    model = SSQFMT(cfg)
    model.train()
    batch = make_batch(mmax=2)
    out = model(
        batch["surface_measurements_packed"],
        batch["query_coordinates_mm"],
        batch=batch,
        return_diagnostics=True,
    )
    target = torch.rand_like(out["density"])
    loss_dict = MorphologyAwareDensityLoss(
        lambda_sdf=0.0,
        pos_weight=10.0,
        candidate_branch_density_weight=1.0,
        candidate_branch_dice_weight=0.25,
    )(out["density"], target, out["aux_outputs"])
    assert torch.isfinite(loss_dict["candidate_branch_density_loss"])
    assert torch.isfinite(loss_dict["candidate_branch_dice_loss"])
    loss_dict["total_loss"].backward()
    if out["aux_outputs"]["p_all"][..., 1:].sum() > 0:
        assert loss_dict["candidate_branch_density_loss"].detach() > 0.0
        assert _has_grad(model.candidate_density_decoder)
        assert _has_grad(model.candidate_density_decoder)


def test_candidate_branch_best_positive_prior_loss_reaches_candidate_decoder():
    cfg = make_cfg(mmax=2, lambda_sdf=0.0)
    model = SSQFMT(cfg)
    model.train()
    batch = make_batch(mmax=2)
    out = model(
        batch["surface_measurements_packed"],
        batch["query_coordinates_mm"],
        batch=batch,
        return_diagnostics=True,
    )
    target = torch.zeros_like(out["density"])
    target[:, :2] = 1.0
    loss_dict = MorphologyAwareDensityLoss(
        lambda_sdf=0.0,
        pos_weight=10.0,
        candidate_branch_density_weight=1.0,
        candidate_branch_dice_weight=0.25,
        candidate_branch_target_mode="best_positive_prior",
    )(out["density"], target, out["aux_outputs"])
    assert torch.isfinite(loss_dict["candidate_branch_density_loss"])
    assert torch.isfinite(loss_dict["candidate_branch_dice_loss"])
    loss_dict["total_loss"].backward()
    if out["aux_outputs"]["p_all"][..., 1:].sum() > 0:
        assert loss_dict["candidate_branch_density_loss"].detach() > 0.0


def test_positive_candidate_assignment_loss_reaches_assignment_head():
    cfg = make_cfg(mmax=2, lambda_sdf=0.0)
    model = SSQFMT(cfg)
    model.train()
    batch = make_batch(mmax=2)
    out = model(
        batch["surface_measurements_packed"],
        batch["query_coordinates_mm"],
        batch=batch,
        return_diagnostics=False,
    )
    target = torch.zeros_like(out["density"])
    target[:, :4] = 1.0
    loss = MorphologyAwareDensityLoss(
        candidate_assignment_weight=1.0,
        candidate_assignment_min_prior=0.0,
    )
    loss_dict = loss(out["density"], target, out["aux_outputs"])
    loss_dict["total_loss"].backward()
    assert torch.isfinite(loss_dict["candidate_assignment_loss"])
    assert loss_dict["candidate_assignment_loss"].detach() > 0.0
    assert _has_grad(model.assignment_head.candidate_assignment_head)
    assert _has_grad(model.assignment_head.compensation_assignment_head)


def test_hungarian_component_loss_matches_fields_and_reaches_candidate_modules():
    cfg = make_cfg(mmax=2, lambda_sdf=0.0)
    model = SSQFMT(cfg)
    model.train()
    batch = make_batch(mmax=2)
    out = model(
        batch["surface_measurements_packed"],
        batch["query_coordinates_mm"],
        batch=batch,
        return_diagnostics=False,
    )
    target = torch.zeros_like(out["density"])
    target[:, :4] = 1.0
    component_ids = torch.zeros(target.shape[:2], dtype=torch.long)
    component_ids[:, :2] = 1
    component_ids[:, 2:4] = 2
    component_centers = torch.tensor(
        [[[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]], [[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]]]
    )
    component_valid = torch.ones(2, 2, dtype=torch.bool)
    loss = MorphologyAwareDensityLoss(
        pos_weight=10.0,
        candidate_branch_density_weight=0.5,
        candidate_branch_dice_weight=0.5,
        candidate_branch_target_mode="hungarian_component",
        candidate_assignment_weight=0.1,
        candidate_assignment_target_mode="hungarian_component",
    )
    loss_dict = loss(
        out["density"],
        target,
        out["aux_outputs"],
        query_component_ids=component_ids,
        gt_component_centers_mm=component_centers,
        gt_component_valid_mask=component_valid,
    )
    assert torch.isfinite(loss_dict["total_loss"])
    assert loss_dict["component_match_coverage"] > 0
    loss_dict["total_loss"].backward()
    assert _has_grad(model.candidate_density_decoder)
    assert _has_grad(model.assignment_head.candidate_assignment_head)
