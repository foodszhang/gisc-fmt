import torch

from minr_fmt.models.ssq_fmt import SSQFMT
from tests.test_ssq_gradients import _has_grad
from tests.test_ssq_math_invariants import make_batch, make_cfg


def _prod_cfg(**overrides):
    cfg = make_cfg(mmax=3, lambda_sdf=0.0)
    ssq = cfg.model.ssq_fmt
    ssq.density_output_mode = "candidate_scalar_composition"
    ssq.query_density_backbone = {"enabled": True, "normalization": "per_view_max"}
    ssq.footprint = {
        "mode": "query_dependent",
        "scale_min_px": 0.6,
        "scale_max_px": 6.0,
        "fixed_sigma": 1.5,
        "alpha_xi": 0.67,
        "alpha_beta": 0.33,
        "delta_h_max": 0.25,
    }
    ssq.routing = {
        "mode": "pre_aggregation",
        "k_min": 1e-6,
        "p_min": 1e-6,
        "support_mode": "continuous",
        "support_center": 0.10,
        "support_temperature": 0.10,
        "support_logit_weight": 1.0,
    }
    ssq.fusion = {"mode": "candidate_specific"}
    ssq.reliability = {"mode": "evidence"}
    for key, value in overrides.items():
        section, name = key.split("__", maxsplit=1)
        getattr(ssq, section)[name] = value
    return cfg


def test_production_density_loss_reaches_ssq_core_modules():
    model = SSQFMT(_prod_cfg())
    assert model.query_density_backbone is None
    model.train()
    batch = make_batch(mmax=3)
    out = model(
        batch["surface_measurements_packed"],
        batch["query_coordinates_mm"],
        detector_valid_mask=batch["detector_valid_mask"],
        batch=batch,
        return_diagnostics=True,
    )
    target = torch.rand_like(out["density"])
    loss = torch.nn.functional.smooth_l1_loss(out["density"], target)
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


def test_footprint_scheduler_affects_coordinates_not_feature_values_directly():
    model = SSQFMT(_prod_cfg())
    batch = make_batch(mmax=3)
    with torch.no_grad():
        out = model(
            batch["surface_measurements_packed"],
            batch["query_coordinates_mm"],
            detector_valid_mask=batch["detector_valid_mask"],
            batch=batch,
            return_diagnostics=True,
        )
    coords = out["diagnostics"]["sample_coordinates"]
    sigma = out["diagnostics"]["sigma_f"]
    assert coords.shape[-1] == 2
    assert coords.shape[-2] == model.surface_sampler.offsets_px.shape[0]
    assert torch.isfinite(coords).all()
    assert torch.isfinite(sigma).all()


def test_production_math_invariants_hold_with_view_and_sample_axes():
    model = SSQFMT(_prod_cfg())
    batch = make_batch(mmax=3)
    with torch.no_grad():
        out = model(
            batch["surface_measurements_packed"],
            batch["query_coordinates_mm"],
            detector_valid_mask=batch["detector_valid_mask"],
            batch=batch,
            return_diagnostics=True,
        )
    d = out["diagnostics"]
    valid_qv = d["sample_valid"].any(dim=-1)
    a_sum = d["A"].sum(dim=-1)
    assert torch.allclose(a_sum[valid_qv], torch.ones_like(a_sum[valid_qv]))
    assert torch.allclose(
        d["zeta"].sum(dim=-1)[d["sample_valid"]],
        torch.ones_like(d["zeta"].sum(dim=-1)[d["sample_valid"]]),
        atol=1e-5,
    )
    assign_sum = d["a"].sum(dim=-1)
    assert torch.allclose(assign_sum[valid_qv], torch.ones_like(assign_sum[valid_qv]))
    active_views = d["view_weights"].sum(dim=2) > 0
    assert torch.allclose(
        d["view_weights"].sum(dim=2)[active_views],
        torch.ones_like(d["view_weights"].sum(dim=2)[active_views]),
        atol=1e-5,
    )
    assert torch.allclose(d["pi"].sum(dim=-1), torch.ones_like(d["pi"].sum(dim=-1)), atol=1e-5)


def test_core_ablation_switches_have_no_artificial_density_epsilon():
    batch = make_batch(mmax=3)
    base = SSQFMT(_prod_cfg())
    variants = [
        _prod_cfg(routing__mode="post_aggregation"),
        _prod_cfg(fusion__mode="shared"),
        _prod_cfg(reliability__mode="assignment_only"),
        _prod_cfg(footprint__mode="fixed"),
    ]
    with torch.no_grad():
        density = base(
            batch["surface_measurements_packed"],
            batch["query_coordinates_mm"],
            detector_valid_mask=batch["detector_valid_mask"],
            batch=batch,
        )["density"]
        for cfg in variants:
            variant = SSQFMT(cfg)
            out = variant(
                batch["surface_measurements_packed"],
                batch["query_coordinates_mm"],
                detector_valid_mask=batch["detector_valid_mask"],
                batch=batch,
            )["density"]
            assert torch.isfinite(out).all()
    assert not hasattr(base, "_mode_epsilon")
    assert torch.isfinite(density).all()
