import copy

import torch

from minr_fmt.models.ssq_fmt import SSQFMT, compose_quotient_residual_density
from minr_fmt.network.ssq_fusion import SourceHypothesisQuotientAggregator
from tests.test_ssq_gradients import _has_grad
from tests.test_ssq_math_invariants import make_batch, make_cfg


def _shq_cfg(mmax=3, consistency=False):
    cfg = make_cfg(mmax=mmax)
    cfg.model.ssq_fmt.composition = {
        "mode": "quotient_residual",
        "delta_l_max": 3.0,
        "tau_u": 1.0,
    }
    cfg.model.ssq_fmt.routing = {"compensation_routing_mode": "proposal_only"}
    cfg.model.ssq_fmt.quotient = {
        "subset_consistency": {
            "enabled": consistency,
            "seed": 17,
            "min_support": 0.0,
        }
    }
    return cfg


def _composition_inputs(m=3):
    return {
        "shared_logit": torch.randn(2, 4, 1),
        "delta_logit": torch.randn(2, 4, m, 1).clamp(-3.0, 3.0),
        "relative_query": torch.randn(2, 4, m, 3),
        "candidate_scores": torch.rand(2, m),
        "candidate_support": torch.rand(2, 4, m),
        "quotient_dispersion": torch.rand(2, 4, m),
        "candidate_valid": torch.ones(2, m, dtype=torch.bool),
        "tau_u": 1.0,
    }


def test_candidate_permutation_invariance():
    inputs = _composition_inputs()
    expected = compose_quotient_residual_density(**inputs)["density"]
    permutation = torch.tensor([2, 0, 1])
    for key in ("delta_logit", "relative_query", "candidate_support", "quotient_dispersion"):
        inputs[key] = inputs[key][:, :, permutation]
    for key in ("candidate_scores", "candidate_valid"):
        inputs[key] = inputs[key][:, permutation]
    actual = compose_quotient_residual_density(**inputs)["density"]
    assert torch.allclose(actual, expected, atol=1.0e-6)


def test_fallback_and_correction_suppression_invariants():
    inputs = _composition_inputs()
    shared = torch.sigmoid(inputs["shared_logit"])
    for key, value in (
        ("candidate_valid", torch.zeros_like(inputs["candidate_valid"])),
        ("candidate_support", torch.zeros_like(inputs["candidate_support"])),
        ("delta_logit", torch.zeros_like(inputs["delta_logit"])),
    ):
        case = dict(inputs)
        case[key] = value
        assert torch.allclose(compose_quotient_residual_density(**case)["density"], shared)
    empty = _composition_inputs(m=0)
    assert torch.allclose(
        compose_quotient_residual_density(**empty)["density"],
        torch.sigmoid(empty["shared_logit"]),
    )
    low = compose_quotient_residual_density(**inputs)["residual_correction"].abs().mean()
    inputs["quotient_dispersion"] = torch.full_like(inputs["quotient_dispersion"], 100.0)
    high = compose_quotient_residual_density(**inputs)["residual_correction"].abs().mean()
    assert high < low * 1.0e-4


def test_quotient_masks_weights_and_dispersion():
    aggregator = SourceHypothesisQuotientAggregator(4, hidden_dim=8)
    evidence = torch.ones(1, 2, 3, 2, 4)
    geometry = torch.zeros(1, 2, 3, 4)
    support = torch.ones(1, 2, 3, 2)
    view_valid = torch.tensor([[[True, False, True], [False, False, False]]])
    proposal_valid = torch.tensor([[True, False]])
    out = aggregator(evidence, geometry, support, view_valid, proposal_valid)
    assert torch.allclose(out["view_weights"][0, 0, :, 0].sum(), torch.tensor(1.0))
    assert torch.count_nonzero(out["view_weights"][..., 1]) == 0
    assert torch.count_nonzero(out["view_weights"][0, 1]) == 0
    assert out["dispersion"][0, 0, 0] < 1.0e-7
    assert torch.isfinite(out["quotient"]).all()


def test_subset_split_is_nonempty_and_consistency_is_finite():
    valid = torch.tensor([[[True, True, True], [True, False, False]]])
    subset_a, subset_b = SSQFMT._deterministic_view_subsets(valid, 17)
    assert subset_a[0, 0].any() and subset_b[0, 0].any()
    assert not subset_a[0, 1].any() and not subset_b[0, 1].any()
    assert not (subset_a & subset_b).any()
    cfg = _shq_cfg(mmax=2, consistency=True)
    batch = make_batch(2)
    out = SSQFMT(cfg)(
        batch["surface_measurements_packed"],
        batch["query_coordinates_mm"],
        batch=batch,
    )
    assert torch.isfinite(out["aux_outputs"]["quotient_consistency_loss"])


def test_shq_forward_bypasses_assignment_and_envelope_and_preserves_shared_path():
    model = SSQFMT(_shq_cfg(mmax=3)).eval()
    model.assignment_head.forward = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("legacy assignment called")
    )
    model.candidate_support_envelope_power = 10.0
    batch = make_batch(3)
    first = model(
        batch["surface_measurements_packed"],
        batch["query_coordinates_mm"],
        batch=batch,
        return_diagnostics=True,
    )
    invalid_batch = copy.deepcopy(batch)
    invalid_batch["candidate_valid_mask"].zero_()
    fallback = model(
        invalid_batch["surface_measurements_packed"],
        invalid_batch["query_coordinates_mm"],
        batch=invalid_batch,
        return_diagnostics=True,
    )
    assert torch.allclose(
        first["diagnostics"]["shared_density"],
        fallback["diagnostics"]["shared_density"],
        atol=1.0e-6,
    )
    assert torch.allclose(fallback["density"], fallback["diagnostics"]["shared_density"])

    one_candidate = copy.deepcopy(batch)
    for key in (
        "candidate_centers_mm",
        "candidate_scores",
        "candidate_support_scales_mm",
        "candidate_scales_mm",
        "candidate_valid_mask",
    ):
        one_candidate[key] = one_candidate[key][:, :1]
    one = model(
        one_candidate["surface_measurements_packed"],
        one_candidate["query_coordinates_mm"],
        batch=one_candidate,
        return_diagnostics=True,
    )
    assert torch.allclose(
        first["diagnostics"]["shared_density"],
        one["diagnostics"]["shared_density"],
        atol=1.0e-6,
    )


def test_shq_gradients_reach_required_modules_and_candidates_are_detached():
    model = SSQFMT(_shq_cfg(mmax=2)).train()
    batch = make_batch(2)
    batch["candidate_scores"].requires_grad_()
    out = model(
        batch["surface_measurements_packed"],
        batch["query_coordinates_mm"],
        batch=batch,
    )
    loss = out["density"].mean() + out["aux_outputs"]["residual_regularization_loss"]
    loss.backward()
    assert _has_grad(model.surface_encoder)
    assert _has_grad(model.surface_sampler)
    assert _has_grad(model.quotient_aggregator.reliability_residual)
    assert _has_grad(model.shared_density_logit_decoder)
    assert _has_grad(model.source_hypothesis_residual_decoder)
    assert batch["candidate_scores"].grad is None


def test_shq_smoke_m_zero_one_five_and_all_views_invalid():
    for m in (0, 1, 5):
        model = SSQFMT(_shq_cfg(mmax=m)).eval()
        batch = make_batch(m)
        if m == 5:
            batch["detector_valid_mask"][:, 0] = False
        out = model(
            batch["surface_measurements_packed"],
            batch["query_coordinates_mm"],
            batch=batch,
        )
        assert out["density"].shape == (2, 5, 1)
        assert torch.isfinite(out["density"]).all()
    batch["detector_valid_mask"].zero_()
    out = model(
        batch["surface_measurements_packed"], batch["query_coordinates_mm"], batch=batch
    )
    assert torch.isfinite(out["density"]).all()
