import torch

from minr_fmt.models.ssq_fmt import SSQFMT
from minr_fmt.network.a3v2_routing import BoundedHypothesisViewRouting
from minr_fmt.network.diverse_candidate_constructor import DiverseCandidateConstructor
from minr_fmt.network.unified_density_decoder import UnifiedDensityDecoder
from tests.test_ssq_math_invariants import make_batch
from tests.test_view_complementary_model import _cfg


def _routing_inputs(far=False, existence=1.0):
    points = torch.full((2, 7, 3), 1000.0 if far else 0.0)
    centers = torch.randn(2, 3, 3)
    covariance = torch.eye(3)[None, None].expand(2, 3, -1, -1).clone()
    probabilities = torch.full((2, 3), existence)
    slots = torch.ones(2, 3, dtype=torch.bool)
    support = torch.rand(2, 3, 4)
    geometry = torch.rand(2, 4, 3)
    valid = torch.ones(2, 4, 7, dtype=torch.bool)
    valid[:, -1] = False
    return points, centers, covariance, probabilities, slots, support, geometry, valid


def _uniform(valid):
    weight = valid.float()
    return weight / weight.sum(dim=1, keepdim=True).clamp_min(1)


def test_uniform_fallback_for_zero_gain_existence_and_far_queries():
    for kwargs, gain in (({}, 0.0), ({"existence": 0.0}, 1.0), ({"far": True}, 1.0)):
        module = BoundedHypothesisViewRouting()
        with torch.no_grad():
            module.routing_gain_raw.fill_(gain)
        inputs = _routing_inputs(**kwargs)
        result = module(*inputs)
        assert torch.allclose(result["view_weights"], _uniform(inputs[-1]), atol=1e-6)


def test_view_weights_are_finite_masked_and_normalized():
    module = BoundedHypothesisViewRouting()
    with torch.no_grad():
        module.routing_gain_raw.fill_(0.7)
    inputs = _routing_inputs()
    weight = module(*inputs)["view_weights"]
    assert torch.isfinite(weight).all()
    assert torch.count_nonzero(weight[:, -1]) == 0
    assert torch.allclose(weight.sum(dim=1), torch.ones_like(weight[:, 0]), atol=1e-6)


def test_continuous_context_gate_vanishes_for_weak_hypotheses():
    decoder = UnifiedDensityDecoder(8, 10, 12)
    shared = torch.randn(1, 5, 8)
    candidate = torch.randn(1, 2, 8)
    points = torch.zeros(1, 5, 3)
    encoded = torch.randn(1, 5, 10)
    centers = torch.zeros(1, 2, 3)
    covariance = torch.eye(3)[None, None].expand(1, 2, -1, -1).clone()
    valid = torch.ones(1, 2, dtype=torch.bool)
    out = decoder(
        shared,
        candidate,
        points,
        encoded,
        centers,
        covariance,
        torch.zeros(1, 2),
        valid,
        continuous_applicability=True,
    )
    assert torch.count_nonzero(out["candidate_context"]) == 0


def test_analysis_threshold_changes_mask_not_reconstruction_slots():
    constructor = DiverseCandidateConstructor(mmax=2, descriptor_dim=4)
    proposal = {
        "proposal_points_mm": torch.randn(1, 2, 3, 3),
        "proposal_scores": torch.rand(1, 2, 3),
        "proposal_descriptors": torch.randn(1, 2, 3, 4),
        "proposal_valid_mask": torch.ones(1, 2, 3, dtype=torch.bool),
    }
    low = constructor(proposal)
    constructor.candidate_conf_threshold = 0.99
    high = constructor(proposal)
    assert torch.equal(low["candidate_valid_mask"], high["candidate_valid_mask"])
    assert torch.allclose(low["candidate_centers_mm"], high["candidate_centers_mm"])
    assert torch.allclose(
        low["candidate_existence_probability"], high["candidate_existence_probability"]
    )


def test_grid_nms_is_structured_and_deterministic():
    from minr_fmt.network.view_candidate_evidence import ViewCandidateEvidence

    scores = torch.zeros(1, 2, 27)
    scores[..., 13] = 1
    maxima = ViewCandidateEvidence._grid_local_maxima(scores, (3, 3, 3))
    assert maxima[..., 13].all()
    assert maxima.sum() == 2


def test_standalone_phase_b_eval_uses_full_context_and_training_warms_up():
    model = SSQFMT(_cfg(ablation="a2", separability_mode="none"))
    model.set_view_training_phase("phase_b")
    batch = make_batch(1)
    model.eval()
    with torch.no_grad():
        evaluated = model(
            batch["surface_measurements_packed"],
            batch["query_coordinates_mm"],
            detector_valid_mask=batch["detector_valid_mask"],
            depth_maps=batch["depth_maps"],
            batch=batch,
            return_diagnostics=True,
        )
    assert evaluated["diagnostics"]["candidate_context_scale"].item() == 1.0
    model.train()
    model.set_training_step(0)
    trained = model(
        batch["surface_measurements_packed"],
        batch["query_coordinates_mm"],
        detector_valid_mask=batch["detector_valid_mask"],
        depth_maps=batch["depth_maps"],
        batch=batch,
        return_diagnostics=True,
    )
    assert trained["diagnostics"]["candidate_context_scale"].item() == 0.0


def test_fixed_grid_hypotheses_are_density_query_invariant_and_threshold_is_analysis_only():
    cfg = _cfg(ablation="a3_v2_bounded_routing", separability_mode="geometry_only")
    cfg.model.ssq_fmt.view_complementary.hypothesis_grid = {
        "enabled": True,
        "spacing_mm": 6.0,
    }
    cfg.model.ssq_fmt.view_complementary.routing = {
        "enabled": True,
        "delta_logit_max": 1.0,
        "zero_init": True,
    }
    cfg.model.ssq_fmt.view_complementary.continuous_applicability = True
    model = SSQFMT(cfg).eval()
    model.set_view_training_phase("phase_b")
    batch = make_batch(1)

    def forward(points, threshold):
        model.diverse_candidate_constructor.candidate_conf_threshold = threshold
        with torch.no_grad():
            return model(
                batch["surface_measurements_packed"],
                points,
                detector_valid_mask=batch["detector_valid_mask"],
                depth_maps=batch["depth_maps"],
                batch=batch,
                return_diagnostics=True,
            )

    short = forward(batch["query_coordinates_mm"][:, :3], 0.2)
    long = forward(batch["query_coordinates_mm"], 0.5)
    for key in (
        "candidate_centers_mm",
        "candidate_existence_probability",
        "candidate_covariances_mm",
        "candidate_view_support",
        "candidate_slot_valid_mask",
    ):
        assert torch.allclose(short["diagnostics"][key], long["diagnostics"][key], atol=1e-6)
    same_queries_low = forward(batch["query_coordinates_mm"], 0.2)
    same_queries_high = forward(batch["query_coordinates_mm"], 0.5)
    assert torch.allclose(same_queries_low["density"], same_queries_high["density"], atol=1e-7)
