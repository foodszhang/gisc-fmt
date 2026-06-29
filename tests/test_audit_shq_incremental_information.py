import importlib.util
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from minr_fmt.loss import MorphologyAwareDensityLoss

SPEC = importlib.util.spec_from_file_location(
    "audit_shq", Path(__file__).parents[1] / "scripts" / "audit_shq_incremental_information.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def cached(m=5, q=11):
    torch.manual_seed(3)
    d = {
        "q_s": torch.randn(q, 4),
        "q_m": torch.randn(q, m, 4),
        "query_coordinate_encoding": torch.randn(q, 6),
        "relative_query": torch.randn(q, m, 3),
        "candidate_score": torch.rand(q, m),
        "spatial_prior": torch.rand(q, m),
        "quotient_support": torch.rand(q, m),
        "quotient_dispersion": torch.rand(q, m),
        "candidate_covariance_eigen_scales": torch.rand(q, m, 3),
        "candidate_valid_mask": torch.rand(q, m) > 0.2,
        "alpha": torch.rand(q, m),
        "shared_logit": torch.randn(q, 1),
        "shared_density": torch.rand(q, 1),
        "ground_truth_density": (torch.rand(q, 1) > 0.7).float(),
        "measurement_supported": torch.ones(q, 1, dtype=torch.bool),
        "sample_id": torch.arange(q) // 3,
        "query_id": torch.arange(q),
        "source_count": torch.arange(q) % 3 + 1,
    }
    d["q_m_minus_q_s"] = d["q_m"] - d["q_s"][:, None]
    d["alpha"] *= d["candidate_valid_mask"]
    d["alpha"] /= d["alpha"].sum(-1, keepdim=True).clamp_min(1e-6)
    return d


def loss_config():
    return {
        "pos_weight": 3.0,
        "dice_weight": 0.5,
        "sparse_weight": 0.1,
        "density_bce_weight": 0.2,
        "tversky_weight": 0.3,
        "tversky_alpha": 0.6,
        "tversky_beta": 0.4,
        "tversky_gamma": 1.33,
    }


def test_s0_s1_s2_parameter_identity_and_masks_only_zero_intended_groups():
    data = cached()
    features, dims = audit.build_candidate_features(data)
    counts = []
    slices = audit.feature_slices(dims)
    for variant in audit.VARIANTS:
        counts.append(audit.parameter_count(audit.NestedCandidateProbe(features.shape[-1])))
        masked = audit.apply_feature_mask(features, variant, dims)
        for group, enabled in audit.FEATURE_MASKS[variant].items():
            actual = masked[..., slices[group]]
            expected = features[..., slices[group]] if enabled else torch.zeros_like(actual)
            assert torch.equal(actual, expected)
    assert len(set(counts)) == 1


def test_same_cached_tensors_and_query_ids_across_variants_and_seeds():
    data = cached()
    features, dims = audit.build_candidate_features(data)
    pointers = {name: value.data_ptr() for name, value in data.items()}
    query_ids = data["query_id"].clone()
    for seed in range(5):
        torch.manual_seed(seed)
        for variant in audit.VARIANTS:
            audit.apply_feature_mask(features, variant, dims)
            assert torch.equal(data["query_id"], query_ids)
            assert pointers == {name: value.data_ptr() for name, value in data.items()}


def test_candidate_permutation_invariance_invalid_masking_and_m_0_1_5():
    for m in (0, 1, 5):
        data = cached(m=m)
        features, _ = audit.build_candidate_features(data)
        probe = audit.NestedCandidateProbe(features.shape[-1])
        actual = probe(features, data["alpha"], data["candidate_valid_mask"])
        assert actual.shape == (11, 1)
        if m:
            permutation = torch.randperm(m)
            expected = probe(
                features[:, permutation],
                data["alpha"][:, permutation],
                data["candidate_valid_mask"][:, permutation],
            )
            assert torch.allclose(actual, expected, atol=1e-6)
            changed = features.clone()
            changed[~data["candidate_valid_mask"]] += 1000
            alpha = data["alpha"].clone()
            alpha[~data["candidate_valid_mask"]] = 1000
            assert torch.allclose(
                actual, probe(changed, alpha, data["candidate_valid_mask"]), atol=1e-6
            )
        none = torch.zeros_like(data["candidate_valid_mask"])
        assert torch.equal(
            probe(features, torch.ones_like(data["alpha"]), none), torch.zeros(11, 1)
        )


def test_formal_density_loss_reuses_final_density_formula_components():
    data = cached()
    pred = torch.sigmoid(data["shared_logit"])
    total, parts = audit.formal_density_loss(
        pred,
        data["ground_truth_density"],
        data["measurement_supported"],
        data["sample_id"],
        loss_config(),
    )
    expected = (
        parts["density"]
        + 0.2 * parts["density_bce"]
        + 0.5 * parts["dice"]
        + 0.3 * parts["tversky"]
        + 0.1 * parts["sparse"]
    )
    assert torch.allclose(total, expected)
    assert set(parts) == set(audit.LOSS_COMPONENTS)

    # Compare directly with the production loss on the same complete per-sample layout.
    prediction = pred[:9].reshape(3, 3, 1)
    target = data["ground_truth_density"][:9].reshape(3, 3, 1)
    support = data["measurement_supported"][:9].reshape(3, 3)
    production = MorphologyAwareDensityLoss(**loss_config())(
        prediction,
        target,
        {"measurement_supported": support},
    )
    flattened, flattened_parts = audit.formal_density_loss(
        prediction.flatten(0, 1),
        target.flatten(0, 1),
        support.flatten().unsqueeze(-1),
        torch.arange(3).repeat_interleave(3),
        loss_config(),
    )
    assert torch.allclose(flattened, production["total_loss"])
    for local_name, production_name in {
        "density": "density_loss",
        "density_bce": "density_bce_loss",
        "dice": "dice_loss",
        "tversky": "tversky_loss",
        "sparse": "sparse_loss",
    }.items():
        assert torch.allclose(flattened_parts[local_name], production[production_name])


def test_task_gradient_direction_is_negative_formal_loss_gradient():
    data = cached()
    direction = audit.task_descent_direction(
        data["shared_logit"],
        data["ground_truth_density"],
        data["measurement_supported"],
        data["sample_id"],
        loss_config(),
    )
    step = 1e-3
    before, _ = audit.formal_density_loss(
        torch.sigmoid(data["shared_logit"]),
        data["ground_truth_density"],
        data["measurement_supported"],
        data["sample_id"],
        loss_config(),
    )
    after, _ = audit.formal_density_loss(
        torch.sigmoid(data["shared_logit"] + step * direction),
        data["ground_truth_density"],
        data["measurement_supported"],
        data["sample_id"],
        loss_config(),
    )
    assert after < before


def test_sample_level_paired_bootstrap_not_query_level():
    left = np.array([0.1, 0.2, 0.3])
    right = np.array([0.2, 0.3, 0.4])
    result = audit.paired_bootstrap(left, right, repetitions=1000, seed=7)
    assert np.isclose(result["mean_difference"], 0.1)
    assert np.isclose(result["ci_lower"], 0.1)
    assert np.isclose(result["ci_upper"], 0.1)


def decision_args():
    return Namespace(
        quotient_dice_gain=0.005,
        metadata_dice_gain=0.005,
        negligible_gain=0.003,
        required_positive_seeds=4,
    )


def comparison(mean, positive, low, high, loss=-0.01):
    return {
        "dice": {"mean": mean, "positive_seed_count": positive},
        "formal_density_loss": {"mean": loss},
        "bootstrap_dice": {"ci_lower": low, "ci_upper": high},
    }


def test_automatic_conclusion_rules():
    variants = {name: {"descent_alignment": {"mean": 0.2}} for name in audit.VARIANTS}
    case1 = {
        "S1_minus_S0": comparison(0.006, 5, 0.002, 0.01),
        "S2_minus_S1": comparison(0.006, 4, 0.001, 0.011),
        "S2_minus_S0": comparison(0.012, 5, 0.005, 0.02),
    }
    text, _ = audit.automatic_conclusion(case1, variants, decision_args())
    assert "quotient difference provides reproducible" in text
    case2 = {
        "S1_minus_S0": comparison(0.006, 5, 0.002, 0.01),
        "S2_minus_S1": comparison(0.001, 3, -0.002, 0.004),
        "S2_minus_S0": comparison(0.007, 5, 0.002, 0.012),
    }
    text, _ = audit.automatic_conclusion(case2, variants, decision_args())
    assert "metadata provide incremental value" in text
    case3 = {
        "S1_minus_S0": comparison(0.001, 3, -0.002, 0.003),
        "S2_minus_S1": comparison(0.001, 3, -0.002, 0.003),
        "S2_minus_S0": comparison(0.002, 3, -0.002, 0.004),
    }
    text, _ = audit.automatic_conclusion(case3, variants, decision_args())
    assert "additional decoder capacity" in text


def test_protocol_hash_stability_and_sensitivity():
    payload = {"b": [2, 3], "a": {"x": 1}}
    assert audit.protocol_hash(payload) == audit.protocol_hash({"a": {"x": 1}, "b": [2, 3]})
    assert audit.protocol_hash(payload) != audit.protocol_hash({"a": {"x": 2}, "b": [2, 3]})


def test_no_gradient_into_frozen_shq_model_or_alpha():
    main = torch.nn.Linear(3, 4)
    main.requires_grad_(False)
    values = main(torch.randn(7, 5, 3)).detach()
    alpha = torch.rand(7, 5, requires_grad=True)
    probe = audit.NestedCandidateProbe(4)
    probe(values, alpha, torch.ones(7, 5, dtype=torch.bool)).sum().backward()
    assert all(parameter.grad is None for parameter in main.parameters())
    assert alpha.grad is None
    assert all(parameter.grad is not None for parameter in probe.parameters())
