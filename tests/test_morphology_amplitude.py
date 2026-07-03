from __future__ import annotations

import torch

from minr_fmt.network.morphology_amplitude import (
    MorphologyAmplitudeLossConfig,
    MorphologyAmplitudeObjective,
)
from minr_fmt.network.unified_density_decoder import UnifiedDensityDecoder


def _decoder_inputs(batch: int = 2, queries: int = 7, candidates: int = 3):
    representation_dim = 8
    position_dim = 6
    shared = torch.randn(batch, queries, representation_dim)
    candidate = torch.randn(batch, candidates, representation_dim)
    points = torch.randn(batch, queries, 3)
    encoded = torch.randn(batch, queries, position_dim)
    centers = torch.randn(batch, candidates, 3)
    covariance = torch.eye(3).reshape(1, 1, 3, 3).expand(batch, candidates, -1, -1).clone()
    scores = torch.sigmoid(torch.randn(batch, candidates))
    valid = torch.ones(batch, candidates, dtype=torch.bool)
    return shared, candidate, points, encoded, centers, covariance, scores, valid


def test_disabled_factorization_preserves_scalar_density_exactly():
    torch.manual_seed(3)
    decoder = UnifiedDensityDecoder(8, 6, hidden_dim=12, fusion_mode="joint_nonresidual")
    out = decoder(*_decoder_inputs(), ablation="shared_only", context_scale=0.0)
    assert torch.equal(out["density"], out["amplitude"])
    assert torch.equal(out["support_probability"], torch.ones_like(out["density"]))


def test_factorized_density_is_product_and_phase_a_initialization_is_near_identity():
    torch.manual_seed(4)
    decoder = UnifiedDensityDecoder(8, 6, hidden_dim=12, fusion_mode="joint_nonresidual")
    inputs = _decoder_inputs()
    baseline = decoder(*inputs, ablation="shared_only", context_scale=0.0)["density"].detach()
    decoder.enable_factorized_output(compose_density=True, support_init_logit=8.0)
    out = decoder(*inputs, ablation="shared_only", context_scale=0.0)
    assert torch.allclose(out["density"], out["support_probability"] * out["amplitude"])
    assert torch.equal(out["amplitude"], baseline)
    assert (out["density"] - baseline).abs().max().item() < 4.0e-4


def test_support_auxiliary_mode_does_not_change_density_composition():
    torch.manual_seed(5)
    decoder = UnifiedDensityDecoder(8, 6, hidden_dim=12, fusion_mode="additive")
    decoder.enable_factorized_output(compose_density=False, support_init_logit=0.0)
    out = decoder(*_decoder_inputs(), ablation="shared_only", context_scale=0.0)
    assert torch.equal(out["density"], out["amplitude"])
    assert not torch.equal(out["support_probability"], torch.ones_like(out["density"]))


def test_factorized_objective_is_finite_and_rewards_correct_decomposition():
    target = torch.tensor([[[0.0], [0.2], [0.8], [0.0]]])
    component_ids = torch.tensor([[-1, 0, 1, -1]])
    objective = MorphologyAmplitudeObjective(
        MorphologyAmplitudeLossConfig(
            support_threshold=0.05,
            support_weight=1.0,
            amplitude_weight=1.0,
            component_weight=1.0,
        )
    )
    good_support = torch.tensor([[[0.01], [0.99], [0.99], [0.01]]])
    bad_support = 1.0 - good_support
    good = objective(
        support_probability=good_support,
        amplitude=target,
        density=good_support * target,
        target_density=target,
        component_ids=component_ids,
    )
    bad = objective(
        support_probability=bad_support,
        amplitude=1.0 - target,
        density=bad_support * (1.0 - target),
        target_density=target,
        component_ids=component_ids,
    )
    assert torch.isfinite(good["total"])
    assert torch.isfinite(bad["total"])
    assert good["total"] < bad["total"]
