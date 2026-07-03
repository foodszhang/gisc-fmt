from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from minr_fmt.network.morphology_amplitude import (
    MorphologyAmplitudeLossConfig,
    MorphologyAmplitudeObjective,
    activate_legacy_phase_a_decoder,
    load_legacy_phase_a_checkpoint,
)
from minr_fmt.network.ssq_decoder import SharedDensityLogitDecoder


class _DummyAggregation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.projection = nn.Linear(dim, dim)

    def aggregate_shared(self, per_view: torch.Tensor, valid: torch.Tensor):
        weight = valid.to(per_view.dtype)
        weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        shared = (per_view * weight[..., None]).sum(dim=1)
        return self.projection(shared), weight


class _DummyLegacyPhaseANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.surface_encoder = nn.Linear(2, 2)
        self.surface_sampler = nn.Linear(2, 2)
        self.complementary_aggregation = _DummyAggregation(8)
        self.shared_density_logit_decoder = SharedDensityLogitDecoder(
            8,
            6,
            hidden_dim=12,
            positive_ratio=0.03,
            query_chunk_size=0,
        )
        self.unified_density_decoder = nn.Linear(8, 1)

    def _encoded_query_points(self, points_mm: torch.Tensor) -> torch.Tensor:
        return torch.cat([points_mm, points_mm], dim=-1)

    def _forward_clean_shared_ablation(self, *args, **kwargs):
        raise AssertionError("activation should replace this method")


def _forward_inputs():
    torch.manual_seed(4)
    per_view = torch.randn(2, 3, 7, 8)
    valid = torch.ones(2, 3, 7, dtype=torch.bool)
    points = torch.randn(2, 7, 3)
    norm = torch.ones(2, 3)
    return per_view, valid, points, norm


def test_legacy_phase_a_disabled_mode_preserves_old_scalar_function():
    net = _DummyLegacyPhaseANet()
    per_view, valid, points, norm = _forward_inputs()
    shared, _ = net.complementary_aggregation.aggregate_shared(per_view, valid)
    expected = torch.sigmoid(
        net.shared_density_logit_decoder(shared, net._encoded_query_points(points))
    )
    activate_legacy_phase_a_decoder(net, {"enabled": False, "compose_density": True})
    out = net._forward_clean_shared_ablation(per_view, valid, points, norm, False)
    assert torch.equal(out["density"], expected)
    assert torch.equal(net.last_factorized_outputs["amplitude"], expected)


def test_legacy_factorized_density_is_product_and_starts_near_identity():
    net = _DummyLegacyPhaseANet()
    per_view, valid, points, norm = _forward_inputs()
    shared, _ = net.complementary_aggregation.aggregate_shared(per_view, valid)
    baseline = torch.sigmoid(
        net.shared_density_logit_decoder(shared, net._encoded_query_points(points))
    ).detach()
    activate_legacy_phase_a_decoder(
        net,
        {"enabled": True, "compose_density": True, "support_init_logit": 8.0},
    )
    out = net._forward_clean_shared_ablation(per_view, valid, points, norm, False)
    cached = net.last_factorized_outputs
    assert torch.allclose(out["density"], cached["support_probability"] * cached["amplitude"])
    assert torch.equal(cached["amplitude"], baseline)
    assert (out["density"] - baseline).abs().max().item() < 4.0e-4


def test_support_auxiliary_mode_keeps_historical_density_output():
    net = _DummyLegacyPhaseANet()
    per_view, valid, points, norm = _forward_inputs()
    activate_legacy_phase_a_decoder(
        net,
        {"enabled": True, "compose_density": False, "support_init_logit": 0.0},
    )
    out = net._forward_clean_shared_ablation(per_view, valid, points, norm, False)
    cached = net.last_factorized_outputs
    assert torch.equal(out["density"], cached["amplitude"])
    assert not torch.equal(cached["support_probability"], torch.ones_like(out["density"]))


def test_controlled_checkpoint_load_allows_only_inactive_unified_branch(tmp_path: Path):
    source = _DummyLegacyPhaseANet()
    state = {
        f"net.{key}": value.detach().clone()
        for key, value in source.state_dict().items()
        if not key.startswith("unified_density_decoder.")
    }
    checkpoint = tmp_path / "phase_a.ckpt"
    torch.save({"state_dict": state, "epoch": 8, "global_step": 123}, checkpoint)

    target = _DummyLegacyPhaseANet()
    report = load_legacy_phase_a_checkpoint(target, checkpoint)
    assert report["epoch"] == 8
    assert report["allowed_missing_keys"]
    assert all(key.startswith("unified_density_decoder.") for key in report["allowed_missing_keys"])
    for key, value in source.state_dict().items():
        if key.startswith("unified_density_decoder."):
            continue
        assert torch.equal(target.state_dict()[key], value)


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
