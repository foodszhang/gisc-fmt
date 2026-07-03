from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from minr_fmt.network.morphology_amplitude import (
    MorphologyAmplitudeLossConfig,
    MorphologyAmplitudeObjective,
    attach_factorized_unified_output,
    load_factorized_or_historical_checkpoint,
    load_historical_phase_a_checkpoint,
)
from minr_fmt.network.unified_density_decoder import UnifiedDensityDecoder


class _DummyPhaseANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.surface_encoder = nn.Linear(2, 2)
        self.surface_sampler = nn.Linear(2, 2)
        self.complementary_aggregation = nn.Linear(8, 8)
        self.unified_density_decoder = UnifiedDensityDecoder(
            representation_dim=8,
            position_dim=6,
            hidden_dim=12,
            fusion_mode="joint_nonresidual",
        )


def _decoder_inputs(batch: int = 2, queries: int = 7, candidates: int = 3):
    shared = torch.randn(batch, queries, 8)
    candidate = torch.randn(batch, candidates, 8)
    points = torch.randn(batch, queries, 3)
    encoded = torch.randn(batch, queries, 6)
    centers = torch.randn(batch, candidates, 3)
    covariance = (
        torch.eye(3)
        .reshape(1, 1, 3, 3)
        .expand(batch, candidates, -1, -1)
        .clone()
    )
    scores = torch.sigmoid(torch.randn(batch, candidates))
    valid = torch.ones(batch, candidates, dtype=torch.bool)
    return shared, candidate, points, encoded, centers, covariance, scores, valid


def test_factorized_unified_output_preserves_phase_a_amplitude_at_attachment():
    torch.manual_seed(4)
    net = _DummyPhaseANet()
    inputs = _decoder_inputs()
    baseline = net.unified_density_decoder(
        *inputs,
        ablation="shared_only",
        context_scale=0.0,
    )["density"].detach()
    attach_factorized_unified_output(
        net,
        {"enabled": True, "compose_density": True, "support_init_logit": 8.0},
    )
    out = net.unified_density_decoder(
        *inputs,
        ablation="shared_only",
        context_scale=0.0,
    )
    assert torch.equal(out["amplitude"], baseline)
    assert torch.equal(
        out["support_probability"],
        torch.sigmoid(out["support_logits"]),
    )
    assert torch.allclose(
        out["density"],
        out["support_probability"] * out["amplitude"],
    )
    assert (out["density"] - baseline).abs().max().item() < 4.0e-4


def test_support_auxiliary_mode_keeps_original_density_output():
    torch.manual_seed(5)
    net = _DummyPhaseANet()
    attach_factorized_unified_output(
        net,
        {"enabled": True, "compose_density": False, "support_init_logit": 0.0},
    )
    out = net.unified_density_decoder(
        *_decoder_inputs(),
        ablation="shared_only",
        context_scale=0.0,
    )
    assert torch.equal(out["density"], out["amplitude"])
    assert not torch.equal(
        out["support_probability"],
        torch.ones_like(out["density"]),
    )


def test_factorization_rejects_candidate_conditioned_phase_b_or_c_calls():
    net = _DummyPhaseANet()
    attach_factorized_unified_output(
        net,
        {"enabled": True, "compose_density": True, "support_init_logit": 8.0},
    )
    with pytest.raises(RuntimeError, match="cannot be used in Phase B/C"):
        net.unified_density_decoder(
            *_decoder_inputs(),
            ablation="full",
            context_scale=1.0,
        )


def test_historical_loader_allows_only_later_candidate_heads(tmp_path: Path):
    source = _DummyPhaseANet()
    state = {
        f"net.{key}": value.detach().clone()
        for key, value in source.state_dict().items()
        if not key.startswith(
            (
                "unified_density_decoder.candidate_residual_head.",
                "unified_density_decoder.candidate_branch_head.",
            )
        )
    }
    checkpoint = tmp_path / "phase_a.ckpt"
    torch.save({"state_dict": state, "epoch": 35, "global_step": 100}, checkpoint)

    target = _DummyPhaseANet()
    report = load_historical_phase_a_checkpoint(target, checkpoint)
    assert report["epoch"] == 35
    assert report["allowed_current_extensions"]
    assert all(
        key.startswith(
            (
                "unified_density_decoder.candidate_residual_head.",
                "unified_density_decoder.candidate_branch_head.",
            )
        )
        for key in report["allowed_current_extensions"]
    )
    for key, value in source.state_dict().items():
        if key.startswith(
            (
                "unified_density_decoder.candidate_residual_head.",
                "unified_density_decoder.candidate_branch_head.",
            )
        ):
            continue
        assert torch.equal(target.state_dict()[key], value)


def test_factorized_checkpoint_load_is_strict(tmp_path: Path):
    source = _DummyPhaseANet()
    attach_factorized_unified_output(
        source,
        {"enabled": True, "compose_density": True, "support_init_logit": 8.0},
    )
    checkpoint = tmp_path / "factorized.ckpt"
    torch.save(
        {"state_dict": {f"net.{k}": v for k, v in source.state_dict().items()}},
        checkpoint,
    )

    target = _DummyPhaseANet()
    attach_factorized_unified_output(
        target,
        {"enabled": True, "compose_density": True, "support_init_logit": 8.0},
    )
    report = load_factorized_or_historical_checkpoint(target, checkpoint)
    assert report["strict_factorized"] is True
    for key, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], value)


def test_factorized_objective_is_amp_safe_with_support_logits():
    target = torch.tensor([[[0.0], [0.2], [0.8], [0.0]]])
    support_logits = torch.tensor(
        [[[-4.0], [4.0], [4.0], [-4.0]]],
        requires_grad=True,
    )
    amplitude = target.detach().clone().requires_grad_(True)
    objective = MorphologyAmplitudeObjective(
        MorphologyAmplitudeLossConfig(
            support_threshold=0.05,
            support_weight=1.0,
            amplitude_weight=1.0,
        )
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        support_probability = torch.sigmoid(support_logits)
        result = objective(
            support_logits=support_logits,
            support_probability=support_probability,
            amplitude=amplitude,
            density=support_probability * amplitude,
            target_density=target,
        )
    assert result["support_bce"].dtype == torch.float32
    assert torch.isfinite(result["total"])
    result["total"].backward()
    assert support_logits.grad is not None
    assert torch.isfinite(support_logits.grad).all()


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
        support_logits=torch.logit(good_support),
        support_probability=good_support,
        amplitude=target,
        density=good_support * target,
        target_density=target,
        component_ids=component_ids,
    )
    bad = objective(
        support_logits=torch.logit(bad_support),
        support_probability=bad_support,
        amplitude=1.0 - target,
        density=bad_support * (1.0 - target),
        target_density=target,
        component_ids=component_ids,
    )
    assert torch.isfinite(good["total"])
    assert torch.isfinite(bad["total"])
    assert good["total"] < bad["total"]
