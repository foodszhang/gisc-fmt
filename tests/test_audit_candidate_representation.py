import copy
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

SPEC = importlib.util.spec_from_file_location(
    "candidate_audit", Path(__file__).parents[1] / "scripts" / "audit_candidate_representation.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_hungarian_matching_and_duplicate_detection():
    candidates = torch.tensor([[0.0, 0, 0], [0.2, 0, 0], [10.0, 0, 0]])
    components = torch.tensor([[0.0, 0, 0], [10.0, 0, 0]])
    result = audit.hungarian_candidate_matching(
        candidates,
        torch.ones(3, dtype=torch.bool),
        components,
        torch.ones(2, dtype=torch.bool),
        1.0,
    )
    assert result["component_coverage_rate"] == 1.0
    assert result["candidate_matching_rate"] == 2 / 3
    assert result["duplicate_candidate_rate"] == 1 / 3
    assert result["unmatched_candidate_rate"] == 1 / 3


def test_cosine_l2_metrics():
    result = audit.paired_feature_metrics(
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    assert torch.allclose(result["cosine"], torch.tensor([1.0, 0.0]))
    assert torch.allclose(result["l2"], torch.tensor([0.0, 2.0**0.5]))


def test_routing_entropy():
    valid = torch.ones(2, 3, dtype=torch.bool)
    entropy = audit.routing_entropy(torch.tensor([[1 / 3, 1 / 3, 1 / 3], [1.0, 0.0, 0.0]]), valid)
    assert torch.isclose(entropy[0], torch.tensor(1.0))
    assert entropy[1] < 1e-5


def test_top_bottom_gap_and_ranking_stability():
    score = torch.tensor([[0.0, 0.1, 0.8, 0.9]])
    valid = torch.ones_like(score, dtype=torch.bool)
    assert torch.allclose(audit.top_bottom_score_gap(score, valid), torch.tensor([0.8]))
    stable = audit.ranking_perturbation_stability(score, valid, noise_std=0.001, repeats=20)
    assert stable.item() == 1.0


def test_gradient_norm_and_gated_ungated_backward_without_weight_update():
    decoder = nn.Linear(3, 1)
    before = copy.deepcopy(decoder.state_dict())
    context = torch.randn(2, 4, 3, 3)
    encoded = torch.empty(0)  # ignored by this adapter

    class Adapter(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, x, _encoded):
            return self.layer(x)

    adapter = Adapter(decoder)
    result = audit.compare_gated_ungated_backward(
        adapter,
        context,
        encoded,
        torch.zeros(2, 4, 1),
        torch.full((2, 4), 0.01),
        torch.full((2, 4, 3), 1 / 3),
        torch.rand(2, 4, 1),
    )
    assert result["gated"] > 0
    assert result["ungated_over_gated"] > 5
    assert audit.collect_gradient_norm(adapter) > 0
    for key, value in decoder.state_dict().items():
        assert torch.equal(value, before[key])
