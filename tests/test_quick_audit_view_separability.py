import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

SPEC = importlib.util.spec_from_file_location(
    "quick_sep", Path(__file__).parents[1] / "scripts" / "quick_audit_view_separability.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def fixture():
    evidence = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]]]
    )
    centers = torch.tensor(
        [[[[0.0, 0.0], [2.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [3.0, 0.0]]]]
    )
    scales = torch.ones(1, 3, 2)
    valid = torch.ones(1, 3, 2, dtype=torch.bool)
    return evidence, centers, scales, valid


def test_score_range_and_pair_symmetry():
    evidence, centers, scales, valid = fixture()
    geo, meas, combined, usable = audit.pairwise_separability(evidence, centers, scales, valid)
    for score in (geo, meas, combined):
        assert bool(((score >= 0) & (score <= 1)).all())
        assert torch.allclose(score[..., 0], score[..., 1])
    assert usable.all()


def test_top_bottom_two_and_invalid_masking():
    score = torch.tensor([[[0.1], [0.8], [0.4], [0.9]]])
    valid = torch.ones_like(score, dtype=torch.bool)
    high = audit.select_view_weights(score, valid, "high")
    low = audit.select_view_weights(score, valid, "low")
    assert high.squeeze().nonzero().flatten().tolist() == [1, 3]
    assert low.squeeze().nonzero().flatten().tolist() == [0, 2]
    valid[:, 3] = False
    weights = audit.select_view_weights(score, valid, "sep_aware")
    assert weights[:, 3].eq(0).all()
    assert torch.allclose(weights.sum(dim=-2), torch.ones_like(weights.sum(dim=-2)))


def test_shuffle_control_preserves_valid_values_and_invalid_slots():
    score = torch.arange(14.0).reshape(1, 7, 2)
    valid = torch.ones_like(score, dtype=torch.bool)
    valid[:, -1, 0] = False
    shuffled = audit.shuffle_view_scores(score, valid, torch.Generator().manual_seed(4))
    for candidate in range(2):
        assert sorted(shuffled[..., candidate][valid[..., candidate]].tolist()) == sorted(
            score[..., candidate][valid[..., candidate]].tolist()
        )
    assert shuffled[..., 0][~valid[..., 0]].eq(score[..., 0][~valid[..., 0]]).all()
    assert not torch.equal(shuffled[valid], score[valid])


def test_no_gradient_into_main_model():
    main = nn.Linear(3, 4)
    for parameter in main.parameters():
        parameter.requires_grad_(False)
    frozen = main(torch.randn(5, 3)).detach()
    probe = audit.DensityProbe(4, hidden_dim=8)
    probe(frozen).sum().backward()
    assert all(parameter.grad is None for parameter in main.parameters())
    assert all(parameter.grad is not None for parameter in probe.parameters())
