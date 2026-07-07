from __future__ import annotations

import numpy as np
import torch

from scripts.eval_view_complementary_full_volume_paired import (
    deterministic_proposal_indices,
    linear_indices_to_points_mm,
    paired_statistics,
    replace_forward,
)


def test_deterministic_proposal_indices_are_reproducible_unique_and_sample_specific():
    shape = (10, 11, 12)
    first = deterministic_proposal_indices(shape, 128, 42, "sample_0001")
    repeated = deterministic_proposal_indices(shape, 128, 42, "sample_0001")
    other = deterministic_proposal_indices(shape, 128, 42, "sample_0002")
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, other)
    assert len(np.unique(first)) == 128
    assert first.min() >= 0
    assert first.max() < np.prod(shape)


def test_linear_indices_to_points_mm_uses_voxel_centers():
    points = linear_indices_to_points_mm(np.asarray([0, 7]), (2, 2, 2), 0.2)
    expected = torch.tensor([[[0.1, 0.1, 0.1], [0.3, 0.3, 0.3]]])
    assert torch.allclose(points, expected)


def test_replace_forward_restores_original_method():
    module = torch.nn.Identity()
    original = module(torch.tensor([1.0]))
    with replace_forward(module, lambda value: value + 3.0):
        changed = module(torch.tensor([1.0]))
    restored = module(torch.tensor([1.0]))
    assert torch.equal(original, restored)
    assert torch.equal(changed, torch.tensor([4.0]))


def test_paired_statistics_respects_metric_direction():
    rows_a = [
        {"sample_id": "a", "dice": 0.8, "cle": 1.0},
        {"sample_id": "b", "dice": 0.7, "cle": 2.0},
    ]
    rows_b = [
        {"sample_id": "a", "dice": 0.6, "cle": 2.0},
        {"sample_id": "b", "dice": 0.6, "cle": 3.0},
    ]
    dice = paired_statistics(rows_a, rows_b, "dice", 100, 42)
    cle = paired_statistics(rows_a, rows_b, "cle", 100, 42)
    assert dice["mean_delta_a_minus_b"] > 0
    assert dice["win_rate_a_over_b"] == 1.0
    assert cle["mean_delta_a_minus_b"] < 0
    assert cle["win_rate_a_over_b"] == 1.0
