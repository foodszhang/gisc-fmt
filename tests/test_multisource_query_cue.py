import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minr_fmt.models.gisc_multisource import MultiSourceGISCFMT


def _make_net(dim: int = 10) -> MultiSourceGISCFMT:
    net = MultiSourceGISCFMT.__new__(MultiSourceGISCFMT)
    nn.Module.__init__(net)
    net.source_instance_cue_enabled = True
    net.source_instance_cue_dim = dim
    net.source_instance_tau_s_mm = 5.0
    net.source_cue_aggregation = "all_candidates_null"
    net.source_cue_null_logit = 0.0
    net.source_cue_score_weight = 2.0
    net.source_cue_min_scale_mm = 1.0
    net.source_cue_max_scale_mm = 20.0
    net.last_source_cue_stats = {}
    net.last_source_nearest_dist = None
    return net


def _hypotheses():
    return {
        "centers": torch.tensor(
            [[[0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [16.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
        "peak_scores": torch.tensor([[1.0, 0.8, 0.7]], dtype=torch.float32),
        "scales": torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32),
        "valid": torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
    }


def test_all_candidate_cue_is_permutation_invariant():
    net = _make_net()
    points = torch.tensor([[[2.0, 0.0, 0.0], [14.0, 0.0, 0.0]]])
    hyp = _hypotheses()
    cue = net._source_instance_cue(points, hyp)

    perm = torch.tensor([2, 0, 1])
    permuted = {key: value[:, perm] for key, value in hyp.items()}
    cue_permuted = net._source_instance_cue(points, permuted)

    assert cue.shape == (1, 2, 10)
    assert torch.allclose(cue, cue_permuted, atol=1e-6, rtol=1e-6)


def test_third_candidate_changes_query_conditioning():
    net = _make_net()
    points = torch.tensor([[[16.0, 0.0, 0.0]]])
    hyp = _hypotheses()
    cue_with_third = net._source_instance_cue(points, hyp)

    hyp_without_third = {key: value.clone() for key, value in hyp.items()}
    hyp_without_third["valid"][0, 2] = 0.0
    cue_without_third = net._source_instance_cue(points, hyp_without_third)

    # The support channel is first. A nearby third source must materially increase it.
    assert cue_with_third[..., 0].item() > cue_without_third[..., 0].item() + 0.1
    assert not torch.allclose(cue_with_third, cue_without_third)


def test_null_support_dominates_far_from_all_candidates():
    net = _make_net()
    points = torch.tensor([[[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]])
    cue = net._source_instance_cue(points, _hypotheses())

    near_support = cue[0, 0, 0]
    far_support = cue[0, 1, 0]
    assert near_support > far_support
    assert far_support < 0.05
    assert torch.isfinite(cue).all()


def test_empty_candidate_set_returns_zero_conditioning():
    net = _make_net(dim=8)
    points = torch.tensor([[[3.0, 2.0, 1.0]]])
    hyp = _hypotheses()
    hyp["valid"].zero_()

    cue = net._source_instance_cue(points, hyp)
    assert cue.shape == (1, 1, 8)
    assert torch.allclose(cue, torch.zeros_like(cue), atol=1e-6)
    assert torch.isfinite(cue).all()
