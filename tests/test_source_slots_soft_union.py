import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minr_fmt.models.minr_fmt import (
    EvidenceAwareOwnershipNet,
    PointDensityNet,
    SourceInstanceFieldDecoder,
)


def _make_net(merge="soft_union", ownership_mode="evidence_aware", use_all_slots=True):
    net = PointDensityNet.__new__(PointDensityNet)
    nn.Module.__init__(net)
    net.source_instance_decoder_enabled = True
    net.source_instance_decoder_use_all_slots = use_all_slots
    net.source_instance_decoder_top_k = None if use_all_slots else 2
    net.source_instance_decoder_merge = merge
    net.source_instance_ownership_mode = ownership_mode
    net.source_instance_ownership_temperature = 1.0
    net.source_instance_detach_ownership_feature = False
    net.source_instance_tau_s_mm = 5.0
    net.source_instance_decoder_local_dim = 8
    net.source_instance_decoder_output_components = True
    net.source_instance_decoder = SourceInstanceFieldDecoder(
        feature_dim=4, hidden_dim=8, local_dim=8
    )
    if ownership_mode == "evidence_aware":
        net.source_ownership_net = EvidenceAwareOwnershipNet(
            feature_dim=4, hidden_dim=8, geom_dim=6
        )
    net.last_source_decoder_stats = {}
    return net


def _inputs():
    points_mm = torch.tensor(
        [[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]]], dtype=torch.float32
    )
    source_hypotheses = {
        "centers": torch.tensor([[[0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [40.0, 0.0, 0.0]]]),
        "peak_scores": torch.tensor([[1.0, 0.8, 10.0]]),
        "scales": torch.tensor([[1.0, 1.0, 1.0]]),
        "valid": torch.tensor([[1.0, 1.0, 0.0]]),
    }
    canonical = torch.randn(1, 3, 4)
    pcfs = torch.randn(1, 3, 4)
    return points_mm, source_hypotheses, canonical, pcfs


def test_soft_union_shape():
    net = _make_net()
    points_mm, source_hypotheses, canonical, pcfs = _inputs()
    decoded = net._source_instance_decode(points_mm, source_hypotheses, canonical, pcfs)
    assert decoded is not None
    logits, aux = decoded
    assert logits.shape == (1, 3, 1)
    assert torch.isfinite(logits).all()
    assert aux["source_component_logits"].shape == (1, 3, 3, 1)
    assert aux["source_component_ownership"].shape == (1, 3, 3)
    assert aux["source_component_valid"].shape == (1, 3)


def test_invalid_slot_masking():
    net = _make_net()
    points_mm, source_hypotheses, canonical, pcfs = _inputs()
    logits, aux = net._source_instance_decode(points_mm, source_hypotheses, canonical, pcfs)
    invalid = (~aux["source_component_valid"][:, None, :, None]).expand_as(
        aux["source_component_prob"]
    )
    assert torch.allclose(aux["source_component_ownership"][..., 2], torch.zeros(1, 3))
    assert torch.allclose(
        aux["source_component_prob"][invalid],
        torch.zeros_like(aux["source_component_prob"][invalid]),
    )
    assert torch.isfinite(logits).all()


def test_no_hypothesis_fallback():
    net = _make_net()
    points_mm, _source_hypotheses, canonical, pcfs = _inputs()
    assert net._source_instance_decode(points_mm, None, canonical, pcfs) is None
    assert net.last_source_decoder_stats["fallback_density_head"].item() == 1.0


def test_distance_ownership_compatibility():
    net = _make_net(ownership_mode="distance", use_all_slots=False)
    points_mm, source_hypotheses, canonical, pcfs = _inputs()
    logits, aux = net._source_instance_decode(points_mm, source_hypotheses, canonical, pcfs)
    assert logits.shape == (1, 3, 1)
    assert aux["source_component_logits"].shape == (1, 3, 2, 1)
    assert torch.isfinite(logits).all()


def test_weighted_sum_compatibility():
    net = _make_net(merge="weighted_sum", ownership_mode="distance", use_all_slots=False)
    points_mm, source_hypotheses, canonical, pcfs = _inputs()
    logits, aux = net._source_instance_decode(points_mm, source_hypotheses, canonical, pcfs)
    assert logits.shape == (1, 3, 1)
    assert aux["source_component_ownership"].shape == (1, 3, 2)
    assert torch.isfinite(logits).all()
