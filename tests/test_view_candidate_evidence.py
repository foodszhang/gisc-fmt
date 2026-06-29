import torch

from minr_fmt.network.view_candidate_evidence import ViewCandidateEvidence


def _inputs():
    torch.manual_seed(7)
    features = torch.randn(1, 3, 12, 8)
    geometry = torch.randn(1, 3, 12, 4)
    points = torch.stack(
        [torch.linspace(0, 11, 12), torch.zeros(12), torch.zeros(12)], dim=-1
    )[None]
    valid = torch.ones(1, 3, 12, dtype=torch.bool)
    return features, geometry, points, valid


def test_per_view_proposals_are_independent_before_association():
    module = ViewCandidateEvidence(8, hidden_dim=16, topk_per_view=4, nms_radius_mm=0.4)
    features, geometry, points, valid = _inputs()
    first = module(features, geometry, points, valid)
    changed = features.clone()
    changed[:, 1] += 4.0
    second = module(changed, geometry, points, valid)

    assert torch.equal(first["evidence"][:, 0], second["evidence"][:, 0])
    assert not torch.equal(first["evidence"][:, 1], second["evidence"][:, 1])
    assert torch.equal(first["evidence"][:, 2], second["evidence"][:, 2])


def test_local_maximum_selection_avoids_neighboring_duplicates():
    module = ViewCandidateEvidence(8, hidden_dim=16, topk_per_view=6, nms_radius_mm=1.1)
    features, geometry, points, valid = _inputs()
    out = module(features, geometry, points, valid)
    for view in range(out["proposal_query_indices"].shape[1]):
        selected = out["proposal_query_indices"][0, view][
            out["proposal_valid_mask"][0, view]
        ]
        if selected.numel() > 1:
            distance = (selected[:, None] - selected[None, :]).abs()
            distance.fill_diagonal_(2)
            assert torch.all(distance > 1)
