import torch

from minr_fmt.network.diverse_candidate_constructor import DiverseCandidateConstructor


def _proposals():
    points = torch.tensor(
        [[[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], [[8.0, 0.0, 0.0], [8.2, 0.0, 0.0]]]]
    )
    scores = torch.tensor([[[0.9, 0.8], [0.85, 0.7]]])
    descriptors = torch.nn.functional.normalize(torch.randn(1, 2, 2, 6), dim=-1)
    return {
        "proposal_points_mm": points,
        "proposal_scores": scores,
        "proposal_descriptors": descriptors,
        "proposal_valid_mask": torch.ones(1, 2, 2, dtype=torch.bool),
    }


def test_diverse_initialization_suppresses_near_duplicates_and_assignment_competes():
    constructor = DiverseCandidateConstructor(
        mmax=2, score_threshold=0.01, sigma_nms_mm=2.0, refinement_steps=1
    )
    out = constructor(_proposals())
    distance = torch.linalg.vector_norm(
        out["candidate_centers_mm"][:, 0] - out["candidate_centers_mm"][:, 1], dim=-1
    )
    assert torch.all(distance > 4.0)
    assignment = out["proposal_assignment"]
    assert torch.allclose(assignment.sum(dim=1), torch.ones_like(assignment[:, 0]), atol=1e-6)
    covariance = out["candidate_covariances_mm"]
    diagonal = torch.diagonal(covariance, dim1=-2, dim2=-1)
    assert torch.count_nonzero(covariance - torch.diag_embed(diagonal)) == 0
    assert torch.all((out["candidate_view_support"] >= 0) & (out["candidate_view_support"] <= 1))


def test_proposal_order_does_not_change_constructed_candidate_set():
    constructor = DiverseCandidateConstructor(
        mmax=2, score_threshold=0.01, sigma_nms_mm=2.0, refinement_steps=2
    )
    proposals = _proposals()
    first = constructor(proposals)["candidate_centers_mm"]
    permutation = torch.tensor([1, 0])
    shuffled = {
        key: value[:, :, permutation]
        for key, value in proposals.items()
    }
    second = constructor(shuffled)["candidate_centers_mm"]
    first = first.sort(dim=1).values
    second = second.sort(dim=1).values
    assert torch.allclose(first, second, atol=1e-5)
