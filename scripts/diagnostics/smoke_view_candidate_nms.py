"""Smoke checks for large-query-safe ViewCandidateEvidence NMS."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from minr_fmt.network.view_candidate_evidence import ViewCandidateEvidence  # noqa: E402


def _inputs(batch: int, views: int, queries: int, feature_dim: int, *, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    query_features = torch.randn(batch, views, queries, feature_dim, generator=generator)
    geometry_features = torch.randn(batch, views, queries, 4, generator=generator)
    points = torch.rand(batch, queries, 3, generator=generator) * 40.0
    valid = torch.rand(batch, views, queries, generator=generator) > 0.1
    return query_features, geometry_features, points, valid


def _module(
    *,
    feature_dim: int = 4,
    topk_per_view: int = 8,
    exact_threshold: int = 4096,
    pre_nms_topk: int = 1024,
    max_nms_candidates: int = 2048,
) -> ViewCandidateEvidence:
    return ViewCandidateEvidence(
        feature_dim=feature_dim,
        hidden_dim=16,
        descriptor_dim=8,
        delta_max_mm=1.0,
        topk_per_view=topk_per_view,
        nms_radius_mm=2.0,
        exact_nms_threshold=exact_threshold,
        pre_nms_topk=pre_nms_topk,
        pre_nms_factor=64,
        max_nms_candidates=max_nms_candidates,
    )


def _assert_shapes(out: dict[str, torch.Tensor], batch: int, views: int, topk: int) -> None:
    assert out["proposal_scores"].shape == (batch, views, topk)
    assert out["proposal_points_mm"].shape == (batch, views, topk, 3)
    assert out["proposal_valid_mask"].shape == (batch, views, topk)
    assert out["proposal_query_indices"].shape == (batch, views, topk)


def smoke_small_exact() -> None:
    module = _module(topk_per_view=8, exact_threshold=4096)
    inputs = _inputs(batch=1, views=2, queries=128, feature_dim=4, seed=11)
    with torch.no_grad():
        out = module(*inputs)
    _assert_shapes(out, batch=1, views=2, topk=8)
    assert out["proposal_valid_mask"].any()


def smoke_large_prefilter() -> None:
    module = _module(topk_per_view=8, exact_threshold=4096, pre_nms_topk=1024)
    inputs = _inputs(batch=1, views=3, queries=32768, feature_dim=4, seed=12)
    with torch.no_grad():
        out = module(*inputs)
    _assert_shapes(out, batch=1, views=3, topk=8)
    assert out["proposal_query_indices"].max() < 32768


def smoke_small_consistency() -> None:
    torch.manual_seed(13)
    scores = torch.rand(1, 2, 512)
    points = torch.rand(1, 512, 3) * 20.0
    valid = torch.ones_like(scores, dtype=torch.bool)
    exact = ViewCandidateEvidence._local_maxima_exact(scores, points, radius=1.5)
    prefiltered = ViewCandidateEvidence._local_maxima_topk_prefilter(
        scores,
        points,
        radius=1.5,
        valid_mask=valid,
        topk_per_view=8,
        pre_nms_topk=512,
        pre_nms_factor=64,
        max_nms_candidates=512,
    )
    assert torch.equal(exact, prefiltered)
    exact_top = scores.masked_fill(~exact, -1.0).topk(1, dim=-1).values
    pre_top = scores.masked_fill(~prefiltered, -1.0).topk(1, dim=-1).values
    assert torch.allclose(exact_top, pre_top)


def smoke_grid_path() -> None:
    module = _module(topk_per_view=8, exact_threshold=0)
    grid_shape = (32, 32, 32)
    queries = grid_shape[0] * grid_shape[1] * grid_shape[2]
    inputs = _inputs(batch=1, views=2, queries=queries, feature_dim=4, seed=14)
    with torch.no_grad():
        out = module(*inputs, grid_shape=grid_shape)
    _assert_shapes(out, batch=1, views=2, topk=8)
    assert out["proposal_query_indices"].max() < queries


def main() -> None:
    smoke_small_exact()
    smoke_large_prefilter()
    smoke_small_consistency()
    smoke_grid_path()
    print("ViewCandidateEvidence NMS smoke checks passed")


if __name__ == "__main__":
    main()
