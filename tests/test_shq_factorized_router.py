import time

import pytest
import torch

from minr_fmt.models.ssq_fmt import SSQFMT
from minr_fmt.module import E15_INIT_WHITELIST, load_e15_compatible_weights
from minr_fmt.network.ssq_diagnostics import masked_softmax
from minr_fmt.network.ssq_routing import CandidateSurfaceRouter
from tests.test_shq_quotient_field import _shq_cfg


def _router_inputs(candidate_count: int = 5):
    batch, views, queries, samples, channels = 2, 3, 7, 4, 8
    sample_valid = torch.ones(batch, views, queries, samples, dtype=torch.bool)
    sample_valid[:, :, :, -1] = False
    sample_data = {
        "sample_features": torch.randn(
            batch, views, queries, samples, channels, requires_grad=True
        ),
        "sample_valid": sample_valid,
        "A": torch.softmax(torch.randn(batch, views, queries, samples), dim=-1),
        "sample_measurements": torch.rand(batch, views, queries, samples, 1),
        "sample_coordinates_px": torch.rand(batch, views, queries, samples, 2) * 16.0,
    }
    candidate_valid = torch.ones(batch, candidate_count, dtype=torch.bool)
    detector_valid = torch.ones(batch, views, candidate_count, dtype=torch.bool)
    if candidate_count:
        detector_valid[:, 0, -1] = False
    candidates = {
        "candidate_centers_mm": torch.rand(batch, candidate_count, 3),
        "candidate_support_inverse_sqrt_mm": torch.eye(3)
        .view(1, 1, 3, 3)
        .expand(batch, candidate_count, -1, -1),
        "candidate_valid_mask": candidate_valid,
        "candidate_uv_px": torch.rand(batch, views, candidate_count, 2) * 16.0,
        "candidate_detector_valid_mask": detector_valid,
        "candidate_detector_support_scales_px": torch.ones(batch, views, candidate_count) * 3.0,
    }
    mapped = {
        "detector_side_path_proxy": torch.rand(batch, views, queries),
        "boundary_distance": torch.rand(batch, views, queries),
        "valid_mask": torch.ones(batch, views, queries, dtype=torch.bool),
    }
    return sample_data, torch.rand(batch, queries, 3), candidates, mapped


def _run_router(candidate_count: int = 5):
    inputs = _router_inputs(candidate_count)
    router = CandidateSurfaceRouter(
        8,
        sample_embedding_dim=16,
        candidate_embedding_dim=4,
        compensation_routing_mode="proposal_only",
    )
    return router, inputs, router(*inputs, return_diagnostics=True)


def test_factorized_router_preserves_legacy_shape_contract_and_shared_zeta():
    _, inputs, output = _run_router(5)
    samples = inputs[0]
    assert output["zeta"].shape == (2, 3, 7, 4, 6)
    assert output["a"].shape == (2, 3, 7, 6)
    assert output["e"].shape == output["a"].shape
    assert torch.equal(output["zeta"][..., 0].bool(), samples["sample_valid"])
    assert torch.allclose(output["zeta"][..., 0], samples["sample_valid"].float())


def test_candidate_zeta_normalization_visibility_and_permutation():
    router, inputs, output = _run_router(5)
    samples, points, candidates, mapped = inputs
    visible = candidates["candidate_detector_valid_mask"][:, :, None, None]
    valid_sample = samples["sample_valid"][..., None]
    sums = output["zeta"][..., 1:].sum(dim=-1)
    assert torch.allclose(
        sums[samples["sample_valid"]], torch.ones_like(sums[samples["sample_valid"]])
    )
    assert torch.count_nonzero(output["zeta"][..., 1:].masked_select(~visible & valid_sample)) == 0
    assert torch.count_nonzero(output["a"][:, 0, :, -1]) == 0

    permutation = torch.tensor([3, 0, 4, 1, 2])
    permuted = dict(candidates)
    for key in (
        "candidate_centers_mm",
        "candidate_support_inverse_sqrt_mm",
        "candidate_valid_mask",
    ):
        permuted[key] = candidates[key][:, permutation]
    for key in (
        "candidate_uv_px",
        "candidate_detector_valid_mask",
        "candidate_detector_support_scales_px",
    ):
        permuted[key] = candidates[key][:, :, permutation]
    actual = router(samples, points, permuted, mapped)["zeta"][..., 1:]
    assert torch.allclose(actual, output["zeta"][..., 1:][..., permutation], atol=1.0e-6)


@pytest.mark.parametrize("candidate_count", [0, 1, 5])
def test_factorized_router_forward_backward(candidate_count):
    router, inputs, output = _run_router(candidate_count)
    loss = output["r"].sum() + output["measurement_consistency"].sum()
    loss.backward()
    assert any(parameter.grad is not None for parameter in router.parameters())
    assert inputs[0]["sample_features"].grad is not None


def test_e15_initialization_loads_only_whitelisted_modules(tmp_path):
    model = SSQFMT(_shq_cfg(mmax=1))
    state = model.state_dict()
    allowed = next(key for key in state if key.startswith(E15_INIT_WHITELIST))
    forbidden = next(key for key in state if key.startswith("assignment_head."))
    checkpoint = tmp_path / "e15.ckpt"
    torch.save(
        {
            "state_dict": {
                f"net.{allowed}": torch.full_like(state[allowed], 0.125),
                f"net.{forbidden}": torch.full_like(state[forbidden], 0.25),
            }
        },
        checkpoint,
    )
    forbidden_before = state[forbidden].clone()
    report = load_e15_compatible_weights(model, str(checkpoint))
    assert report["loaded_keys"] == [allowed]
    assert torch.all(model.state_dict()[allowed] == 0.125)
    assert torch.equal(model.state_dict()[forbidden], forbidden_before)


def test_factorized_router_single_step_benchmark():
    router, inputs, _ = _run_router(5)
    samples, points, candidates, mapped = inputs
    start = time.perf_counter()
    router(samples, points, candidates, mapped)
    vectorized_seconds = time.perf_counter() - start

    k = torch.rand(2, 3, 7, 4, 5)
    valid = torch.ones_like(k, dtype=torch.bool)
    start = time.perf_counter()
    legacy_parts = [
        masked_softmax(k[..., index : index + 1], valid[..., index : index + 1], -1)
        for index in range(5)
    ]
    legacy_loop_seconds = time.perf_counter() - start
    assert torch.cat(legacy_parts, dim=-1).shape == (2, 3, 7, 4, 5)
    assert vectorized_seconds >= 0.0 and legacy_loop_seconds >= 0.0
