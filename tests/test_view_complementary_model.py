import torch

from minr_fmt.models.ssq_fmt import SSQFMT
from minr_fmt.network.unified_density_decoder import UnifiedDensityDecoder
from minr_fmt.network.view_separability import ViewSeparability
from tests.test_ssq_math_invariants import make_batch, make_cfg


def _cfg(ablation="full", separability_mode="geometry_measurement"):
    cfg = make_cfg(mmax=3)
    cfg.model.ssq_fmt.encoder = {"type": "shallow", "channels": 8, "output_channels": 12}
    cfg.model.ssq_fmt.density_output_mode = "view_complementary"
    cfg.model.ssq_fmt.composition = {"mode": "view_complementary"}
    cfg.model.ssq_fmt.view_complementary = {
        "ablation": ablation,
        "separability_mode": separability_mode,
        "topk_per_view": 3,
        "mmax": 3,
        "score_threshold": 0.001,
    }
    return cfg


def test_geometry_and_measurement_separability_are_symmetric_and_bounded():
    torch.manual_seed(3)
    module = ViewSeparability(6)
    features = torch.randn(2, 3, 4, 6)
    centers = torch.randn(2, 3, 4, 2)
    scales = torch.ones(2, 3, 4)
    valid = torch.ones(2, 4, dtype=torch.bool)
    geometry = module(features, centers, scales, valid, mode="geometry_only")
    with torch.no_grad():
        module.correction[-1].bias.fill_(0.1)
    full = module(features, centers, scales, valid, mode="geometry_measurement")
    for out in (geometry, full):
        pair = out["pair_separability"]
        assert torch.allclose(pair, pair.transpose(-1, -2), atol=1e-6)
        assert torch.all((pair >= 0.0) & (pair <= 1.0))
    assert not torch.allclose(geometry["pair_separability"], full["pair_separability"])


def test_geometry_separability_uses_consistent_pixel_units():
    module = ViewSeparability(4)
    features = torch.zeros(1, 1, 2, 4)
    centers = torch.tensor([[[[0.0, 0.0], [10.0, 0.0]]]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    narrow = module(
        features, centers, torch.full((1, 1, 2), 2.0), valid, mode="geometry_only"
    )["pair_separability"][0, 0, 0, 1]
    broad = module(
        features, centers, torch.full((1, 1, 2), 20.0), valid, mode="geometry_only"
    )["pair_separability"][0, 0, 0, 1]
    scaled = module(
        features,
        centers * 2.0,
        torch.full((1, 1, 2), 4.0),
        valid,
        mode="geometry_only",
    )["pair_separability"][0, 0, 0, 1]
    assert narrow > 0.99
    assert broad < 0.2
    assert torch.allclose(narrow, scaled, atol=1.0e-6)


def test_unified_decoder_has_exact_shared_fallback_and_candidate_permutation_invariance():
    torch.manual_seed(4)
    decoder = UnifiedDensityDecoder(8, 10, 12)
    shared = torch.randn(2, 7, 8)
    candidate = torch.randn(2, 3, 8)
    points = torch.randn(2, 7, 3)
    encoded = torch.randn(2, 7, 10)
    centers = torch.randn(2, 3, 3)
    covariance = torch.eye(3)[None, None].expand(2, 3, -1, -1).clone()
    scores = torch.rand(2, 3)
    valid = torch.ones(2, 3, dtype=torch.bool)
    full = decoder(shared, candidate, points, encoded, centers, covariance, scores, valid)
    permutation = torch.tensor([2, 0, 1])
    shuffled = decoder(
        shared,
        candidate[:, permutation],
        points,
        encoded,
        centers[:, permutation],
        covariance[:, permutation],
        scores[:, permutation],
        valid[:, permutation],
    )
    assert torch.allclose(full["density"], shuffled["density"], atol=1e-6)

    empty = valid & False
    fallback = decoder(shared, candidate, points, encoded, centers, covariance, scores, empty)
    shared_only = decoder(
        shared, candidate, points, encoded, centers, covariance, scores, valid, "shared_only"
    )
    assert torch.equal(fallback["density"], shared_only["density"])
    assert torch.count_nonzero(fallback["candidate_context"]) == 0


def test_candidate_centers_change_detector_sampling_grid_and_no_residual_gate_is_returned():
    model = SSQFMT(_cfg())
    assert model.source_hypothesis_residual_decoder is None
    batch = make_batch(3)
    out = model(
        batch["surface_measurements_packed"],
        batch["query_coordinates_mm"],
        detector_valid_mask=batch["detector_valid_mask"],
        depth_maps=batch["depth_maps"],
        batch=batch,
        return_diagnostics=True,
    )
    diagnostics = out["diagnostics"]
    centers = diagnostics["candidate_detector_centers"]
    valid = diagnostics["candidate_valid_mask"]
    for sample in range(valid.shape[0]):
        selected = centers[sample, :, valid[sample]]
        if selected.shape[1] > 1:
            flattened = selected.reshape(selected.shape[0], selected.shape[1], -1)
            assert torch.unique(flattened, dim=1).shape[1] > 1
    assert "proposal_gate" not in out["aux_outputs"]
    assert "residual_correction" not in out["aux_outputs"]
    assert out["density"].shape == (*batch["query_coordinates_mm"].shape[:2], 1)
