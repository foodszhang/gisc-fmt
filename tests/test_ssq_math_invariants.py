import torch
from omegaconf import OmegaConf

from minr_fmt.models.ssq_fmt import SSQFMT


def make_cfg(mmax=3, lambda_sdf=0.0):
    return OmegaConf.create(
        {
            "model": {
                "name": "ssq_fmt",
                "in_channels": 1,
                "ssq_fmt": {
                    "hidden_dim": 16,
                    "normalization": {"percentile": 99.9, "eps": 1e-6},
                    "encoder": {"channels": 8, "feature_dim": 12},
                    "geometry": {
                        "camera_distance_mm": 200.0,
                        "fov_mm": 80.0,
                        "detector_resolution": [16, 16],
                        "volume_center_world": [1.0, 1.0, 1.0],
                    },
                    "local_sampling": {"offsets_px": [[0.0, 0.0], [1.0, 0.0]]},
                    "candidates": {
                        "mmax": mmax,
                        "threshold": 0.0,
                        "scale_min_mm": 0.1,
                        "scale_max_mm": 4.0,
                        "trunk_size_mm": [2.0, 2.0, 2.0],
                    },
                },
            },
            "data": {"view_angles": [0, 90]},
            "loss": {"lambda_sdf": lambda_sdf},
        }
    )


def make_batch(mmax=3, candidate_valid=None):
    B, V, H, W, N = 2, 2, 16, 16, 5
    y = torch.rand(B, V, 1, H, W)
    points = torch.rand(B, N, 3) * 2.0
    if candidate_valid is None:
        candidate_valid = torch.ones(B, mmax, dtype=torch.bool)
    centers = torch.rand(B, mmax, 3) * 2.0
    return {
        "surface_measurements_packed": y,
        "query_coordinates_mm": points,
        "detector_valid_mask": torch.ones(B, V, H, W, dtype=torch.bool),
        "candidate_centers_mm": centers,
        "candidate_scores": torch.ones(B, mmax),
        "candidate_scales_mm": torch.ones(B, mmax),
        "candidate_valid_mask": candidate_valid,
    }


def test_ssq_probability_and_mass_invariants():
    model = SSQFMT(make_cfg(mmax=3))
    out = model(
        make_batch(3)["surface_measurements_packed"],
        make_batch(3)["query_coordinates_mm"],
        batch=make_batch(3),
        return_diagnostics=True,
    )
    d = out["diagnostics"]

    assert torch.allclose(
        d["zeta"].sum(dim=-1)[d["sample_valid"]],
        torch.ones_like(d["zeta"].sum(dim=-1)[d["sample_valid"]]),
        atol=1e-5,
    )
    assert torch.allclose(d["a"].sum(dim=-1), torch.ones_like(d["a"].sum(dim=-1)), atol=1e-5)
    assert torch.allclose(d["pi"].sum(dim=-1), torch.ones_like(d["pi"].sum(dim=-1)), atol=1e-5)
    for key in ("e", "nu", "r", "pi", "branch_density", "branch_contributions"):
        value = d[key]
        assert torch.all(value >= 0.0)
        assert torch.all(value <= 1.0)
    assert torch.all(out["density"] >= 0.0)
    assert torch.all(out["density"] <= 1.0)


def test_padding_candidate_weights_and_contributions_are_zero():
    valid = torch.tensor([[True, False, True], [False, False, True]])
    batch = make_batch(3, candidate_valid=valid)
    out = SSQFMT(make_cfg(mmax=3))(
        batch["surface_measurements_packed"],
        batch["query_coordinates_mm"],
        batch=batch,
        return_diagnostics=True,
    )
    d = out["diagnostics"]
    invalid = ~valid
    assert torch.all(d["pi"][:, :, 1:][invalid[:, None, :].expand_as(d["pi"][:, :, 1:])] == 0.0)
    assert torch.all(
        d["branch_contributions"][:, :, 1:][
            invalid[:, None, :, None].expand_as(d["branch_contributions"][:, :, 1:])
        ]
        == 0.0
    )
