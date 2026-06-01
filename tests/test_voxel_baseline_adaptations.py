from __future__ import annotations

import torch
from omegaconf import OmegaConf

from minr_fmt.models.voxel_baselines import D2RecSTAdapted, FEM2VoxUNet


def _config(model_name: str):
    return OmegaConf.create(
        {
            "data": {
                "voxel_ranges": {"x": [0, 16], "y": [0, 16], "z": [0, 8]},
                "view_angles": [-90, -60, -30, 0, 30, 60, 90],
            },
            "model": {
                "name": model_name,
                "num_views": 7,
                "geometry": {"global_voxel_shape": [16, 16, 8]},
                "fem2vox_unet": {
                    "refine_channels": 2,
                    "refine_blocks": 2,
                    "residual_prior": True,
                    "allow_projection_fallback": False,
                },
                "d2_recst": {
                    "base_channels": 2,
                    "latent_channels": 4,
                    "internal_shape": [8, 8, 4],
                },
            },
        }
    )


def test_fem2vox_residual_initialization_preserves_prior():
    prior = torch.rand(1, 16, 16, 8)
    model = FEM2VoxUNet(_config("fem2vox_unet")).eval()

    output = model({}, batch={"stage1_voxel": prior})["pred_voxel"]

    assert torch.allclose(torch.sigmoid(output[:, 0]), prior.clamp(1e-4, 1.0 - 1e-4), atol=1e-6)


def test_projection_shell_lift_preserves_spatial_cues_and_restores_full_shape():
    config = _config("d2_recst")
    projections = {str(angle): torch.rand(1, 16, 16) for angle in config.data.view_angles}
    model = D2RecSTAdapted(config).eval()

    coarse = model.coarse_volume(projections)
    output = model(projections)["pred_voxel"]

    assert coarse.shape == (1, 4, 8, 8, 4)
    assert coarse.std(dim=(2, 3, 4)).mean() > 0
    assert output.shape == (1, 1, 16, 16, 8)
