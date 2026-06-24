import pytest
import torch
from omegaconf import OmegaConf

from minr_fmt.module import TrainingLightningModule
from tests.test_ssq_math_invariants import make_batch


def _cfg(lambda_sdf=0.0):
    return OmegaConf.create(
        {
            "model": {
                "name": "ssq_fmt",
                "in_channels": 1,
                "output_type": "point_probability",
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
                        "mmax": 2,
                        "threshold": 0.0,
                        "scale_min_mm": 0.1,
                        "scale_max_mm": 4.0,
                        "trunk_size_mm": [2.0, 2.0, 2.0],
                    },
                },
            },
            "data": {"view_angles": [0, 90]},
            "loss": {
                "lambda_sdf": lambda_sdf,
                "tau_s": 3.0,
                "sdf_boundary_weight": 1.0,
                "center_weight": 0.0,
                "distance_weight": 0.0,
                "empty_slot_weight": 0.0,
                "aux_projection_weight": 0.0,
                "target_aux_projection_weight": 0.0,
                "start_decay_epoch": 1,
                "decay_epochs": 1,
                "pos_weight": 1.0,
                "sparse_weight": 0.0,
                "dice_weight": 0.0,
                "light_weight": 1.0,
            },
            "optim": {
                "_target_": "torch.optim.Adam",
                "lr": 1e-3,
                "betas": [0.9, 0.999],
                "weight_decay": 0.0,
                "eps": 1e-8,
            },
            "trainer": {"max_epochs": 1, "torch_compile": False},
            "validation": {"pred_threshold": 0.5},
        }
    )


def _lightning_batch(device="cpu"):
    batch = make_batch(mmax=2)
    batch["projections"] = {}
    batch["projections_packed"] = batch["surface_measurements_packed"]
    batch["surface_measurements"] = {}
    batch["points_mm"] = batch["query_coordinates_mm"]
    batch["points"] = batch["query_coordinates_mm"] / 2.0
    batch["point_densities"] = torch.rand(2, 5)
    batch["points_ijk"] = (batch["points"] * 3.0).clamp(0, 3)
    batch["gt_voxels"] = torch.zeros(2, 4, 4, 4)
    batch["gt_voxels"][:, 1:3, 1:3, 1:3] = 1.0
    batch["feasible_voxel_shape"] = (
        torch.tensor([4, 4]),
        torch.tensor([4, 4]),
        torch.tensor([4, 4]),
    )
    return {
        key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()
    }


def test_lightning_train_val_inference_and_chunking_cpu():
    module = TrainingLightningModule(_cfg(lambda_sdf=0.5))
    batch = _lightning_batch()
    loss = module.training_step(batch, 0)
    loss.backward()
    opt = module.configure_optimizers()["optimizer"]
    opt.step()
    opt.zero_grad(set_to_none=True)

    module.eval()
    with torch.no_grad():
        val = module.validation_step(batch, 0)
        out = module.net(
            batch["surface_measurements_packed"],
            batch["query_coordinates_mm"],
            batch=batch,
            return_diagnostics=False,
        )
        out_diag = module.net(
            batch["surface_measurements_packed"],
            batch["query_coordinates_mm"],
            batch=batch,
            return_diagnostics=True,
        )
        chunks = []
        for start in range(0, batch["query_coordinates_mm"].shape[1], 2):
            chunk_batch = dict(batch)
            chunk_batch["query_coordinates_mm"] = batch["query_coordinates_mm"][
                :, start : start + 2
            ]
            chunk_batch["points_mm"] = chunk_batch["query_coordinates_mm"]
            chunks.append(
                module.net(
                    batch["surface_measurements_packed"],
                    chunk_batch["query_coordinates_mm"],
                    batch=chunk_batch,
                )["density"]
            )
        chunked = torch.cat(chunks, dim=1)

    assert "dice" in val
    assert out["density"].shape == (2, 5, 1)
    assert "diagnostics" in out_diag
    assert chunked.shape == out["density"].shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_ssq_cuda_amp_smoke():
    module = TrainingLightningModule(_cfg()).cuda()
    batch = _lightning_batch(device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        out = module.net(
            batch["surface_measurements_packed"],
            batch["query_coordinates_mm"],
            batch=batch,
        )
        loss = out["density"].mean()
    loss.backward()
    assert torch.isfinite(loss.detach())
