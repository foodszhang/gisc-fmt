from types import SimpleNamespace as NS

import pytest
import torch

from minr_fmt.benchmark import get_baseline_spec, validate_baseline_protocol
from minr_fmt.models.native_grid_baselines import NativeGridCNN3DBaseline


def _cfg(model_name: str, *, table_tier: str = "development"):
    data = NS(
        view_angles=[-90, -60, -30, 0, 30, 60, 90],
        voxel_ranges=NS(x=[0, 10], y=[0, 12], z=[0, 6]),
    )
    geometry = NS(global_voxel_shape=[10, 12, 6])
    model = NS(
        name=model_name,
        geometry=geometry,
        benchmark=NS(native_output_shape=[8, 8, 4]),
        cnn3d_baseline=NS(
            base_channels=2,
            native_output_shape=[8, 8, 4],
            return_reference_grid=True,
        ),
    )
    return NS(
        data=data,
        model=model,
        benchmark_protocol=NS(
            table_tier=table_tier,
            strict_fidelity=False,
            allow_unsafe_fully_connected_head=False,
        ),
    )


def test_controlled_native_grid_model_returns_reference_grid():
    cfg = _cfg("cnn3d_baseline")
    model = NativeGridCNN3DBaseline(cfg).eval()
    projections = {
        str(angle): torch.rand(1, 16, 16)
        for angle in cfg.data.view_angles
    }
    with torch.no_grad():
        out = model(projections)
    assert out["pred_voxel"].shape == (1, 1, 10, 12, 6)
    assert out["aux_outputs"]["native_prediction_shape"] == (8, 8, 4)
    assert out["aux_outputs"]["fixed_output_resampling"] is True


def test_main_table_rejects_architecture_proxy():
    cfg = _cfg("map_pgan", table_tier="main")
    with pytest.raises(ValueError, match="not approved for the TMI main table"):
        validate_baseline_protocol(cfg)


def test_development_allows_proxy_but_keeps_fidelity_label():
    cfg = _cfg("d2_recst", table_tier="development")
    spec = validate_baseline_protocol(cfg)
    assert spec.fidelity == "architecture_proxy"
    assert spec.main_table_allowed is False


def test_uhr_is_explicitly_labeled_adapted():
    spec = get_baseline_spec("uhr_deepfmt")
    assert "adapted" in spec.display_name.lower()
    assert spec.fidelity == "mechanism_preserving_adaptation"


def test_vox_dmrn_full_grid_head_is_blocked():
    cfg = _cfg("vox_dmrn")
    cfg.model.geometry.global_voxel_shape = [190, 200, 104]
    with pytest.raises(ValueError, match="fully connected output head"):
        validate_baseline_protocol(cfg)
