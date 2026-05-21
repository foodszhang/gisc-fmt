import numpy as np
import torch

from minr_fmt.dataset.fmt_simgen_dataset import FmtSimGenProjDataset
from minr_fmt.loss import AuxProjectionLightLoss


def _write_sample(root, with_descatter: bool, descatter_name: str = "proj_noscatter.npz"):
    sample = root / "sample_000"
    sample.mkdir(parents=True)
    proj = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=np.float32)
    np.savez(sample / "proj.npz", **{"0": proj})
    np.save(sample / "gt_voxels.npy", np.ones((2, 2, 2), dtype=np.float32))
    if with_descatter:
        descatter = np.array([[2.0, 4.0], [0.0, 2.0]], dtype=np.float32)
        np.savez(sample / descatter_name, **{"0": descatter})


def test_fmt_simgen_loads_optional_descatter_targets(tmp_path):
    _write_sample(tmp_path, with_descatter=True)
    ds = FmtSimGenProjDataset(
        str(tmp_path),
        config={
            "view_angles": [0],
            "sample_num": 4,
            "projection_norm": "per_view_max",
            "descatter_target_scale": 4.0,
        },
        split="all",
    )

    item = ds[0]

    assert "descatter_targets" in item
    expected = torch.tensor([[0.5, 1.0], [0.0, 0.5]])
    assert torch.allclose(item["descatter_targets"]["0"], expected)


def test_fmt_simgen_without_descatter_targets_remains_valid(tmp_path):
    _write_sample(tmp_path, with_descatter=False)
    ds = FmtSimGenProjDataset(
        str(tmp_path),
        config={"view_angles": [0], "sample_num": 4, "projection_norm": "per_view_max"},
        split="all",
    )

    item = ds[0]

    assert "descatter_targets" not in item


def test_fmt_simgen_falls_back_to_legacy_no_proj_name(tmp_path):
    _write_sample(tmp_path, with_descatter=True, descatter_name="no_proj.npz")
    ds = FmtSimGenProjDataset(
        str(tmp_path),
        config={
            "view_angles": [0],
            "sample_num": 4,
            "projection_norm": "per_view_max",
            "descatter_target_scale": 4.0,
        },
        split="all",
    )

    item = ds[0]

    assert "descatter_targets" in item


def test_aux_projection_loss_is_zero_without_target():
    loss_fn = AuxProjectionLightLoss()
    pred_aux = {"0": torch.ones(1, 2, 2)}
    pred_density = torch.zeros(1, 4, 1)
    gt_density = torch.zeros(1, 4, 1)

    loss_dict = loss_fn(pred_aux, None, pred_density, gt_density)

    assert loss_dict["aux_projection_loss"].item() == 0.0
