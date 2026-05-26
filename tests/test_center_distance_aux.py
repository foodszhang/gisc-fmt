import sys
from types import SimpleNamespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minr_fmt.module import TrainingLightningModule


def test_center_distance_targets_and_losses():
    module = TrainingLightningModule.__new__(TrainingLightningModule)
    module.cfg = SimpleNamespace(data=SimpleNamespace(voxel_size_mm=1.0))
    module._center_distance_sigma_mm = 0.5

    gt = torch.zeros(1, 4, 4, 4)
    gt[0, 1:3, 1:3, 1:3] = 1.0
    batch = {"gt_voxels": gt}
    points_mm = torch.tensor([[[1.5, 1.5, 1.5], [0.5, 0.5, 0.5]]], dtype=torch.float32)
    points_ijk = torch.tensor([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]], dtype=torch.float32)
    center, distance, fg = module._center_distance_query_targets(batch, points_mm, points_ijk)

    assert center.shape == (1, 2, 1)
    assert distance.shape == (1, 2, 1)
    assert fg.shape == (1, 2, 1)
    assert center[0, 0, 0] > center[0, 1, 0]
    assert distance[0, 0, 0] > distance[0, 1, 0]
