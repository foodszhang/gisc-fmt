#!/usr/bin/env python
"""Save two-stage projection/profile/volume tensors for axis sanity checks."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import hydra
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from minr_fmt.dataset.fmt_simgen_dataset import FmtSimGenProjDataset
from minr_fmt.model_factory import ModelFactory

SAVE_DIR = ROOT / "outputs" / "fmt_simgen_v2_3k_20k" / "debug" / "two_stage"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-axis", choices=("h", "w"), default="h")
    args = parser.parse_args()
    slice_axis = "w" if args.profile_axis == "h" else "h"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                "model=two_stage_deepfmt", "exp=fmt_simgen_v2_3k_20k_common",
                "data.dataset_type=fmt_simgen", "data.train_max_samples=2",
                f"model.two_stage_deepfmt.profile_axis={args.profile_axis}",
                f"model.two_stage_deepfmt.slice_axis={slice_axis}",
            ],
        )
    ds = FmtSimGenProjDataset(str(cfg.data.train_dir), config=cfg, split="train", is_training=False)
    net = ModelFactory.create_model(cfg.model.name, config=cfg).eval()
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    for index in range(2):
        item = ds[index]
        projections = {key: value.unsqueeze(0) for key, value in item["projections"].items()}
        with torch.no_grad():
            out = net(projections)
        pred = out["pred_voxel"][0, 0].numpy()
        restored = out["aux_outputs"]["restored_projections"][0].numpy()
        profiles = out["aux_outputs"]["selected_profiles"][0].numpy()
        gt = item["gt_voxels"].numpy()
        print(f"{item['sample_id']}: projections={restored.shape} gt={gt.shape} pred={pred.shape}")
        if pred.shape != gt.shape:
            raise ValueError(f"pred shape {pred.shape} != gt shape {gt.shape}")
        np.savez_compressed(
            SAVE_DIR / f"{item['sample_id']}_{args.profile_axis}{slice_axis}.npz",
            restored_projection=restored, selected_profiles=profiles, pred_voxel=pred, gt_voxels=gt,
        )

if __name__ == "__main__":
    main()
