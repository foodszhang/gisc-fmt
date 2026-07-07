#!/usr/bin/env python
"""Inspect the FMT-SimGen dataset adapter and DataLoader collation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.dataset.fmt_simgen_dataset import FmtSimGenProjDataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--num_queries", type=int, default=4096)
    args = parser.parse_args()

    cfg = OmegaConf.create(
        {
            "data": {
                "view_angles": [-90, -60, -30, 0, 30, 60, 90],
                "sample_num": args.num_queries,
                "num_queries": args.num_queries,
                "voxel_size_mm": 0.2,
                "projection_norm": "per_view_max",
                "projection_eps": 1e-8,
            }
        }
    )

    all_ds = {
        split: FmtSimGenProjDataset(
            args.data_dir, config=cfg, split=split, is_training=split == "train"
        )
        for split in ("train", "val", "test")
    }
    print("split sizes:", {k: len(v) for k, v in all_ds.items()})

    ds = all_ds["train"]
    sample_dir = ds.dirs[0]
    discovered_samples = len(ds._scan_samples())
    gt = ds._load_gt(sample_dir)
    item = ds[0]
    pp = item["projections_packed"]
    pd = item["point_densities"]
    print("num samples:", discovered_samples)
    print("first sample id:", item["sample_id"])
    print("proj shape:", tuple(pp.shape))
    print("proj min/max/mean:", float(pp.min()), float(pp.max()), float(pp.mean()))
    print(
        "gt_voxels shape:",
        gt.shape,
        "gt min/max/mean/nonzero:",
        float(gt.min()),
        float(gt.max()),
        float(gt.mean()),
        int(np.count_nonzero(gt)),
    )
    print("points shape:", tuple(item["points"].shape))
    print("point_densities shape:", tuple(pd.shape))
    print("point_densities positive ratio:", float((pd > 0).float().mean()))
    print("projection keys:", list(item["projections"].keys()))
    print("projection_scales:", item["projection_scales"].tolist())

    loader = DataLoader(ds, batch_size=2, num_workers=0)
    batch = next(iter(loader))
    print("batch projections_packed shape:", tuple(batch["projections_packed"].shape))
    print("batch points shape:", tuple(batch["points"].shape))
    print("batch point_densities shape:", tuple(batch["point_densities"].shape))


if __name__ == "__main__":
    main()
