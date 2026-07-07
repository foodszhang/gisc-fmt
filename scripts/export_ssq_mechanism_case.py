#!/usr/bin/env python
"""Export SSQ-FMT mechanism diagnostics for one FMT-SimGen sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import hydra
import numpy as np
import torch

from minr_fmt.datamodule import TrainingDataModule
from minr_fmt.module import TrainingLightningModule


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--exp", default="fmt_simgen_v2_ssq_final")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--save-path", required=True)
    args = parser.parse_args()

    with hydra.initialize(config_path="../configs", version_base=None):
        cfg = hydra.compose(
            config_name="config",
            overrides=["model=ssq_fmt", f"exp={args.exp}", "data.dataset_type=fmt_simgen"],
        )
    dm = TrainingDataModule(cfg)
    setup = {"train": dm.setup, "val": dm.setup, "test": dm.setup}
    setup[args.split](stage="fit" if args.split != "test" else "test")
    datasets = {"train": dm.train_dataset, "val": dm.val_dataset, "test": dm.test_dataset}
    dataset = datasets[args.split]
    index = next(i for i, sample in enumerate(dataset.dirs) if sample.name == args.sample_id)
    batch = dataset[index]
    batch = {
        key: value.unsqueeze(0) if torch.is_tensor(value) else [value]
        for key, value in batch.items()
    }

    module = TrainingLightningModule.load_from_checkpoint(
        args.checkpoint, cfg=cfg, map_location="cpu"
    )
    module.eval()
    with torch.no_grad():
        out = module.net(
            batch["surface_measurements_packed"],
            batch["query_coordinates_mm"],
            detector_valid_mask=batch.get("detector_valid_mask"),
            depth_maps=batch.get("depth_maps"),
            batch=batch,
            return_diagnostics=True,
        )

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"diagnostics/{key}": _to_numpy(value) for key, value in out["diagnostics"].items()}
    payload["density"] = _to_numpy(out["density"])
    payload["sample_id"] = np.asarray([args.sample_id])
    np.savez_compressed(save_path, **payload)
    print(f"saved {save_path}")


if __name__ == "__main__":
    main()
