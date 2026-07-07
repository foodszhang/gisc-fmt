#!/usr/bin/env python3
"""Training entrypoint for the audited sample-level PHSA path."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import hydra
from torch.utils.data import default_collate
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.datamodule import TrainingDataModule
from minr_fmt.phsa_sample_level import activate_phsa_sample_level_hypotheses
from minr_fmt.utils.logging_utils import setup_logger


def _phsa_collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate PHSA samples without transferring an unused full GT volume.

    Density targets, query indices and component-supervision tensors are created in
    ``__getitem__`` before collation. The active PHSA loss has ``lambda_sdf=0`` and
    does not consume ``gt_voxels``; keeping the full 190x200x104 tensor in every
    batch only adds host memory, pinning and host-to-device transfer overhead.
    """

    batch = default_collate(items)
    batch.pop("gt_voxels", None)
    return batch


def _activate_phsa_loader_contract() -> None:
    original = TrainingDataModule._loader_kwargs
    if getattr(TrainingDataModule, "_phsa_loader_patch", False):
        return

    def patched(self, shuffle: bool, batch_size: int) -> dict[str, Any]:
        kwargs = original(self, shuffle=shuffle, batch_size=batch_size)
        kwargs["collate_fn"] = _phsa_collate
        if int(kwargs.get("num_workers", 0)) > 0:
            # ``FmtSimGenProjDataset.current_epoch`` is ordinary process-local state.
            # Recreating workers each epoch propagates ``set_epoch`` and therefore
            # makes resample_queries_each_epoch effective. Persistent workers would
            # otherwise keep their epoch-0 dataset copies for the whole run.
            kwargs["persistent_workers"] = False
        return kwargs

    TrainingDataModule._loader_kwargs = patched  # type: ignore[method-assign]
    TrainingDataModule._phsa_loader_patch = True


activate_phsa_sample_level_hypotheses()
_activate_phsa_loader_contract()

from train import _rewrite_positional_task, run  # noqa: E402


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def hydra_main(cfg: DictConfig) -> float | None:
    warnings.filterwarnings("ignore", category=UserWarning)
    setup_logger("ssq_fmt")
    return run(cfg)


if __name__ == "__main__":
    _rewrite_positional_task(sys.argv)
    hydra_main()
