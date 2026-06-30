#!/usr/bin/env python3
"""Training entrypoint for the audited sample-level PHSA path."""

from __future__ import annotations

import sys
from typing import Any

from torch.utils.data._utils.collate import default_collate

from minr_fmt.datamodule import TrainingDataModule
from minr_fmt.phsa_sample_level import activate_phsa_sample_level_hypotheses


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


def _activate_phsa_collate() -> None:
    original = TrainingDataModule._loader_kwargs
    if getattr(TrainingDataModule, "_phsa_collate_patch", False):
        return

    def patched(self, shuffle: bool, batch_size: int) -> dict[str, Any]:
        kwargs = original(self, shuffle=shuffle, batch_size=batch_size)
        kwargs["collate_fn"] = _phsa_collate
        return kwargs

    TrainingDataModule._loader_kwargs = patched  # type: ignore[method-assign]
    TrainingDataModule._phsa_collate_patch = True


activate_phsa_sample_level_hypotheses()
_activate_phsa_collate()

from train import _rewrite_positional_task, hydra_main  # noqa: E402


if __name__ == "__main__":
    _rewrite_positional_task(sys.argv)
    hydra_main()
