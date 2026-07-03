#!/usr/bin/env python3
"""Train the Phase-A morphology-amplitude factorization experiment.

This entrypoint deliberately keeps the production Lightning module untouched. It
loads the audited Phase-A checkpoint first, then attaches the support head and adds
only the factorization-specific auxiliary objective.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import default_collate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.datamodule import TrainingDataModule
from minr_fmt.module import TrainingLightningModule as BaseTrainingLightningModule
from minr_fmt.network.morphology_amplitude import (
    MorphologyAmplitudeLossConfig,
    MorphologyAmplitudeObjective,
)
from minr_fmt.phsa_sample_level import activate_phsa_sample_level_hypotheses
from minr_fmt.utils.logging_utils import setup_logger


def _morphology_collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate query supervision without moving unused full GT volumes."""

    batch = default_collate(items)
    batch.pop("gt_voxels", None)
    return batch


def _activate_loader_contract() -> None:
    original = TrainingDataModule._loader_kwargs
    if getattr(TrainingDataModule, "_morphology_amplitude_loader_patch", False):
        return

    def patched(self, shuffle: bool, batch_size: int) -> dict[str, Any]:
        kwargs = original(self, shuffle=shuffle, batch_size=batch_size)
        kwargs["collate_fn"] = _morphology_collate
        if int(kwargs.get("num_workers", 0)) > 0:
            kwargs["persistent_workers"] = False
        return kwargs

    TrainingDataModule._loader_kwargs = patched  # type: ignore[method-assign]
    TrainingDataModule._morphology_amplitude_loader_patch = True


class MorphologyAmplitudeTrainingModule(BaseTrainingLightningModule):
    """Add a support-amplitude objective to the existing Phase-A reconstruction."""

    def __init__(self, cfg: DictConfig):
        # Base initialization intentionally happens first. This loads the scalar
        # Phase-A checkpoint against the original architecture before the new head
        # is attached, avoiding any non-strict or ad-hoc checkpoint migration.
        super().__init__(cfg)
        view_cfg = cfg.model.ssq_fmt.view_complementary
        factor_cfg = view_cfg.get("factorized_reconstruction", {}) or {}
        self.factorized_reconstruction_enabled = bool(factor_cfg.get("enabled", False))
        self.factorized_objective: MorphologyAmplitudeObjective | None = None
        self.factorized_aux_warmup_steps = int(factor_cfg.get("aux_warmup_steps", 500))

        if not self.factorized_reconstruction_enabled:
            return
        if str(view_cfg.get("training_phase", "")) != "phase_a":
            raise ValueError(
                "The first morphology-amplitude experiment is restricted to Phase A. "
                "Do not mix it with PHSA/candidate-path changes in the same run."
            )
        decoder = getattr(self.net, "unified_density_decoder", None)
        if decoder is None or not hasattr(decoder, "enable_factorized_output"):
            raise RuntimeError("SSQ-FMT unified density decoder lacks factorized-output support")
        decoder.enable_factorized_output(
            compose_density=bool(factor_cfg.get("compose_density", True)),
            support_init_logit=float(factor_cfg.get("support_init_logit", 8.0)),
        )
        self.factorized_objective = MorphologyAmplitudeObjective(
            MorphologyAmplitudeLossConfig(
                support_threshold=float(factor_cfg.get("support_threshold", 0.05)),
                support_weight=float(factor_cfg.get("support_weight", 0.10)),
                support_dice_weight=float(factor_cfg.get("support_dice_weight", 1.0)),
                amplitude_weight=float(factor_cfg.get("amplitude_weight", 0.25)),
                component_weight=float(factor_cfg.get("component_weight", 0.0)),
                eps=float(factor_cfg.get("eps", 1.0e-6)),
            )
        )

    def training_step(self, batch, batch_idx):
        base_loss = super().training_step(batch, batch_idx)
        if not self.factorized_reconstruction_enabled:
            return base_loss
        if self.factorized_objective is None:
            raise RuntimeError("factorized objective was not initialized")

        decoder = self.net.unified_density_decoder
        cached = decoder.last_factorized_outputs
        required = {"support_probability", "amplitude", "density"}
        missing = sorted(required - set(cached))
        if missing:
            raise RuntimeError(
                "Factorized decoder outputs were not produced by the current forward pass: "
                f"missing={missing}"
            )
        target = batch["point_densities"].to(cached["density"]).unsqueeze(-1)
        objective = self.factorized_objective(
            support_probability=cached["support_probability"],
            amplitude=cached["amplitude"],
            density=cached["density"],
            target_density=target,
            component_ids=batch.get("query_component_ids"),
        )
        if self.factorized_aux_warmup_steps > 0:
            auxiliary_scale = min(
                1.0,
                float(self.global_step + 1) / float(self.factorized_aux_warmup_steps),
            )
        else:
            auxiliary_scale = 1.0
        scaled_auxiliary = objective["total"] * auxiliary_scale
        total_loss = base_loss + scaled_auxiliary
        if not torch.isfinite(total_loss.detach()):
            raise FloatingPointError("morphology-amplitude objective produced a non-finite loss")
        for key, value in objective.items():
            self.log(
                f"train_factorized_{key}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        self.log(
            "train_factorized_auxiliary_scale",
            cached["density"].new_tensor(auxiliary_scale),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train_factorized_scaled_total",
            scaled_auxiliary,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log(
            "train_loss_with_factorization",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return total_loss


activate_phsa_sample_level_hypotheses()
_activate_loader_contract()

# train.run resolves this global at runtime, so replacing it here makes fit,
# validate, and test instantiate the extension-aware Lightning module.
import train as train_entry  # noqa: E402

train_entry.TrainingLightningModule = MorphologyAmplitudeTrainingModule


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def hydra_main(cfg: DictConfig) -> float | None:
    warnings.filterwarnings("ignore", category=UserWarning)
    setup_logger("ssq_fmt")
    return train_entry.run(cfg)


if __name__ == "__main__":
    train_entry._rewrite_positional_task(sys.argv)
    hydra_main()
