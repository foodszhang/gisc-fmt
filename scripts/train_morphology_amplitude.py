#!/usr/bin/env python3
"""Train morphology-amplitude factorization on the original high-Dice Phase-A path."""

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
    activate_legacy_phase_a_decoder,
    load_legacy_phase_a_checkpoint,
)
from minr_fmt.phsa_sample_level import activate_phsa_sample_level_hypotheses
from minr_fmt.utils.logging_utils import setup_logger


def _morphology_collate(items: list[dict[str, Any]]) -> dict[str, Any]:
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
    """Continue the historical Phase-A decoder and add an optional support branch."""

    def _apply_finetune_setup(self):
        """Load the old checkpoint before the support branch is attached.

        The current refactor always instantiates ``unified_density_decoder``. The
        existing approximately 0.75 Phase-A checkpoint predates that inactive branch,
        so the generic strict loader rejects it. Here we permit missing tensors only
        under that inactive prefix and require exact matching for the historical active
        reconstruction path.
        """

        finetune_cfg = getattr(self.cfg.model, "finetune", None)
        if finetune_cfg is None:
            return
        init_from = str(getattr(finetune_cfg, "init_from_ckpt", "") or "")
        if not init_from:
            return
        report = load_legacy_phase_a_checkpoint(
            self.net,
            init_from,
            allowed_missing_prefixes=("unified_density_decoder.",),
        )
        print(
            "[morphology-amplitude] loaded historical Phase-A checkpoint: "
            f"{report}"
        )

    def __init__(self, cfg: DictConfig):
        view_cfg = cfg.model.ssq_fmt.view_complementary
        factor_cfg = view_cfg.get("factorized_reconstruction", {}) or {}
        if str(factor_cfg.get("base_decoder", "legacy_shared_logit")) != "legacy_shared_logit":
            raise ValueError(
                "This experiment must use base_decoder=legacy_shared_logit so it remains "
                "functionally compatible with the existing Phase-A checkpoint."
            )
        if str(view_cfg.get("training_phase", "")) != "phase_a":
            raise ValueError("Morphology-amplitude screening is restricted to Phase A")
        if str(view_cfg.get("ablation", "")) not in {"shared_only", "a0"}:
            raise ValueError("Historical Phase-A screening requires ablation=shared_only")

        super().__init__(cfg)
        activate_legacy_phase_a_decoder(self.net, factor_cfg)

        self.factorized_reconstruction_enabled = bool(factor_cfg.get("enabled", False))
        self.factorized_aux_warmup_steps = int(factor_cfg.get("aux_warmup_steps", 500))
        self.factorized_objective: MorphologyAmplitudeObjective | None = None
        if self.factorized_reconstruction_enabled:
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

        # The experiment continues only the old active Phase-A representation and
        # density decoder. Candidate construction, PHSA, and the newer unified decoder
        # are not optimized and cannot affect the result.
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)
        for module in (
            self.net.complementary_aggregation,
            self.net.shared_density_logit_decoder,
            getattr(self.net, "factorized_support_decoder", None),
        ):
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

    def training_step(self, batch, batch_idx):
        base_loss = super().training_step(batch, batch_idx)
        if not self.factorized_reconstruction_enabled:
            return base_loss
        if self.factorized_objective is None:
            raise RuntimeError("factorized objective was not initialized")

        cached = getattr(self.net, "last_factorized_outputs", {})
        required = {"support_probability", "amplitude", "density"}
        missing = sorted(required - set(cached))
        if missing:
            raise RuntimeError(
                "Historical Phase-A factorized outputs were not produced: "
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
        auxiliary_scale = (
            min(1.0, float(self.global_step + 1) / float(self.factorized_aux_warmup_steps))
            if self.factorized_aux_warmup_steps > 0
            else 1.0
        )
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
