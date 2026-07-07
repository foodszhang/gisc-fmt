#!/usr/bin/env python3
"""Screen morphology-amplitude factorization on the reproduced Phase-A function."""

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
    attach_factorized_unified_output,
    load_historical_phase_a_checkpoint,
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
    """Fine-tune only the reproduced shared reconstruction and support factorization."""

    def _apply_finetune_setup(self):
        finetune_cfg = getattr(self.cfg.model, "finetune", None)
        if finetune_cfg is None:
            return
        init_from = str(getattr(finetune_cfg, "init_from_ckpt", "") or "")
        if not init_from:
            return
        report = load_historical_phase_a_checkpoint(self.net, init_from)
        print(f"[morphology-amplitude] historical Phase-A load: {report}")

    @staticmethod
    def _require_contract(cfg: DictConfig) -> None:
        view_cfg = cfg.model.ssq_fmt.view_complementary
        sample_cfg = view_cfg.get("sample_level_hypotheses", {}) or {}
        routing_cfg = view_cfg.get("routing", {}) or {}
        checks = {
            "training_phase=phase_a": str(view_cfg.get("training_phase", "")) == "phase_a",
            "ablation=full": str(view_cfg.get("ablation", "")) == "full",
            "strong_shared_fusion=true": bool(view_cfg.get("strong_shared_fusion", False)),
            "decoder_fusion_mode=joint_nonresidual": str(
                view_cfg.get("decoder_fusion_mode", "")
            )
            == "joint_nonresidual",
            "sample_level_hypotheses.enabled=true": bool(sample_cfg.get("enabled", False)),
            "routing.enabled=false": not bool(routing_cfg.get("enabled", False)),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(
                "Morphology-amplitude screening must preserve the reproduced Phase-A "
                f"contract; failed={failed}"
            )

    def __init__(self, cfg: DictConfig):
        self._require_contract(cfg)
        super().__init__(cfg)

        factor_cfg = cfg.model.ssq_fmt.view_complementary.get(
            "factorized_reconstruction", {}
        ) or {}
        attach_factorized_unified_output(self.net, factor_cfg)
        self.factorized_reconstruction_enabled = bool(factor_cfg.get("enabled", False))
        self.factorized_aux_warmup_steps = int(factor_cfg.get("aux_warmup_steps", 500))
        self.factorized_support_lr = float(
            factor_cfg.get("support_lr", max(float(cfg.optim.lr), 1.0e-4))
        )
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

        # Do not repeat the failed Phase-B/C strategy. The encoder, sampler,
        # hypothesis construction, PHSA, candidate adapters, and candidate heads are
        # all frozen. Only the already validated shared fusion/decoder function and
        # the new support head can change.
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)

        shared_modules = [
            self.net.complementary_aggregation.feature_norm,
            self.net.complementary_aggregation.common_projection,
            self.net.complementary_aggregation.output_norm,
            getattr(self.net.complementary_aggregation, "shared_attention", None),
            getattr(self.net.complementary_aggregation, "shared_set_fusion", None),
            self.net.unified_density_decoder.shared_norm,
            self.net.unified_density_decoder.shared_input,
            self.net.unified_density_decoder.head,
        ]
        for module in shared_modules:
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)

        support_head = getattr(
            self.net.unified_density_decoder,
            "factorized_support_head",
            None,
        )
        if support_head is not None:
            for parameter in support_head.parameters():
                parameter.requires_grad_(True)

        trainable_names = [
            name for name, parameter in self.net.named_parameters() if parameter.requires_grad
        ]
        forbidden_tokens = (
            "surface_encoder.",
            "surface_sampler.",
            "view_candidate_evidence.",
            "diverse_candidate_constructor.",
            "candidate_projection.",
            "candidate_context.",
            "candidate_input.",
            "candidate_residual_head.",
            "candidate_branch_head.",
        )
        leaked = [
            name for name in trainable_names if any(token in name for token in forbidden_tokens)
        ]
        if leaked:
            raise RuntimeError(f"candidate/PHSA parameters leaked into training: {leaked[:20]}")
        print(
            "[morphology-amplitude] trainable shared-path parameters: "
            f"{len(trainable_names)} tensors / "
            f"{sum(p.numel() for p in self.net.parameters() if p.requires_grad)} values"
        )

    def configure_optimizers(self):
        support_parameters = []
        support_ids: set[int] = set()
        support_head = getattr(
            self.net.unified_density_decoder,
            "factorized_support_head",
            None,
        )
        if support_head is not None:
            support_parameters = [
                parameter for parameter in support_head.parameters() if parameter.requires_grad
            ]
            support_ids = {id(parameter) for parameter in support_parameters}
        base_parameters = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in support_ids
        ]
        groups = []
        if base_parameters:
            groups.append(
                {
                    "params": base_parameters,
                    "lr": float(self.cfg.optim.lr),
                    "name": "reproduced_shared_path",
                }
            )
        if support_parameters:
            groups.append(
                {
                    "params": support_parameters,
                    "lr": self.factorized_support_lr,
                    "name": "factorized_support_head",
                }
            )
        if not groups:
            raise ValueError("No trainable morphology-amplitude parameters")

        optimizer = torch.optim.AdamW(
            groups,
            lr=float(self.cfg.optim.lr),
            betas=tuple(self.cfg.optim.betas),
            weight_decay=float(self.cfg.optim.weight_decay),
            eps=float(self.cfg.optim.eps),
        )
        scheduler_cfg = self.cfg.optim.get("scheduler", None)
        if scheduler_cfg is None:
            return {"optimizer": optimizer}
        eta_min = float(scheduler_cfg.get("eta_min", 0.0))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(self.cfg.trainer.max_epochs), 1),
            eta_min=eta_min,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx):
        base_loss = super().training_step(batch, batch_idx)
        if not self.factorized_reconstruction_enabled:
            return base_loss
        if self.factorized_objective is None:
            raise RuntimeError("factorized objective was not initialized")

        cached = self.net.unified_density_decoder.last_factorized_outputs
        required = {"support_logits", "support_probability", "amplitude", "density"}
        missing = sorted(required - set(cached))
        if missing:
            raise RuntimeError(
                "Factorized UnifiedDensityDecoder outputs were not produced: "
                f"missing={missing}"
            )
        target = batch["point_densities"].to(cached["density"]).unsqueeze(-1)
        objective = self.factorized_objective(
            support_logits=cached["support_logits"],
            support_probability=cached["support_probability"],
            amplitude=cached["amplitude"],
            density=cached["density"],
            target_density=target,
            component_ids=batch.get("query_component_ids"),
        )
        auxiliary_scale = (
            min(
                1.0,
                float(self.global_step + 1) / float(self.factorized_aux_warmup_steps),
            )
            if self.factorized_aux_warmup_steps > 0
            else 1.0
        )
        scaled_auxiliary = objective["total"] * auxiliary_scale
        total_loss = base_loss + scaled_auxiliary
        if not torch.isfinite(total_loss.detach()):
            raise FloatingPointError("morphology-amplitude objective became non-finite")
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
