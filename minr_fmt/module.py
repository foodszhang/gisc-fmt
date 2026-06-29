"""LightningModule with Hydra configuration support"""

import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from .loss import (
    AuxProjectionLightLoss,
    MorphologyAwareDensityLoss,
    VoxelReconstructionLoss,
    compute_dice,
    dice_coefficient,
)
from .model_factory import ModelFactory
from .utils.utils import get_psnr_3d, get_ssim_3d

E15_INIT_WHITELIST = (
    "surface_encoder.",
    "surface_sampler.context_net.",
    "surface_sampler.sample_projection.",
    "view_encoder.",
    "shared_density_logit_decoder.",
)

E15_EXPLICIT_KEY_MAP = {
    "shared_unet.feature_fusion.0.double_conv.0.weight": "surface_encoder.fuse.1.net.0.weight",
    "shared_unet.feature_fusion.0.double_conv.1.weight": "surface_encoder.fuse.1.net.1.weight",
    "shared_unet.feature_fusion.0.double_conv.1.bias": "surface_encoder.fuse.1.net.1.bias",
    "density_head.mlp_out.3.weight": "shared_density_logit_decoder.fusion.6.weight",
    "density_head.mlp_out.3.bias": "shared_density_logit_decoder.fusion.6.bias",
}


def load_e15_compatible_weights(model: torch.nn.Module, checkpoint_path: str) -> dict[str, list]:
    """Load only explicitly whitelisted, name- and shape-identical E15 tensors."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = checkpoint.get("state_dict", checkpoint)
    source = {key.removeprefix("net."): value for key, value in source.items()}
    target = model.state_dict()
    loaded: dict[str, torch.Tensor] = {}
    mismatched: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    intentionally_skipped: list[str] = []
    for source_key, value in source.items():
        key = E15_EXPLICIT_KEY_MAP.get(source_key, source_key)
        explicitly_mapped = source_key in E15_EXPLICIT_KEY_MAP
        if not key.startswith(E15_INIT_WHITELIST):
            intentionally_skipped.append(source_key)
            continue
        if not explicitly_mapped and not source_key.startswith(E15_INIT_WHITELIST):
            intentionally_skipped.append(key)
            continue
        if key in target and target[key].shape == value.shape:
            loaded[key] = value
        elif key in target:
            mismatched.append((key, tuple(value.shape), tuple(target[key].shape)))
        else:
            intentionally_skipped.append(source_key)
    missing = sorted(
        key for key in target if key.startswith(E15_INIT_WHITELIST) and key not in loaded
    )
    model.load_state_dict(loaded, strict=False)
    report = {
        "loaded_keys": sorted(loaded),
        "missing_keys": missing,
        "shape_mismatched_keys": mismatched,
        "intentionally_skipped_keys": sorted(intentionally_skipped),
    }
    print(f"[e15-init] checkpoint={checkpoint_path}")
    for name, values in report.items():
        print(f"[e15-init] {name} ({len(values)}): {values}")
    return report


class TrainingLightningModule(LightningModule):
    """
    PyTorch Lightning training module with Hydra config support

    Key design principles:
    1. All hyperparameters come from OmegaConf DictConfig
    2. forward() directly calls the original model's forward
    3. training_step/validation_step only call loss functions (no math changes)
    4. Optimizer and scheduler configuration come from config
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize LightningModule

        Args:
            cfg: Complete Hydra config (DictConfig)
        """
        super().__init__()
        self.cfg = cfg

        # Save hyperparameters for logging/reproducibility
        # Note: We save a copy of the config dict, not the DictConfig object itself
        self.save_hyperparameters(ignore=["cfg"])

        # Extract commonly used parameters
        self.learning_rate = cfg.optim.lr
        self.max_epochs = cfg.trainer.max_epochs
        self._center_distance_sigma_mm: float = 0.5
        self._center_distance_center_weight: float = 0.2
        self._center_distance_weight: float = 0.1
        self._empty_slot_weight: float = 0.0
        self._backbone_logit_loss_weight: float = 0.0

        # Create model and loss
        self._setup_model()
        self._setup_loss()

        # Training metrics tracking
        self.best_dice = -1.0
        self._last_gradient_norm_step = -1

        # Test-time state (initialized in on_test_start)
        self._test_out_dir: Path | None = None
        self._test_recon_dir: Path | None = None
        self._test_seg_dir: Path | None = None
        self._test_proj_dir: Path | None = None
        self._test_angles_to_save: list[str] = []
        self._test_angle_saved_counts: dict[str, int] = {}
        self._test_base_seg: np.ndarray | None = None
        self._test_new_label: int | None = None
        self._test_pred_threshold: float = 0.5
        self._test_max_samples_per_angle: int = 10
        self._test_save_recon_roi: bool = True
        self._test_save_registered_seg: bool = True
        self._test_save_proj_comparisons: bool = True
        validation_cfg = cfg.get("validation", None)
        test_cfg = cfg.get("test", None)
        if validation_cfg is not None and "pred_threshold" in validation_cfg:
            self._validation_pred_threshold = float(validation_cfg.pred_threshold)
        elif test_cfg is not None and "pred_threshold" in test_cfg:
            self._validation_pred_threshold = float(test_cfg.pred_threshold)
        else:
            self._validation_pred_threshold = 0.5

    @staticmethod
    def _center_focal_loss(
        logits: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 2.0,
        beta: float = 4.0,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        pred = torch.sigmoid(logits).clamp(eps, 1.0 - eps)

        pos_mask = target >= 0.5
        neg_mask = target < 0.5

        pos_loss = -torch.log(pred) * torch.pow(1.0 - pred, alpha) * pos_mask.float()
        neg_weight = torch.pow(1.0 - target, beta)
        neg_loss = -torch.log(1.0 - pred) * torch.pow(pred, alpha) * neg_weight * neg_mask.float()

        pos_count = pos_mask.float().sum().clamp_min(1.0)
        return (pos_loss.sum() + neg_loss.sum()) / pos_count

    def _setup_model(self):
        """Create model from config using ModelFactory"""
        self.net = ModelFactory.create_model(
            model_type=self.cfg.model.name,
            config=self.cfg,
        )
        self._apply_finetune_setup()

        # Optional torch.compile for speed (PyTorch 2.x)
        if getattr(self.cfg.trainer, "torch_compile", False) and hasattr(torch, "compile"):
            mode = str(getattr(self.cfg.trainer, "torch_compile_mode", "reduce-overhead"))

            # Inductor/Triton can fail on some shapes/kernels, e.g. int32 indexing overflow.
            # Suppress compile errors so Dynamo falls back to eager instead of crashing.
            try:
                from torch import _dynamo  # type: ignore

                _dynamo.config.suppress_errors = True
            except Exception:
                pass

            self.net = torch.compile(self.net, mode=mode)  # type: ignore[attr-defined]

    def _apply_finetune_setup(self):
        """Optional non-strict initialization/freezing for targeted fine-tuning configs."""
        finetune_cfg = getattr(self.cfg.model, "finetune", None)
        if finetune_cfg is None:
            return

        def apply_ssq_controls(initial_state: dict[str, torch.Tensor]) -> None:
            reset_modules = [str(name) for name in getattr(finetune_cfg, "reset_modules", []) or []]
            if reset_modules:
                reset_prefixes = tuple(f"{name}." for name in reset_modules)
                reset_state = {
                    key: value
                    for key, value in initial_state.items()
                    if key.startswith(reset_prefixes)
                }
                missing, unexpected = self.net.load_state_dict(reset_state, strict=False)
                if unexpected:
                    raise RuntimeError(
                        f"SSQ module reset produced unexpected keys: {unexpected[:20]}"
                    )
                print(
                    f"[finetune] reset modules to fresh initialization: {reset_modules}; "
                    f"keys={len(reset_state)}"
                )
            freeze_modules = [
                str(name) for name in getattr(finetune_cfg, "freeze_modules", []) or []
            ]
            for module_name in freeze_modules:
                for parameter in self.net.get_submodule(module_name).parameters():
                    parameter.requires_grad = False
            train_modules_only = [
                str(name) for name in getattr(finetune_cfg, "train_modules_only", []) or []
            ]
            if train_modules_only:
                for parameter in self.net.parameters():
                    parameter.requires_grad = False
                for module_name in train_modules_only:
                    for parameter in self.net.get_submodule(module_name).parameters():
                        parameter.requires_grad = True
            if reset_modules or freeze_modules or train_modules_only:
                trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
                total = sum(p.numel() for p in self.net.parameters())
                print(
                    f"[finetune] SSQ controls: freeze={freeze_modules} "
                    f"train_only={train_modules_only}; trainable={trainable}/{total}"
                )

        init_from = str(getattr(finetune_cfg, "init_from_ckpt", "") or "")
        if init_from:
            if self._is_ssq_model():
                if bool(getattr(finetune_cfg, "e15_compatible_init", False)):
                    initial_state = {
                        key: value.detach().clone() for key, value in self.net.state_dict().items()
                    }
                    load_e15_compatible_weights(self.net, init_from)
                    apply_ssq_controls(initial_state)
                    return
                ckpt = torch.load(init_from, map_location="cpu", weights_only=False)
                state = ckpt.get("state_dict", ckpt)
                net_state = {
                    key[len("net.") :]: value
                    for key, value in state.items()
                    if key.startswith("net.")
                }
                if not net_state:
                    net_state = state
                current_state = self.net.state_dict()
                allow_missing_prefixes = tuple(
                    str(prefix) for prefix in getattr(finetune_cfg, "allow_missing_prefixes", [])
                )
                matching_state = {
                    key: value
                    for key, value in net_state.items()
                    if key in current_state and current_state[key].shape == value.shape
                }
                missing_current = sorted(set(current_state) - set(matching_state))
                unexpected_source = sorted(set(net_state) - set(matching_state))
                allowed_extension = (
                    bool(allow_missing_prefixes)
                    and not unexpected_source
                    and all(key.startswith(allow_missing_prefixes) for key in missing_current)
                )
                if allowed_extension:
                    missing, unexpected = self.net.load_state_dict(matching_state, strict=False)
                    if unexpected or sorted(missing) != missing_current:
                        raise RuntimeError(
                            "SSQ extension initialization produced inconsistent keys: "
                            f"missing={missing[:20]} unexpected={unexpected[:20]}"
                        )
                    print(
                        f"[finetune] initialized SSQ-FMT extension from {init_from}; "
                        f"new_keys={len(missing_current)}"
                    )
                    apply_ssq_controls(current_state)
                    return
                if all(
                    key in current_state and current_state[key].shape == value.shape
                    for key, value in net_state.items()
                ):
                    self.net.load_state_dict(net_state, strict=True)
                    print(f"[finetune] strictly initialized SSQ-FMT net from {init_from}")
                else:
                    mapped_state = dict(current_state)
                    mapped = {}
                    for key, value in net_state.items():
                        mapped_key = f"query_density_backbone.{key}"
                        if (
                            mapped_key in current_state
                            and current_state[mapped_key].shape == value.shape
                        ):
                            mapped[mapped_key] = value
                    backbone_keys = [
                        key for key in current_state if key.startswith("query_density_backbone.")
                    ]
                    missing_backbone = [key for key in backbone_keys if key not in mapped]
                    if missing_backbone:
                        raise RuntimeError(
                            "SSQ-FMT strict backbone init failed; missing mapped keys: "
                            f"{missing_backbone[:20]}"
                        )
                    mapped_state.update(mapped)
                    self.net.load_state_dict(mapped_state, strict=True)
                    print(
                        f"[finetune] strictly initialized SSQ-FMT query backbone from "
                        f"{init_from}; mapped_backbone_keys={len(mapped)}"
                    )
                apply_ssq_controls(current_state)
                return
            ckpt = torch.load(init_from, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt)
            net_state = {}
            for key, value in state.items():
                if key.startswith("net."):
                    net_state[key[len("net.") :]] = value
            if not net_state:
                net_state = state
            current_state = self.net.state_dict()
            skipped = []
            filtered_state = {}
            for key, value in net_state.items():
                current_value = current_state.get(key)
                if current_value is not None and current_value.shape != value.shape:
                    if (
                        value.ndim == 2
                        and current_value.ndim == 2
                        and current_value.shape[0] == value.shape[0]
                        and current_value.shape[1] > value.shape[1]
                    ):
                        adapted = torch.zeros_like(current_value)
                        adapted[:, : value.shape[1]] = value.to(
                            dtype=current_value.dtype, device=current_value.device
                        )
                        filtered_state[key] = adapted
                        print(
                            f"[finetune] expanded linear weight {key}: "
                            f"{tuple(value.shape)} -> {tuple(current_value.shape)}"
                        )
                        continue
                    skipped.append((key, tuple(value.shape), tuple(current_value.shape)))
                    continue
                filtered_state[key] = value
            net_state = filtered_state
            missing, unexpected = self.net.load_state_dict(net_state, strict=False)
            print(
                f"[finetune] initialized net from {init_from}; "
                f"missing={len(missing)} unexpected={len(unexpected)} "
                f"skipped_shape_mismatch={len(skipped)}"
            )
            if skipped:
                print(f"[finetune] skipped shape-mismatched keys: {skipped}")
            if missing:
                print(f"[finetune] missing keys: {missing}")
            if unexpected:
                print(f"[finetune] unexpected keys: {unexpected}")

        if bool(getattr(finetune_cfg, "freeze_except_mean_prior_gate", False)):
            for param in self.net.parameters():
                param.requires_grad = False
            if not hasattr(self.net, "mean_prior_residual_gate"):
                raise ValueError(
                    "model.finetune.freeze_except_mean_prior_gate=true requires "
                    "mean_prior_residual_gate"
                )
            for param in self.net.mean_prior_residual_gate.parameters():
                param.requires_grad = True
            trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.net.parameters())
            print(f"[finetune] trainable parameters: {trainable}/{total}")

        freeze_modules = list(getattr(finetune_cfg, "freeze_modules", []) or [])
        for module_name in freeze_modules:
            module = self.net.get_submodule(str(module_name))
            for param in module.parameters():
                param.requires_grad = False
        if freeze_modules:
            trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.net.parameters())
            print(
                f"[finetune] frozen modules: {freeze_modules}; "
                f"trainable parameters: {trainable}/{total}"
            )

        train_modules_only = list(getattr(finetune_cfg, "train_modules_only", []) or [])
        if train_modules_only:
            for param in self.net.parameters():
                param.requires_grad = False
            for module_name in train_modules_only:
                module = self.net.get_submodule(str(module_name))
                for param in module.parameters():
                    param.requires_grad = True
            trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.net.parameters())
            print(
                f"[finetune] trainable modules only: {train_modules_only}; "
                f"trainable parameters: {trainable}/{total}"
            )

    def _setup_loss(self):
        """Create loss function from config"""
        loss_cfg = self.cfg.loss
        self._center_distance_sigma_mm = float(loss_cfg.get("center_sigma_mm", 0.5))
        self._center_distance_center_weight = float(loss_cfg.get("center_weight", 0.2))
        self._center_distance_weight = float(loss_cfg.get("distance_weight", 0.1))
        self._empty_slot_weight = float(loss_cfg.get("empty_slot_weight", 0.0))
        self._backbone_logit_loss_weight = float(loss_cfg.get("backbone_logit_loss_weight", 0.0))
        self.loss_func = AuxProjectionLightLoss(
            init_scatter_weight=loss_cfg.get(
                "aux_projection_weight", loss_cfg.get("scatter_weight", 1.0)
            ),
            target_scatter_weight=loss_cfg.get(
                "target_aux_projection_weight", loss_cfg.get("target_scatter_weight", 0.5)
            ),
            start_decay_epoch=loss_cfg.start_decay_epoch,
            decay_epochs=loss_cfg.decay_epochs,
            pos_weight=loss_cfg.pos_weight,
            sparse_weight=loss_cfg.sparse_weight,
            lambda_dice=loss_cfg.dice_weight,
            use_tversky=loss_cfg.get("use_tversky", True),
            tversky_alpha=loss_cfg.get("tversky_alpha", 0.6),
            tversky_beta=loss_cfg.get("tversky_beta", 0.4),
            tversky_gamma=loss_cfg.get("tversky_gamma", 1.33),
            tversky_weight=loss_cfg.get("tversky_weight", None),
            light_weight=loss_cfg.get("light_weight", 1.0),
        )
        self.voxel_loss_func = VoxelReconstructionLoss(
            pos_weight=loss_cfg.pos_weight,
            sparse_weight=loss_cfg.sparse_weight,
            lambda_dice=loss_cfg.dice_weight,
            use_tversky=loss_cfg.get("use_tversky", True),
            tversky_alpha=loss_cfg.get("tversky_alpha", 0.6),
            tversky_beta=loss_cfg.get("tversky_beta", 0.4),
            tversky_gamma=loss_cfg.get("tversky_gamma", 1.33),
            tversky_weight=loss_cfg.get("tversky_weight", None),
        )
        self.ssq_loss_func = MorphologyAwareDensityLoss(
            lambda_sdf=loss_cfg.get("lambda_sdf", 0.0),
            tau_s=loss_cfg.get("tau_s", 3.0),
            boundary_weight=loss_cfg.get("sdf_boundary_weight", 1.0),
            pos_weight=loss_cfg.get("pos_weight", 1.0),
            dice_weight=loss_cfg.get("dice_weight", 0.0),
            sparse_weight=loss_cfg.get("sparse_weight", 0.0),
            density_bce_weight=loss_cfg.get("density_bce_weight", 0.0),
            tversky_weight=loss_cfg.get("tversky_weight", 0.0),
            tversky_alpha=loss_cfg.get("tversky_alpha", 0.6),
            tversky_beta=loss_cfg.get("tversky_beta", 0.4),
            tversky_gamma=loss_cfg.get("tversky_gamma", 1.33),
            candidate_branch_density_weight=loss_cfg.get("candidate_branch_density_weight", 0.0),
            candidate_branch_dice_weight=loss_cfg.get("candidate_branch_dice_weight", 0.0),
            candidate_branch_prior_power=loss_cfg.get("candidate_branch_prior_power", 1.0),
            candidate_branch_min_prior=loss_cfg.get("candidate_branch_min_prior", 0.0),
            candidate_branch_target_mode=loss_cfg.get("candidate_branch_target_mode", "soft_prior"),
            candidate_assignment_weight=loss_cfg.get("candidate_assignment_weight", 0.0),
            candidate_assignment_min_prior=loss_cfg.get("candidate_assignment_min_prior", 0.05),
            candidate_assignment_target_mode=loss_cfg.get(
                "candidate_assignment_target_mode", "best_prior"
            ),
            component_match_center_weight=loss_cfg.get("component_match_center_weight", 0.25),
            component_unmatched_weight=loss_cfg.get("component_unmatched_weight", 0.25),
            lambda_shared=loss_cfg.get("lambda_shared", 0.0),
            lambda_quot=loss_cfg.get("lambda_quot", 0.0),
            lambda_res=loss_cfg.get("lambda_res", 0.0),
        )

    def _is_ssq_model(self) -> bool:
        return str(getattr(self.cfg.model, "name", "")).lower() == "ssq_fmt"

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        routing_key = "net.bounded_view_routing.routing_gain_raw"
        if routing_key in self.state_dict() and routing_key not in state_dict:
            state_dict = dict(state_dict)
            state_dict[routing_key] = self.state_dict()[routing_key]
        if self._is_ssq_model() and not strict:
            missing = sorted(set(self.state_dict().keys()) - set(state_dict.keys()))
            unexpected = sorted(set(state_dict.keys()) - set(self.state_dict().keys()))
            raise RuntimeError(
                "SSQ-FMT checkpoint loading requires strict=True; "
                f"missing_keys={missing[:20]} unexpected_keys={unexpected[:20]}"
            )
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(
        self,
        projections,
        points,
        points_mm=None,
        depth_maps=None,
        source_hypotheses=None,
        aux_only: bool = False,
    ):
        """Forward pass."""
        return self.net(
            projections,
            points,
            points_mm=points_mm,
            depth_maps=depth_maps,
            source_hypotheses=source_hypotheses,
            aux_only=aux_only,
        )

    def _call_model(
        self,
        projections,
        points,
        points_mm=None,
        depth_maps=None,
        source_hypotheses=None,
        aux_only: bool = False,
    ):
        if hasattr(self.net, "set_training_epoch"):
            self.net.set_training_epoch(int(self.current_epoch))
        out = self(
            projections,
            points,
            points_mm=points_mm,
            depth_maps=depth_maps,
            source_hypotheses=source_hypotheses,
            aux_only=aux_only,
        )
        if isinstance(out, tuple) and len(out) >= 2:
            return out[0], out[1]
        raise RuntimeError(f"Unexpected model output: {type(out)}")

    def _call_ssq_model(self, batch, return_diagnostics: bool = False):
        if hasattr(self.net, "set_training_epoch"):
            self.net.set_training_epoch(int(self.current_epoch))
        if hasattr(self.net, "set_training_step"):
            self.net.set_training_step(int(self.global_step))
        surface = batch.get("surface_measurements_packed", batch.get("projections_packed"))
        if surface is None:
            raise KeyError(
                "SSQ-FMT requires batch.surface_measurements_packed or projections_packed"
            )
        query_mm = batch.get("query_coordinates_mm", batch.get("points_mm"))
        if query_mm is None:
            raise KeyError("SSQ-FMT requires batch.query_coordinates_mm or points_mm")
        out = self.net(
            surface,
            query_mm,
            detector_valid_mask=batch.get("detector_valid_mask"),
            depth_maps=batch.get("depth_maps"),
            batch=batch,
            return_diagnostics=return_diagnostics,
        )
        if not isinstance(out, dict) or "density" not in out:
            raise RuntimeError(f"Unexpected SSQ-FMT model output: {type(out)}")
        return out

    @staticmethod
    def _source_hypotheses_from_batch(batch):
        if "source_hypothesis_centers" not in batch:
            return None
        return {
            "centers": batch["source_hypothesis_centers"],
            "peak_scores": batch.get("source_hypothesis_peak_scores"),
            "scales": batch.get("source_hypothesis_scales"),
            "valid": batch.get("source_hypothesis_valid"),
        }

    def _log_source_cue_stats(self, prefix: str, batch, point_densities: torch.Tensor) -> None:
        stats = getattr(self.net, "last_source_cue_stats", None)
        if stats:
            for key, value in stats.items():
                if torch.is_tensor(value):
                    self.log(
                        f"{prefix}_source_cue_{key}",
                        value.detach(),
                        prog_bar=False,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                    )
        decoder_stats = getattr(self.net, "last_source_decoder_stats", None)
        if decoder_stats:
            for key, value in decoder_stats.items():
                if torch.is_tensor(value):
                    self.log(
                        f"{prefix}_source_decoder_{key}",
                        value.detach(),
                        prog_bar=False,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                    )
        nearest_dist = getattr(self.net, "last_source_nearest_dist", None)
        if torch.is_tensor(nearest_dist):
            mask = point_densities > 0.0
            if mask.any():
                self.log(
                    f"{prefix}_positive_query_nearest_peak_dist",
                    nearest_dist.to(point_densities.device)[mask].mean().detach(),
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

    def _is_voxel_model(self) -> bool:
        return (
            str(getattr(self.cfg.model, "output_type", "")).lower() == "voxel"
            or str(getattr(self.net, "output_type", "")).lower() == "voxel"
        )

    def _call_voxel_model(self, projections, batch):
        if hasattr(self.net, "set_training_epoch"):
            self.net.set_training_epoch(int(self.current_epoch))
        out = self.net(
            projections,
            points=batch.get("points"),
            points_mm=batch.get("points_mm"),
            depth_maps=batch.get("depth_maps"),
            batch=batch,
        )
        if isinstance(out, dict) and "pred_voxel" in out:
            return out["pred_voxel"], out.get("aux_outputs", {})
        if torch.is_tensor(out):
            return out, {}
        raise RuntimeError(f"Unexpected voxel model output: {type(out)}")

    def _voxel_target(self, batch) -> torch.Tensor:
        if "gt_voxels" in batch:
            return batch["gt_voxels"].float()
        point_densities = batch["point_densities"]
        voxel_shape = batch["feasible_voxel_shape"]
        shape = self._shape_tuple_from_batch(voxel_shape, point_densities.shape[0])
        if int(np.prod(shape[1:])) != point_densities.shape[1]:
            raise ValueError("Voxel baseline requires batch.gt_voxels or full-grid point labels")
        return point_densities.reshape(shape).float()

    def _standardize_pred_voxel(self, pred_voxel, target_voxel):
        if pred_voxel.dim() == 5 and pred_voxel.size(1) == 1:
            pred_voxel = pred_voxel[:, 0]
        if target_voxel.dim() == 5 and target_voxel.size(1) == 1:
            target_voxel = target_voxel[:, 0]
        if pred_voxel.shape[1:] != target_voxel.shape[1:]:
            raise ValueError(
                "Voxel prediction and target shapes differ during evaluation. "
                f"pred={tuple(pred_voxel.shape)}, target={tuple(target_voxel.shape)}. "
                "Use full-volume output, crop_target, paste_pred, or mesh_to_voxel alignment."
            )
        return pred_voxel, target_voxel

    def _prepare_projection_input(self, batch):
        projections = batch["projections"]
        if (
            self.cfg.model.name
            in {
                "minr_fmt",
                "gisc_fmt",
                "point_cqr",
                "fixed_footprint_cqr",
                "depth_footprint_cqr",
                "unconstrained_adaptive_cqr",
            }
            and "projections_packed" in batch
        ):
            p = batch["projections_packed"]  # [B,V,1,H,W]
            B, V = p.shape[0], p.shape[1]
            # GISC-FMT expects view-major flattening: [V*B,1,H,W]
            return projections, p.permute(1, 0, 2, 3, 4).reshape(B * V, 1, p.shape[-2], p.shape[-1])
        if self._is_ssq_model() and "surface_measurements_packed" in batch:
            return batch["surface_measurements"], batch["surface_measurements_packed"]
        return projections, projections

    def _shape_tuple_from_batch(self, voxel_shape, batch_size: int) -> tuple[int, int, int, int]:
        return (
            batch_size,
            int(voxel_shape[0][0].detach().cpu().numpy()),
            int(voxel_shape[1][0].detach().cpu().numpy()),
            int(voxel_shape[2][0].detach().cpu().numpy()),
        )

    @staticmethod
    def _connected_components_from_voxels(
        gt_voxels: torch.Tensor,
    ) -> list[tuple[torch.Tensor, np.ndarray, float]]:
        from scipy import ndimage

        gt_np = gt_voxels.detach().to(dtype=torch.float32).cpu().numpy()
        if gt_np.ndim == 4 and gt_np.shape[0] == 1:
            gt_np = gt_np[0]
        structure = ndimage.generate_binary_structure(3, 1)
        labeled, num = ndimage.label(gt_np > 0.0, structure=structure)
        comps: list[tuple[torch.Tensor, np.ndarray, float]] = []
        for label_id in range(1, num + 1):
            mask = labeled == label_id
            if not mask.any():
                continue
            coords = np.argwhere(mask)
            center = coords.astype(np.float32).mean(axis=0)
            dist_map = ndimage.distance_transform_edt(mask).astype(np.float32)
            radius = float(dist_map[mask].max())
            comps.append((torch.tensor(center, dtype=torch.float32), dist_map, max(radius, 1.0)))
        return comps

    def _center_distance_query_targets(
        self,
        batch: dict,
        points_mm: torch.Tensor,
        points_ijk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            "center_target" in batch
            and "distance_target" in batch
            and "center_distance_fg_mask" in batch
        ):
            return (
                batch["center_target"],
                batch["distance_target"],
                batch["center_distance_fg_mask"],
            )

        gt_voxels = batch.get("gt_voxels")
        if gt_voxels is None:
            raise ValueError("Center-distance auxiliary supervision requires batch.gt_voxels")
        voxel_size_mm = float(getattr(self.cfg.data, "voxel_size_mm", 0.2))
        gt = gt_voxels.detach().to(device=points_mm.device, dtype=torch.float32)
        B, N, _ = points_mm.shape
        center_targets = torch.zeros((B, N, 1), device=points_mm.device, dtype=torch.float32)
        distance_targets = torch.zeros((B, N, 1), device=points_mm.device, dtype=torch.float32)
        fg_mask = torch.zeros((B, N, 1), device=points_mm.device, dtype=torch.float32)
        for b in range(B):
            comps = self._connected_components_from_voxels(gt[b])
            if not comps:
                continue
            pts = points_ijk[b].detach().to(device=points_mm.device, dtype=torch.float32)
            gt_b = gt[b]
            center_vals = []
            dist_vals = []
            fg_vals = []
            for center_ijk, dist_map_np, radius_vox in comps:
                center_mm = (center_ijk.to(device=points_mm.device) + 0.5) * voxel_size_mm
                diff = points_mm[b] - center_mm[None, :]
                dist_sq = (diff * diff).sum(dim=-1, keepdim=True)
                center_vals.append(
                    torch.exp(-dist_sq / (2.0 * (self._center_distance_sigma_mm**2)))
                )
                x = pts[:, 0].round().long().clamp(0, gt_b.shape[0] - 1)
                y = pts[:, 1].round().long().clamp(0, gt_b.shape[1] - 1)
                z = pts[:, 2].round().long().clamp(0, gt_b.shape[2] - 1)
                mask = gt_b[x, y, z] > 0.0
                fg_vals.append(mask.unsqueeze(-1).float())
                if mask.any():
                    dist_map = torch.tensor(
                        dist_map_np, device=points_mm.device, dtype=torch.float32
                    )
                    dist_to_boundary = dist_map[x, y, z].unsqueeze(-1) * voxel_size_mm
                    dist_vals.append(
                        (dist_to_boundary / (radius_vox * voxel_size_mm)).clamp(0.0, 1.0)
                    )
                else:
                    dist_vals.append(
                        torch.zeros((N, 1), device=points_mm.device, dtype=torch.float32)
                    )
            center_targets[b] = torch.stack(center_vals, dim=0).amax(dim=0)
            distance_targets[b] = torch.stack(dist_vals, dim=0).amax(dim=0)
            fg_mask[b] = torch.stack(fg_vals, dim=0).amax(dim=0)
        return center_targets, distance_targets, fg_mask

    def _center_distance_aux_losses(self, batch, aux_outputs, points_mm, points_ijk):
        if not isinstance(aux_outputs, dict):
            return {}
        center_logits = aux_outputs.get("center_logits")
        distance_logits = aux_outputs.get("distance_logits")
        use_center_loss = center_logits is not None and self._center_distance_center_weight > 0.0
        use_distance_loss = distance_logits is not None and self._center_distance_weight > 0.0
        if not use_center_loss and not use_distance_loss:
            return {}
        center_target, distance_target, fg_mask = self._center_distance_query_targets(
            batch, points_mm, points_ijk
        )
        losses = {}
        if use_center_loss:
            center_pred = torch.sigmoid(center_logits)
            # center_loss = F.mse_loss(center_pred, center_target)
            center_loss = self._center_focal_loss(center_logits, center_target)
            losses["center_loss"] = center_loss
            losses["center_target_pos_ratio"] = (center_target > 0.5).float().mean()
            losses["center_target_mean"] = center_target.mean()
            losses["pred_center_mean"] = center_pred.mean()
        if use_distance_loss:
            distance_pred = torch.sigmoid(distance_logits)
            fg = fg_mask > 0.5
            if fg.any():
                distance_loss = F.smooth_l1_loss(distance_pred[fg], distance_target[fg])
                losses["distance_loss"] = distance_loss
                losses["distance_target_mean_fg"] = distance_target[fg].mean()
                losses["pred_distance_mean"] = distance_pred[fg].mean()
            else:
                losses["distance_loss"] = torch.zeros((), device=points_mm.device)
                losses["distance_target_mean_fg"] = torch.zeros((), device=points_mm.device)
                losses["pred_distance_mean"] = torch.zeros((), device=points_mm.device)
        return losses

    def _empty_slot_suppression_loss(self, aux_outputs) -> torch.Tensor | None:
        if self._empty_slot_weight <= 0.0 or not isinstance(aux_outputs, dict):
            return None
        component_logits = aux_outputs.get("source_component_logits")
        component_valid = aux_outputs.get("source_component_valid")
        if component_logits is None or component_valid is None:
            return None
        invalid = ~component_valid.to(device=component_logits.device, dtype=torch.bool)
        invalid = invalid[:, None, :, None].expand_as(component_logits)
        if invalid.any():
            component_logits = torch.nan_to_num(
                component_logits, nan=0.0, posinf=30.0, neginf=-30.0
            ).clamp(-30.0, 30.0)
            return torch.sigmoid(component_logits)[invalid].mean()
        return torch.zeros((), dtype=component_logits.dtype, device=component_logits.device)

    def _log_query_sampling_stats(self, prefix: str, batch, point_densities: torch.Tensor) -> None:
        """Log sampled-query composition for stability audits."""
        pos_ratio = (point_densities > 0.0).float().mean()
        self.log(
            f"{prefix}_pos_ratio",
            pos_ratio,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{prefix}_pos_count",
            (point_densities > 0.0).float().sum().detach(),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        query_src_tag = batch.get("query_src_tag")
        if query_src_tag is None:
            return
        tags = query_src_tag.to(device=point_densities.device)
        valid = tags >= 0
        denom = valid.float().sum().clamp_min(1.0)
        trunk_ratio = ((tags == 0).float().sum() / denom).detach()
        proposal_ratio = ((tags == 1).float().sum() / denom).detach()
        peak_ratio = ((tags == 2).float().sum() / denom).detach()
        self.log(
            f"{prefix}_query_trunk_ratio",
            trunk_ratio,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{prefix}_query_peak_balanced_ratio",
            peak_ratio,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        for tag_value, tag_name in (
            (0, "trunk"),
            (1, "meas_proposal"),
            (2, "peak_balanced"),
        ):
            mask = tags == tag_value
            tag_denom = mask.float().sum().clamp_min(1.0)
            tag_pos_ratio = ((point_densities > 0.0).float()[mask].sum() / tag_denom).detach()
            self.log(
                f"{prefix}_{tag_name}_pos_ratio",
                tag_pos_ratio,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        self.log(
            f"{prefix}_query_proposal_ratio",
            proposal_ratio,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def training_step(self, batch, batch_idx):
        """
        Single training step

        Lightning automatically handles:
        - Device transfer
        - Backward pass
        - Gradient accumulation
        - Optimizer step

        Args:
            batch: Batch from dataloader
            batch_idx: Batch index

        Returns:
            Loss value
        """
        projections, proj_in = self._prepare_projection_input(batch)

        if self._is_voxel_model():
            pred_voxel, aux_outputs = self._call_voxel_model(projections, batch)
            if aux_outputs.get("training_space") == "mesh":
                target_voxel = batch.get("gt_nodes")
                if target_voxel is None:
                    raise ValueError("Mesh-space training requires gt_nodes in the batch")
            else:
                target_voxel = self._voxel_target(batch)
            loss_dict = self.voxel_loss_func(pred_voxel, target_voxel, aux_outputs)
            total_loss = loss_dict["total_loss"]
            self.log(
                "train_loss",
                total_loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            for key, value in loss_dict.items():
                if key != "total_loss" and isinstance(value, torch.Tensor):
                    self.log(f"train_{key}", value, on_step=False, on_epoch=True, sync_dist=True)
            self.loss_func.update_epoch(self.current_epoch)
            return total_loss

        if self._is_ssq_model():
            points_mm = batch.get("query_coordinates_mm", batch.get("points_mm"))
            density = batch["point_densities"].unsqueeze(-1)
            self._log_query_sampling_stats("train", batch, batch["point_densities"])
            out = self._call_ssq_model(batch, return_diagnostics=False)
            density_pred = out["density"]
            aux_outputs = out.get("aux_outputs", {})
            if isinstance(aux_outputs, dict):
                pi = aux_outputs.get("pi")
                if torch.is_tensor(pi) and pi.numel() > 0:
                    self.log(
                        "train_ssq_pi0_mean",
                        pi[..., 0].mean(),
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                    )
                    if pi.shape[-1] > 1:
                        self.log(
                            "train_ssq_candidate_pi_mean",
                            pi[..., 1:].sum(dim=-1).mean(),
                            on_step=False,
                            on_epoch=True,
                            sync_dist=True,
                        )
                self.log(
                    "train_ssq_density_mean",
                    density_pred.detach().mean(),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
            loss_dict = self.ssq_loss_func(
                density_pred,
                density,
                aux_outputs,
                gt_voxels=batch.get("gt_voxels"),
                points_ijk=batch.get("points_ijk"),
                sdf_targets=batch.get("sdf_targets"),
                query_component_ids=batch.get("query_component_ids"),
                gt_component_centers_mm=batch.get("gt_component_centers_mm"),
                gt_component_valid_mask=batch.get("gt_component_valid_mask"),
            )
            density_logits = (
                aux_outputs.get("density_logits") if isinstance(aux_outputs, dict) else None
            )
            if torch.is_tensor(density_logits) and self._backbone_logit_loss_weight > 0.0:
                backbone_logit_loss = self.loss_func.sparse_light_loss(density_logits, density)
                loss_dict["backbone_logit_loss"] = backbone_logit_loss
                loss_dict["total_loss"] = (
                    loss_dict["total_loss"] + self._backbone_logit_loss_weight * backbone_logit_loss
                )
            total_loss = loss_dict["total_loss"]
            if not torch.isfinite(total_loss.detach()):
                if getattr(self.net, "composition_mode", None) == "view_complementary":
                    self._dump_view_nonfinite(batch, aux_outputs, loss_dict)
                    raise FloatingPointError(
                        "view_complementary produced a nonfinite training loss; "
                        "diagnostics were dumped before optimizer.step"
                    )
                self.log(
                    "train_nonfinite_loss_skip",
                    torch.ones((), dtype=total_loss.dtype, device=total_loss.device),
                    prog_bar=False,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                )
                total_loss = (
                    torch.nan_to_num(density_pred, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
                )
                loss_dict["total_loss"] = total_loss
            else:
                self.log(
                    "train_nonfinite_loss_skip",
                    torch.zeros((), dtype=total_loss.dtype, device=total_loss.device),
                    prog_bar=False,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                )
            self.log(
                "train_loss",
                total_loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            for key, value in loss_dict.items():
                if key != "total_loss" and isinstance(value, torch.Tensor):
                    self.log(f"train_{key}", value, on_step=False, on_epoch=True, sync_dist=True)
            if points_mm is not None:
                self.log(
                    "train_query_mm_abs_mean",
                    points_mm.detach().abs().mean(),
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
            return total_loss

        aux_only = float(self.cfg.loss.get("light_weight", 1.0)) == 0.0
        points = batch["points"]
        points_mm = batch.get("points_mm")
        depth_maps = batch.get("depth_maps")
        source_hypotheses = self._source_hypotheses_from_batch(batch)
        density = batch["point_densities"].unsqueeze(-1)
        if not aux_only:
            self._log_query_sampling_stats("train", batch, batch["point_densities"])

        # Forward pass (use packed tensor when available)
        density_pred, aux_outputs = self._call_model(
            proj_in,
            points,
            points_mm=points_mm,
            depth_maps=depth_maps,
            source_hypotheses=source_hypotheses,
            aux_only=aux_only,
        )
        self._log_source_cue_stats("train", batch, batch["point_densities"])

        # Compute loss
        descatter_targets = batch.get("descatter_targets")
        loss_dict = self.loss_func(aux_outputs, descatter_targets, density_pred, density)
        center_distance_losses = self._center_distance_aux_losses(
            batch, aux_outputs, points_mm, batch["points_ijk"]
        )
        if center_distance_losses:
            total_center_distance = torch.zeros(
                (), dtype=density_pred.dtype, device=density_pred.device
            )
            if "center_loss" in center_distance_losses:
                total_center_distance = (
                    total_center_distance
                    + self._center_distance_center_weight * center_distance_losses["center_loss"]
                )
            if "distance_loss" in center_distance_losses:
                total_center_distance = (
                    total_center_distance
                    + self._center_distance_weight * center_distance_losses["distance_loss"]
                )
            loss_dict.update(center_distance_losses)
            loss_dict["center_distance_aux_loss"] = total_center_distance
            loss_dict["total_loss"] = loss_dict["total_loss"] + total_center_distance
        empty_slot_loss = self._empty_slot_suppression_loss(aux_outputs)
        if empty_slot_loss is not None:
            loss_dict["empty_slot_loss"] = empty_slot_loss
            loss_dict["total_loss"] = (
                loss_dict["total_loss"] + self._empty_slot_weight * empty_slot_loss
            )
        total_loss = loss_dict["total_loss"]
        anchor_loss = getattr(self.net, "last_feature_refinement_anchor_loss", None)
        if isinstance(anchor_loss, torch.Tensor):
            total_loss = total_loss + anchor_loss
            loss_dict["total_loss"] = total_loss
            loss_dict["feature_refinement_anchor_loss"] = anchor_loss.detach()

        nonfinite_loss = ~torch.isfinite(total_loss.detach())
        if bool(nonfinite_loss.item()):
            self.log(
                "train_nonfinite_loss_skip",
                torch.ones((), dtype=total_loss.dtype, device=total_loss.device),
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            total_loss = torch.nan_to_num(density_pred, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
            loss_dict["total_loss"] = total_loss
        else:
            self.log(
                "train_nonfinite_loss_skip",
                torch.zeros((), dtype=total_loss.dtype, device=total_loss.device),
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

        # Log metrics
        self.log(
            "train_loss",
            total_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        for key, value in loss_dict.items():
            if key != "total_loss" and isinstance(value, torch.Tensor):
                self.log(
                    f"train_{key}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

        # Update loss function's epoch for dynamic weight adjustment
        self.loss_func.update_epoch(self.current_epoch)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Single validation step

        Lightning automatically handles:
        - no_grad context
        - Device transfer

        Args:
            batch: Batch from dataloader
            batch_idx: Batch index

        Returns:
            Dictionary with metrics
        """
        _projections, proj_in = self._prepare_projection_input(batch)

        if self._is_voxel_model():
            projections = batch["projections"]
            target_voxel = self._voxel_target(batch)
            pred_voxel, _ = self._call_voxel_model(projections, batch)
            pred_voxel, target_voxel = self._standardize_pred_voxel(pred_voxel, target_voxel)
            dice = dice_coefficient(
                torch.sigmoid(pred_voxel),
                (target_voxel > 0.0).float(),
            )
            self.log(
                "val_full_dice",
                dice,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "val_dice",
                dice,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            return {"dice": dice}

        if self._is_ssq_model():
            point_densities = batch["point_densities"]
            voxel_shape = batch["feasible_voxel_shape"]
            self._log_query_sampling_stats("val", batch, point_densities)
            B = point_densities.shape[0]
            voxel_shape_tuple = self._shape_tuple_from_batch(voxel_shape, B)
            out = self._call_ssq_model(batch, return_diagnostics=False)
            pred_prob = out["density"].clamp(0.0, 1.0)
            diagnostics = out.get("diagnostics", out.get("aux_outputs", {}))
            validation_loss = self.ssq_loss_func(
                pred_prob,
                point_densities.unsqueeze(-1),
                diagnostics,
                gt_voxels=batch.get("gt_voxels"),
                points_ijk=batch.get("points_ijk"),
                query_component_ids=batch.get("query_component_ids"),
                gt_component_centers_mm=batch.get("gt_component_centers_mm"),
                gt_component_valid_mask=batch.get("gt_component_valid_mask"),
            )
            self.log(
                "val_formal_density_loss",
                validation_loss["total_loss"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            if torch.is_tensor(diagnostics.get("candidate_centers_mm")):
                self._log_view_candidate_metrics(diagnostics, batch)
            full_grid = int(np.prod(voxel_shape_tuple[1:])) == point_densities.shape[1]
            if full_grid:
                density_gt = point_densities.reshape(voxel_shape_tuple)
                density_gt_bin = (density_gt > 0.0).to(dtype=torch.float32)
                density_pred = pred_prob.reshape(voxel_shape_tuple)
                dice = dice_coefficient(
                    density_pred,
                    density_gt_bin,
                    threshold=self._validation_pred_threshold,
                )
                metric_name = "val_full_dice"
            else:
                pred_bin = (pred_prob.squeeze(-1) >= self._validation_pred_threshold).float()
                gt_bin = (point_densities > 0.0).float()
                intersection = (pred_bin * gt_bin).sum(dim=1)
                dice = (
                    (2.0 * intersection + 1e-8) / (pred_bin.sum(dim=1) + gt_bin.sum(dim=1) + 1e-8)
                ).mean()
                metric_name = "val_query_dice"
            if "shared_density" in diagnostics:
                shared_prob = diagnostics["shared_density"].clamp(0.0, 1.0)
                if full_grid:
                    shared_dice = dice_coefficient(
                        shared_prob.reshape(voxel_shape_tuple),
                        density_gt_bin,
                        threshold=self._validation_pred_threshold,
                    )
                else:
                    shared_bin = (
                        shared_prob.squeeze(-1) >= self._validation_pred_threshold
                    ).float()
                    shared_intersection = (shared_bin * gt_bin).sum(dim=1)
                    shared_dice = (
                        (2.0 * shared_intersection + 1e-8)
                        / (shared_bin.sum(dim=1) + gt_bin.sum(dim=1) + 1e-8)
                    ).mean()
                self.log(
                    "val_shared_dice", shared_dice, on_step=False, on_epoch=True, sync_dist=True
                )
                self.log(
                    "val_final_minus_shared_dice",
                    dice - shared_dice,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
            if "proposal_gate" in diagnostics:
                gate = diagnostics["proposal_gate"].detach().float().reshape(-1)
                applicability = diagnostics["candidate_applicability"].detach().float()
                dispersion = diagnostics["quotient_dispersion"].detach().float()
                residual = diagnostics["residual_correction"].detach().float()
                diagnostic_scalars = {
                    "val_proposal_gate_mean": gate.mean(),
                    "val_proposal_gate_p10": torch.quantile(gate, 0.1),
                    "val_proposal_gate_p90": torch.quantile(gate, 0.9),
                    "val_candidate_applicability_mean": applicability.mean()
                    if applicability.numel()
                    else gate.new_zeros(()),
                    "val_quotient_dispersion_mean": dispersion.mean()
                    if dispersion.numel()
                    else gate.new_zeros(()),
                    "val_residual_correction_abs_mean": residual.abs().mean(),
                }
                for name, value in diagnostic_scalars.items():
                    self.log(name, value, on_step=False, on_epoch=True, sync_dist=True)
            self.log(
                metric_name,
                dice,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "val_dice",
                dice,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "val_pred_threshold",
                torch.tensor(
                    self._validation_pred_threshold,
                    dtype=dice.dtype,
                    device=dice.device,
                ),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            return {"dice": dice}

        points = batch["points"]
        points_mm = batch.get("points_mm")
        depth_maps = batch.get("depth_maps")
        source_hypotheses = self._source_hypotheses_from_batch(batch)
        point_densities = batch["point_densities"]
        voxel_shape = batch["feasible_voxel_shape"]
        self._log_query_sampling_stats("val", batch, point_densities)

        B = points.shape[0]

        # Reconstruct voxel shape
        voxel_shape_tuple = self._shape_tuple_from_batch(voxel_shape, B)

        # Inference
        pred, aux_outputs = self._call_model(
            proj_in,
            points,
            points_mm=points_mm,
            depth_maps=depth_maps,
            source_hypotheses=source_hypotheses,
        )
        self._log_source_cue_stats("val", batch, point_densities)
        empty_slot_loss = self._empty_slot_suppression_loss(aux_outputs)
        if empty_slot_loss is not None:
            self.log(
                "val_empty_slot_loss",
                empty_slot_loss.detach(),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        full_grid = int(np.prod(voxel_shape_tuple[1:])) == point_densities.shape[1]
        pred_prob = torch.sigmoid(pred)
        if full_grid:
            density_gt = point_densities.reshape(voxel_shape_tuple)
            density_gt_bin = (density_gt > 0.0).to(dtype=torch.float32)
            density_pred = pred_prob.reshape(voxel_shape_tuple)
            dice = dice_coefficient(
                density_pred,
                density_gt_bin,
                threshold=self._validation_pred_threshold,
            )
            metric_name = "val_full_dice"
        else:
            pred_bin = (pred_prob.squeeze(-1) >= self._validation_pred_threshold).float()
            gt_bin = (point_densities > 0.0).float()
            intersection = (pred_bin * gt_bin).sum(dim=1)
            dice = (
                (2.0 * intersection + 1e-8) / (pred_bin.sum(dim=1) + gt_bin.sum(dim=1) + 1e-8)
            ).mean()
            metric_name = "val_query_dice"

        # FMT-SimGen regular validation uses sampled queries; val_dice is kept as a
        # checkpoint-compatible alias and is usually equivalent to val_query_dice.
        self.log(
            metric_name,
            dice,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val_dice",
            dice,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val_pred_threshold",
            torch.tensor(
                self._validation_pred_threshold,
                dtype=dice.dtype,
                device=dice.device,
            ),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return {"dice": dice}

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch"""
        # Track best validation metric
        avg_dice = self.trainer.callback_metrics.get("val_dice", torch.tensor(-1.0))
        if isinstance(avg_dice, torch.Tensor):
            avg_dice = avg_dice.item()

        if avg_dice > self.best_dice:
            self.best_dice = avg_dice

        # Restore training mode after evaluation to keep behavior consistent
        self.net.train()

    @staticmethod
    def _module_gradient_norm(module: torch.nn.Module) -> torch.Tensor:
        gradients = [
            parameter.grad.detach()
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            return next(module.parameters()).new_zeros(())
        per_tensor = torch._foreach_norm(gradients, 2.0)
        return torch.stack([value.float() for value in per_tensor]).norm(2.0)

    def _dump_view_nonfinite(self, batch: dict, diagnostics: dict, losses: dict) -> None:
        output_dir = Path(str(self.cfg.paths.output_dir)) / "nonfinite_diagnostics"
        output_dir.mkdir(parents=True, exist_ok=True)
        keys = (
            "per_view_evidence",
            "proposal_offsets_mm",
            "proposal_points_mm",
            "proposal_compatibility",
            "proposal_assignment",
            "candidate_centers_mm",
            "candidate_covariance_eigenvalues",
            "candidate_view_support",
            "candidate_detector_scales",
            "separability",
            "candidate_context",
            "decoder_pre_activation",
        )
        payload = {
            "global_step": int(self.global_step),
            "sample_id": batch.get("sample_id"),
            "diagnostics": {
                key: diagnostics[key].detach().float().cpu()
                for key in keys
                if torch.is_tensor(diagnostics.get(key))
            },
            "losses": {
                key: value.detach().float().cpu()
                for key, value in losses.items()
                if torch.is_tensor(value)
            },
        }
        torch.save(payload, output_dir / f"step_{int(self.global_step):08d}.pt")

    def _log_view_candidate_metrics(self, diagnostics: dict, batch: dict) -> None:
        gt_centers = batch.get("gt_component_centers_mm")
        gt_valid = batch.get("gt_component_valid_mask")
        if not torch.is_tensor(gt_centers) or not torch.is_tensor(gt_valid):
            return
        centers = diagnostics["candidate_centers_mm"].detach().float()
        valid = (
            diagnostics.get("candidate_analysis_valid_mask", diagnostics["candidate_valid_mask"])
            .detach()
            .bool()
        )
        scores: dict[str, list[torch.Tensor]] = {
            "coverage_6mm": [],
            "coverage_8mm": [],
            "coverage_10mm": [],
            "duplicate_rate": [],
            "unmatched_rate": [],
            "matched_center_error_mm": [],
        }
        for sample in range(centers.shape[0]):
            predicted = centers[sample, valid[sample]]
            target = gt_centers[sample, gt_valid[sample]].float()
            if predicted.numel() == 0 or target.numel() == 0:
                zero = centers.new_zeros(())
                for radius in (6, 8, 10):
                    scores[f"coverage_{radius}mm"].append(zero)
                scores["duplicate_rate"].append(zero)
                scores["unmatched_rate"].append(centers.new_ones(()))
                scores["matched_center_error_mm"].append(centers.new_tensor(float("nan")))
                continue
            distance = torch.cdist(predicted, target)
            nearest_target = distance.amin(dim=0)
            nearest_candidate = distance.amin(dim=1)
            for radius in (6, 8, 10):
                scores[f"coverage_{radius}mm"].append((nearest_target <= radius).float().mean())
            assigned = distance.argmin(dim=1)
            duplicate = torch.zeros(len(predicted), dtype=torch.bool, device=centers.device)
            for target_index in range(len(target)):
                members = torch.where((assigned == target_index) & (nearest_candidate <= 8.0))[0]
                if len(members) > 1:
                    duplicate[members] = True
                    best = members[distance[members, target_index].argmin()]
                    duplicate[best] = False
            scores["duplicate_rate"].append(duplicate.float().mean())
            scores["unmatched_rate"].append((nearest_candidate > 8.0).float().mean())
            matched = nearest_candidate <= 8.0
            scores["matched_center_error_mm"].append(
                nearest_candidate[matched].mean()
                if matched.any()
                else centers.new_tensor(float("nan"))
            )
        for name, values in scores.items():
            value = torch.stack(values)
            finite = value[torch.isfinite(value)]
            self.log(
                f"val_candidate_{name}",
                finite.mean() if finite.numel() else centers.new_zeros(()),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=centers.shape[0],
            )
        count = valid.sum(dim=-1)
        self.log(
            "val_candidate_count_mean",
            count.float().mean(),
            on_epoch=True,
            sync_dist=True,
            batch_size=centers.shape[0],
        )
        for number in range(1, centers.shape[1] + 1):
            self.log(
                f"val_candidate_count_{number}_ratio",
                (count == number).float().mean(),
                on_epoch=True,
                sync_dist=True,
                batch_size=centers.shape[0],
            )
        support = diagnostics["candidate_view_support"].detach()
        support_count = ((support > 1.0e-4) & valid[:, :, None]).sum(dim=-1).float()
        self.log(
            "val_candidate_support_view_count",
            support_count[valid].mean() if valid.any() else centers.new_zeros(()),
            on_epoch=True,
            sync_dist=True,
            batch_size=centers.shape[0],
        )
        if torch.is_tensor(diagnostics.get("routing_residual")):
            residual = diagnostics["routing_residual"].detach().float().abs().flatten()
            gate = diagnostics["hypothesis_gate"].detach().float().flatten()
            self.log(
                "val_routing_residual_abs_mean", residual.mean(), on_epoch=True, sync_dist=True
            )
            self.log(
                "val_routing_residual_abs_p90",
                torch.quantile(residual, 0.9),
                on_epoch=True,
                sync_dist=True,
            )
            self.log("val_hypothesis_gate_mean", gate.mean(), on_epoch=True, sync_dist=True)
            for q in (0.1, 0.5, 0.9):
                self.log(
                    f"val_hypothesis_gate_p{int(q * 100)}",
                    torch.quantile(gate, q),
                    on_epoch=True,
                    sync_dist=True,
                )
            self.log(
                "val_routing_gain",
                self.net.bounded_view_routing.routing_gain_raw.detach(),
                on_epoch=True,
                sync_dist=True,
            )
        eigen = (
            diagnostics["candidate_covariance_eigenvalues"]
            .detach()
            .float()[valid[..., None].expand_as(diagnostics["candidate_covariance_eigenvalues"])]
        )
        if eigen.numel():
            for quantile in (0.1, 0.5, 0.9):
                self.log(
                    f"val_covariance_eigenvalue_p{int(quantile * 100)}",
                    torch.quantile(eigen, quantile),
                    on_epoch=True,
                    sync_dist=True,
                )
        for name in ("lower", "upper"):
            hit = diagnostics[f"covariance_{name}_bound_hit"].detach()
            self.log(
                f"val_covariance_{name}_bound_hit_ratio",
                hit[valid[..., None].expand_as(hit)].float().mean()
                if valid.any()
                else centers.new_zeros(()),
                on_epoch=True,
                sync_dist=True,
            )

    def on_after_backward(self) -> None:
        if (
            self._is_ssq_model()
            and getattr(self.net, "composition_mode", None) == "view_complementary"
        ):
            modules = {
                "2d_encoder": self.net.surface_encoder,
                "view_evidence_head": self.net.view_candidate_evidence.evidence,
                "offset_head": self.net.view_candidate_evidence.offset,
                "descriptor_head": self.net.view_candidate_evidence.descriptor,
                "cross_view_association": self.net.diverse_candidate_constructor,
                "candidate_view_encoder": self.net.complementary_aggregation.candidate_projection,
                "separability_correction": self.net.view_separability.correction,
                "candidate_context": self.net.unified_density_decoder.candidate_context,
                "shared_decoder": self.net.unified_density_decoder.head,
            }
            norms = {name: self._module_gradient_norm(module) for name, module in modules.items()}
            for name, value in norms.items():
                self.log(
                    f"train_grad_norm_{name}", value, on_step=False, on_epoch=True, sync_dist=True
                )
            candidate_norm = torch.stack(
                [
                    value
                    for name, value in norms.items()
                    if name not in {"2d_encoder", "shared_decoder"}
                ]
            ).norm()
            self.log(
                "train_candidate_to_shared_gradient_ratio",
                candidate_norm / norms["shared_decoder"].clamp_min(1.0e-12),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            nonfinite_gradients = [
                name
                for name, module in modules.items()
                if any(
                    parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                    for parameter in module.parameters()
                )
            ]
            if nonfinite_gradients:
                output_dir = Path(str(self.cfg.paths.output_dir)) / "nonfinite_diagnostics"
                output_dir.mkdir(parents=True, exist_ok=True)
                gradient_stats = {}
                for name, module in modules.items():
                    gradients = [
                        parameter.grad.detach().float()
                        for parameter in module.parameters()
                        if parameter.grad is not None
                    ]
                    total = sum(gradient.numel() for gradient in gradients)
                    finite = sum(
                        int(torch.isfinite(gradient).sum().item()) for gradient in gradients
                    )
                    gradient_stats[name] = {
                        "norm": float(norms[name].detach().cpu()),
                        "finite_ratio": finite / max(total, 1),
                    }
                torch.save(
                    gradient_stats,
                    output_dir / f"step_{int(self.global_step):08d}_gradients.pt",
                )
                raise FloatingPointError(
                    "view_complementary produced nonfinite gradients before optimizer.step: "
                    + ", ".join(nonfinite_gradients)
                )
            return
        if (
            not self._is_ssq_model()
            or getattr(self.net, "composition_mode", None) != "quotient_residual"
        ):
            return
        step = int(self.global_step)
        diagnostics_cfg = self.cfg.get("diagnostics", {}) or {}
        log_interval = max(int(diagnostics_cfg.get("gradient_norm_every_n_steps", 100)), 1)
        if step == 0 or step == self._last_gradient_norm_step or step % log_interval != 0:
            return
        self._last_gradient_norm_step = step
        modules = {
            "surface_encoder": self.net.surface_encoder,
            "shared_decoder": self.net.shared_density_logit_decoder,
            "quotient_reliability": self.net.quotient_aggregator.reliability_residual,
            "residual_decoder": self.net.source_hypothesis_residual_decoder,
        }
        for name, module in modules.items():
            self.log(
                f"train_grad_norm_{name}",
                self._module_gradient_norm(module),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler

        Returns:
            Dictionary with optimizer and scheduler config
        """
        # Get optimizer config
        optim_cfg = self.cfg.optim

        view_phase = getattr(self.net, "view_training_phase", None)
        if (
            self._is_ssq_model()
            and getattr(self.net, "composition_mode", None) == "view_complementary"
            and view_phase in {"phase_b", "full"}
        ):
            lr_cfg = self.cfg.model.ssq_fmt.view_complementary.lr
            grouped_modules = {
                "encoder": [self.net.surface_encoder, self.net.surface_sampler],
                "constructor": [
                    self.net.view_candidate_evidence,
                    self.net.diverse_candidate_constructor,
                ],
                "candidate_encoder": [
                    self.net.complementary_aggregation,
                ],
                "candidate_context": [
                    self.net.unified_density_decoder.candidate_context,
                ],
                "decoder": [
                    self.net.unified_density_decoder.candidate_input,
                    self.net.unified_density_decoder.head,
                ],
                "separability": [self.net.view_separability],
                "routing": [self.net.bounded_view_routing]
                if self.net.bounded_view_routing is not None
                else [],
            }
            optimizer_params = []
            used: set[int] = set()
            for name, modules in grouped_modules.items():
                parameters = [
                    parameter
                    for module in modules
                    for parameter in module.parameters()
                    if parameter.requires_grad and id(parameter) not in used
                ]
                used.update(id(parameter) for parameter in parameters)
                if parameters:
                    optimizer_params.append(
                        {
                            "params": parameters,
                            "lr": float(lr_cfg[name]),
                            "name": name,
                        }
                    )
            remaining = [
                parameter
                for parameter in self.parameters()
                if parameter.requires_grad and id(parameter) not in used
            ]
            if remaining:
                optimizer_params.append(
                    {
                        "params": remaining,
                        "lr": float(lr_cfg.get("shared", 0.0)),
                        "name": "shared_decoder",
                    }
                )
        else:
            optimizer_params = None

        # Mean-prior residual view correction is intentionally conservative; train its
        # gate with a smaller LR when configured, without changing older experiments.
        gate_lr_mult = float(getattr(self.net, "mean_prior_gate_lr_mult", 1.0))
        gate_params = []
        gate_param_ids = set()
        if gate_lr_mult != 1.0 and hasattr(self.net, "mean_prior_residual_gate"):
            for param in self.net.mean_prior_residual_gate.parameters():
                if param.requires_grad:
                    gate_params.append(param)
                    gate_param_ids.add(id(param))
        base_params = [
            param
            for param in self.parameters()
            if param.requires_grad and id(param) not in gate_param_ids
        ]
        if optimizer_params is not None:
            pass
        elif gate_params:
            optimizer_params = []
            if base_params:
                optimizer_params.append({"params": base_params})
            optimizer_params.append({"params": gate_params, "lr": optim_cfg.lr * gate_lr_mult})
        else:
            optimizer_params = [param for param in self.parameters() if param.requires_grad]
        if not optimizer_params:
            raise ValueError("No trainable parameters available for optimizer")

        # Create optimizer based on _target_
        if "adamw" in optim_cfg._target_.lower():
            optimizer = torch.optim.AdamW(
                optimizer_params,
                lr=optim_cfg.lr,
                betas=optim_cfg.betas,
                weight_decay=optim_cfg.weight_decay,
                eps=optim_cfg.eps,
            )
        elif "adam" in optim_cfg._target_.lower():
            optimizer = torch.optim.Adam(
                optimizer_params,
                lr=optim_cfg.lr,
                betas=optim_cfg.betas,
                weight_decay=optim_cfg.weight_decay,
                eps=optim_cfg.eps,
            )
        elif "sgd" in optim_cfg._target_.lower():
            optimizer = torch.optim.SGD(
                optimizer_params,
                lr=optim_cfg.lr,
                momentum=optim_cfg.momentum,
                weight_decay=optim_cfg.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optim_cfg._target_}")

        # Configure learning rate scheduler
        config_dict = {"optimizer": optimizer}

        if "scheduler" in optim_cfg and optim_cfg.scheduler is not None:
            scheduler_cfg = optim_cfg.scheduler.copy()
            scheduler_class_name = scheduler_cfg._target_.split(".")[-1]

            # Map scheduler target to class
            warmup_epochs = int(scheduler_cfg.get("warmup_epochs", 0))
            warmup_start_factor = float(scheduler_cfg.get("warmup_start_factor", 0.01))
            if "StepLR" in scheduler_class_name:
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=scheduler_cfg.step_size,
                    gamma=scheduler_cfg.gamma,
                )
            elif "CosineAnnealingLR" in scheduler_class_name:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, self.max_epochs - warmup_epochs),
                    eta_min=scheduler_cfg.eta_min,
                )
            elif "ExponentialLR" in scheduler_class_name:
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer,
                    gamma=scheduler_cfg.gamma,
                )
            else:
                scheduler = None

            if scheduler is not None:
                if warmup_epochs > 0:
                    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=warmup_start_factor,
                        end_factor=1.0,
                        total_iters=warmup_epochs,
                    )
                    scheduler = torch.optim.lr_scheduler.SequentialLR(
                        optimizer,
                        schedulers=[warmup_scheduler, scheduler],
                        milestones=[warmup_epochs],
                    )
                config_dict["lr_scheduler"] = {
                    "scheduler": scheduler,
                    "interval": scheduler_cfg.interval,
                    "frequency": scheduler_cfg.frequency,
                }

        return config_dict

    def on_train_epoch_start(self):
        """Called at start of training epoch"""
        trainer = getattr(self, "trainer", None)
        datamodule = getattr(trainer, "datamodule", None) if trainer is not None else None
        if datamodule is not None and hasattr(datamodule, "set_epoch"):
            datamodule.set_epoch(int(self.current_epoch))
            train_dataset = getattr(datamodule, "train_dataset", None)
            resample_enabled = bool(
                getattr(train_dataset, "resample_queries_each_epoch", False)
                and getattr(train_dataset, "is_training", False)
            )
            self.log(
                "train_query_resample_enabled",
                float(resample_enabled),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train_query_epoch",
                float(self.current_epoch),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        if torch.cuda.is_available() and bool(
            getattr(getattr(self.cfg, "trainer", {}), "empty_cache_each_epoch", False)
        ):
            gc.collect()
            torch.cuda.empty_cache()

    def on_validation_epoch_start(self):
        """Called at start of validation epoch"""
        self.net.eval()

    def on_test_start(self):
        """Initialize test-time outputs (metrics + saving recon/figures)."""
        self.net.eval()

        test_cfg = getattr(self.cfg, "test", None)
        save_dir = (
            str(test_cfg.save_dir)
            if test_cfg is not None and "save_dir" in test_cfg
            else os.path.join(self.cfg.paths.output_dir, "test")
        )
        self._test_pred_threshold = (
            float(test_cfg.pred_threshold)
            if test_cfg is not None and "pred_threshold" in test_cfg
            else 0.5
        )
        self._test_max_samples_per_angle = (
            int(test_cfg.max_samples_per_angle)
            if test_cfg is not None and "max_samples_per_angle" in test_cfg
            else 10
        )
        self._test_save_recon_roi = (
            bool(test_cfg.save_recon_roi)
            if test_cfg is not None and "save_recon_roi" in test_cfg
            else True
        )
        self._test_save_registered_seg = (
            bool(test_cfg.save_registered_seg)
            if test_cfg is not None and "save_registered_seg" in test_cfg
            else True
        )
        self._test_save_proj_comparisons = (
            bool(test_cfg.save_proj_comparisons)
            if test_cfg is not None and "save_proj_comparisons" in test_cfg
            else True
        )

        # Segmentation metrics options
        self._test_min_region_size = (
            int(getattr(test_cfg, "min_region_size"))
            if test_cfg is not None and "min_region_size" in test_cfg
            else 10
        )
        self._test_voxel_spacing = (
            tuple(float(x) for x in getattr(test_cfg, "voxel_spacing"))
            if test_cfg is not None and "voxel_spacing" in test_cfg
            else (1.0, 1.0, 1.0)
        )

        # Connected components options for region counting.
        raw_conn = (
            int(getattr(test_cfg, "cc_connectivity"))
            if test_cfg is not None and "cc_connectivity" in test_cfg
            else 6
        )
        if raw_conn in (1, 2, 3):
            # ndimage.generate_binary_structure connectivity parameter
            self._test_cc_connectivity = int(raw_conn)
        elif raw_conn == 6:
            self._test_cc_connectivity = 1
        elif raw_conn == 18:
            self._test_cc_connectivity = 2
        elif raw_conn == 26:
            self._test_cc_connectivity = 3
        else:
            self._test_cc_connectivity = 1

        self._test_cc_dilation_iters = (
            int(getattr(test_cfg, "cc_dilation_iters"))
            if test_cfg is not None and "cc_dilation_iters" in test_cfg
            else 0
        )

        # Per-sample metrics written on rank 0 (initialized each test run)
        self._test_sample_metrics = []
        # Recon ROI storage stats (rank 0)
        self._test_recon_file_sizes = {}

        view_angles = [str(v) for v in getattr(self.cfg.data, "view_angles", [])]
        if len(view_angles) > 0:
            mid_idx = len(view_angles) // 2
            self._test_angles_to_save = list(
                dict.fromkeys([view_angles[0], view_angles[mid_idx], view_angles[-1]])
            )
            self._test_angle_saved_counts = {a: 0 for a in self._test_angles_to_save}

        self._test_out_dir = Path(save_dir)
        self._test_recon_dir = self._test_out_dir / "recon_roi"
        self._test_seg_dir = self._test_out_dir / "seg_registered"
        self._test_proj_dir = self._test_out_dir / "proj_compare"

        if self.trainer is not None and self.trainer.is_global_zero:
            self._test_out_dir.mkdir(parents=True, exist_ok=True)
            if self._test_save_recon_roi:
                self._test_recon_dir.mkdir(parents=True, exist_ok=True)
            if self._test_save_registered_seg:
                self._test_seg_dir.mkdir(parents=True, exist_ok=True)
            if self._test_save_proj_comparisons:
                self._test_proj_dir.mkdir(parents=True, exist_ok=True)

            # Load base segmentation (vox_file) once, best-effort.
            data_dir = self.cfg.data.test_dir or self.cfg.data.val_dir or self.cfg.data.train_dir
            vox_path = (
                os.path.join(str(data_dir), str(self.cfg.data.voxel_file)) if data_dir else None
            )
            base_seg_error = None
            try:
                if vox_path is not None and os.path.exists(vox_path):
                    self._test_base_seg = np.load(vox_path)
                    self._test_new_label = int(np.max(self._test_base_seg)) + 1
            except Exception as e:
                base_seg_error = f"{type(e).__name__}: {e}"
                self._test_base_seg = None
                self._test_new_label = None

            # Always write meta so missing outputs are debuggable
            try:
                (self._test_out_dir / "meta.json").write_text(
                    json.dumps(
                        {
                            "vox_path": vox_path,
                            "vox_exists": bool(vox_path and os.path.exists(vox_path)),
                            "base_seg_loaded": self._test_base_seg is not None,
                            "base_seg_error": base_seg_error,
                            "new_label": self._test_new_label,
                            "pred_threshold": self._test_pred_threshold,
                            "angles_saved": self._test_angles_to_save,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            except Exception:
                pass

    def _as_pair(self, v):
        if torch.is_tensor(v):
            v = v.detach().cpu().tolist()
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return int(v[0]), int(v[1])
        raise ValueError(f"Expected a pair, got: {type(v)} {v}")

    def _label_components(self, mask: np.ndarray) -> tuple[np.ndarray, int]:
        """Label connected components with optional test-time dilation/connectivity."""
        from scipy import ndimage

        mask = mask.astype(bool)
        conn = int(getattr(self, "_test_cc_connectivity", 1))
        conn = max(1, min(int(conn), mask.ndim))
        structure = ndimage.generate_binary_structure(mask.ndim, conn)

        dilate = int(getattr(self, "_test_cc_dilation_iters", 0))
        if dilate > 0:
            # Use a full neighborhood to bridge small gaps before CC labeling.
            dil_struct = np.ones((3,) * mask.ndim, dtype=bool)
            mask = ndimage.binary_dilation(mask, structure=dil_struct, iterations=int(dilate))

        labeled, num = ndimage.label(mask, structure=structure)
        return labeled, int(num)

    def _filter_small_components(self, mask: np.ndarray, min_size: int) -> tuple[np.ndarray, int]:
        """Return (filtered_mask, num_components_kept).

        Connected-components are computed on an optional dilated mask (test.cc_dilation_iters)
        with configurable connectivity (test.cc_connectivity), but the returned mask itself is
        not dilated.
        """
        labeled, num = self._label_components(mask)

        if min_size <= 1:
            return mask.astype(bool), int(num)

        if num == 0:
            return mask.astype(bool), 0

        from scipy import ndimage

        sizes = ndimage.sum(mask.astype(np.uint8), labeled, index=list(range(1, num + 1)))
        keep = np.asarray(sizes) >= int(min_size)
        if not np.any(keep):
            return np.zeros_like(mask, dtype=bool), 0

        keep_ids = np.nonzero(keep)[0] + 1
        filtered = np.isin(labeled, keep_ids) & mask.astype(bool)
        return filtered, int(len(keep_ids))

    def _assd_hd95(
        self, pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float]
    ) -> tuple[float, float]:
        """Compute ASSD + HD95 between two binary 3D masks."""
        from scipy import ndimage

        pred = pred.astype(bool)
        gt = gt.astype(bool)

        if not pred.any() and not gt.any():
            return 0.0, 0.0
        if not pred.any() or not gt.any():
            # Undefined for surface distance; caller should ignore.
            return float("nan"), float("nan")

        # Surface voxels
        struct = np.ones((3, 3, 3), dtype=bool)
        pred_s = pred ^ ndimage.binary_erosion(pred, structure=struct)
        gt_s = gt ^ ndimage.binary_erosion(gt, structure=struct)

        # Distances to the other surface
        dt_gt = ndimage.distance_transform_edt(~gt_s, sampling=spacing)
        dt_pred = ndimage.distance_transform_edt(~pred_s, sampling=spacing)

        d_pred_to_gt = dt_gt[pred_s]
        d_gt_to_pred = dt_pred[gt_s]
        d = np.concatenate([d_pred_to_gt, d_gt_to_pred], axis=0)

        assd = float(np.mean(d))
        hd95 = float(np.percentile(d, 95))
        return assd, hd95

    def _volume_metrics(self, pred, gt, gt_bin, threshold: float) -> dict[str, torch.Tensor]:
        pred_bin = (pred >= threshold).float()
        intersection = (pred_bin * gt_bin).sum(dim=(1, 2, 3))
        union = ((pred_bin + gt_bin) > 0).float().sum(dim=(1, 2, 3))
        iou = ((intersection + 1e-8) / (union + 1e-8)).mean()

        nrmse = (
            torch.sqrt(torch.mean((pred - gt).square(), dim=(1, 2, 3)))
            / (gt.amax(dim=(1, 2, 3)) - gt.amin(dim=(1, 2, 3))).clamp_min(1e-8)
        ).mean()
        volume_error = (
            (pred_bin.sum(dim=(1, 2, 3)) - gt_bin.sum(dim=(1, 2, 3))).abs()
            / gt_bin.sum(dim=(1, 2, 3)).clamp_min(1.0)
        ).mean()

        coords = torch.stack(
            torch.meshgrid(
                torch.arange(pred.shape[1], device=pred.device, dtype=pred.dtype),
                torch.arange(pred.shape[2], device=pred.device, dtype=pred.dtype),
                torch.arange(pred.shape[3], device=pred.device, dtype=pred.dtype),
                indexing="ij",
            ),
            dim=-1,
        )
        flat_coords = coords.reshape(-1, 3)
        pred_w = pred.reshape(pred.shape[0], -1).clamp_min(0.0)
        gt_w = gt.reshape(gt.shape[0], -1).clamp_min(0.0)
        pred_centroid = pred_w @ flat_coords / pred_w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        gt_centroid = gt_w @ flat_coords / gt_w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        cle = torch.linalg.norm(pred_centroid - gt_centroid, dim=1).mean()

        pred_peak = flat_coords[pred.reshape(pred.shape[0], -1).argmax(dim=1)]
        gt_peak = flat_coords[gt.reshape(gt.shape[0], -1).argmax(dim=1)]
        ple = torch.linalg.norm(pred_peak - gt_peak, dim=1).mean()

        return {
            "iou": iou,
            "nrmse": nrmse,
            "volume_error": volume_error,
            "cle": cle,
            "ple": ple,
        }

    @rank_zero_only
    def _save_proj_compare(self, gt_img, pred_img, out_path: Path, title: str):
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        def _to_img(x):
            if torch.is_tensor(x):
                x = x.detach().cpu().numpy()
            x = x.astype(np.float32)
            x_min, x_max = float(np.min(x)), float(np.max(x))
            if x_max > x_min:
                x = (x - x_min) / (x_max - x_min)
            return x

        g = _to_img(gt_img)
        p = _to_img(pred_img)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(g, cmap="gray")
        axes[0].set_title("gt")
        axes[0].axis("off")
        axes[1].imshow(p, cmap="gray")
        axes[1].set_title("pred")
        axes[1].axis("off")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    def test_step(self, batch, batch_idx):
        projections = batch["projections"]
        descatter_targets = batch.get("descatter_targets")
        _projections, proj_in = self._prepare_projection_input(batch)
        points = batch["points"]
        points_mm = batch.get("points_mm")
        depth_maps = batch.get("depth_maps")
        point_densities = batch["point_densities"]
        voxel_shape = batch["feasible_voxel_shape"]

        B = points.shape[0]
        if self._is_voxel_model():
            target_voxel = self._voxel_target(batch)
            pred_voxel, _aux_projections = self._call_voxel_model(projections, batch)
            density_pred, density_gt = self._standardize_pred_voxel(pred_voxel, target_voxel)
            density_pred = torch.sigmoid(density_pred)
            density_gt_bin = (density_gt > 0.0).to(dtype=torch.float32)
            full_grid = True
            dice = dice_coefficient(
                density_pred, density_gt_bin, threshold=self._test_pred_threshold
            )
        elif self._is_ssq_model():
            out = self._call_ssq_model(batch, return_diagnostics=False)
            pred_density = out["density"].clamp(0.0, 1.0)
            _aux_projections = out.get("aux_outputs", {})
            voxel_shape_tuple = self._shape_tuple_from_batch(voxel_shape, B)
            full_grid = int(np.prod(voxel_shape_tuple[1:])) == point_densities.shape[1]
            if full_grid:
                density_gt = point_densities.reshape(voxel_shape_tuple)
                density_gt_bin = (density_gt > 0.0).to(dtype=torch.float32)
                density_pred = pred_density.reshape(voxel_shape_tuple)
                dice = dice_coefficient(
                    density_pred, density_gt_bin, threshold=self._test_pred_threshold
                )
            else:
                density_gt = None
                density_gt_bin = None
                density_pred = pred_density
        else:
            pred_density, _aux_projections = self._call_model(
                proj_in,
                points,
                points_mm=points_mm,
                depth_maps=depth_maps,
                source_hypotheses=self._source_hypotheses_from_batch(batch),
            )
            voxel_shape_tuple = self._shape_tuple_from_batch(voxel_shape, B)
            full_grid = int(np.prod(voxel_shape_tuple[1:])) == point_densities.shape[1]
            if full_grid:
                density_gt = point_densities.reshape(voxel_shape_tuple)
                density_gt_bin = (density_gt > 0.0).to(dtype=torch.float32)
                density_pred = torch.sigmoid(pred_density).reshape(voxel_shape_tuple)
                dice = dice_coefficient(
                    density_pred, density_gt_bin, threshold=self._test_pred_threshold
                )
            else:
                density_gt = None
                density_gt_bin = None
                density_pred = torch.sigmoid(pred_density)

        if not full_grid:
            pred_bin = (density_pred.squeeze(-1) >= self._test_pred_threshold).float()
            gt_bin = (point_densities > 0.0).float()
            intersection = (pred_bin * gt_bin).sum(dim=1)
            dice = (
                (2.0 * intersection + 1e-8) / (pred_bin.sum(dim=1) + gt_bin.sum(dim=1) + 1e-8)
            ).mean()
        self.log(
            "test_dice",
            dice,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        if not full_grid:
            return {"dice": dice}

        extra_metrics = self._volume_metrics(
            density_pred, density_gt, density_gt_bin, self._test_pred_threshold
        )
        for name, value in extra_metrics.items():
            self.log(f"test_{name}", value, on_step=False, on_epoch=True, sync_dist=True)

        # Reconstruction metrics (PSNR/SSIM) on [0,1] via gt-based min-max normalization.
        gt_min = density_gt.amin(dim=(1, 2, 3), keepdim=True)
        gt_max = density_gt.amax(dim=(1, 2, 3), keepdim=True)
        denom = (gt_max - gt_min).clamp_min(1e-8)
        density_gt_n = ((density_gt - gt_min) / denom).clamp(0.0, 1.0)
        density_pred_n = ((density_pred - gt_min) / denom).clamp(0.0, 1.0)

        psnr_vals, ssim_vals = [], []
        for i in range(B):
            psnr_vals.append(float(get_psnr_3d(density_pred_n[i], density_gt_n[i])))
            ssim_vals.append(float(get_ssim_3d(density_pred_n[i], density_gt_n[i])))

        psnr = torch.tensor(float(np.mean(psnr_vals)), device=density_pred.device)
        ssim = torch.tensor(float(np.mean(ssim_vals)), device=density_pred.device)
        self.log("test_psnr", psnr, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_ssim", ssim, on_step=False, on_epoch=True, sync_dist=True)

        # Segmentation metrics: region count + ASSD + HD95
        thr = float(self._test_pred_threshold)
        min_sz = int(getattr(self, "_test_min_region_size", 10))
        spacing = tuple(getattr(self, "_test_voxel_spacing", (1.0, 1.0, 1.0)))

        pred_regions, gt_regions = [], []
        assd_vals, hd95_vals = [], []
        prec_vals, rec_vals = [], []
        valid_pairs = 0

        # aux_projections contains predicted auxiliary per-view projections when provided.

        for i in range(B):
            pm = density_pred[i].detach().to(dtype=torch.float32).cpu().numpy() >= thr
            # GT is a reconstruction volume; use >0 as foreground.
            gm = density_gt_bin[i].detach().to(dtype=torch.float32).cpu().numpy() > 0.0

            pm_f, n_pred = self._filter_small_components(pm, min_sz)
            gm_f, n_gt = self._filter_small_components(gm, min_sz)
            pred_regions.append(float(n_pred))
            gt_regions.append(float(n_gt))

            # Voxel-level Precision / Recall (on filtered masks)
            tp = float(np.logical_and(pm_f, gm_f).sum())
            fp = float(np.logical_and(pm_f, np.logical_not(gm_f)).sum())
            fn = float(np.logical_and(np.logical_not(pm_f), gm_f).sum())
            if (tp + fp) == 0.0:
                prec = 1.0 if gm_f.sum() == 0 else 0.0
            else:
                prec = tp / (tp + fp)
            if (tp + fn) == 0.0:
                rec = 1.0
            else:
                rec = tp / (tp + fn)
            prec_vals.append(float(prec))
            rec_vals.append(float(rec))

            a, h = self._assd_hd95(pm_f, gm_f, spacing)
            if np.isfinite(a) and np.isfinite(h):
                valid_pairs += 1
                assd_vals.append(float(a))
                hd95_vals.append(float(h))

            # per-sample record (rank 0 only)
            if self.trainer is not None and self.trainer.is_global_zero:
                sid = None
                sids = batch.get("sample_id")
                if sids is not None:
                    sid = str(sids[i])
                else:
                    sid = f"{batch_idx:06d}_{i}"
                pred_i = density_pred[i]
                gt_i = density_gt[i]
                gt_bin_i = density_gt_bin[i]
                pred_bin_i = (pred_i >= thr).float()
                intersection_i = (pred_bin_i * gt_bin_i).sum()
                union_i = ((pred_bin_i + gt_bin_i) > 0).float().sum()
                iou_i = float(((intersection_i + 1e-8) / (union_i + 1e-8)).detach().cpu())
                nrmse_i = torch.sqrt(torch.mean((pred_i - gt_i).square())) / (
                    gt_i.amax() - gt_i.amin()
                ).clamp_min(1e-8)
                volume_error_i = (
                    pred_bin_i.sum() - gt_bin_i.sum()
                ).abs() / gt_bin_i.sum().clamp_min(1.0)
                coords_i = torch.stack(
                    torch.meshgrid(
                        torch.arange(pred_i.shape[0], device=pred_i.device, dtype=pred_i.dtype),
                        torch.arange(pred_i.shape[1], device=pred_i.device, dtype=pred_i.dtype),
                        torch.arange(pred_i.shape[2], device=pred_i.device, dtype=pred_i.dtype),
                        indexing="ij",
                    ),
                    dim=-1,
                ).reshape(-1, 3)
                pred_w_i = pred_i.reshape(-1).clamp_min(0.0)
                gt_w_i = gt_i.reshape(-1).clamp_min(0.0)
                pred_centroid_i = pred_w_i @ coords_i / pred_w_i.sum().clamp_min(1e-8)
                gt_centroid_i = gt_w_i @ coords_i / gt_w_i.sum().clamp_min(1e-8)
                cle_i = torch.linalg.norm(pred_centroid_i - gt_centroid_i)
                pred_peak_i = coords_i[pred_i.reshape(-1).argmax()]
                gt_peak_i = coords_i[gt_i.reshape(-1).argmax()]
                ple_i = torch.linalg.norm(pred_peak_i - gt_peak_i)
                # Instance-level metrics (only when #lights>1)
                mr = ms = delta_cc = None
                if int(n_gt) > 1:
                    gt_lab, gt_n = self._label_components(gm_f)
                    pred_lab, pred_n = self._label_components(pm_f)

                    missed = 0
                    splits = 0
                    for gid in range(1, int(gt_n) + 1):
                        overlaps = np.unique(pred_lab[gt_lab == gid])
                        overlaps = overlaps[overlaps != 0]
                        if overlaps.size == 0:
                            missed += 1
                        elif overlaps.size > 1:
                            splits += int(overlaps.size - 1)

                    merges = 0
                    for pid in range(1, int(pred_n) + 1):
                        overlaps = np.unique(gt_lab[pred_lab == pid])
                        overlaps = overlaps[overlaps != 0]
                        if overlaps.size > 1:
                            merges += int(overlaps.size - 1)

                    mr = float(missed / max(int(gt_n), 1))
                    ms = float((splits + merges) / max(int(gt_n), 1))
                    delta_cc = float(abs(int(pred_n) - int(gt_n)))

                self._test_sample_metrics.append(
                    {
                        "sample_id": sid,
                        "pred_regions": int(n_pred),
                        "gt_regions": int(n_gt),
                        "dice": float(compute_dice(pm_f, gm_f)),
                        "iou": iou_i,
                        "nrmse": float(nrmse_i.detach().cpu()),
                        "precision": float(prec),
                        "recall": float(rec),
                        "mr": mr,
                        "ms": ms,
                        "delta_cc": delta_cc,
                        "assd": float(a) if np.isfinite(a) else None,
                        "hd95": float(h) if np.isfinite(h) else None,
                        "psnr": float(psnr_vals[i]),
                        "ssim": float(ssim_vals[i]),
                        "cle": float(cle_i.detach().cpu()),
                        "ple": float(ple_i.detach().cpu()),
                        "volume_error": float(volume_error_i.detach().cpu()),
                        "pred_positive_ratio": float(pred_bin_i.mean().detach().cpu()),
                        "gt_positive_ratio": float(gt_bin_i.mean().detach().cpu()),
                    }
                )

        test_regions = torch.tensor(float(np.mean(pred_regions)), device=density_pred.device)
        test_gt_regions = torch.tensor(float(np.mean(gt_regions)), device=density_pred.device)
        test_valid_surface_frac = torch.tensor(
            float(valid_pairs / max(B, 1)), device=density_pred.device
        )

        assd_mean = float(np.mean(assd_vals)) if len(assd_vals) else 0.0
        hd95_mean = float(np.mean(hd95_vals)) if len(hd95_vals) else 0.0
        test_assd = torch.tensor(assd_mean, device=density_pred.device)
        test_hd95 = torch.tensor(hd95_mean, device=density_pred.device)
        test_precision = torch.tensor(
            float(np.mean(prec_vals)) if len(prec_vals) else 0.0, device=density_pred.device
        )
        test_recall = torch.tensor(
            float(np.mean(rec_vals)) if len(rec_vals) else 0.0, device=density_pred.device
        )

        self.log(
            "test_regions",
            test_regions,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log("test_gt_regions", test_gt_regions, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_assd", test_assd, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_hd95", test_hd95, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_precision", test_precision, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_recall", test_recall, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "test_surface_valid_frac",
            test_valid_surface_frac,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # Saving (rank 0 only)
        if self.trainer is not None and self.trainer.is_global_zero:
            sample_ids = batch.get("sample_id")
            if sample_ids is None:
                sample_ids = [f"{batch_idx:06d}_{i}" for i in range(B)]

            for i in range(B):
                sid = str(sample_ids[i])

                if self._test_save_recon_roi and self._test_recon_dir is not None:
                    npz_path = self._test_recon_dir / f"{sid}.npz"
                    np.savez_compressed(
                        npz_path,
                        pred=density_pred[i].detach().to(dtype=torch.float32).cpu().numpy(),
                        gt=density_gt[i].detach().to(dtype=torch.float32).cpu().numpy(),
                        pred_threshold=np.float32(self._test_pred_threshold),
                    )

                    # Also store raw uint8 binary masks as .bin
                    # - pred: pred >= pred_threshold
                    # - gt:   gt > 0
                    pred_bin_path = self._test_recon_dir / f"{sid}_pred.bin"
                    gt_bin_path = self._test_recon_dir / f"{sid}_gt.bin"

                    pred_bin = (
                        density_pred[i].detach().to(dtype=torch.float32).cpu().numpy()
                        >= float(self._test_pred_threshold)
                    ).astype(np.uint8)
                    gt_bin = (
                        density_gt[i].detach().to(dtype=torch.float32).cpu().numpy() > 0.0
                    ).astype(np.uint8)

                    pred_bin.tofile(pred_bin_path)
                    gt_bin.tofile(gt_bin_path)

                    # Track storage sizes
                    try:
                        self._test_recon_file_sizes[sid] = {
                            "npz_bytes": int(npz_path.stat().st_size),
                            "pred_bin_bytes": int(pred_bin_path.stat().st_size),
                            "gt_bin_bytes": int(gt_bin_path.stat().st_size),
                            "shape": [int(x) for x in pred_bin.shape],
                            "dtype": "uint8",
                        }
                    except Exception:
                        pass

                if self._test_save_registered_seg and self._test_seg_dir is not None:
                    if self._test_base_seg is None or self._test_new_label is None:
                        self._test_sample_metrics.append(
                            {
                                "sample_id": sid,
                                "warning": "registered_seg_base_seg_missing",
                            }
                        )
                    else:
                        rx = self._as_pair(batch["range_x"][i])
                        ry = self._as_pair(batch["range_y"][i])
                        rz = self._as_pair(batch["range_z"][i])

                        roi_mask = density_pred[i].detach().to(
                            dtype=torch.float32
                        ).cpu().numpy() >= float(self._test_pred_threshold)

                        new_seg = self._test_base_seg.copy()

                        # IMPORTANT: keep ROI slicing consistent with MultiProjDataset:
                        # Matches MultiProjDataset ROI slicing order.
                        x0, x1 = rx
                        y0, y1 = ry
                        z0, z1 = rz
                        roi_view = new_seg[x0:x1, y0:y1, z0:z1]

                        # Skip instead of truncating; shape mismatch means ROI misalignment.
                        if roi_view.shape != roi_mask.shape:
                            self._test_sample_metrics.append(
                                {
                                    "sample_id": sid,
                                    "warning": "registered_seg_roi_shape_mismatch",
                                    "range_x": [int(x0), int(x1)],
                                    "range_y": [int(y0), int(y1)],
                                    "range_z": [int(z0), int(z1)],
                                    "roi_view_shape": [int(s) for s in roi_view.shape],
                                    "roi_mask_shape": [int(s) for s in roi_mask.shape],
                                }
                            )
                        else:
                            roi_view[roi_mask] = self._test_new_label
                            new_seg[x0:x1, y0:y1, z0:z1] = roi_view
                            np.save(self._test_seg_dir / f"{sid}.npy", new_seg)
                            # Also save raw binary dump (same dtype as numpy array)
                            new_seg.tofile(self._test_seg_dir / f"{sid}.bin")

                if (
                    self._test_save_proj_comparisons
                    and self._test_proj_dir is not None
                    and descatter_targets is not None
                    and len(self._test_angles_to_save) > 0
                ):
                    for angle in self._test_angles_to_save:
                        if (
                            self._test_angle_saved_counts.get(angle, 0)
                            >= self._test_max_samples_per_angle
                        ):
                            continue
                        out_path = self._test_proj_dir / f"{sid}_angle{angle}.png"
                        self._save_proj_compare(
                            projections[angle][i],
                            descatter_targets[angle][i],
                            out_path,
                            title=f"{sid} angle={angle}",
                        )
                        self._test_angle_saved_counts[angle] = (
                            self._test_angle_saved_counts.get(angle, 0) + 1
                        )

        return {
            "dice": dice,
            "psnr": psnr,
            "ssim": ssim,
            "regions": test_regions,
            "gt_regions": test_gt_regions,
            "assd": test_assd,
            "hd95": test_hd95,
        }

    def on_test_end(self):
        """Write a small metrics summary file (rank 0 only)."""
        if self.trainer is None or not self.trainer.is_global_zero or self._test_out_dir is None:
            return

        metrics = {}
        for k, v in self.trainer.callback_metrics.items():
            if not str(k).startswith("test_"):
                continue
            if torch.is_tensor(v):
                v = v.detach().cpu().item()
            metrics[str(k)] = v

        try:
            (self._test_out_dir / "metrics_summary.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2)
            )
        except Exception:
            pass

        # Per-sample + grouped summaries
        try:
            import csv

            with open(self._test_out_dir / "metrics_summary.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["metric", "value"])
                w.writeheader()
                for key in sorted(metrics):
                    w.writerow({"metric": key, "value": metrics[key]})

            rows = list(getattr(self, "_test_sample_metrics", []) or [])
            if rows:
                # De-duplicate by sample_id (saving stage may append warning-only rows)
                by_id = {}
                for r in rows:
                    sid = r.get("sample_id")
                    if sid is None:
                        continue
                    if sid not in by_id:
                        by_id[sid] = dict(r)
                        continue

                    # Merge fields (prefer non-None); accumulate warnings
                    for k, v in r.items():
                        if k == "warning":
                            if v is None:
                                continue
                            prev = by_id[sid].get("warning")
                            if not prev:
                                by_id[sid]["warning"] = v
                            elif v not in str(prev):
                                by_id[sid]["warning"] = f"{prev};{v}"
                            continue

                        if by_id[sid].get(k) is None and v is not None:
                            by_id[sid][k] = v

                rows = list(by_id.values())
                # 1) per-sample csv
                keys = [
                    "sample_id",
                    "case_id",
                    "Dice",
                    "IoU",
                    "NRMSE",
                    "PSNR",
                    "SSIM",
                    "CLE",
                    "LE",
                    "PLE",
                    "Volume Error",
                    "pred_positive_ratio",
                    "gt_positive_ratio",
                    "pred_regions",
                    "gt_regions",
                    "dice",
                    "iou",
                    "nrmse",
                    "precision",
                    "recall",
                    "mr",
                    "ms",
                    "delta_cc",
                    "assd",
                    "hd95",
                    "psnr",
                    "ssim",
                    "cle",
                    "ple",
                    "volume_error",
                    "warning",
                ]
                for r in rows:
                    r["case_id"] = r.get("sample_id")
                    r["Dice"] = r.get("dice")
                    r["IoU"] = r.get("iou")
                    r["NRMSE"] = r.get("nrmse")
                    r["PSNR"] = r.get("psnr")
                    r["SSIM"] = r.get("ssim")
                    r["CLE"] = r.get("cle")
                    r["LE"] = r.get("cle")
                    r["PLE"] = r.get("ple")
                    r["Volume Error"] = r.get("volume_error")
                with open(self._test_out_dir / "metrics.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=keys)
                    w.writeheader()
                    for r in rows:
                        w.writerow({k: r.get(k) for k in keys})
                with open(self._test_out_dir / "metrics_per_sample.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=keys)
                    w.writeheader()
                    for r in rows:
                        w.writerow({k: r.get(k) for k in keys})

                # 2) grouped by gt_regions
                grouped = {}
                for r in rows:
                    g = int(r.get("gt_regions", 0))
                    grouped.setdefault(g, []).append(r)

                summary_by_gt = {}
                for g, rs in grouped.items():

                    def _mean(field):
                        vals = [x[field] for x in rs if x.get(field) is not None]
                        return float(sum(vals) / len(vals)) if vals else None

                    summary_by_gt[str(g)] = {
                        "count": len(rs),
                        "dice_mean": _mean("dice"),
                        "precision_mean": _mean("precision"),
                        "recall_mean": _mean("recall"),
                        "mr_mean": _mean("mr"),
                        "ms_mean": _mean("ms"),
                        "delta_cc_mean": _mean("delta_cc"),
                        "assd_mean": _mean("assd"),
                        "hd95_mean": _mean("hd95"),
                        "pred_regions_mean": _mean("pred_regions"),
                        "psnr_mean": _mean("psnr"),
                        "ssim_mean": _mean("ssim"),
                    }

                (self._test_out_dir / "summary_by_gt_regions.json").write_text(
                    json.dumps(summary_by_gt, ensure_ascii=False, indent=2)
                )

                # 3) required summary table: 1-light / 2-light / 3-light / overall
                def _mean_over(rs, field):
                    vals = [x.get(field) for x in rs if x.get(field) is not None]
                    return float(sum(vals) / len(vals)) if vals else None

                groups = {
                    "1": [r for r in rows if int(r.get("gt_regions", 0)) == 1],
                    "2": [r for r in rows if int(r.get("gt_regions", 0)) == 2],
                    "3": [r for r in rows if int(r.get("gt_regions", 0)) == 3],
                    "overall": rows,
                }

                table_rows = []
                for name, rs in groups.items():
                    table_rows.append(
                        {
                            "group": name,
                            "count": len(rs),
                            "dsc": _mean_over(rs, "dice"),
                            "precision": _mean_over(rs, "precision"),
                            "recall": _mean_over(rs, "recall"),
                            "assd": _mean_over(rs, "assd"),
                            "hd95": _mean_over(rs, "hd95"),
                            "mr": _mean_over(rs, "mr"),
                            "ms": _mean_over(rs, "ms"),
                            "delta_cc": _mean_over(rs, "delta_cc"),
                        }
                    )

                with open(self._test_out_dir / "summary_by_num_lights.csv", "w", newline="") as f:
                    w = csv.DictWriter(
                        f,
                        fieldnames=[
                            "group",
                            "count",
                            "dsc",
                            "precision",
                            "recall",
                            "assd",
                            "hd95",
                            "mr",
                            "ms",
                            "delta_cc",
                        ],
                    )
                    w.writeheader()
                    for r in table_rows:
                        w.writerow(r)

                (self._test_out_dir / "summary_by_num_lights.json").write_text(
                    json.dumps(table_rows, ensure_ascii=False, indent=2)
                )

            # Recon ROI storage summary
            if self._test_recon_dir is not None and getattr(self, "_test_recon_file_sizes", None):
                files = dict(self._test_recon_file_sizes)
                total_npz = sum(v.get("npz_bytes", 0) for v in files.values())
                total_pred_bin = sum(v.get("pred_bin_bytes", 0) for v in files.values())
                total_gt_bin = sum(v.get("gt_bin_bytes", 0) for v in files.values())
                total_bin = total_pred_bin + total_gt_bin
                summary = {
                    "num_samples": len(files),
                    "total_npz_bytes": int(total_npz),
                    "total_pred_bin_bytes": int(total_pred_bin),
                    "total_gt_bin_bytes": int(total_gt_bin),
                    "total_bin_bytes": int(total_bin),
                    "total_bytes": int(total_npz + total_bin),
                    "total_mb": float((total_npz + total_bin) / (1024**2)),
                    "files": files,
                }
                (self._test_recon_dir / "storage_size.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2)
                )

        except Exception:
            pass
