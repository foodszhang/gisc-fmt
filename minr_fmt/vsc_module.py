from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .module import TrainingLightningModule


class VSCTrainingLightningModule(TrainingLightningModule):
    """SSQ-FMT trainer with view-subset consistency regularization.

    The implementation is intentionally confined to the training objective. It does
    not introduce a new density decoder, candidate-conditioned final prediction,
    PHSA path, or boundary forward operator. The full-view forward path remains the
    configured SSQ-FMT model; subset passes only change the valid-view mask and the
    corresponding surface measurements supplied to the same model.
    """

    def _vsc_cfg(self) -> Any:
        ssq_cfg = getattr(self.cfg.model, "ssq_fmt", None)
        if ssq_cfg is None:
            return {}
        return getattr(ssq_cfg, "view_subset_consistency", {}) or {}

    def _vsc_enabled(self) -> bool:
        cfg = self._vsc_cfg()
        return bool(cfg.get("enabled", False))

    def _vsc_weight_schedule(self) -> tuple[float, float]:
        cfg = self._vsc_cfg()
        epoch = float(self.current_epoch)
        warmup_epochs = float(cfg.get("supervised_warmup_epochs", 0.0))
        subset_weight = float(cfg.get("subset_supervised_weight", 0.0))
        if epoch < warmup_epochs:
            subset_weight = 0.0

        max_consistency = float(cfg.get("consistency_weight", 0.0))
        ramp_start = float(cfg.get("consistency_ramp_start_epoch", warmup_epochs))
        ramp_epochs = float(cfg.get("consistency_ramp_epochs", 1.0))
        if epoch < ramp_start:
            consistency_weight = 0.0
        elif ramp_epochs <= 0.0:
            consistency_weight = max_consistency
        else:
            ramp = min(1.0, max(0.0, (epoch - ramp_start + 1.0) / ramp_epochs))
            consistency_weight = max_consistency * ramp
        return subset_weight, consistency_weight

    @staticmethod
    def _as_view_mask(detector_valid_mask: torch.Tensor | None, surface: torch.Tensor) -> torch.Tensor:
        batch_size, num_views = surface.shape[:2]
        if detector_valid_mask is None:
            return torch.ones((batch_size, num_views), dtype=torch.bool, device=surface.device)
        mask = detector_valid_mask.to(device=surface.device, dtype=torch.bool)
        if mask.dim() == 5:
            mask = mask.squeeze(2)
        if mask.dim() == 4:
            return mask.flatten(start_dim=2).any(dim=-1)
        if mask.dim() == 3:
            return mask.flatten(start_dim=2).any(dim=-1)
        if mask.dim() == 2:
            return mask
        raise ValueError(f"Unsupported detector_valid_mask shape: {tuple(mask.shape)}")

    def _sample_subset_size(self, num_views: int, device: torch.device) -> int:
        cfg = self._vsc_cfg()
        raw_sizes = list(cfg.get("subset_sizes", [max(num_views - 1, 1)]))
        if not raw_sizes:
            raw_sizes = [max(num_views - 1, 1)]
        idx = int(torch.randint(len(raw_sizes), (1,), device=device).item())
        value = raw_sizes[idx]
        if isinstance(value, str) and value.lower() in {"leave_one", "leave-one", "loo"}:
            subset_size = num_views - 1
        else:
            subset_size = int(value)
        return max(1, min(int(subset_size), int(num_views)))

    def _sample_view_subset_mask(self, batch: dict[str, Any]) -> torch.Tensor:
        surface = batch.get("surface_measurements_packed", batch.get("projections_packed"))
        if surface is None:
            raise KeyError("VSC requires surface_measurements_packed or projections_packed")
        base_view_mask = self._as_view_mask(batch.get("detector_valid_mask"), surface)
        batch_size, num_views = base_view_mask.shape
        subset_size = self._sample_subset_size(num_views, surface.device)
        subset = torch.zeros_like(base_view_mask, dtype=torch.bool)
        for batch_index in range(batch_size):
            valid_idx = torch.nonzero(base_view_mask[batch_index], as_tuple=False).flatten()
            if valid_idx.numel() <= subset_size:
                chosen = valid_idx
            else:
                perm = torch.randperm(valid_idx.numel(), device=surface.device)[:subset_size]
                chosen = valid_idx[perm]
            if chosen.numel() == 0:
                # Degenerate safety fallback: keep all views rather than producing an empty forward.
                chosen = torch.arange(num_views, device=surface.device)
            subset[batch_index, chosen] = True
        return subset & base_view_mask

    @staticmethod
    def _detector_mask_from_view_mask(
        detector_valid_mask: torch.Tensor | None,
        view_mask: torch.Tensor,
        surface: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_views = surface.shape[:2]
        height, width = int(surface.shape[-2]), int(surface.shape[-1])
        view_pixel_mask = view_mask[:, :, None, None].expand(batch_size, num_views, height, width)
        if detector_valid_mask is None:
            return view_pixel_mask
        base = detector_valid_mask.to(device=surface.device, dtype=torch.bool)
        if base.dim() == 5:
            base = base.squeeze(2)
        elif base.dim() == 2:
            base = base[:, :, None, None].expand(batch_size, num_views, height, width)
        elif base.dim() == 3:
            base = base[:, :, None].expand(batch_size, num_views, height, width)
        elif base.dim() != 4:
            raise ValueError(f"Unsupported detector_valid_mask shape: {tuple(base.shape)}")
        return base & view_pixel_mask

    def _subset_batch(self, batch: dict[str, Any], view_mask: torch.Tensor) -> dict[str, Any]:
        surface = batch.get("surface_measurements_packed", batch.get("projections_packed"))
        if surface is None:
            raise KeyError("VSC requires surface_measurements_packed or projections_packed")
        mask5 = view_mask[:, :, None, None, None].to(device=surface.device, dtype=surface.dtype)
        subset_surface = surface * mask5
        subset = dict(batch)
        if "surface_measurements_packed" in batch:
            subset["surface_measurements_packed"] = subset_surface
        if "projections_packed" in batch:
            subset["projections_packed"] = batch["projections_packed"] * mask5.to(batch["projections_packed"].dtype)
        if "surface_measurements" in batch and torch.is_tensor(batch["surface_measurements"]):
            sm = batch["surface_measurements"]
            if sm.dim() == 5:
                subset["surface_measurements"] = sm * mask5.to(sm.dtype)
            elif sm.dim() == 4:
                subset["surface_measurements"] = sm * view_mask[:, :, None, None].to(sm.dtype)
        subset["detector_valid_mask"] = self._detector_mask_from_view_mask(
            batch.get("detector_valid_mask"), view_mask, surface
        )
        return subset

    def _ssq_density_loss_dict(
        self,
        batch: dict[str, Any],
        pred_density: torch.Tensor,
        target_density: torch.Tensor,
        aux_outputs: dict[str, Any] | None,
        *,
        include_auxiliary_terms: bool,
    ) -> dict[str, torch.Tensor]:
        if include_auxiliary_terms:
            loss_aux = aux_outputs or {}
        else:
            loss_aux = {}
            if isinstance(aux_outputs, dict) and torch.is_tensor(aux_outputs.get("measurement_supported")):
                loss_aux["measurement_supported"] = aux_outputs["measurement_supported"]
        return self.ssq_loss_func(
            pred_density,
            target_density,
            loss_aux,
            gt_voxels=batch.get("gt_voxels"),
            points_ijk=batch.get("points_ijk"),
            sdf_targets=batch.get("sdf_targets"),
            query_component_ids=batch.get("query_component_ids"),
            gt_component_centers_mm=batch.get("gt_component_centers_mm"),
            gt_component_valid_mask=batch.get("gt_component_valid_mask"),
        )

    @staticmethod
    def _density_logit(out: dict[str, Any], eps: float = 1.0e-5) -> torch.Tensor:
        aux = out.get("aux_outputs", {}) if isinstance(out, dict) else {}
        density = out["density"]
        logit = aux.get("density_logits") if isinstance(aux, dict) else None
        if torch.is_tensor(logit) and tuple(logit.shape) == tuple(density.shape):
            return torch.nan_to_num(logit, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30.0, 30.0)
        return torch.logit(density.clamp(eps, 1.0 - eps))

    def _vsc_distance(self, source_out: dict[str, Any], target_out: dict[str, Any], *, detach_target: bool) -> torch.Tensor:
        cfg = self._vsc_cfg()
        mode = str(cfg.get("distance", "logit_mse")).lower()
        if mode == "prob_mse":
            source = source_out["density"].clamp(0.0, 1.0)
            target = target_out["density"].clamp(0.0, 1.0)
            if detach_target:
                target = target.detach()
            return torch.nn.functional.mse_loss(source, target)
        source_logit = self._density_logit(source_out)
        target_logit = self._density_logit(target_out)
        if detach_target:
            target_logit = target_logit.detach()
        if mode == "bce":
            target_prob = torch.sigmoid(target_logit).detach() if detach_target else torch.sigmoid(target_logit)
            return torch.nn.functional.binary_cross_entropy_with_logits(source_logit, target_prob)
        if mode == "kl":
            source_log_prob = torch.nn.functional.logsigmoid(source_logit)
            target_prob = torch.sigmoid(target_logit)
            if detach_target:
                target_prob = target_prob.detach()
            # Bernoulli KL target || source, averaged over query samples.
            source_prob = torch.sigmoid(source_logit).clamp(1.0e-6, 1.0 - 1.0e-6)
            target_prob = target_prob.clamp(1.0e-6, 1.0 - 1.0e-6)
            kl = target_prob * (target_prob.log() - source_log_prob) + (1.0 - target_prob) * (
                (1.0 - target_prob).log() - (1.0 - source_prob).log()
            )
            return kl.mean()
        if mode != "logit_mse":
            raise ValueError("view_subset_consistency.distance must be logit_mse, prob_mse, bce, or kl")
        return torch.nn.functional.mse_loss(source_logit, target_logit)

    def _apply_vsc_terms(
        self,
        batch: dict[str, Any],
        full_out: dict[str, Any],
        target_density: torch.Tensor,
        loss_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        cfg = self._vsc_cfg()
        subset_weight, consistency_weight = self._vsc_weight_schedule()
        num_subsets = int(cfg.get("num_subsets", 2))
        if num_subsets <= 0 or (subset_weight <= 0.0 and consistency_weight <= 0.0):
            loss_dict["vsc_subset_supervised_weight"] = full_out["density"].new_tensor(subset_weight)
            loss_dict["vsc_consistency_weight"] = full_out["density"].new_tensor(consistency_weight)
            return loss_dict

        include_subset_aux = bool(cfg.get("include_aux_in_subset_supervision", False))
        subset_outs: list[dict[str, Any]] = []
        subset_losses: list[torch.Tensor] = []
        subset_sizes: list[torch.Tensor] = []
        for _ in range(num_subsets):
            subset_mask = self._sample_view_subset_mask(batch)
            subset_sizes.append(subset_mask.float().sum(dim=1).mean())
            subset_batch = self._subset_batch(batch, subset_mask)
            subset_out = self._call_ssq_model(subset_batch, return_diagnostics=False)
            subset_outs.append(subset_out)
            if subset_weight > 0.0:
                subset_aux = subset_out.get("aux_outputs", {})
                sub_loss = self._ssq_density_loss_dict(
                    subset_batch,
                    subset_out["density"],
                    target_density,
                    subset_aux if isinstance(subset_aux, dict) else {},
                    include_auxiliary_terms=include_subset_aux,
                )["total_loss"]
                subset_losses.append(sub_loss)

        if subset_losses:
            subset_supervised_loss = torch.stack(subset_losses).mean()
            loss_dict["vsc_subset_supervised_loss"] = subset_supervised_loss
            loss_dict["total_loss"] = loss_dict["total_loss"] + subset_weight * subset_supervised_loss
        else:
            loss_dict["vsc_subset_supervised_loss"] = full_out["density"].sum() * 0.0

        if consistency_weight > 0.0 and subset_outs:
            terms = [self._vsc_distance(subset_out, full_out, detach_target=True) for subset_out in subset_outs]
            if len(subset_outs) >= 2:
                terms.append(self._vsc_distance(subset_outs[0], subset_outs[1], detach_target=False))
            vsc_loss = torch.stack(terms).mean()
            loss_dict["vsc_consistency_loss"] = vsc_loss
            loss_dict["total_loss"] = loss_dict["total_loss"] + consistency_weight * vsc_loss
            with torch.no_grad():
                disagreements = [
                    (subset_out["density"].detach() - full_out["density"].detach()).abs().mean()
                    for subset_out in subset_outs
                ]
                if len(subset_outs) >= 2:
                    disagreements.append(
                        (subset_outs[0]["density"].detach() - subset_outs[1]["density"].detach()).abs().mean()
                    )
                loss_dict["vsc_prob_disagreement"] = torch.stack(disagreements).mean()
        else:
            loss_dict["vsc_consistency_loss"] = full_out["density"].sum() * 0.0
            loss_dict["vsc_prob_disagreement"] = full_out["density"].sum() * 0.0

        loss_dict["vsc_subset_supervised_weight"] = full_out["density"].new_tensor(subset_weight)
        loss_dict["vsc_consistency_weight"] = full_out["density"].new_tensor(consistency_weight)
        loss_dict["vsc_subset_size_mean"] = (
            torch.stack(subset_sizes).mean() if subset_sizes else full_out["density"].new_zeros(())
        )
        return loss_dict

    def training_step(self, batch, batch_idx):
        if not self._is_ssq_model() or not self._vsc_enabled():
            return super().training_step(batch, batch_idx)

        points_mm = batch.get("query_coordinates_mm", batch.get("points_mm"))
        target_density = batch["point_densities"].unsqueeze(-1)
        self._log_query_sampling_stats("train", batch, batch["point_densities"])

        full_out = self._call_ssq_model(batch, return_diagnostics=False)
        density_pred = full_out["density"]
        aux_outputs = full_out.get("aux_outputs", {})
        if isinstance(aux_outputs, dict):
            pi = aux_outputs.get("pi")
            if torch.is_tensor(pi) and pi.numel() > 0:
                self.log("train_ssq_pi0_mean", pi[..., 0].mean(), on_step=False, on_epoch=True, sync_dist=True)
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

        loss_dict = self._ssq_density_loss_dict(
            batch,
            density_pred,
            target_density,
            aux_outputs if isinstance(aux_outputs, dict) else {},
            include_auxiliary_terms=True,
        )
        density_logits = aux_outputs.get("density_logits") if isinstance(aux_outputs, dict) else None
        if torch.is_tensor(density_logits) and self._backbone_logit_loss_weight > 0.0:
            backbone_logit_loss = self.loss_func.sparse_light_loss(density_logits, target_density)
            loss_dict["backbone_logit_loss"] = backbone_logit_loss
            loss_dict["total_loss"] = loss_dict["total_loss"] + self._backbone_logit_loss_weight * backbone_logit_loss

        loss_dict = self._apply_vsc_terms(batch, full_out, target_density, loss_dict)
        total_loss = loss_dict["total_loss"]
        if not torch.isfinite(total_loss.detach()):
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

        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
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
