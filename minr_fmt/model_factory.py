"""Model factory for configured GISC-FMT and baseline models."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from .models.ssq_fmt import SSQFMT


class Patch2ComplementaryAggregation(nn.Module):
    """Encode shared and candidate view evidence in a common latent space."""

    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        epsilon_s: float = 0.1,
        strong_shared_fusion: bool = False,
        support_weighted_reliability: bool = True,
    ) -> None:
        super().__init__()
        self.epsilon_s = float(epsilon_s)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.common_projection = nn.Linear(feature_dim, output_dim)
        # Keep this name because the optimizer and gradient diagnostics already use it.
        self.candidate_projection = nn.Sequential(
            nn.Linear(2, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )
        self.output_norm = nn.LayerNorm(output_dim)
        self.strong_shared_fusion = bool(strong_shared_fusion)
        self.support_weighted_reliability = bool(support_weighted_reliability)
        if self.strong_shared_fusion:
            self.shared_attention = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.SiLU(),
                nn.Linear(output_dim, 1),
            )
            self.shared_set_fusion = nn.Sequential(
                nn.Linear(output_dim * 4, output_dim * 2),
                nn.SiLU(),
                nn.Linear(output_dim * 2, output_dim),
                nn.LayerNorm(output_dim),
            )
        self.last_candidate_view_features: torch.Tensor | None = None
        self.last_shared_view_weights: torch.Tensor | None = None

    @staticmethod
    def _normalize(weight: torch.Tensor, dim: int = 1) -> torch.Tensor:
        denominator = weight.sum(dim=dim, keepdim=True)
        normalized = weight / denominator.clamp_min(1.0e-8)
        return torch.where(
            denominator > 0.0,
            normalized,
            torch.zeros_like(normalized),
        )

    def aggregate_shared(
        self,
        shared_per_view: torch.Tensor,
        view_valid: torch.Tensor,
        attention_logit_residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        per_view = self.common_projection(self.feature_norm(shared_per_view))
        valid = view_valid.to(dtype=torch.bool, device=shared_per_view.device)
        uniform = self._normalize(valid.to(shared_per_view.dtype), dim=1)
        if self.strong_shared_fusion:
            logits = self.shared_attention(per_view).squeeze(-1)
            if attention_logit_residual is not None:
                if attention_logit_residual.shape != logits.shape:
                    raise ValueError("attention_logit_residual must match [B,V,N]")
                logits = logits + attention_logit_residual.to(logits.dtype)
            logits = logits.masked_fill(~valid, -1.0e4)
            has_valid = valid.any(dim=1, keepdim=True)
            logits = torch.where(has_valid, logits, torch.zeros_like(logits))
            weight = torch.softmax(logits, dim=1) * valid.to(logits.dtype)
            weight = self._normalize(weight, dim=1)
            attended = (per_view * weight[..., None]).sum(dim=1)
            mean = (per_view * uniform[..., None]).sum(dim=1)
            centered = per_view - mean[:, None]
            std = (
                centered.square() * uniform[..., None]
            ).sum(dim=1).clamp_min(1.0e-8).sqrt()
            masked = per_view.masked_fill(~valid[..., None], -torch.inf)
            maximum = masked.amax(dim=1)
            maximum = torch.where(
                has_valid.movedim(1, -1), maximum, torch.zeros_like(maximum)
            )
            shared = self.shared_set_fusion(torch.cat([attended, mean, std, maximum], dim=-1))
        else:
            weight = uniform
            shared = (per_view * weight[..., None]).sum(dim=1)
        self.last_shared_view_weights = weight
        return self.output_norm(shared), weight

    def forward(
        self,
        shared_per_view: torch.Tensor,
        candidate_per_view: torch.Tensor,
        view_valid: torch.Tensor,
        candidate_support: torch.Tensor,
        separability: torch.Tensor,
        candidate_valid: torch.Tensor,
        candidate_view_valid: torch.Tensor | None = None,
        uniform_views: bool = False,
        geometry_only: bool = False,
        aggregation_strategy: str | None = None,
    ) -> dict[str, torch.Tensor]:
        shared, shared_weight = self.aggregate_shared(shared_per_view, view_valid)
        if candidate_view_valid is None:
            candidate_view_valid = candidate_valid[:, None].expand_as(candidate_support)
        valid_float = candidate_view_valid.to(candidate_support.dtype)

        candidate_encoded = self.common_projection(self.feature_norm(candidate_per_view))
        condition = torch.stack(
            [candidate_support.clamp(0.0, 1.0), separability.clamp(0.0, 1.0)],
            dim=-1,
        )
        candidate_encoded = self.output_norm(
            candidate_encoded + self.candidate_projection(condition)
        )
        self.last_candidate_view_features = candidate_encoded

        strategy = aggregation_strategy
        if strategy is None or strategy == "legacy":
            strategy = "uniform" if uniform_views else "geometry" if geometry_only else "full"
            if strategy == "full" and not self.support_weighted_reliability:
                strategy = "geometry"
        if strategy in {"uniform", "oracle"}:
            reliability = valid_float
        elif strategy == "support":
            reliability = candidate_support * valid_float
        elif strategy == "geometry":
            reliability = (self.epsilon_s + separability) * valid_float
        elif strategy == "full":
            reliability = (
                candidate_support
                * (self.epsilon_s + separability)
                * valid_float
            )
        elif strategy == "shuffled":
            keys = torch.rand_like(separability).masked_fill(~candidate_view_valid, 2.0)
            permutation = keys.argsort(dim=1)
            shuffled = separability.gather(1, permutation)
            reliability = (self.epsilon_s + shuffled) * valid_float
        else:
            raise ValueError(f"unknown aggregation strategy: {strategy}")
        weight = self._normalize(reliability, dim=1)
        weight = weight * candidate_valid[:, None].to(weight.dtype)
        candidate = (candidate_encoded * weight[..., None]).sum(dim=1)
        candidate = torch.where(
            candidate_valid[..., None],
            candidate,
            torch.zeros_like(candidate),
        )
        return {
            "shared": shared,
            "candidate": candidate,
            "view_weights": weight,
            "shared_view_weights": shared_weight,
            "candidate_view_features": candidate_encoded,
        }


def _clean_support_statistics(support: torch.Tensor) -> torch.Tensor:
    support_safe = support.clamp_min(1.0e-8)
    normalized = support_safe / support_safe.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    entropy = -(normalized * normalized.log()).sum(dim=-1) / torch.log(
        support.new_tensor(float(max(support.shape[-1], 2)))
    )
    return torch.stack(
        [
            support.mean(dim=-1),
            support.amax(dim=-1),
            support.std(dim=-1, unbiased=False),
            entropy,
        ],
        dim=-1,
    )


class SSQFMTPatch2(SSQFMT):
    """View-complementary SSQ-FMT with Patch-2 training semantics."""

    def __init__(self, config: Any):
        super().__init__(config)
        old_aggregation = self.complementary_aggregation
        feature_dim = old_aggregation.shared_projection.in_features
        output_dim = old_aggregation.shared_projection.out_features
        epsilon_s = float(old_aggregation.epsilon_s)
        view_cfg = config.model.ssq_fmt.view_complementary
        self.complementary_aggregation = Patch2ComplementaryAggregation(
            feature_dim,
            output_dim,
            epsilon_s=epsilon_s,
            strong_shared_fusion=bool(view_cfg.get("strong_shared_fusion", False)),
            support_weighted_reliability=bool(
                view_cfg.get("support_weighted_reliability", True)
            ),
        )

        constructor = self.diverse_candidate_constructor
        descriptor_dim = int(self.view_candidate_evidence.descriptor.out_features)
        first_layer = constructor.existence_head[0]
        if int(first_layer.in_features) != descriptor_dim + 7:
            hidden_dim = int(first_layer.out_features)
            constructor.existence_head = nn.Sequential(
                nn.Linear(descriptor_dim + 7, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
        # Remove a redundant scalar conversion in the intermediate constructor commit.
        constructor._support_statistics = _clean_support_statistics

        # The intermediate commit exposed the target for diagnostics. The parent
        # forward treats every returned item as a weighted loss, so filter it here.
        original_supervision_losses = constructor.supervision_losses

        def filtered_supervision_losses(*args, **kwargs):
            losses = original_supervision_losses(*args, **kwargs)
            losses.pop("candidate_existence_target", None)
            return losses

        constructor.supervision_losses = filtered_supervision_losses

    def _view_evidence_heatmap_loss(
        self,
        evidence: torch.Tensor,
        points_mm: torch.Tensor,
        query_valid: torch.Tensor,
        gt_centers: torch.Tensor,
        gt_covariances: torch.Tensor,
        gt_valid: torch.Tensor,
        depth_maps: torch.Tensor | None,
        detector_valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Supervise view-conditioned detector compatibility, not 3-D localization."""
        query_mapped = self.geometry_mapper(
            points_mm,
            depth_maps=depth_maps,
            detector_valid_mask=detector_valid_mask,
        )
        gt_mapped = self.geometry_mapper(
            gt_centers,
            depth_maps=depth_maps,
            detector_valid_mask=detector_valid_mask,
        )
        query_uv = query_mapped["uv_px"]
        gt_uv = gt_mapped["uv_px"]
        detector_delta = query_uv[:, :, :, None] - gt_uv[:, :, None]

        gt_variance = torch.diagonal(
            gt_covariances.float(), dim1=-2, dim2=-1
        ).clamp_min(1.0e-4)
        gt_scale_mm = gt_variance.mean(dim=-1).sqrt()
        pixels_per_mm = gt_mapped["view_pixels_per_mm"].to(evidence.dtype)
        detector_scale = (
            gt_scale_mm[:, None]
            .square()
            .add(float(self.evidence_sigma_mm) ** 2)
            .sqrt()
            * pixels_per_mm[None, :, None]
        ).clamp_min(1.0)

        pair_delta = gt_uv[:, :, :, None] - gt_uv[:, :, None]
        pair_variance = (
            detector_scale[:, :, :, None].square()
            + detector_scale[:, :, None, :].square()
        )
        collision = torch.exp(
            -0.5
            * pair_delta.square().sum(dim=-1)
            / pair_variance.clamp_min(1.0e-6)
        )
        pair_valid = gt_valid[:, None, :, None] & gt_valid[:, None, None, :]
        collision = torch.where(pair_valid, collision, torch.zeros_like(collision))
        eye = torch.eye(
            gt_valid.shape[1],
            device=evidence.device,
            dtype=torch.bool,
        )[None, None]
        separability = 1.0 - collision.masked_fill(eye, 0.0).amax(dim=-1)
        amplitude = self.evidence_eta0 + (1.0 - self.evidence_eta0) * separability
        component_valid = gt_valid[:, None] & gt_mapped["valid_mask"]
        amplitude = amplitude * component_valid.to(amplitude.dtype)

        detector_mahal = (
            detector_delta.float().square().sum(dim=-1)
            / detector_scale[:, :, None].float().square().clamp_min(1.0e-6)
        )
        target = torch.exp(-0.5 * detector_mahal) * amplitude[:, :, None]
        target = target.masked_fill(~component_valid[:, :, None], 0.0).amax(dim=-1)

        valid = query_valid & query_mapped["valid_mask"]
        mask = valid.to(evidence.dtype)
        prediction = evidence.float().clamp(1.0e-6, 1.0 - 1.0e-6)
        target = target.float().clamp(0.0, 1.0)
        positive_weight = 1.0 + 9.0 * target
        loss = -(
            target * prediction.log()
            + (1.0 - target) * (1.0 - prediction).log()
        )
        loss = loss * positive_weight * mask.float()
        return loss.sum() / (positive_weight * mask).sum().clamp_min(1.0)

    def _encoded_query_points(self, points_mm: torch.Tensor) -> torch.Tensor:
        trunk = torch.tensor(
            self.trunk_size_mm,
            device=points_mm.device,
            dtype=points_mm.dtype,
        )
        return self.position_encoding(
            (points_mm / trunk.clamp_min(1.0e-6)).mul(2.0).sub(1.0)
        )

    def _forward_clean_shared_ablation(
        self,
        query_per_view: torch.Tensor,
        query_view_valid: torch.Tensor,
        points_mm: torch.Tensor,
        norm_scale: torch.Tensor,
        return_diagnostics: bool,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Run A0/A1 without candidate construction or candidate supervision."""
        shared, shared_weights = self.complementary_aggregation.aggregate_shared(
            query_per_view,
            query_view_valid,
        )
        b = points_mm.shape[0]
        empty_candidate = shared.new_zeros((b, 0, shared.shape[-1]))
        empty_centers = points_mm.new_zeros((b, 0, 3))
        empty_covariance = points_mm.new_zeros((b, 0, 3, 3))
        empty_scores = shared.new_zeros((b, 0))
        empty_valid = torch.zeros((b, 0), dtype=torch.bool, device=points_mm.device)
        encoded_points = self._encoded_query_points(points_mm)
        decoder_ablation = (
            "shared_capacity"
            if self.view_complementary_ablation == "shared_capacity"
            else "shared_only"
        )
        decoded = self.unified_density_decoder(
            shared,
            empty_candidate,
            points_mm,
            encoded_points,
            empty_centers,
            empty_covariance,
            empty_scores,
            empty_valid,
            ablation=decoder_ablation,
            context_scale=1.0,
        )
        aux_outputs: dict[str, torch.Tensor] = {
            "shared_density": decoded["density"],
            "candidate_context": decoded["candidate_context"],
            "candidate_context_scale": decoded["density"].new_tensor(0.0),
            "decoder_pre_activation": decoded["decoder_pre_activation"],
            "measurement_supported": query_view_valid.any(dim=1),
            "shared_view_weights": shared_weights,
            "view_complementary_mode": torch.ones((), device=points_mm.device),
            "clean_shared_ablation": torch.ones((), device=points_mm.device),
        }
        out: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "density": decoded["density"],
            "aux_outputs": aux_outputs,
        }
        if return_diagnostics:
            diagnostics = dict(aux_outputs)
            diagnostics["normalization_scale"] = norm_scale
            out["diagnostics"] = diagnostics
        return out

    def _forward_view_complementary(self, *args, **kwargs):
        samples = args[2] if len(args) > 2 else kwargs["samples"]
        points_mm = args[4] if len(args) > 4 else kwargs["points_mm"]
        norm_scale = args[7] if len(args) > 7 else kwargs["norm_scale"]
        return_diagnostics = (
            args[8] if len(args) > 8 else kwargs.get("return_diagnostics", False)
        )
        if self.view_complementary_ablation in {
            "a0",
            "shared_only",
            "shared_capacity",
        }:
            query_per_view = (
                samples["sample_features"] * samples["A"][..., None]
            ).sum(dim=3)
            return self._forward_clean_shared_ablation(
                query_per_view,
                samples["query_view_valid"],
                points_mm,
                norm_scale,
                return_diagnostics,
            )
        out = super()._forward_view_complementary(*args, **kwargs)
        encoded = self.complementary_aggregation.last_candidate_view_features
        shared_weights = self.complementary_aggregation.last_shared_view_weights
        if encoded is not None:
            out["aux_outputs"]["candidate_view_features"] = encoded
            if "diagnostics" in out:
                out["diagnostics"]["candidate_view_features"] = encoded
        if shared_weights is not None:
            out["aux_outputs"]["shared_view_weights"] = shared_weights
            if "diagnostics" in out:
                out["diagnostics"]["shared_view_weights"] = shared_weights
        return out


class ModelFactory:
    """Create configured reconstruction models."""

    @staticmethod
    def create_minr_fmt_model(config: Optional[Any] = None, **kwargs):
        return ModelFactory.create_gisc_fmt_model(config=config, **kwargs)

    @staticmethod
    def create_gisc_fmt_model(config: Optional[Any] = None, **kwargs):
        from .models.gisc_multisource import GISCFMT
        if config is None:
            raise ValueError("GISC-FMT 模型需要配置对象")
        return GISCFMT(config=config)

    @staticmethod
    def create_ssq_fmt_model(config: Optional[Any] = None, **kwargs):
        if config is None:
            raise ValueError("SSQ-FMT 模型需要配置对象")
        ssq_cfg = getattr(getattr(config, "model", None), "ssq_fmt", None)
        density_mode = str(getattr(ssq_cfg, "density_output_mode", ""))
        composition = getattr(ssq_cfg, "composition", None)
        composition_mode = str(getattr(composition, "mode", ""))
        cls = SSQFMTPatch2 if "view_complementary" in {
            density_mode,
            composition_mode,
        } else SSQFMT
        return cls(config=config)

    @staticmethod
    def create_uhr_deepfmt_model(config: Optional[Any] = None, **kwargs):
        from .models.uhr_deepfmt import UHRDeepFMT3DUNet
        if config is None:
            raise ValueError("UHR-DeepFMT 模型需要配置对象")
        return UHRDeepFMT3DUNet(config=config)

    @staticmethod
    def create_vox_dmrn_model(config: Optional[Any] = None, **kwargs):
        from .models.vox_dmrn import VoxDMRN
        if config is None:
            raise ValueError("VoxDMRN 模型需要配置对象")
        return VoxDMRN(config=config)

    @staticmethod
    def create_voxel_baseline_model(
        model_type: str, config: Optional[Any] = None, **kwargs
    ):
        if config is None:
            raise ValueError("Voxel baseline 模型需要配置对象")
        from .models import voxel_baselines as baselines
        names = {
            "map_pgan": "MAPPGANAdapted",
            "d2_recst": "D2RecSTAdapted",
            "two_stage_deepfmt": "TwoStageDeepFMTAdapted",
            "fmt_reconnet": "FMTReconNetAdapted",
            "pgdpnn": "PGDPNNAdapted",
            "pgd_pnn": "PGDPNNAdapted",
            "pgd-pnn": "PGDPNNAdapted",
            "dspgn": "DSPGNAdapted",
            "fem2vox_unet": "FEM2VoxUNet",
            "stage1_unet": "FEM2VoxUNet",
            "stage1_interpolation": "Stage1InterpolationBaseline",
            "cnn3d_baseline": "CNN3DBaseline",
            "transunet3d_baseline": "TransUNet3DBaseline",
        }
        cls = getattr(baselines, names.get(model_type, "GenericVoxelBaseline"))
        return cls(config)

    @staticmethod
    def create_fem_baseline_model(
        model_type: str, config: Optional[Any] = None, **kwargs
    ):
        if config is None:
            raise ValueError("FEM baseline 模型需要配置对象")
        from .models import fem_baselines as baselines
        names = {
            "fem_coarse": "FEMCoarseBaseline",
            "fem_to_voxel": "FEMToVoxelBaseline",
            "stage1_fem": "FEMCoarseBaseline",
            "stage1_to_voxel": "FEMToVoxelBaseline",
            "tikhonov_fem": "TikhonovFEM",
            "l1_fem": "L1FEM",
            "elasticnet_fem": "ElasticNetFEM",
            "fista_fem": "FISTAFEM",
            "stomp_fem": "StOMPFEM",
            "gaicn": "GAICNLikeFEM",
        }
        return getattr(baselines, names[model_type])(config)

    @staticmethod
    def create_model(model_type: str, config: Optional[Any] = None, **kwargs):
        if config is None:
            raise ValueError("模型创建需要配置对象")
        model_type = model_type.lower()
        if model_type == "ssq_fmt":
            return ModelFactory.create_ssq_fmt_model(config, **kwargs)
        if model_type in {
            "gisc_fmt", "minr_fmt", "point_cqr", "fixed_footprint_cqr",
            "depth_footprint_cqr", "unconstrained_adaptive_cqr",
        }:
            return ModelFactory.create_gisc_fmt_model(config, **kwargs)
        if model_type == "uhr_deepfmt":
            return ModelFactory.create_uhr_deepfmt_model(config, **kwargs)
        if model_type == "vox_dmrn":
            return ModelFactory.create_vox_dmrn_model(config, **kwargs)
        if model_type == "pah2t_former":
            from .models.pah2t_former import PAH2TFormer
            return PAH2TFormer(config)
        voxel = {
            "map_pgan", "d2_recst", "two_stage_deepfmt", "fmt_reconnet",
            "pgdpnn", "pgd_pnn", "pgd-pnn", "dspgn", "fem2vox_unet",
            "stage1_unet", "stage1_interpolation", "cnn3d_baseline",
            "transunet3d_baseline",
        }
        if model_type in voxel:
            return ModelFactory.create_voxel_baseline_model(model_type, config, **kwargs)
        fem = {
            "fem_coarse", "fem_to_voxel", "stage1_fem", "stage1_to_voxel",
            "tikhonov_fem", "l1_fem", "elasticnet_fem", "fista_fem",
            "stomp_fem", "gaicn",
        }
        if model_type in fem:
            return ModelFactory.create_fem_baseline_model(model_type, config, **kwargs)
        raise ValueError(f"Unknown model type: {model_type}")
