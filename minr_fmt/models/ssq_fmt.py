"""SSQ-FMT final-method point model.

The model file is intentionally thin: it builds modules, executes the formal
candidate-conditioned scalar-composition information flow, and returns density,
auxiliary tensors, and diagnostics. Implementation details live in
``minr_fmt.network.ssq_*`` modules.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from minr_fmt.models.gisc_multisource import GISCFMT
from minr_fmt.network.a3v2_routing import BoundedHypothesisViewRouting
from minr_fmt.network.complementary_aggregation import ComplementaryAggregation
from minr_fmt.network.diverse_candidate_constructor import DiverseCandidateConstructor
from minr_fmt.network.ssq_candidates import (
    MeasurementDerivedCandidateBuilder,
    attach_candidate_detector_priors,
)
from minr_fmt.network.ssq_decoder import (
    CandidateDensityDecoder,
    CandidateDensityResidualDecoder,
    CandidateFieldCalibrator,
    CandidateMorphologyDecoder,
    CompensationDensityDecoder,
    CompensationMorphologyDecoder,
    FourierPositionEncoding,
    SharedDensityLogitDecoder,
    SourceHypothesisResidualDecoder,
)
from minr_fmt.network.ssq_encoder import (
    ShallowSurfaceEncoder,
    SharedResidualUNetPyramidEncoder,
)
from minr_fmt.network.ssq_fusion import (
    CandidateAssignmentHead,
    CandidateSpecificViewFusion,
    CandidateViewEncoder,
    SourceHypothesisQuotientAggregator,
)
from minr_fmt.network.ssq_geometry import GeometryQueryMapper, infer_detector_margin_map
from minr_fmt.network.ssq_sampler import QueryDependentSurfaceSampler, local_template
from minr_fmt.network.unified_density_decoder import UnifiedDensityDecoder
from minr_fmt.network.view_candidate_evidence import ViewCandidateEvidence
from minr_fmt.network.view_separability import ViewSeparability


def _to_container(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)
    if isinstance(cfg, dict):
        return cfg
    return {}


def _cfg_get(cfg: dict[str, Any], path: str, default: Any) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


_ENCODER_PROFILES = {
    "small": {
        "stem_channels": 16,
        "stage_channels": [24, 48, 96, 128],
        "stage_blocks": [1, 1, 2, 2],
        "pyramid_channels": 32,
        "output_channels": 64,
    },
    "efficient": {
        "stem_channels": 24,
        "stage_channels": [32, 64, 128, 192],
        "stage_blocks": [1, 2, 2, 2],
        "pyramid_channels": 48,
        "output_channels": 64,
    },
    "base": {
        "stem_channels": 32,
        "stage_channels": [48, 96, 192, 256],
        "stage_blocks": [2, 2, 3, 3],
        "pyramid_channels": 64,
        "output_channels": 96,
    },
}


def compose_quotient_residual_density(
    shared_logit: torch.Tensor,
    delta_logit: torch.Tensor,
    relative_query: torch.Tensor,
    candidate_scores: torch.Tensor,
    candidate_support: torch.Tensor,
    quotient_dispersion: torch.Tensor,
    candidate_valid: torch.Tensor,
    tau_u: float,
) -> dict[str, torch.Tensor]:
    """Apply the permutation-invariant SHQ bounded residual composition."""
    if delta_logit.shape[2] == 0:
        empty = shared_logit.new_zeros((*shared_logit.shape[:2], 0))
        return {
            "density": torch.sigmoid(shared_logit),
            "residual_correction": torch.zeros_like(shared_logit),
            "proposal_gate": shared_logit.new_zeros(shared_logit.shape[:2]),
            "alpha": empty,
            "applicability": empty,
        }
    spatial_prior = torch.exp(-0.5 * relative_query.square().sum(dim=-1))
    applicability = (
        candidate_valid[:, None].to(delta_logit.dtype)
        * candidate_scores[:, None].detach().to(delta_logit.dtype).clamp(0.0, 1.0)
        * spatial_prior.clamp(0.0, 1.0)
        * candidate_support.clamp(0.0, 1.0)
        * torch.exp(-quotient_dispersion.clamp_min(0.0) / max(float(tau_u), 1.0e-6))
    )
    applicability = torch.nan_to_num(applicability, nan=0.0, posinf=0.0).clamp(0.0, 1.0)
    applicability_sum = applicability.sum(dim=-1, keepdim=True)
    alpha = applicability / applicability_sum.clamp_min(1.0e-8)
    proposal_gate = 1.0 - torch.exp(-applicability_sum.squeeze(-1))
    residual_correction = proposal_gate[..., None] * (alpha[..., None] * delta_logit).sum(dim=2)
    return {
        "density": torch.sigmoid(shared_logit + residual_correction),
        "residual_correction": residual_correction,
        "proposal_gate": proposal_gate,
        "alpha": alpha,
        "applicability": applicability,
    }


class SurfaceMeasurementNormalizer(nn.Module):
    """Normalize valid detector measurements with an explicit scaling policy."""

    def __init__(
        self,
        percentile: float = 99.9,
        eps: float = 1e-6,
        mode: str = "sample_percentile",
    ):
        super().__init__()
        self.percentile = float(percentile)
        self.eps = float(eps)
        self.mode = str(mode)
        if self.mode not in {"sample_percentile", "per_view_max"}:
            raise ValueError(
                "surface normalization mode must be 'sample_percentile' or 'per_view_max'"
            )

    def forward(
        self, measurements: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if measurements.dim() != 5:
            raise ValueError(
                f"surface measurements must be [B,V,C,H,W], got {tuple(measurements.shape)}"
            )
        y = torch.nan_to_num(measurements.float(), nan=0.0, posinf=0.0, neginf=0.0)
        y = y.clamp_min(0.0)
        b = y.shape[0]
        if valid_mask is None:
            mask = torch.isfinite(measurements).all(dim=2) & (y.abs().sum(dim=2) > 0.0)
        else:
            mask = valid_mask.squeeze(2) if valid_mask.dim() == 5 else valid_mask
            mask = mask.to(device=y.device, dtype=torch.bool)
        if self.mode == "per_view_max":
            masked = torch.where(mask[:, :, None], y, torch.zeros_like(y))
            scale = masked.amax(dim=(2, 3, 4)).clamp_min(self.eps)
            has_valid = mask.flatten(start_dim=2).any(dim=-1)
            scale = torch.where(has_valid, scale, torch.ones_like(scale))
            y_norm = (y / scale[:, :, None, None, None]).clamp(0.0, 1.0)
        else:
            scale = torch.empty((b,), dtype=y.dtype, device=y.device)
            flat_y = y.amax(dim=2).reshape(b, -1)
            flat_m = mask.reshape(b, -1)
            q = max(0.0, min(1.0, self.percentile / 100.0))
            for idx in range(b):
                vals = flat_y[idx][flat_m[idx]]
                scale[idx] = torch.quantile(vals, q).clamp_min(self.eps) if vals.numel() else 1.0
            y_norm = (y / scale[:, None, None, None, None]).clamp(0.0, 1.0)
        y_norm = torch.where(mask[:, :, None], y_norm, torch.zeros_like(y_norm))
        return y_norm, scale


class SSQFMT(nn.Module):
    """Final SSQ-FMT query model returning probability-domain density."""

    output_type = "point_probability"

    def __init__(self, config: Any):
        super().__init__()
        root = _to_container(config)
        model_cfg = root.get("model", root)
        ssq = model_cfg.get("ssq_fmt", {})
        requested_composition_mode = str(_cfg_get(ssq, "composition.mode", "legacy_branch_mixture"))
        requested_density_mode = str(
            _cfg_get(ssq, "density_output_mode", "candidate_scalar_composition")
        )
        requested_view_complementary = (
            requested_composition_mode == "view_complementary"
            or requested_density_mode == "view_complementary"
        )
        data = root.get("data", {})
        view_angles = [int(v) for v in data.get("view_angles", [-90, -60, -30, 0, 30, 60, 90])]
        in_channels = int(model_cfg.get("in_channels", ssq.get("in_channels", 1)))
        hidden_dim = int(_cfg_get(ssq, "hidden_dim", 128))
        self.trunk_size_mm = tuple(
            float(x) for x in _cfg_get(ssq, "candidates.trunk_size_mm", [38.0, 40.0, 20.8])
        )
        self.position_encoding = FourierPositionEncoding(
            num_frequencies=int(_cfg_get(ssq, "fourier.num_frequencies", 4))
        )
        normalization_mode = str(_cfg_get(ssq, "normalization.mode", "per_view_max"))
        self.normalizer = SurfaceMeasurementNormalizer(
            percentile=float(_cfg_get(ssq, "normalization.percentile", 99.9)),
            eps=float(_cfg_get(ssq, "normalization.eps", 1e-6)),
            mode=normalization_mode,
        )
        self.candidate_normalizer = SurfaceMeasurementNormalizer(
            percentile=float(_cfg_get(ssq, "normalization.candidate_percentile", 99.9)),
            eps=float(_cfg_get(ssq, "normalization.eps", 1e-6)),
            mode="sample_percentile",
        )
        geom = ssq.get("geometry", {})
        footprint_cfg = ssq.get("footprint", {})
        detector = geom.get(
            "detector_resolution", model_cfg.get("geometry", {}).get("detector_size", [256, 256])
        )
        self.geometry_mapper = GeometryQueryMapper(
            view_angles=view_angles,
            camera_distance_mm=float(
                geom.get(
                    "camera_distance_mm",
                    model_cfg.get("geometry", {}).get("camera_distance", 200.0),
                )
            ),
            fov_mm=float(geom.get("fov_mm", 80.0)),
            detector_resolution=(int(detector[0]), int(detector[1])),
            volume_center_world=tuple(geom.get("volume_center_world", [19.0, 20.0, 10.4])),
            fd_step_mm=float(geom.get("jacobian_fd_step_mm", 0.2)),
            path_max_mm=float(footprint_cfg.get("path_max_mm", 20.0)),
            path_sign=str(
                geom.get("depth_path_sign", footprint_cfg.get("path_sign", "query_minus_surface"))
            ),
            depth_valid_weight_min=float(geom.get("depth_valid_weight_min", 1.0e-4)),
        )

        enc_cfg = ssq.get("encoder", {})
        memory_cfg = ssq.get("memory", {})
        enc_type = str(enc_cfg.get("type", "residual_unet_pyramid_fusion"))
        self.resolved_architecture: dict[str, Any] = {}
        if enc_type in {"shallow", "shallow_encoder_ablation"}:
            feature_dim = int(enc_cfg.get("output_channels", enc_cfg.get("feature_dim", 48)))
            self.surface_encoder = ShallowSurfaceEncoder(
                in_channels=in_channels,
                channels=int(enc_cfg.get("channels", 32)),
                out_channels=feature_dim,
            )
            self.resolved_architecture = {
                "encoder_type": enc_type,
                "profile": "shallow",
                "output_channels": feature_dim,
            }
        else:
            profile = str(enc_cfg.get("profile", "base")).lower()
            if profile not in _ENCODER_PROFILES:
                raise ValueError(
                    f"Unknown SSQ encoder profile '{profile}'. "
                    f"Expected one of {sorted(_ENCODER_PROFILES)}."
                )
            resolved = dict(_ENCODER_PROFILES[profile])
            for key in (
                "stem_channels",
                "stage_channels",
                "stage_blocks",
                "pyramid_channels",
                "output_channels",
            ):
                if key in enc_cfg:
                    resolved[key] = enc_cfg[key]
            stage_channels = tuple(resolved["stage_channels"])
            stage_blocks = tuple(resolved["stage_blocks"])
            stem_channels = int(resolved["stem_channels"])
            feature_dim = int(resolved["output_channels"])
            self.surface_encoder = SharedResidualUNetPyramidEncoder(
                in_channels=in_channels,
                stem_channels=stem_channels,
                stage_channels=stage_channels,  # type: ignore[arg-type]
                stage_blocks=stage_blocks,  # type: ignore[arg-type]
                pyramid_channels=int(resolved["pyramid_channels"]),
                output_channels=feature_dim,
                dropout=float(enc_cfg.get("dropout", 0.0)),
                use_pyramid_fusion=enc_type
                not in {"residual_unet_no_pyramid", "residual_unet_without_pyramid"},
                checkpoint_encoder=bool(_cfg_get(ssq, "memory.checkpoint_encoder", True)),
            )
            self.resolved_architecture = {
                "encoder_type": enc_type,
                "profile": profile,
                "stem_channels": stem_channels,
                "stage_channels": list(stage_channels),
                "stage_blocks": list(stage_blocks),
                "pyramid_channels": int(resolved["pyramid_channels"]),
                "output_channels": feature_dim,
                "checkpoint_encoder": bool(_cfg_get(ssq, "memory.checkpoint_encoder", True)),
            }

        offsets = local_template(
            str(_cfg_get(ssq, "local_sampling.template", "grid3x3")),
            _cfg_get(ssq, "local_sampling.offsets_px", None),
        )
        sample_feature_dim = int(_cfg_get(ssq, "local_sampling.sample_feature_dim", 128))
        self.surface_sampler = QueryDependentSurfaceSampler(
            offsets,
            feature_channels=feature_dim,
            sample_feature_dim=sample_feature_dim,
            hidden_dim=hidden_dim,
            sigma_min_px=float(
                footprint_cfg.get("sigma_min_px", footprint_cfg.get("scale_min_px", 1.0))
            ),
            sigma_max_px=float(
                footprint_cfg.get("sigma_max_px", footprint_cfg.get("scale_max_px", 8.0))
            ),
            alpha_xi=float(footprint_cfg.get("alpha_xi", 0.5)),
            alpha_beta=float(footprint_cfg.get("alpha_beta", 0.25)),
            delta_h_max=float(footprint_cfg.get("delta_h_max", 0.25)),
            boundary_margin_radius_px=float(footprint_cfg.get("boundary_margin_radius_px", 8.0)),
            mode=str(footprint_cfg.get("mode", "query_dependent")),
            fixed_sigma=float(footprint_cfg.get("fixed_sigma", 2.0)),
        )

        cand = ssq.get("candidates", {})
        self._candidate_cfg = dict(cand)
        self.candidate_builder = MeasurementDerivedCandidateBuilder(
            mmax=int(cand.get("mmax", 5)),
            threshold=float(cand.get("threshold", 0.05)),
            scale_min_mm=float(cand.get("scale_min_mm", 1.0)),
            scale_max_mm=float(cand.get("scale_max_mm", 12.0)),
            trunk_size_mm=tuple(cand.get("trunk_size_mm", [38.0, 40.0, 20.8])),
        )
        self.candidate_router = __import__(
            "minr_fmt.network.ssq_routing", fromlist=["CandidateSurfaceRouter"]
        ).CandidateSurfaceRouter(
            sample_feature_dim,
            hidden_dim=hidden_dim,
            tau_K=float(_cfg_get(ssq, "routing.tau_K", _cfg_get(ssq, "routing.temperature", 1.0))),
            delta_q=float(_cfg_get(ssq, "routing.delta_q", 1.0)),
            k_min=float(_cfg_get(ssq, "routing.k_min", 1e-6)),
            mode=str(_cfg_get(ssq, "routing.mode", "pre_aggregation")),
            reliability_mode=str(_cfg_get(ssq, "reliability.mode", "evidence")),
            query_chunk_size=int(_cfg_get(ssq, "routing.query_chunk_size", 4096)),
            checkpoint_routing=bool(memory_cfg.get("checkpoint_routing", False)),
            measurement_consistency_temperature=float(
                _cfg_get(ssq, "routing.measurement_consistency_temperature", 0.15)
            ),
            compensation_routing_mode=str(
                _cfg_get(ssq, "routing.compensation_routing_mode", "joint")
            ),
            sample_embedding_dim=int(_cfg_get(ssq, "routing.sample_embedding_dim", 64)),
            candidate_embedding_dim=int(_cfg_get(ssq, "routing.candidate_embedding_dim", 16)),
            query_coordinate_scale_mm=float(
                _cfg_get(ssq, "routing.query_coordinate_scale_mm", 40.0)
            ),
        )
        self.measurement_consistency_enabled = (
            float(_cfg_get(ssq, "routing.measurement_consistency_logit_weight", 0.0)) != 0.0
        )
        rep_cfg = ssq.get("representation", {})
        representation_dim = int(rep_cfg.get("hidden_dim", hidden_dim))
        self.view_encoder = CandidateViewEncoder(
            sample_feature_dim,
            representation_dim,
            blocks=int(rep_cfg.get("blocks", 3)),
            dropout=float(rep_cfg.get("dropout", 0.0)),
            query_chunk_size=int(rep_cfg.get("query_chunk_size", 4096)),
            checkpoint_representation=bool(memory_cfg.get("checkpoint_representation", False)),
        )
        self.view_fusion = CandidateSpecificViewFusion(
            representation_dim,
            hidden_dim,
            mode=str(_cfg_get(ssq, "fusion.mode", "candidate_specific")),
            query_chunk_size=int(_cfg_get(ssq, "fusion.query_chunk_size", 4096)),
            checkpoint_fusion=bool(memory_cfg.get("checkpoint_fusion", False)),
        )
        composition_cfg = ssq.get("composition", {})
        self.composition_mode = str(composition_cfg.get("mode", "legacy_branch_mixture"))
        if self.composition_mode not in {
            "legacy_branch_mixture",
            "quotient_residual",
            "view_complementary",
        }:
            raise ValueError(
                "composition.mode must be 'legacy_branch_mixture', 'quotient_residual', "
                "or 'view_complementary'"
            )
        quotient_cfg = ssq.get("quotient", {})
        self.quotient_aggregator = SourceHypothesisQuotientAggregator(
            representation_dim,
            hidden_dim=int(quotient_cfg.get("reliability_hidden_dim", 64)),
            temperature=float(quotient_cfg.get("temperature", 1.0)),
            delta_r_max=float(quotient_cfg.get("delta_r_max", 0.25)),
            a_beta=float(quotient_cfg.get("a_beta", 1.0)),
            a_xi=float(quotient_cfg.get("a_xi", 1.0)),
            a_sigma=float(quotient_cfg.get("a_sigma", 0.5)),
            a_center=float(quotient_cfg.get("a_center", 0.25)),
        )
        self.delta_l_max = float(composition_cfg.get("delta_l_max", 3.0))
        self.tau_u = float(composition_cfg.get("tau_u", 1.0))
        consistency_cfg = quotient_cfg.get("subset_consistency", {})
        self.quotient_consistency_enabled = bool(consistency_cfg.get("enabled", False))
        self.quotient_consistency_seed = int(consistency_cfg.get("seed", 20260628))
        self.quotient_consistency_min_support = float(consistency_cfg.get("min_support", 1.0e-4))
        self.assignment_head = CandidateAssignmentHead(
            representation_dim,
            hidden_dim,
            p_min=float(_cfg_get(ssq, "routing.p_min", 1e-4)),
            support_center=float(_cfg_get(ssq, "routing.support_center", 0.10)),
            support_temperature=float(_cfg_get(ssq, "routing.support_temperature", 0.10)),
            support_logit_weight=float(_cfg_get(ssq, "routing.support_logit_weight", 1.0)),
            tau0=float(_cfg_get(ssq, "routing.tau0", 0.25)),
            tau_p=float(_cfg_get(ssq, "routing.tau_p", 1.0)),
            measurement_consistency_logit_weight=float(
                _cfg_get(ssq, "routing.measurement_consistency_logit_weight", 0.0)
            ),
            active_candidate_top_k=int(_cfg_get(ssq, "routing.active_candidate_top_k", 0)),
            occupancy_evidence_weight=float(
                _cfg_get(ssq, "routing.occupancy_evidence_weight", 0.0)
            ),
            occupancy_evidence_floor=float(_cfg_get(ssq, "routing.occupancy_evidence_floor", 0.05)),
        )
        positive_ratio = float(_cfg_get(ssq, "decoder.positive_ratio_init", 0.03))
        decoder_chunk_size = int(_cfg_get(ssq, "decoder.query_chunk_size", 4096))
        self.compensation_density_decoder = CompensationDensityDecoder(
            representation_dim,
            self.position_encoding.out_dim,
            hidden_dim,
            positive_ratio,
            query_chunk_size=decoder_chunk_size,
            checkpoint_decoder=bool(memory_cfg.get("checkpoint_decoder", False)),
        )
        self.shared_density_logit_decoder = SharedDensityLogitDecoder(
            representation_dim,
            self.position_encoding.out_dim,
            hidden_dim,
            positive_ratio,
            query_chunk_size=decoder_chunk_size,
            checkpoint_decoder=bool(memory_cfg.get("checkpoint_decoder", False)),
        )
        self.source_hypothesis_residual_decoder = (
            None
            if requested_view_complementary
            else SourceHypothesisResidualDecoder(
                representation_dim * 2 + 6,
                self.position_encoding.out_dim,
                hidden_dim,
                positive_ratio,
                query_chunk_size=decoder_chunk_size,
                checkpoint_decoder=bool(memory_cfg.get("checkpoint_decoder", False)),
            )
        )
        view_cfg = ssq.get("view_complementary", {})
        self.view_complementary_ablation = str(view_cfg.get("ablation", "full"))
        self.hypothesis_source = str(view_cfg.get("hypothesis_source", "learned"))
        if self.hypothesis_source not in {"learned", "gt"}:
            raise ValueError("view_complementary.hypothesis_source must be learned or gt")
        self.hypothesis_aggregation_strategy = str(
            view_cfg.get("aggregation_strategy", "legacy")
        )
        self.view_candidate_loss_weights = {
            "candidate_covariance_loss": float(view_cfg.get("lambda_cov", 0.1)),
            "candidate_center_loss": float(view_cfg.get("lambda_center", 1.0)),
            "candidate_existence_loss": float(view_cfg.get("lambda_exist", 0.5)),
            "candidate_coverage_loss": float(view_cfg.get("lambda_cover", 0.5)),
            "candidate_duplicate_loss": float(view_cfg.get("lambda_dup", 0.1)),
        }
        self.view_separability_mode = str(view_cfg.get("separability_mode", "geometry_measurement"))
        self.view_training_phase = str(view_cfg.get("training_phase", "phase_a"))
        self.view_training_epoch = 0
        self.view_training_step = 0
        self.context_warmup_steps = int(view_cfg.get("context_warmup_steps", 50))
        self.context_warmup_enabled = bool(view_cfg.get("context_warmup_enabled", True))
        grid_cfg = view_cfg.get("hypothesis_grid", {})
        self.hypothesis_grid_enabled = bool(grid_cfg.get("enabled", False))
        self.hypothesis_grid_spacing_mm = float(grid_cfg.get("spacing_mm", 3.0))
        self.hypothesis_grid_chunk_size = int(grid_cfg.get("chunk_size", 4096))
        grid_axes = [
            torch.arange(0.0, size, self.hypothesis_grid_spacing_mm) for size in self.trunk_size_mm
        ]
        grid = torch.stack(torch.meshgrid(*grid_axes, indexing="ij"), dim=-1)
        self.hypothesis_grid_shape = tuple(int(x) for x in grid.shape[:3])
        self.register_buffer("hypothesis_grid_points_mm", grid.reshape(-1, 3), persistent=False)
        routing_cfg = view_cfg.get("routing", {})
        self.a3v2_routing_enabled = bool(routing_cfg.get("enabled", False))
        self.routing_counterfactual_diagnostics = bool(
            routing_cfg.get("counterfactual_diagnostics", False)
        )
        self.routing_oracle_gain = float(routing_cfg.get("oracle_gain", 0.5))
        self.continuous_applicability_enabled = bool(
            view_cfg.get("continuous_applicability", False)
        )
        self.candidate_hidden_injection_enabled = bool(
            view_cfg.get("candidate_hidden_injection", True)
        )
        self.bounded_view_routing = (
            BoundedHypothesisViewRouting(
                delta_logit_max=float(routing_cfg.get("delta_logit_max", 1.0)),
                zero_init=bool(routing_cfg.get("zero_init", True)),
            )
            if self.a3v2_routing_enabled
            else None
        )
        self.evidence_eta0 = float(view_cfg.get("evidence_eta0", 0.25))
        self.evidence_sigma_mm = float(view_cfg.get("evidence_sigma_mm", 3.0))
        self.lambda_separability_measurement = float(
            view_cfg.get("lambda_separability_measurement", 0.05)
        )
        self.phase_a_aux_loss_scale = float(view_cfg.get("phase_a_aux_loss_scale", 1.0))
        self.phase_b_aux_loss_scale = float(view_cfg.get("phase_b_aux_loss_scale", 1.0))
        self.full_aux_loss_scale = float(view_cfg.get("full_aux_loss_scale", 1.0))
        self.phase_b_separability_loss_scale = float(
            view_cfg.get("phase_b_separability_loss_scale", 1.0)
        )
        self.full_separability_loss_scale = float(
            view_cfg.get("full_separability_loss_scale", 1.0)
        )
        evidence_nms_cfg = view_cfg.get("evidence_nms", {})
        self.view_candidate_evidence = ViewCandidateEvidence(
            sample_feature_dim,
            hidden_dim=int(view_cfg.get("hidden_dim", hidden_dim)),
            descriptor_dim=int(view_cfg.get("descriptor_dim", 32)),
            delta_max_mm=float(view_cfg.get("delta_max_mm", 3.0)),
            topk_per_view=int(view_cfg.get("topk_per_view", 8)),
            nms_radius_mm=float(view_cfg.get("nms_radius_mm", 3.0)),
            exact_nms_threshold=int(evidence_nms_cfg.get("exact_threshold", 4096)),
            pre_nms_topk=int(evidence_nms_cfg.get("pre_nms_topk", 2048)),
            pre_nms_factor=int(evidence_nms_cfg.get("pre_nms_factor", 64)),
            max_nms_candidates=int(evidence_nms_cfg.get("max_nms_candidates", 4096)),
        )
        self.diverse_candidate_constructor = DiverseCandidateConstructor(
            mmax=int(view_cfg.get("mmax", 5)),
            score_threshold=float(view_cfg.get("score_threshold", 0.05)),
            candidate_conf_threshold=float(view_cfg.get("candidate_conf_threshold", 0.4)),
            sigma_nms_mm=float(view_cfg.get("sigma_nms_mm", 4.0)),
            sigma_assoc_mm=float(view_cfg.get("sigma_assoc_mm", 5.0)),
            sigma_min_mm=float(view_cfg.get("sigma_min_mm", 1.0)),
            sigma_max_mm=float(view_cfg.get("sigma_max_mm", 8.0)),
        )
        self.view_separability = ViewSeparability(
            sample_feature_dim,
            hidden_dim=int(view_cfg.get("separability_hidden_dim", 64)),
            delta_max=float(view_cfg.get("delta_s", 0.25)),
        )
        self.complementary_aggregation = ComplementaryAggregation(
            sample_feature_dim,
            representation_dim,
            epsilon_s=float(view_cfg.get("epsilon_s", 0.1)),
        )
        self.unified_density_decoder = UnifiedDensityDecoder(
            representation_dim,
            self.position_encoding.out_dim,
            hidden_dim=hidden_dim,
            fusion_mode=str(view_cfg.get("decoder_fusion_mode", "additive")),
        )
        candidate_field_mode = str(_cfg_get(ssq, "candidate_field.mode", "independent"))
        if candidate_field_mode not in {"independent", "shared_residual"}:
            raise ValueError("candidate_field.mode must be 'independent' or 'shared_residual'")
        self.candidate_field_mode = candidate_field_mode
        candidate_decoder_cls = (
            CandidateDensityResidualDecoder
            if candidate_field_mode == "shared_residual"
            else CandidateDensityDecoder
        )
        self.candidate_density_decoder = candidate_decoder_cls(
            representation_dim,
            self.position_encoding.out_dim,
            hidden_dim,
            positive_ratio,
            query_chunk_size=decoder_chunk_size,
            checkpoint_decoder=bool(memory_cfg.get("checkpoint_decoder", False)),
        )
        context_cfg = ssq.get("candidate_context", {})
        self.candidate_context_enabled = bool(context_cfg.get("enabled", False))
        if self.candidate_context_enabled:
            context_hidden = int(context_cfg.get("hidden_dim", hidden_dim))
            self.candidate_context_encoder = nn.Sequential(
                nn.Linear(11, context_hidden),
                nn.SiLU(),
                nn.Linear(context_hidden, representation_dim),
            )
            nn.init.zeros_(self.candidate_context_encoder[-1].weight)
            nn.init.zeros_(self.candidate_context_encoder[-1].bias)
            self.candidate_field_calibrator = CandidateFieldCalibrator(
                representation_dim,
                hidden_dim=int(context_cfg.get("field_hidden_dim", 64)),
            )
        else:
            self.candidate_context_encoder = None
            self.candidate_field_calibrator = None
        self.compensation_morphology_decoder = CompensationMorphologyDecoder(
            representation_dim, self.position_encoding.out_dim, hidden_dim
        )
        self.candidate_morphology_decoder = CandidateMorphologyDecoder(
            representation_dim, self.position_encoding.out_dim, hidden_dim
        )
        self.lambda_sdf = float(
            _cfg_get(root, "loss.lambda_sdf", _cfg_get(ssq, "sdf.lambda_sdf", 0.0))
        )
        self.candidate_prior_weight = float(_cfg_get(ssq, "candidates.prior_density_weight", 0.0))
        self.candidate_support_envelope_power = float(
            _cfg_get(ssq, "candidate_field.support_envelope_power", 0.0)
        )
        self.density_output_mode = str(
            _cfg_get(ssq, "density_output_mode", "candidate_scalar_composition")
        )
        if self.density_output_mode not in {
            "candidate_scalar_composition",
            "compensation_direct_probe",
            "e15_backbone_baseline",
            "view_complementary",
        }:
            raise ValueError(
                "SSQ-FMT density_output_mode must be 'candidate_scalar_composition' "
                "'compensation_direct_probe', 'e15_backbone_baseline', or "
                "'view_complementary'."
            )
        self.query_density_backbone = (
            GISCFMT(config) if self.density_output_mode == "e15_backbone_baseline" else None
        )
        if requested_view_complementary:
            legacy_only = (
                self.candidate_router,
                self.view_encoder,
                self.view_fusion,
                self.quotient_aggregator,
                self.assignment_head,
                self.compensation_density_decoder,
                self.shared_density_logit_decoder,
                self.candidate_density_decoder,
                self.compensation_morphology_decoder,
                self.candidate_morphology_decoder,
            )
            for module in legacy_only:
                for parameter in module.parameters():
                    parameter.requires_grad = False
        data_cfg = _to_container(root.get("data", {}))
        default_angles = [-90, -60, -30, 0, 30, 60, 90]
        self.view_angles = [int(v) for v in data_cfg.get("view_angles", default_angles)]
        self.resolved_architecture.update(
            {
                "sample_feature_dim": sample_feature_dim,
                "representation_hidden_dim": representation_dim,
                "routing_query_chunk_size": int(_cfg_get(ssq, "routing.query_chunk_size", 4096)),
                "representation_query_chunk_size": int(rep_cfg.get("query_chunk_size", 4096)),
                "fusion_query_chunk_size": int(_cfg_get(ssq, "fusion.query_chunk_size", 4096)),
                "decoder_query_chunk_size": decoder_chunk_size,
                "checkpoint_routing": bool(memory_cfg.get("checkpoint_routing", False)),
                "checkpoint_representation": bool(
                    memory_cfg.get("checkpoint_representation", False)
                ),
                "checkpoint_fusion": bool(memory_cfg.get("checkpoint_fusion", False)),
                "checkpoint_decoder": bool(memory_cfg.get("checkpoint_decoder", False)),
                "positive_ratio_init": positive_ratio,
                "density_output_mode": self.density_output_mode,
                "candidate_context_enabled": self.candidate_context_enabled,
                "encoder_normalization": normalization_mode,
                "candidate_normalization": "sample_percentile",
                "candidate_support_envelope_power": self.candidate_support_envelope_power,
                "candidate_field_mode": self.candidate_field_mode,
                "composition_mode": self.composition_mode,
                "delta_l_max": self.delta_l_max,
                "tau_u": self.tau_u,
            }
        )
        if requested_view_complementary:
            self.set_view_training_phase(self.view_training_phase)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        rank_zero_info(
            "SSQ-FMT resolved architecture: "
            f"{self.resolved_architecture}; params={total_params:,}; "
            f"trainable={trainable_params:,}"
        )

    def _surface_pack_to_projection_dict(
        self, surface_measurements: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if surface_measurements.dim() == 5 and surface_measurements.shape[2] == 1:
            surface_measurements = surface_measurements.squeeze(2)
        if surface_measurements.dim() != 4:
            raise ValueError(
                "query_density_backbone expects packed surface measurements with shape "
                f"[B,V,H,W] or [B,V,1,H,W], got {tuple(surface_measurements.shape)}"
            )
        if surface_measurements.shape[1] < len(self.view_angles):
            raise ValueError(
                f"packed surface has {surface_measurements.shape[1]} views, "
                f"but config declares {len(self.view_angles)} view angles"
            )
        return {
            str(angle): surface_measurements[:, idx]
            for idx, angle in enumerate(self.view_angles[: surface_measurements.shape[1]])
        }

    def forward(
        self,
        surface_measurements: torch.Tensor,
        query_coordinates_mm: torch.Tensor,
        *,
        detector_valid_mask: torch.Tensor | None = None,
        depth_maps: torch.Tensor | None = None,
        batch: dict[str, Any] | None = None,
        return_diagnostics: bool = False,
        **_: Any,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if batch is not None:
            detector_valid_mask = (
                detector_valid_mask
                if detector_valid_mask is not None
                else batch.get("detector_valid_mask")
            )
            depth_maps = depth_maps if depth_maps is not None else batch.get("depth_maps")
        if self.query_density_backbone is not None:
            projections = self._surface_pack_to_projection_dict(surface_measurements)
            x3d = batch.get("points") if batch is not None else query_coordinates_mm
            logits, aux_projections = self.query_density_backbone(
                projections,
                x3d,
                points_mm=query_coordinates_mm,
                depth_maps=depth_maps,
            )
            density = torch.sigmoid(logits).clamp(0.0, 1.0)
            aux_outputs: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
                "density_logits": logits,
                "aux_projections": aux_projections,
                "measurement_supported": torch.ones(
                    density.shape[:2], dtype=torch.bool, device=density.device
                ),
                "density_output_mode": torch.tensor(1, device=density.device),
            }
            out: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
                "density": density,
                "aux_outputs": aux_outputs,
            }
            if return_diagnostics:
                out["diagnostics"] = aux_outputs
            return out
        y_norm, norm_scale = self.normalizer(surface_measurements, detector_valid_mask)
        candidate_y_norm, candidate_norm_scale = self.candidate_normalizer(
            surface_measurements, detector_valid_mask
        )
        features = self.surface_encoder(y_norm)
        margin_map = infer_detector_margin_map(batch)
        mapped = self.geometry_mapper(
            query_coordinates_mm,
            depth_maps=depth_maps,
            detector_margin_map=margin_map,
            detector_valid_mask=detector_valid_mask,
        )
        samples = self.surface_sampler(
            features,
            y_norm,
            mapped,
            detector_valid_mask=detector_valid_mask,
            depth_maps=depth_maps,
        )
        if (
            self.density_output_mode == "view_complementary"
            or self.composition_mode == "view_complementary"
        ):
            return self._forward_view_complementary(
                features,
                y_norm,
                samples,
                mapped,
                query_coordinates_mm,
                detector_valid_mask,
                depth_maps,
                norm_scale,
                return_diagnostics,
                batch,
            )
        candidates = self.candidate_builder(candidate_y_norm, batch=batch)
        if candidates["candidate_centers_mm"].shape[1] > 0:
            cand_mapped = self.geometry_mapper(
                candidates["candidate_centers_mm"],
                depth_maps=depth_maps,
                detector_valid_mask=detector_valid_mask,
            )
            candidates = attach_candidate_detector_priors(
                candidates,
                cand_mapped,
                float(self._candidate_cfg.get("detector_scale_min_px", 0.5)),
                float(self._candidate_cfg.get("detector_scale_max_px", 20.0)),
                measurements=candidate_y_norm
                if self.measurement_consistency_enabled or self.candidate_context_enabled
                else None,
            )
        if self.composition_mode == "quotient_residual":
            candidates = {
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in candidates.items()
            }
        router = self.candidate_router(
            samples,
            query_coordinates_mm,
            candidates,
            mapped,
            return_diagnostics=return_diagnostics,
        )
        per_view = self.view_encoder(samples, router, mapped)
        if self.composition_mode == "quotient_residual":
            return self._forward_quotient_residual(
                per_view,
                samples,
                mapped,
                router,
                candidates,
                query_coordinates_mm,
                norm_scale,
                candidate_norm_scale,
                return_diagnostics,
            )
        z, view_weights, Lambda = self.view_fusion(
            per_view, router["r"], mapped["valid_mask"], router["branch_valid"]
        )
        candidate_context = None
        if self.candidate_context_enabled and candidates["candidate_centers_mm"].shape[1] > 0:
            covariance = candidates["candidate_support_covariances_mm"]
            eigen_scales = torch.linalg.eigvalsh(covariance.float()).clamp_min(1.0e-8).sqrt()
            eigen_scales = eigen_scales.to(dtype=z.dtype) / max(
                self._candidate_cfg.get("scale_max_mm", 12.0), 1.0e-6
            )
            context_trunk = torch.tensor(
                self.trunk_size_mm,
                device=z.device,
                dtype=z.dtype,
            )
            center_norm = candidates["candidate_centers_mm"].to(dtype=z.dtype) / context_trunk
            center_measurements = candidates["candidate_center_measurements"].to(dtype=z.dtype)
            center_valid = candidates["candidate_detector_valid_mask"]
            valid_float = center_valid.to(dtype=z.dtype)
            valid_count = valid_float.sum(dim=1).clamp_min(1.0)
            measurement_mean = (center_measurements * valid_float).sum(dim=1) / valid_count
            centered = center_measurements - measurement_mean[:, None]
            measurement_std = ((centered.square() * valid_float).sum(dim=1) / valid_count).sqrt()
            measurement_max = center_measurements.masked_fill(~center_valid, 0.0).amax(dim=1)
            visibility = valid_float.mean(dim=1)
            context_input = torch.cat(
                [
                    center_norm.mul(2.0).sub(1.0),
                    candidates["candidate_scores"][..., None].to(dtype=z.dtype),
                    eigen_scales,
                    measurement_mean[..., None],
                    measurement_std[..., None],
                    measurement_max[..., None],
                    visibility[..., None],
                ],
                dim=-1,
            )
            candidate_context = self.candidate_context_encoder(context_input)
            z = torch.cat([z[:, :, :1], z[:, :, 1:] + candidate_context[:, None]], dim=2)
        router["Lambda"] = Lambda
        router["valid_view_count"] = mapped["valid_mask"].sum(dim=1)
        if candidates["candidate_centers_mm"].shape[1] > 0:
            candidate_visible = candidates["candidate_detector_valid_mask"].any(dim=1)
            router["candidate_geometric_visible"] = candidate_visible[:, None, :].expand(
                -1, query_coordinates_mm.shape[1], -1
            )
        measurement_supported = router["valid_view_count"] > 0

        trunk = torch.tensor(
            self.trunk_size_mm,
            device=query_coordinates_mm.device,
            dtype=query_coordinates_mm.dtype,
        )
        query_norm = (query_coordinates_mm / trunk.clamp_min(1e-6)).mul(2.0).sub(1.0)
        encoded_query = self.position_encoding(query_norm)
        d0 = self.compensation_density_decoder(z[:, :, 0], encoded_query)
        m = z.shape[2] - 1
        if m > 0:
            delta = query_coordinates_mm[:, :, None] - candidates["candidate_centers_mm"][:, None]
            rel = torch.einsum(
                "bnmi,bmij->bnmj",
                delta,
                candidates["candidate_support_inverse_sqrt_mm"],
            )
            encoded_rel = self.position_encoding(rel.clamp(-4.0, 4.0))
            candidate_field = self.candidate_density_decoder(z[:, :, 1:], encoded_rel)
            if self.candidate_field_mode == "shared_residual":
                shared_logit = torch.logit(d0.float().clamp(1.0e-5, 1.0 - 1.0e-5))
                dm = torch.sigmoid(shared_logit[:, :, None] + candidate_field.float()).to(
                    dtype=d0.dtype
                )
            else:
                dm = candidate_field
            if candidate_context is not None and self.candidate_field_calibrator is not None:
                context_expanded = candidate_context[:, None].expand(-1, rel.shape[1], -1, -1)
                field_residual = self.candidate_field_calibrator(context_expanded, rel)
                dm_dtype = dm.dtype
                dm_logit = torch.logit(dm.float().clamp(1.0e-5, 1.0 - 1.0e-5))
                dm = torch.sigmoid(dm_logit + field_residual.float()).to(dtype=dm_dtype)
            candidate_density_pre_envelope = dm
            if self.candidate_support_envelope_power > 0.0:
                support_envelope = (
                    router["p_all"][..., 1:]
                    .clamp(0.0, 1.0)
                    .pow(self.candidate_support_envelope_power)
                )
                dm = dm * support_envelope[..., None]
            dm = torch.where(
                candidates["candidate_valid_mask"][:, None, :, None], dm, torch.zeros_like(dm)
            )
            branch_density = torch.cat([d0[:, :, None], dm], dim=2).clamp(0.0, 1.0)
            branch_density_pre_envelope = torch.cat(
                [d0[:, :, None], candidate_density_pre_envelope], dim=2
            ).clamp(0.0, 1.0)
        else:
            branch_density = d0[:, :, None].clamp(0.0, 1.0)
            branch_density_pre_envelope = branch_density
        prior_pi = self.assignment_head(z, query_coordinates_mm, router, candidates)
        pi = self.assignment_head.apply_occupancy_evidence(prior_pi, branch_density)
        branch_contributions = pi[..., None] * branch_density
        branch_contributions = branch_contributions * measurement_supported[:, :, None, None].to(
            dtype=branch_contributions.dtype
        )
        density = branch_contributions.sum(dim=2).clamp(0.0, 1.0)
        if self.density_output_mode == "compensation_direct_probe":
            density = d0 * measurement_supported[..., None].to(dtype=d0.dtype)
        candidate_prior = torch.zeros_like(density)
        cand_contrib = (
            branch_contributions[:, :, 1:].sum(dim=2) if m > 0 else torch.zeros_like(density)
        )
        comp_contrib = branch_contributions[:, :, :1].sum(dim=2)
        support_f = measurement_supported.to(dtype=density.dtype)
        per_sample_total_mass = (density.squeeze(-1) * support_f).sum(dim=1).clamp_min(1e-8)
        per_sample_branch_mass = (branch_contributions.squeeze(-1) * support_f[:, :, None]).sum(
            dim=1
        )
        valid_branch = torch.ones_like(per_sample_branch_mass, dtype=torch.bool)
        if m > 0:
            valid_branch[:, 1:] = candidates["candidate_valid_mask"]
        utilized = (per_sample_branch_mass / per_sample_total_mass[:, None] > 0.01) & valid_branch
        comp_ratio = (
            (comp_contrib.squeeze(-1) * support_f).sum(dim=1) / per_sample_total_mass
        ).mean()
        cand_ratio = (
            (cand_contrib.squeeze(-1) * support_f).sum(dim=1) / per_sample_total_mass
        ).mean()
        aux_outputs: dict[str, torch.Tensor] = {
            "pi": pi,
            "prior_pi": prior_pi,
            "p_all": router["p_all"],
            "branch_density": branch_density,
            "branch_contributions": branch_contributions,
            "candidate_prior_density": candidate_prior,
            "lightweight_density": density,
            "measurement_supported": measurement_supported,
            "compensation_contribution_ratio": comp_ratio.detach(),
            "candidate_contribution_ratio": cand_ratio.detach(),
            "pi0_mean": pi[..., 0].mean().detach(),
            "candidate_mass_mean": pi[..., 1:].sum(dim=-1).mean().detach()
            if pi.shape[-1] > 1
            else torch.zeros((), device=density.device, dtype=density.dtype),
            "branch_density_mean": branch_density.mean().detach(),
            "branch_density_std": branch_density.std(unbiased=False).detach(),
            "measurement_supported_positive_ratio": measurement_supported.to(dtype=density.dtype)
            .mean()
            .detach(),
            "density_output_mode": torch.tensor(0, device=density.device),
        }
        if self.training:
            aux_outputs["candidate_centers_mm"] = candidates["candidate_centers_mm"]
            aux_outputs["candidate_valid_mask"] = candidates["candidate_valid_mask"]
        branch_sdf = None
        composed_sdf = None
        if (self.lambda_sdf > 0.0 and self.training) or return_diagnostics:
            q0 = self.compensation_morphology_decoder(z[:, :, 0], encoded_query)
            if m > 0:
                qm = self.candidate_morphology_decoder(z[:, :, 1:], encoded_rel)
                qm = torch.where(
                    candidates["candidate_valid_mask"][:, None, :, None],
                    qm,
                    torch.zeros_like(qm),
                )
                branch_sdf = torch.cat([q0[:, :, None], qm], dim=2).clamp(-1.0, 1.0)
            else:
                branch_sdf = q0[:, :, None].clamp(-1.0, 1.0)
            composed_sdf = (pi[..., None] * branch_sdf).sum(dim=2)
            composed_sdf = composed_sdf * measurement_supported[..., None].to(
                dtype=composed_sdf.dtype
            )
            aux_outputs["branch_sdf"] = branch_sdf
            aux_outputs["composed_sdf"] = composed_sdf
            aux_outputs["sdf"] = composed_sdf
        out: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "density": density,
            "aux_outputs": aux_outputs,
        }
        if return_diagnostics:
            pi_entropy = -(pi.clamp_min(1e-8) * pi.clamp_min(1e-8).log()).sum(dim=-1)
            z_norm = torch.nn.functional.normalize(z[:, :, 1:], dim=-1) if m > 0 else z[:, :, 1:]
            pairwise_z_cosine = (
                torch.matmul(z_norm, z_norm.transpose(-1, -2))
                if m > 0
                else torch.empty((*z.shape[:2], 0, 0), device=z.device, dtype=z.dtype)
            )
            branch_prob = branch_contributions[:, :, 1:, 0] if m > 0 else pi[:, :, 1:]
            branch_prob = branch_prob * support_f[:, :, None]
            branch_sum = branch_prob.sum(dim=1)
            pair_intersection = torch.matmul(branch_prob.transpose(1, 2), branch_prob)
            pair_den = branch_sum[:, :, None] + branch_sum[:, None, :]
            pairwise_branch_overlap = (
                (2.0 * pair_intersection + 1.0e-8) / (pair_den + 1.0e-8)
                if m > 0
                else torch.empty((*z.shape[:2], 0, 0), device=z.device, dtype=z.dtype)
            )
            if m > 0:
                eye = torch.eye(m, device=z.device, dtype=torch.bool)[None]
                pairwise_branch_overlap = pairwise_branch_overlap.masked_fill(eye, float("nan"))
            routing_entropy = -(
                router["zeta"].clamp_min(1e-8) * router["zeta"].clamp_min(1e-8).log()
            ).sum(dim=-1)
            diagnostics = {
                "normalization_scale": norm_scale,
                "candidate_normalization_scale": candidate_norm_scale,
                "sigma_f": samples["sigma_f"],
                "sample_coordinates": samples["sample_coordinates"],
                "sample_coordinates_px": samples["sample_coordinates_px"],
                "sample_valid": samples["sample_valid"],
                "query_view_valid": samples["query_view_valid"],
                "A": samples["A"],
                "sample_features": samples["sample_features"],
                "sample_measurements": samples["sample_measurements"],
                "candidate_centers_mm": candidates["candidate_centers_mm"],
                "candidate_scores": candidates["candidate_scores"],
                "candidate_raw_support_scales_mm": candidates["candidate_raw_support_scales_mm"],
                "candidate_support_scales_mm": candidates["candidate_support_scales_mm"],
                "candidate_support_covariances_mm": candidates["candidate_support_covariances_mm"],
                "candidate_raw_scales_mm": candidates["candidate_raw_support_scales_mm"],
                "candidate_scales_mm": candidates["candidate_support_scales_mm"],
                "candidate_valid_mask": candidates["candidate_valid_mask"],
                "candidate_support_scale_low_clip": candidates["candidate_support_scale_low_clip"],
                "candidate_support_scale_high_clip": candidates[
                    "candidate_support_scale_high_clip"
                ],
                "detector_side_path_mm": mapped["detector_side_path_mm"],
                "detector_side_path_proxy": mapped["detector_side_path_proxy"],
                "surface_depth": mapped["surface_depth"],
                "K_all": router["K_all"],
                "zeta": router["zeta"],
                "a": router["a"],
                "e": router["e"],
                "e_sample": router["e_sample"],
                "nu": router["nu"],
                "r": router["r"],
                "p_all": router["p_all"],
                "measurement_consistency": router["measurement_consistency"],
                "view_weights": view_weights,
                "Lambda": Lambda,
                "support_gate": torch.sigmoid(
                    (Lambda[:, :, 1:] - self.assignment_head.support_center)
                    / max(self.assignment_head.support_temperature, 1e-6)
                )
                if Lambda.shape[-1] > 1
                else torch.empty((*Lambda.shape[:2], 0), device=Lambda.device, dtype=Lambda.dtype),
                "pi": pi,
                "measurement_supported": measurement_supported,
                "unsupported_query_ratio": (~measurement_supported).to(dtype=density.dtype).mean(),
                "compensation_contribution_ratio": comp_ratio,
                "candidate_contribution_ratio": cand_ratio,
                "unused_candidate_ratio": (~utilized[:, 1:]).to(dtype=density.dtype).mean()
                if m > 0
                else torch.zeros((), device=density.device),
                "candidate_utilization": utilized[:, 1:].to(dtype=density.dtype).mean()
                if m > 0
                else torch.zeros((), device=density.device),
                "pairwise_z_cosine_similarity": pairwise_z_cosine,
                "pairwise_branch_overlap": pairwise_branch_overlap,
                "pi_entropy": pi_entropy,
                "routing_entropy": routing_entropy,
                "branch_density": branch_density,
                "branch_density_pre_envelope": branch_density_pre_envelope,
                "branch_contributions": branch_contributions,
                "branch_sdf": branch_sdf
                if branch_sdf is not None
                else torch.empty(0, device=density.device),
                "composed_sdf": composed_sdf
                if composed_sdf is not None
                else torch.empty(0, device=density.device),
                "candidate_prior_density": candidate_prior,
                "decoded_density": density,
                "lightweight_density": density,
                "density": density,
            }
            out["diagnostics"] = diagnostics
        return out

    def set_training_epoch(self, epoch: int) -> None:
        self.view_training_epoch = int(epoch)

    def set_training_step(self, step: int) -> None:
        self.view_training_step = int(step)

    def set_view_training_phase(self, phase: str) -> None:
        if phase not in {"phase_a", "phase_b", "full"}:
            raise ValueError(f"unknown view-complementary training phase: {phase}")
        self.view_training_phase = phase

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
        gt_mapped = self.geometry_mapper(
            gt_centers,
            depth_maps=depth_maps,
            detector_valid_mask=detector_valid_mask,
        )
        detector_delta = gt_mapped["grid"][:, :, :, None] - gt_mapped["grid"][:, :, None]
        collision = torch.exp(-detector_delta.square().sum(dim=-1) / 0.02)
        pair_valid = gt_valid[:, None, :, None] & gt_valid[:, None, None, :]
        collision = torch.where(pair_valid, collision, torch.zeros_like(collision))
        eye = torch.eye(gt_valid.shape[1], device=evidence.device, dtype=torch.bool)[None, None]
        max_collision = collision.masked_fill(eye, 0.0).amax(dim=-1)
        separability = 1.0 - max_collision
        amplitude = self.evidence_eta0 + (1.0 - self.evidence_eta0) * separability
        amplitude = amplitude * gt_mapped["valid_mask"].to(amplitude.dtype)
        delta = points_mm[:, None, :, None] - gt_centers[:, None, None]
        variance = torch.diagonal(gt_covariances.float(), dim1=-2, dim2=-1).clamp_min(1.0e-4)
        mahal = (delta.float().square() / variance[:, None, None]).sum(dim=-1)
        target = torch.exp(-0.5 * mahal) * amplitude[:, :, None]
        target = target.masked_fill(~gt_valid[:, None, None], 0.0).amax(dim=-1)
        mask = query_valid.to(evidence.dtype)
        prediction = evidence.float().clamp(1.0e-6, 1.0 - 1.0e-6)
        target = target.float()
        positive_weight = 1.0 + 9.0 * target
        loss = (
            -(target * prediction.log() + (1.0 - target) * (1.0 - prediction).log())
            * positive_weight
            * mask.float()
        )
        return loss.sum() / (positive_weight * mask).sum().clamp_min(1.0)

    def _forward_view_complementary(
        self,
        features: torch.Tensor,
        measurements: torch.Tensor,
        samples: dict[str, torch.Tensor],
        mapped: dict[str, torch.Tensor],
        points_mm: torch.Tensor,
        detector_valid_mask: torch.Tensor | None,
        depth_maps: torch.Tensor | None,
        norm_scale: torch.Tensor,
        return_diagnostics: bool,
        batch: dict[str, Any] | None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Execute the independent-view source-hypothesis information path."""
        query_per_view = (samples["sample_features"] * samples["A"][..., None]).sum(dim=3)
        geometry = torch.stack(
            [
                mapped["detector_side_path_proxy"],
                mapped["boundary_distance"]
                .div(max(self.surface_sampler.boundary_margin_radius_px, 1.0e-6))
                .clamp(0.0, 1.0),
                samples["sigma_f"].div(max(self.surface_sampler.sigma_max_px, 1.0e-6)),
                mapped["grid"].square().sum(dim=-1).sqrt().div(2.0**0.5),
            ],
            dim=-1,
        )
        evidence_points = points_mm
        evidence_per_view = query_per_view
        evidence_geometry = geometry
        evidence_valid = samples["query_view_valid"]
        evidence_grid_shape = None
        if self.hypothesis_grid_enabled:
            evidence_points = self.hypothesis_grid_points_mm.to(
                device=points_mm.device, dtype=points_mm.dtype
            )[None].expand(points_mm.shape[0], -1, -1)
            evidence_mapped = self.geometry_mapper(
                evidence_points,
                depth_maps=depth_maps,
                detector_valid_mask=detector_valid_mask,
            )
            evidence_samples = self.surface_sampler(
                features,
                measurements,
                evidence_mapped,
                detector_valid_mask=detector_valid_mask,
                depth_maps=depth_maps,
            )
            evidence_per_view = (
                evidence_samples["sample_features"] * evidence_samples["A"][..., None]
            ).sum(dim=3)
            evidence_geometry = torch.stack(
                [
                    evidence_mapped["detector_side_path_proxy"],
                    evidence_mapped["boundary_distance"]
                    .div(max(self.surface_sampler.boundary_margin_radius_px, 1.0e-6))
                    .clamp(0.0, 1.0),
                    evidence_samples["sigma_f"].div(max(self.surface_sampler.sigma_max_px, 1.0e-6)),
                    evidence_mapped["grid"].square().sum(dim=-1).sqrt().div(2.0**0.5),
                ],
                dim=-1,
            )
            evidence_valid = evidence_samples["query_view_valid"]
            evidence_grid_shape = self.hypothesis_grid_shape
        proposal = self.view_candidate_evidence(
            evidence_per_view,
            evidence_geometry,
            evidence_points,
            evidence_valid,
            grid_shape=evidence_grid_shape,
        )
        candidates = self.diverse_candidate_constructor(proposal)
        if self.hypothesis_source == "gt":
            if batch is None or not torch.is_tensor(batch.get("gt_component_centers_mm")):
                raise RuntimeError("GT hypothesis study requires component supervision in batch")
            gt_centers = batch["gt_component_centers_mm"].to(points_mm)
            gt_covariance = batch["gt_component_covariances_mm"].to(points_mm)
            gt_valid = batch["gt_component_valid_mask"].to(device=points_mm.device).bool()
            if gt_centers.shape[1] != candidates["candidate_centers_mm"].shape[1]:
                raise RuntimeError("GT and learned hypothesis slot counts must match")
            gt_score = gt_valid.to(points_mm.dtype)
            candidates.update(
                {
                    "candidate_centers_mm": gt_centers,
                    "candidate_covariances_mm": gt_covariance,
                    "candidate_covariance_eigenvalues": torch.linalg.eigvalsh(
                        gt_covariance.float()
                    ).to(points_mm.dtype),
                    "covariance_lower_bound_hit": torch.zeros_like(gt_covariance[..., 0]),
                    "covariance_upper_bound_hit": torch.zeros_like(gt_covariance[..., 0]),
                    "candidate_scores": gt_score,
                    "candidate_existence_probability": gt_score,
                    "candidate_slot_valid_mask": gt_valid,
                    "candidate_analysis_valid_mask": gt_valid,
                    "candidate_view_support": gt_score[:, :, None].expand(
                        -1, -1, len(self.geometry_mapper.view_angles)
                    ),
                }
            )
        reconstruction_valid = candidates["candidate_slot_valid_mask"]
        centers = candidates["candidate_centers_mm"]
        active_phase = self.view_training_phase
        if active_phase == "phase_a" and not return_diagnostics:
            if hasattr(self.complementary_aggregation, "aggregate_shared"):
                shared, shared_weight = self.complementary_aggregation.aggregate_shared(
                    query_per_view, samples["query_view_valid"]
                )
            else:
                shared_weight = samples["query_view_valid"].to(query_per_view.dtype)
                shared_weight = shared_weight / shared_weight.sum(
                    dim=1, keepdim=True
                ).clamp_min(1.0e-8)
                shared = self.complementary_aggregation.shared_projection(
                    (query_per_view * shared_weight[..., None]).sum(dim=1)
                )
            trunk = torch.tensor(
                self.trunk_size_mm, device=points_mm.device, dtype=points_mm.dtype
            )
            encoded_points = self.position_encoding(
                (points_mm / trunk.clamp_min(1.0e-6)).mul(2.0).sub(1.0)
            )
            decoded = self.unified_density_decoder(
                shared,
                shared.new_zeros((shared.shape[0], centers.shape[1], shared.shape[-1])),
                points_mm,
                encoded_points,
                centers,
                candidates["candidate_covariances_mm"],
                candidates["candidate_scores"],
                reconstruction_valid,
                ablation="shared_only",
                context_scale=0.0,
            )
            aux_outputs: dict[str, torch.Tensor] = {
                "shared_density": decoded["density"],
                "candidate_centers_mm": centers,
                "candidate_covariances_mm": candidates["candidate_covariances_mm"],
                "candidate_covariance_eigenvalues": candidates[
                    "candidate_covariance_eigenvalues"
                ],
                "covariance_lower_bound_hit": candidates["covariance_lower_bound_hit"],
                "covariance_upper_bound_hit": candidates["covariance_upper_bound_hit"],
                "candidate_scores": candidates["candidate_scores"],
                "candidate_valid_mask": reconstruction_valid,
                "candidate_slot_valid_mask": candidates["candidate_slot_valid_mask"],
                "candidate_analysis_valid_mask": candidates["candidate_analysis_valid_mask"],
                "candidate_existence_probability": candidates[
                    "candidate_existence_probability"
                ],
                "candidate_view_support": candidates["candidate_view_support"],
                "proposal_assignment": candidates["proposal_assignment"],
                "proposal_compatibility": candidates["proposal_compatibility"],
                "per_view_evidence": proposal["evidence"],
                "query_view_valid": samples["query_view_valid"],
                "proposal_offsets_mm": proposal["offsets_mm"],
                "proposal_points_mm": proposal["proposal_points_mm"],
                "proposal_valid_mask": proposal["proposal_valid_mask"],
                "shared_quotient": shared,
                "view_weights": shared_weight,
                "candidate_context": decoded["candidate_context"],
                "candidate_branch_features": decoded["candidate_branch_features"],
                "candidate_applicability": decoded["candidate_applicability"],
                "candidate_delta_logit": decoded["candidate_delta_logit"],
                "hypothesis_gate": decoded["hypothesis_gate"],
                "branch_density": decoded["branch_density"],
                "p_all": decoded["p_all"],
                "pi": decoded["pi"],
                "decoder_pre_activation": decoded["decoder_pre_activation"],
                "candidate_context_scale": decoded["density"].new_zeros(()),
                "final_density_path": "phase_a_shared_unified_decoder",
                "decoder_ablation": "shared_only",
                "candidate_alpha": decoded["alpha"],
                "measurement_supported": samples["query_view_valid"].any(dim=1),
                "view_complementary_mode": torch.ones((), device=points_mm.device),
            }
            if (
                self.training
                and batch is not None
                and torch.is_tensor(batch.get("gt_component_centers_mm"))
                and torch.is_tensor(batch.get("gt_component_valid_mask"))
            ):
                gt_covariances = batch.get("gt_component_covariances_mm")
                if not torch.is_tensor(gt_covariances):
                    gt_covariances = torch.eye(3, device=points_mm.device)[None, None].expand(
                        points_mm.shape[0], batch["gt_component_centers_mm"].shape[1], -1, -1
                    )
                evidence_loss = self._view_evidence_heatmap_loss(
                    proposal["evidence"],
                    evidence_points,
                    evidence_valid,
                    batch["gt_component_centers_mm"].to(device=points_mm.device),
                    gt_covariances.to(device=points_mm.device),
                    batch["gt_component_valid_mask"].to(device=points_mm.device),
                    depth_maps,
                    detector_valid_mask,
                )
                aux_outputs["view_evidence_heatmap_loss"] = evidence_loss
                candidate_losses = self.diverse_candidate_constructor.supervision_losses(
                    candidates,
                    batch["gt_component_centers_mm"].to(device=points_mm.device),
                    batch["gt_component_valid_mask"].to(device=points_mm.device),
                    gt_covariances.to(device=points_mm.device),
                )
                aux_outputs.update(candidate_losses)
                aux_outputs["view_complementary_aux_loss"] = (
                    self.phase_a_aux_loss_scale * evidence_loss
                )
                if self.view_training_epoch != 0:
                    aux_outputs["view_complementary_aux_loss"] = self.phase_a_aux_loss_scale * (
                        evidence_loss
                        + sum(
                            self.view_candidate_loss_weights[key] * value
                            for key, value in candidate_losses.items()
                        )
                    )
            out: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
                "density": decoded["density"],
                "aux_outputs": aux_outputs,
            }
            if return_diagnostics:
                diagnostics = dict(aux_outputs)
                diagnostics["normalization_scale"] = norm_scale
                out["diagnostics"] = diagnostics
            return out

        candidate_mapped = self.geometry_mapper(
            centers,
            depth_maps=depth_maps,
            detector_valid_mask=detector_valid_mask,
        )
        candidate_samples = self.surface_sampler(
            features,
            measurements,
            candidate_mapped,
            detector_valid_mask=detector_valid_mask,
            depth_maps=depth_maps,
        )
        candidate_per_view = (
            candidate_samples["sample_features"] * candidate_samples["A"][..., None]
        ).sum(dim=3)
        eigen_scale = (
            torch.diagonal(candidates["candidate_covariances_mm"].float(), dim1=-2, dim2=-1)
            .clamp_min(1.0e-8)
            .sqrt()
            .mean(dim=-1)
            .to(candidate_per_view.dtype)
        )
        px_per_mm = candidate_mapped["view_pixels_per_mm"].to(candidate_per_view.dtype)
        detector_scale = (
            eigen_scale[:, None] * px_per_mm[None, :, None] + candidate_samples["sigma_f"]
        )
        candidate_valid_by_view = (
            candidate_mapped["valid_mask"]
            & candidate_samples["query_view_valid"]
            & reconstruction_valid[:, None]
        )
        separability = self.view_separability(
            candidate_per_view,
            candidate_mapped["uv_px"].to(candidate_per_view.dtype),
            detector_scale,
            candidate_valid_by_view,
            mode=self.view_separability_mode,
        )
        support = candidates["candidate_view_support"].transpose(1, 2)
        support = support * candidate_valid_by_view.to(support.dtype)
        if self.hypothesis_aggregation_strategy == "support":
            # A hypothesis-independent signal baseline. This remains meaningful
            # for GT hypotheses, which do not have constructor-derived support.
            support = candidate_per_view.float().norm(dim=-1).to(candidate_per_view.dtype)
            support = support / support.amax(dim=1, keepdim=True).clamp_min(1.0e-8)
            support = support * candidate_valid_by_view.to(support.dtype)
        aggregation = self.complementary_aggregation(
            query_per_view,
            candidate_per_view,
            samples["query_view_valid"],
            support,
            separability["separability"],
            reconstruction_valid,
            candidate_view_valid=candidate_valid_by_view,
            uniform_views=self.view_complementary_ablation
            in {"a2", "uniform_views", "isolated_bounded_routing", "fixed_grid_stabilized"},
            geometry_only=self.view_complementary_ablation == "a3_geometry_only",
            aggregation_strategy=self.hypothesis_aggregation_strategy,
        )
        ungated_shared = aggregation["shared"]
        ungated_view_weights = aggregation.get("shared_view_weights")
        routing_diagnostics = None
        routing_modes = {"a3_v2_bounded_routing", "isolated_bounded_routing"}
        if self.view_complementary_ablation in routing_modes:
            if self.bounded_view_routing is None:
                raise RuntimeError("a3_v2_bounded_routing requires routing.enabled=true")
            # Candidate representation remains an A2-U valid-view mean.
            aggregation = self.complementary_aggregation(
                query_per_view,
                candidate_per_view,
                samples["query_view_valid"],
                support,
                separability["separability"],
                reconstruction_valid,
                candidate_view_valid=candidate_valid_by_view,
                uniform_views=True,
            )
            routing_diagnostics = self.bounded_view_routing(
                points_mm,
                centers,
                candidates["candidate_covariances_mm"],
                candidates["candidate_existence_probability"],
                reconstruction_valid,
                candidates["candidate_view_support"],
                separability["separability"],
                samples["query_view_valid"],
            )
            if hasattr(self.complementary_aggregation, "aggregate_shared"):
                routed_shared, routed_weight = self.complementary_aggregation.aggregate_shared(
                    query_per_view,
                    samples["query_view_valid"],
                    attention_logit_residual=routing_diagnostics[
                        "routing_residual"
                    ].transpose(1, 2),
                )
            else:
                routed_weight = routing_diagnostics["view_weights"]
                routed_shared = self.complementary_aggregation.shared_projection(
                    (query_per_view * routed_weight[..., None]).sum(dim=1)
                )
            aggregation["shared"] = routed_shared
            aggregation["query_view_weights"] = routed_weight
            routing_diagnostics["routed_view_weights"] = routed_weight
        if (
            active_phase == "phase_b"
            and self.view_complementary_ablation not in routing_modes
        ):
            aggregation["shared"] = aggregation["shared"].detach()
        decoder_ablation = self.view_complementary_ablation
        context_scale = 1.0
        if active_phase == "phase_a":
            decoder_ablation = "shared_only"
            context_scale = 0.0
        elif self.view_complementary_ablation in routing_modes:
            decoder_ablation = "shared_only"
            context_scale = 0.0
        elif active_phase == "phase_b" and self.training and self.context_warmup_enabled:
            context_scale = min(
                1.0,
                self.view_training_step / max(self.context_warmup_steps, 1),
            )
        trunk = torch.tensor(self.trunk_size_mm, device=points_mm.device, dtype=points_mm.dtype)
        encoded_points = self.position_encoding(
            (points_mm / trunk.clamp_min(1.0e-6)).mul(2.0).sub(1.0)
        )
        decoded = self.unified_density_decoder(
            aggregation["shared"],
            aggregation["candidate"],
            points_mm,
            encoded_points,
            centers,
            candidates["candidate_covariances_mm"],
            candidates["candidate_scores"],
            reconstruction_valid,
            ablation=decoder_ablation,
            context_scale=context_scale,
            continuous_applicability=self.continuous_applicability_enabled,
        )
        if self.hypothesis_aggregation_strategy == "oracle":
            if batch is None or not torch.is_tensor(batch.get("point_densities")):
                raise RuntimeError("empirical oracle aggregation requires query GT")
            per_view_candidate = aggregation["candidate_view_features"]
            oracle_predictions = []
            for view_index in range(per_view_candidate.shape[1]):
                view_candidate = per_view_candidate[:, view_index]
                view_valid = reconstruction_valid & candidate_valid_by_view[:, view_index]
                view_decoded = self.unified_density_decoder(
                    aggregation["shared"],
                    view_candidate,
                    points_mm,
                    encoded_points,
                    centers,
                    candidates["candidate_covariances_mm"],
                    candidates["candidate_scores"],
                    view_valid,
                    ablation=decoder_ablation,
                    context_scale=context_scale,
                    continuous_applicability=self.continuous_applicability_enabled,
                )
                oracle_predictions.append(view_decoded["density"])
            stacked_oracle = torch.stack(oracle_predictions, dim=2)
            target = batch["point_densities"].to(stacked_oracle)[:, :, None, None]
            valid_view = samples["query_view_valid"].transpose(1, 2)[..., None]
            oracle_error = (stacked_oracle - target).abs().masked_fill(~valid_view, 1.0e4)
            selected_view = oracle_error.squeeze(-1).argmin(dim=2)
            selected_density = stacked_oracle.gather(
                2, selected_view[:, :, None, None]
            ).squeeze(2)
            decoded["density"] = selected_density
            decoded["oracle_selected_view"] = selected_view
        shared_decoded = (
            self.unified_density_decoder(
                ungated_shared,
                aggregation["candidate"].detach() * 0.0,
                points_mm,
                encoded_points,
                centers,
                candidates["candidate_covariances_mm"],
                candidates["candidate_scores"] * 0.0,
                reconstruction_valid & False,
                ablation="shared_only",
                context_scale=0.0,
            )
            if self.view_complementary_ablation in routing_modes
            else decoded
            if decoder_ablation in {"shared_only", "a0"} and context_scale == 0.0
            else self.unified_density_decoder(
                aggregation["shared"],
                aggregation["candidate"].detach() * 0.0,
                points_mm,
                encoded_points,
                centers,
                candidates["candidate_covariances_mm"],
                candidates["candidate_scores"] * 0.0,
                reconstruction_valid & False,
                ablation="shared_only",
                context_scale=0.0,
            )
        )
        aux_outputs: dict[str, torch.Tensor] = {
            "shared_density": shared_decoded["density"],
            "candidate_centers_mm": centers,
            "candidate_covariances_mm": candidates["candidate_covariances_mm"],
            "candidate_covariance_eigenvalues": candidates["candidate_covariance_eigenvalues"],
            "covariance_lower_bound_hit": candidates["covariance_lower_bound_hit"],
            "covariance_upper_bound_hit": candidates["covariance_upper_bound_hit"],
            "candidate_scores": candidates["candidate_scores"],
            "candidate_valid_mask": reconstruction_valid,
            "candidate_slot_valid_mask": candidates["candidate_slot_valid_mask"],
            "candidate_analysis_valid_mask": candidates["candidate_analysis_valid_mask"],
            "candidate_existence_probability": candidates["candidate_existence_probability"],
            "candidate_view_support": candidates["candidate_view_support"],
            "proposal_assignment": candidates["proposal_assignment"],
            "proposal_compatibility": candidates["proposal_compatibility"],
            "per_view_evidence": proposal["evidence"],
            "query_view_valid": samples["query_view_valid"],
            "proposal_offsets_mm": proposal["offsets_mm"],
            "proposal_points_mm": proposal["proposal_points_mm"],
            "proposal_valid_mask": proposal["proposal_valid_mask"],
            "candidate_view_features": candidate_per_view,
            "candidate_quotients": aggregation["candidate"],
            "shared_quotient": aggregation["shared"],
            "view_weights": aggregation["view_weights"],
            "separability": separability["separability"],
            "pair_separability": separability["pair_separability"],
            "geometry_separability": separability["geometry"],
            "measurement_separability_correction": separability["correction"],
            "candidate_detector_scales": detector_scale,
            "candidate_context": decoded["candidate_context"],
            "candidate_branch_features": decoded["candidate_branch_features"],
            "candidate_applicability": decoded["candidate_applicability"],
            "candidate_delta_logit": decoded["candidate_delta_logit"],
            "hypothesis_gate": decoded["hypothesis_gate"],
            "branch_density": decoded["branch_density"],
            "p_all": decoded["p_all"],
            "pi": decoded["pi"],
            "decoder_pre_activation": decoded["decoder_pre_activation"],
            "candidate_context_scale": decoded["density"].new_tensor(context_scale),
            "final_density_path": (
                "phase_a_shared_unified_decoder"
                if active_phase == "phase_a"
                else "view_complementary_unified_decoder"
            ),
            "decoder_ablation": decoder_ablation,
            "candidate_alpha": decoded["alpha"],
            "measurement_supported": samples["query_view_valid"].any(dim=1),
            "view_complementary_mode": torch.ones((), device=points_mm.device),
        }
        if not self.candidate_hidden_injection_enabled:
            decoded = self.unified_density_decoder(
                aggregation["shared"],
                aggregation["candidate"],
                points_mm,
                encoded_points,
                centers,
                candidates["candidate_covariances_mm"],
                candidates["candidate_scores"],
                reconstruction_valid,
                ablation="shared_only",
                context_scale=0.0,
            )
            aux_outputs["candidate_context_scale"] = decoded["density"].new_tensor(0.0)
        if routing_diagnostics is not None:
            aux_outputs.update(routing_diagnostics)
            aux_outputs["routing_residual_only_view_weights"] = routing_diagnostics[
                "view_weights"
            ]
            aux_outputs["view_weights"] = routing_diagnostics["routed_view_weights"]
            if ungated_view_weights is not None:
                aux_outputs["base_view_weights"] = ungated_view_weights
            if self.routing_counterfactual_diagnostics and not self.training:
                for sign, name in ((1.0, "positive"), (-1.0, "negative")):
                    cf_shared, cf_weights = self.complementary_aggregation.aggregate_shared(
                        query_per_view,
                        samples["query_view_valid"],
                        attention_logit_residual=(
                            sign
                            * self.routing_oracle_gain
                            * routing_diagnostics["routing_unit_residual"]
                        ).transpose(1, 2),
                    )
                    cf_decoded = self.unified_density_decoder(
                        cf_shared,
                        aggregation["candidate"].detach() * 0.0,
                        points_mm,
                        encoded_points,
                        centers,
                        candidates["candidate_covariances_mm"],
                        candidates["candidate_scores"] * 0.0,
                        reconstruction_valid & False,
                        ablation="shared_only",
                        context_scale=0.0,
                    )
                    aux_outputs[f"routing_{name}_density"] = cf_decoded["density"].detach()
                    aux_outputs[f"routing_{name}_view_weights"] = cf_weights.detach()
                    del cf_decoded, cf_shared, cf_weights
        normalized_candidate_features = torch.nn.functional.normalize(
            candidate_per_view.float(), dim=-1
        )
        feature_cosine = torch.einsum(
            "bvmd,bvnd->bvmn",
            normalized_candidate_features,
            normalized_candidate_features,
        )
        measurement_correction_target = (
            self.view_separability.delta_max * 0.5 * (1.0 - feature_cosine.detach())
        )
        pair_valid = (
            candidate_valid_by_view[..., :, None]
            & candidate_valid_by_view[..., None, :]
        )
        pair_eye = torch.eye(centers.shape[1], device=points_mm.device, dtype=torch.bool)[
            None, None
        ]
        pair_train = (pair_valid & ~pair_eye).expand_as(separability["pair_separability"])
        separability_measurement_loss = (
            torch.nn.functional.smooth_l1_loss(
                separability["correction"],
                measurement_correction_target,
                reduction="none",
            )[pair_train].mean()
            if pair_train.any()
            else decoded["density"].sum() * 0.0
        )
        aux_outputs["separability_measurement_loss"] = separability_measurement_loss
        if (
            self.training
            and batch is not None
            and torch.is_tensor(batch.get("gt_component_centers_mm"))
            and torch.is_tensor(batch.get("gt_component_valid_mask"))
        ):
            gt_covariances = batch.get("gt_component_covariances_mm")
            if not torch.is_tensor(gt_covariances):
                gt_covariances = torch.eye(3, device=points_mm.device)[None, None].expand(
                    points_mm.shape[0], batch["gt_component_centers_mm"].shape[1], -1, -1
                )
            evidence_loss = self._view_evidence_heatmap_loss(
                proposal["evidence"],
                evidence_points,
                evidence_valid,
                batch["gt_component_centers_mm"].to(device=points_mm.device),
                gt_covariances.to(device=points_mm.device),
                batch["gt_component_valid_mask"].to(device=points_mm.device),
                depth_maps,
                detector_valid_mask,
            )
            aux_outputs["view_evidence_heatmap_loss"] = evidence_loss
            candidate_losses = self.diverse_candidate_constructor.supervision_losses(
                candidates,
                batch["gt_component_centers_mm"].to(device=points_mm.device),
                batch["gt_component_valid_mask"].to(device=points_mm.device),
                gt_covariances.to(device=points_mm.device),
            )
            aux_outputs.update(candidate_losses)
            if active_phase == "phase_a" and self.view_training_epoch == 0:
                aux_outputs["view_complementary_aux_loss"] = (
                    self.phase_a_aux_loss_scale * evidence_loss
                )
            else:
                hypothesis_aux_loss = evidence_loss + sum(
                    self.view_candidate_loss_weights[key] * value
                    for key, value in candidate_losses.items()
                )
                if active_phase == "phase_a":
                    hypothesis_scale = self.phase_a_aux_loss_scale
                    separability_scale = 0.0
                elif active_phase == "phase_b":
                    hypothesis_scale = self.phase_b_aux_loss_scale
                    separability_scale = self.phase_b_separability_loss_scale
                else:
                    hypothesis_scale = self.full_aux_loss_scale
                    separability_scale = self.full_separability_loss_scale
                aux_outputs["view_complementary_aux_loss"] = (
                    hypothesis_scale * hypothesis_aux_loss
                )
                if active_phase in {"phase_b", "full"}:
                    aux_outputs["view_complementary_aux_loss"] = (
                        aux_outputs["view_complementary_aux_loss"]
                        + separability_scale
                        * self.lambda_separability_measurement
                        * separability_measurement_loss
                    )
        out: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "density": decoded["density"],
            "aux_outputs": aux_outputs,
        }
        if return_diagnostics:
            diagnostics = dict(aux_outputs)
            diagnostics.update(
                {
                    "normalization_scale": norm_scale,
                    "candidate_sample_coordinates": candidate_samples["sample_coordinates"],
                    "candidate_detector_centers": candidate_mapped["grid"],
                    "candidate_detector_scales": detector_scale,
                }
            )
            out["diagnostics"] = diagnostics
        return out

    def _quotient_geometry(
        self, samples: dict[str, torch.Tensor], mapped: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        beta = (
            mapped["boundary_distance"]
            / max(self.surface_sampler.boundary_margin_radius_px, 1.0e-6)
        ).clamp(0.0, 1.0)
        xi = mapped["detector_side_path_proxy"].clamp(0.0, 1.0)
        sigma = (
            (samples["sigma_f"] - self.surface_sampler.sigma_min_px)
            / max(self.surface_sampler.sigma_max_px - self.surface_sampler.sigma_min_px, 1.0e-6)
        ).clamp(0.0, 1.0)
        center_distance = mapped["grid"].square().sum(dim=-1).sqrt().div(2.0**0.5).clamp(0.0, 1.0)
        return torch.stack([beta, xi, sigma, center_distance], dim=-1).permute(0, 2, 1, 3)

    @staticmethod
    def _deterministic_view_subsets(
        view_valid: torch.Tensor, seed: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split each query's valid views into deterministic non-empty subsets when possible."""
        v = view_valid.shape[-1]
        order = (torch.arange(v, device=view_valid.device) * 1103515245 + seed) % 2147483647
        keys = order.to(dtype=torch.float32).view(1, 1, v).expand_as(view_valid)
        keys = keys.masked_fill(~view_valid, float("inf"))
        rank = keys.argsort(dim=-1).argsort(dim=-1)
        count = view_valid.sum(dim=-1, keepdim=True)
        split = (count + 1) // 2
        usable = count >= 2
        subset_a = view_valid & (rank < split) & usable
        subset_b = view_valid & (rank >= split) & usable
        return subset_a, subset_b

    def _forward_quotient_residual(
        self,
        per_view: torch.Tensor,
        samples: dict[str, torch.Tensor],
        mapped: dict[str, torch.Tensor],
        router: dict[str, torch.Tensor],
        candidates: dict[str, torch.Tensor],
        points_mm: torch.Tensor,
        norm_scale: torch.Tensor,
        candidate_norm_scale: torch.Tensor,
        return_diagnostics: bool,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        geometry = self._quotient_geometry(samples, mapped)
        view_valid = mapped["valid_mask"].permute(0, 2, 1)
        support = router["r"].permute(0, 2, 1, 3).clamp(0.0, 1.0)
        b = points_mm.shape[0]
        shared_valid = torch.ones((b, 1), dtype=torch.bool, device=points_mm.device)
        shared = self.quotient_aggregator(
            per_view[:, :, :, :1], geometry, support[:, :, :, :1], view_valid, shared_valid
        )
        candidate_valid = candidates["candidate_valid_mask"]
        candidate_view_valid = view_valid[..., None]
        if candidate_valid.shape[1] > 0:
            detector_visible = candidates["candidate_detector_valid_mask"]
            candidate_view_valid = candidate_view_valid & detector_visible[:, None]
        candidate = self.quotient_aggregator(
            per_view[:, :, :, 1:],
            geometry,
            support[:, :, :, 1:],
            candidate_view_valid,
            candidate_valid,
        )
        q_s = shared["quotient"].squeeze(2)
        trunk = torch.tensor(self.trunk_size_mm, device=points_mm.device, dtype=points_mm.dtype)
        query_norm = (points_mm / trunk.clamp_min(1.0e-6)).mul(2.0).sub(1.0)
        shared_logit = self.shared_density_logit_decoder(q_s, self.position_encoding(query_norm))
        shared_density = torch.sigmoid(shared_logit)
        m = candidate_valid.shape[1]
        if m > 0:
            delta = points_mm[:, :, None] - candidates["candidate_centers_mm"][:, None]
            relative = torch.einsum(
                "bnmi,bmij->bnmj", delta, candidates["candidate_support_inverse_sqrt_mm"]
            )
            encoded_relative = self.position_encoding(relative.clamp(-4.0, 4.0))
            q_m = candidate["quotient"]
            dispersion = candidate["dispersion"]
            candidate_score = candidates["candidate_scores"].detach().to(q_m.dtype)
            covariance = candidates["candidate_support_covariances_mm"].detach().float()
            eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(1.0e-8).sqrt().to(q_m.dtype)
            eigenvalues = eigenvalues / max(self._candidate_cfg.get("scale_max_mm", 12.0), 1.0e-6)
            context = torch.cat(
                [
                    q_s[:, :, None].expand(-1, -1, m, -1),
                    q_m - q_s[:, :, None],
                    dispersion[..., None],
                    candidate_score[:, None, :, None].expand(-1, points_mm.shape[1], -1, -1),
                    candidate["support"][..., None],
                    eigenvalues[:, None].expand(-1, points_mm.shape[1], -1, -1),
                ],
                dim=-1,
            )
            raw_residual = self.source_hypothesis_residual_decoder(context, encoded_relative)
            delta_logit = self.delta_l_max * torch.tanh(raw_residual)
            composition = compose_quotient_residual_density(
                shared_logit,
                delta_logit,
                relative,
                candidate_score,
                candidate["support"],
                dispersion,
                candidate_valid,
                self.tau_u,
            )
            alpha = composition["alpha"]
            applicability = composition["applicability"]
            proposal_gate = composition["proposal_gate"]
            residual_correction = composition["residual_correction"]
        else:
            q_m = q_s.new_zeros((*q_s.shape[:2], 0, q_s.shape[-1]))
            dispersion = q_s.new_zeros((*q_s.shape[:2], 0))
            alpha = q_s.new_zeros((*q_s.shape[:2], 0))
            applicability = alpha
            delta_logit = q_s.new_zeros((*q_s.shape[:2], 0, 1))
            proposal_gate = q_s.new_zeros(q_s.shape[:2])
            residual_correction = shared_logit.new_zeros(shared_logit.shape)
        final_logit = shared_logit + residual_correction
        density = torch.sigmoid(final_logit)
        quotient_loss = density.sum() * 0.0
        if self.training and self.quotient_consistency_enabled:
            subset_a, subset_b = self._deterministic_view_subsets(
                view_valid, self.quotient_consistency_seed
            )
            quotient_loss = self._subset_quotient_loss(
                per_view, geometry, support, subset_a, subset_b, candidate_valid
            )
        residual_loss = (
            proposal_gate[..., None] * (alpha[..., None] * delta_logit.abs()).sum(dim=2)
        ).mean()
        no_candidate = applicability.sum(dim=-1) <= 1.0e-8
        aux_outputs: dict[str, torch.Tensor] = {
            "shared_density": shared_density,
            "shared_logit": shared_logit,
            "final_density": density,
            "final_logit": final_logit,
            "residual_correction": residual_correction,
            "proposal_gate": proposal_gate,
            "alpha": alpha,
            "candidate_applicability": applicability,
            "quotient_dispersion": dispersion,
            "quotient_view_weights": candidate["view_weights"],
            "aggregated_candidate_support": candidate["support"],
            "quotient_consistency_loss": quotient_loss,
            "residual_regularization_loss": residual_loss,
            "measurement_supported": shared["valid"].squeeze(-1),
            "candidate_valid_mask": candidate_valid.detach(),
        }
        out: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "density": density,
            "aux_outputs": aux_outputs,
        }
        if return_diagnostics:
            diagnostics = dict(aux_outputs)
            diagnostics.update(
                {
                    "normalization_scale": norm_scale,
                    "candidate_normalization_scale": candidate_norm_scale,
                    "quotient_shared_norm": q_s.norm(dim=-1),
                    "quotient_candidate_norm": q_m.norm(dim=-1),
                    "shared_quotient_dispersion": shared["dispersion"].squeeze(-1),
                    "shared_quotient_view_weights": shared["view_weights"].squeeze(-1),
                    "effective_residual_magnitude": residual_correction.abs().mean(),
                    "final_minus_shared_density_magnitude": (density - shared_density).abs().mean(),
                    "candidate_valid_count": candidate_valid.sum(dim=-1),
                    "no_candidate_fallback_ratio": no_candidate.to(density.dtype).mean(),
                    "candidate_centers_mm": candidates["candidate_centers_mm"],
                    "candidate_scores": candidates["candidate_scores"],
                }
            )
            out["diagnostics"] = diagnostics
        return out

    def _subset_quotient_loss(
        self,
        per_view: torch.Tensor,
        geometry: torch.Tensor,
        support: torch.Tensor,
        subset_a: torch.Tensor,
        subset_b: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> torch.Tensor:
        branch_valid = torch.cat(
            [
                torch.ones((candidate_valid.shape[0], 1), dtype=torch.bool, device=per_view.device),
                candidate_valid,
            ],
            dim=1,
        )
        qa = self.quotient_aggregator(per_view, geometry, support, subset_a, branch_valid)
        qb = self.quotient_aggregator(per_view, geometry, support, subset_b, branch_valid)
        eligible = (
            qa["valid"]
            & qb["valid"]
            & (qa["support"] >= self.quotient_consistency_min_support)
            & (qb["support"] >= self.quotient_consistency_min_support)
        )
        error = (qa["quotient"] - qb["quotient"].detach()).square().mean(dim=-1)
        error = error + (qb["quotient"] - qa["quotient"].detach()).square().mean(dim=-1)
        return error[eligible].mean() if eligible.any() else per_view.sum() * 0.0
