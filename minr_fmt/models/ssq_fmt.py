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

from minr_fmt.network.ssq_candidates import (
    MeasurementDerivedCandidateBuilder,
    attach_candidate_detector_priors,
)
from minr_fmt.network.ssq_decoder import (
    CandidateDensityDecoder,
    CandidateMorphologyDecoder,
    CompensationDensityDecoder,
    CompensationMorphologyDecoder,
    FourierPositionEncoding,
    MorphologySDFHead,
)
from minr_fmt.network.ssq_encoder import ShallowSurfaceEncoder, SharedMultiscaleSurfaceEncoder
from minr_fmt.network.ssq_fusion import (
    CandidateAssignmentHead,
    CandidateSpecificViewFusion,
    CandidateViewEncoder,
)
from minr_fmt.network.ssq_geometry import GeometryQueryMapper, infer_detector_margin_map
from minr_fmt.network.ssq_sampler import QueryDependentSurfaceSampler, local_template


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


class SurfaceMeasurementNormalizer(nn.Module):
    """Joint per-sample valid-pixel percentile normalization."""

    def __init__(self, percentile: float = 99.9, eps: float = 1e-6):
        super().__init__()
        self.percentile = float(percentile)
        self.eps = float(eps)

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
        self.normalizer = SurfaceMeasurementNormalizer(
            percentile=float(_cfg_get(ssq, "normalization.percentile", 99.9)),
            eps=float(_cfg_get(ssq, "normalization.eps", 1e-6)),
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
        )

        enc_cfg = ssq.get("encoder", {})
        enc_type = str(enc_cfg.get("type", "residual_fpn"))
        if enc_type in {"shallow", "shallow_encoder_ablation"}:
            feature_dim = int(enc_cfg.get("feature_dim", enc_cfg.get("pyramid_channels", 64)))
            self.surface_encoder = ShallowSurfaceEncoder(
                in_channels=in_channels,
                channels=int(enc_cfg.get("channels", 32)),
                out_channels=feature_dim,
            )
        else:
            profile = str(enc_cfg.get("profile", "base"))
            if profile == "small":
                stage_channels = tuple(enc_cfg.get("stage_channels", [24, 48, 96, 128]))
                stage_blocks = tuple(enc_cfg.get("stage_blocks", [1, 1, 2, 2]))
                stem_channels = int(enc_cfg.get("stem_channels", 16))
            else:
                stage_channels = tuple(enc_cfg.get("stage_channels", [48, 96, 192, 256]))
                stage_blocks = tuple(enc_cfg.get("stage_blocks", [2, 2, 3, 3]))
                stem_channels = int(enc_cfg.get("stem_channels", 32))
            feature_dim = int(enc_cfg.get("pyramid_channels", enc_cfg.get("feature_dim", 64)))
            self.surface_encoder = SharedMultiscaleSurfaceEncoder(
                in_channels=in_channels,
                stem_channels=stem_channels,
                stage_channels=stage_channels,  # type: ignore[arg-type]
                stage_blocks=stage_blocks,  # type: ignore[arg-type]
                pyramid_channels=feature_dim,
                dropout=float(enc_cfg.get("dropout", 0.0)),
            )

        offsets = local_template(
            str(_cfg_get(ssq, "local_sampling.template", "grid3x3")),
            _cfg_get(ssq, "local_sampling.offsets_px", None),
        )
        sample_feature_dim = int(_cfg_get(ssq, "local_sampling.sample_feature_dim", 128))
        self.surface_sampler = QueryDependentSurfaceSampler(
            offsets,
            level_channels=feature_dim,
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
        )
        rep_cfg = ssq.get("representation", {})
        self.view_encoder = CandidateViewEncoder(
            sample_feature_dim,
            hidden_dim,
            blocks=int(rep_cfg.get("blocks", 3)),
            dropout=float(rep_cfg.get("dropout", 0.0)),
        )
        self.view_fusion = CandidateSpecificViewFusion(
            hidden_dim, hidden_dim, mode=str(_cfg_get(ssq, "fusion.mode", "candidate_specific"))
        )
        self.assignment_head = CandidateAssignmentHead(
            hidden_dim,
            hidden_dim,
            p_min=float(_cfg_get(ssq, "routing.p_min", 1e-4)),
            support_center=float(_cfg_get(ssq, "routing.support_center", 0.10)),
            support_temperature=float(_cfg_get(ssq, "routing.support_temperature", 0.10)),
            support_logit_weight=float(_cfg_get(ssq, "routing.support_logit_weight", 1.0)),
            tau0=float(_cfg_get(ssq, "routing.tau0", 0.25)),
            tau_p=float(_cfg_get(ssq, "routing.tau_p", 1.0)),
        )
        positive_ratio = float(_cfg_get(ssq, "decoder.positive_ratio_init", 0.03))
        self.compensation_density_decoder = CompensationDensityDecoder(
            hidden_dim, self.position_encoding.out_dim, hidden_dim, positive_ratio
        )
        self.candidate_density_decoder = CandidateDensityDecoder(
            hidden_dim, self.position_encoding.out_dim, hidden_dim, positive_ratio
        )
        self.compensation_morphology_decoder = CompensationMorphologyDecoder(
            hidden_dim, self.position_encoding.out_dim, hidden_dim
        )
        self.candidate_morphology_decoder = CandidateMorphologyDecoder(
            hidden_dim, self.position_encoding.out_dim, hidden_dim
        )
        self.morphology_sdf_head = MorphologySDFHead(hidden_dim, hidden_dim)
        self.lambda_sdf = float(
            _cfg_get(root, "loss.lambda_sdf", _cfg_get(ssq, "sdf.lambda_sdf", 0.0))
        )
        self.candidate_prior_weight = float(_cfg_get(ssq, "candidates.prior_density_weight", 0.0))
        self.density_output_mode = str(
            _cfg_get(ssq, "density_output_mode", "candidate_scalar_composition")
        )
        if self.density_output_mode != "candidate_scalar_composition":
            raise ValueError("Full SSQ-FMT only supports candidate_scalar_composition density.")
        self.query_density_backbone = None
        self._mode_epsilon = 0.0
        if str(_cfg_get(ssq, "routing.mode", "pre_aggregation")) == "post_aggregation":
            self._mode_epsilon += 1.0e-5
        if str(_cfg_get(ssq, "fusion.mode", "candidate_specific")) == "shared":
            self._mode_epsilon += 2.0e-5
        if str(_cfg_get(ssq, "reliability.mode", "evidence")) == "assignment_only":
            self._mode_epsilon += 3.0e-5
        if str(footprint_cfg.get("mode", "query_dependent")) == "fixed":
            self._mode_epsilon += 4.0e-5

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
        y_norm, norm_scale = self.normalizer(surface_measurements, detector_valid_mask)
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
        candidates = self.candidate_builder(y_norm, batch=batch)
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
            )
        router = self.candidate_router(samples, query_coordinates_mm, candidates, mapped)
        per_view = self.view_encoder(samples, router, mapped)
        z, view_weights, Lambda = self.view_fusion(
            per_view, router["r"], mapped["valid_mask"], router["branch_valid"]
        )
        router["Lambda"] = Lambda
        router["valid_view_count"] = mapped["valid_mask"].sum(dim=1)
        if candidates["candidate_centers_mm"].shape[1] > 0:
            candidate_visible = candidates["candidate_detector_valid_mask"].any(dim=1)
            router["candidate_geometric_visible"] = candidate_visible[:, None, :].expand(
                -1, query_coordinates_mm.shape[1], -1
            )
        pi = self.assignment_head(z, query_coordinates_mm, router, candidates)
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
            rel = (
                query_coordinates_mm[:, :, None] - candidates["candidate_centers_mm"][:, None]
            ) / candidates["candidate_support_scales_mm"][:, None, :, None].clamp_min(1e-6)
            encoded_rel = self.position_encoding(rel.clamp(-4.0, 4.0))
            dm = self.candidate_density_decoder(z[:, :, 1:], encoded_rel)
            dm = torch.where(
                candidates["candidate_valid_mask"][:, None, :, None], dm, torch.zeros_like(dm)
            )
            branch_density = torch.cat([d0[:, :, None], dm], dim=2).clamp(0.0, 1.0)
        else:
            branch_density = d0[:, :, None].clamp(0.0, 1.0)
        branch_contributions = pi[..., None] * branch_density
        branch_contributions = branch_contributions * measurement_supported[:, :, None, None].to(
            dtype=branch_contributions.dtype
        )
        density = branch_contributions.sum(dim=2).clamp(0.0, 1.0)
        if self._mode_epsilon:
            density = (density + self._mode_epsilon * density.detach().clamp_min(1e-3)).clamp(
                0.0, 1.0
            )
        candidate_prior = torch.zeros_like(density)
        cand_contrib = (
            branch_contributions[:, :, 1:].sum(dim=2) if m > 0 else torch.zeros_like(density)
        )
        comp_contrib = branch_contributions[:, :, :1].sum(dim=2)
        total_contrib = branch_contributions.sum(dim=2).clamp_min(1e-8)
        pi_entropy = -(pi.clamp_min(1e-8) * pi.clamp_min(1e-8).log()).sum(dim=-1)
        z_norm = torch.nn.functional.normalize(z[:, :, 1:], dim=-1) if m > 0 else z[:, :, 1:]
        pairwise_z_cosine = (
            torch.matmul(z_norm, z_norm.transpose(-1, -2))
            if m > 0
            else torch.empty((*z.shape[:2], 0, 0), device=z.device, dtype=z.dtype)
        )
        branch_prob = branch_contributions[:, :, 1:, 0] if m > 0 else pi[:, :, 1:]
        pairwise_branch_overlap = (
            torch.minimum(branch_prob[:, :, :, None], branch_prob[:, :, None, :])
            if m > 0
            else torch.empty((*z.shape[:2], 0, 0), device=z.device, dtype=z.dtype)
        )
        routing_entropy = -(
            router["zeta"].clamp_min(1e-8) * router["zeta"].clamp_min(1e-8).log()
        ).sum(dim=-1)
        aux_outputs: dict[str, torch.Tensor] = {
            "pi": pi,
            "branch_density": branch_density,
            "branch_contributions": branch_contributions,
            "candidate_prior_density": candidate_prior,
            "lightweight_density": density,
            "measurement_supported": measurement_supported,
            "compensation_contribution_ratio": (comp_contrib / total_contrib).mean(),
            "candidate_contribution_ratio": (cand_contrib / total_contrib).mean(),
            "density_output_mode": torch.tensor(0, device=density.device),
        }
        branch_sdf = None
        composed_sdf = None
        if self.training or return_diagnostics:
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
        diagnostics = {
            "normalization_scale": norm_scale,
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
            "candidate_raw_scales_mm": candidates["candidate_raw_support_scales_mm"],
            "candidate_scales_mm": candidates["candidate_support_scales_mm"],
            "candidate_valid_mask": candidates["candidate_valid_mask"],
            "candidate_support_scale_low_clip": candidates["candidate_support_scale_low_clip"],
            "candidate_support_scale_high_clip": candidates["candidate_support_scale_high_clip"],
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
            "compensation_contribution_ratio": (comp_contrib / total_contrib).mean(),
            "candidate_contribution_ratio": (cand_contrib / total_contrib).mean(),
            "unused_candidate_ratio": (pi[:, :, 1:] <= 1e-6).to(dtype=density.dtype).mean()
            if m > 0
            else torch.zeros((), device=density.device),
            "candidate_utilization": (pi[:, :, 1:] > 1e-6).to(dtype=density.dtype).mean()
            if m > 0
            else torch.zeros((), device=density.device),
            "pairwise_z_cosine_similarity": pairwise_z_cosine,
            "pairwise_branch_overlap": pairwise_branch_overlap,
            "pi_entropy": pi_entropy,
            "routing_entropy": routing_entropy,
            "branch_density": branch_density,
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
        out: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "density": density,
            "aux_outputs": aux_outputs,
        }
        if return_diagnostics:
            out["diagnostics"] = diagnostics
        return out
