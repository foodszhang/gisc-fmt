"""
Hydra/OmegaConf configuration extractor utilities.

All configuration values are required to be defined in YAML.
The extractor only converts structured configs into plain Python dictionaries
and never injects default values, keeping configuration fully decoupled from code.
"""

from typing import Any, Dict, List

from omegaconf import DictConfig, OmegaConf


class ConfigExtractor:
    """Helper functions for turning DictConfig objects into plain dicts."""

    @staticmethod
    def _to_dict(config: Any) -> Dict[str, Any]:
        if isinstance(config, DictConfig):
            return OmegaConf.to_container(config, resolve=True)  # type: ignore[arg-type]
        if isinstance(config, dict):
            return config
        raise TypeError("ConfigExtractor expects a DictConfig or dict")

    @staticmethod
    def _model_cfg(config: Any) -> Dict[str, Any]:
        cfg = ConfigExtractor._to_dict(config)
        model_cfg = cfg.get("model", cfg)
        if not isinstance(model_cfg, dict):
            raise ValueError("model section must be a mapping")
        return model_cfg

    @staticmethod
    def _data_cfg(config: Any) -> Dict[str, Any]:
        cfg = ConfigExtractor._to_dict(config)
        data_cfg = cfg.get("data", cfg)
        if not isinstance(data_cfg, dict):
            raise ValueError("data section must be a mapping")
        return data_cfg

    @staticmethod
    def _require_keys(section: Dict[str, Any], keys: List[str], section_name: str) -> None:
        missing = [key for key in keys if key not in section]
        if missing:
            raise KeyError(f"Missing {section_name} keys: {missing}")

    @staticmethod
    def extract_network_params(config: Any) -> Dict[str, Any]:
        model_cfg = ConfigExtractor._model_cfg(config)
        required = ["num_views", "in_channels", "feature_dim", "pos_enc_dim"]
        ConfigExtractor._require_keys(model_cfg, required, "network")
        return {
            "num_views": model_cfg["num_views"],
            "in_channels": model_cfg["in_channels"],
            "feature_dim": model_cfg["feature_dim"],
            "pos_enc_dim": model_cfg["pos_enc_dim"],
        }

    @staticmethod
    def extract_geometry_config(config: Any) -> Dict[str, Any]:
        model_cfg = ConfigExtractor._model_cfg(config)
        geometry = model_cfg.get("geometry")
        if not isinstance(geometry, dict):
            raise KeyError("model.geometry must be provided in the config")
        required = ["camera_distance", "detector_size", "global_voxel_shape"]
        ConfigExtractor._require_keys(geometry, required, "geometry")
        detector_resolution = geometry.get("detector_resolution", geometry["detector_size"])
        return {
            "camera_distance": float(geometry["camera_distance"]),
            "detector_size": tuple(geometry["detector_size"]),
            "detector_resolution": tuple(detector_resolution),
            "fov_mm": float(geometry.get("fov_mm", 80.0)),
            "global_voxel_shape": tuple(geometry["global_voxel_shape"]),
            "volume_center_world": tuple(geometry.get("volume_center_world", (19.0, 20.0, 10.4))),
            "use_fmt_simgen_projection": bool(geometry.get("use_fmt_simgen_projection", False)),
            "transpose_feature_map_for_sampling": bool(
                geometry.get("transpose_feature_map_for_sampling", False)
            ),
        }

    @staticmethod
    def extract_ptfa_config(config: Any) -> Dict[str, Any]:
        model_cfg = ConfigExtractor._model_cfg(config)
        ptfa = model_cfg.get("ptfa", {}) or {}
        if not isinstance(ptfa, dict):
            raise KeyError("model.ptfa must be a mapping when provided")
        return {
            "enabled": bool(ptfa.get("enabled", False)),
            "scales": [str(v) for v in ptfa.get("scales", [])],
            "mode": str(ptfa.get("mode", "fixed_gaussian")),
            "window": int(ptfa.get("window", 5)),
            "sigma_px": float(ptfa.get("sigma_px", 1.0)),
            "sigma_min": float(ptfa.get("sigma_min", 0.8)),
            "sigma_max": float(ptfa.get("sigma_max", 2.5)),
            "exit_depth_max_mm": float(ptfa.get("exit_depth_max_mm", 20.8)),
            "invert_depth": bool(ptfa.get("invert_depth", False)),
        }

    @staticmethod
    def extract_residual_scorer_config(config: Any) -> Dict[str, Any]:
        model_cfg = ConfigExtractor._model_cfg(config)
        scorer = model_cfg.get("residual_scorer", {}) or {}
        if not isinstance(scorer, dict):
            raise KeyError("model.residual_scorer must be a mapping when provided")
        return {
            "enabled": bool(scorer.get("enabled", False)),
            "hidden_dim": int(scorer.get("hidden_dim", 128)),
            "lambda_r": float(scorer.get("lambda_r", 0.0)),
            "input_mode": str(scorer.get("input_mode", "bilinear_s3")),
        }

    @staticmethod
    def extract_feature_refinement_config(config: Any) -> Dict[str, Any]:
        model_cfg = ConfigExtractor._model_cfg(config)
        refinement = model_cfg.get("feature_refinement", {}) or {}
        if not isinstance(refinement, dict):
            raise KeyError("model.feature_refinement must be a mapping when provided")
        gate_cfg = refinement.get("reliability_gate", {}) or {}
        if not isinstance(gate_cfg, dict):
            raise KeyError("model.feature_refinement.reliability_gate must be a mapping")
        mix_cfg = gate_cfg.get("residual_mix", {}) or {}
        if not isinstance(mix_cfg, dict):
            raise KeyError(
                "model.feature_refinement.reliability_gate.residual_mix must be a mapping"
            )
        consensus_cfg = refinement.get("consensus_residual_gate", {}) or {}
        if not isinstance(consensus_cfg, dict):
            raise KeyError(
                "model.feature_refinement.consensus_residual_gate must be a mapping"
            )
        return {
            "enabled": bool(refinement.get("enabled", False)),
            "input_mode": str(refinement.get("input_mode", "s1_ptfa")),
            "ptfa_view_aggregation": str(refinement.get("ptfa_view_aggregation", "masked_mean")),
            "hidden_dim": int(refinement.get("hidden_dim", 128)),
            "geom_dim": int(refinement.get("geom_dim", 5)),
            "zero_init": bool(refinement.get("zero_init", True)),
            "reliability_gate": {
                "hidden_dim": int(gate_cfg.get("hidden_dim", 64)),
                "temperature": float(gate_cfg.get("temperature", 1.5)),
                "zero_init": bool(gate_cfg.get("zero_init", True)),
                "norm": str(gate_cfg.get("norm", "none")),
                "geom_set": str(gate_cfg.get("geom_set", "full")),
                "residual_mix": {
                    "enabled": bool(mix_cfg.get("enabled", False)),
                    "gamma": float(mix_cfg.get("gamma", 1.0)),
                },
            },
            "consensus_residual_gate": {
                "hidden_dim": int(consensus_cfg.get("hidden_dim", 64)),
                "gamma": float(consensus_cfg.get("gamma", 0.1)),
                "norm": str(consensus_cfg.get("norm", "layernorm")),
                "use_evidence_stats": bool(consensus_cfg.get("use_evidence_stats", True)),
            },
        }

    @staticmethod
    def extract_query_aggregation_config(config: Any) -> Dict[str, Any]:
        model_cfg = ConfigExtractor._model_cfg(config)
        gisc_cfg = model_cfg.get("gisc_fmt")
        if not isinstance(gisc_cfg, dict):
            gisc_cfg = model_cfg.get("minr_fmt")
        if not isinstance(gisc_cfg, dict):
            return {
                "aggregation_mode": "legacy_multiscale",
                "hidden_dim": 64,
                "temperature": 1.0,
                "zero_init": True,
            }

        agg_cfg = gisc_cfg.get("query_aggregation", {}) or {}
        if not isinstance(agg_cfg, dict):
            raise KeyError("model.gisc_fmt.query_aggregation must be a mapping when provided")
        return {
            "aggregation_mode": str(gisc_cfg.get("aggregation_mode", "legacy_multiscale")),
            "hidden_dim": int(agg_cfg.get("hidden_dim", 64)),
            "temperature": float(agg_cfg.get("temperature", 1.0)),
            "zero_init": bool(agg_cfg.get("zero_init", True)),
        }

    @staticmethod
    def extract_view_angles(config: Any) -> Any:
        data_cfg = ConfigExtractor._data_cfg(config)
        if "view_angles" not in data_cfg:
            raise KeyError("data.view_angles must be defined")
        return data_cfg["view_angles"]

    @staticmethod
    def extract_minr_fmt_config(config: Any) -> Dict[str, Any]:
        model_cfg = ConfigExtractor._model_cfg(config)
        minr_cfg = model_cfg.get("minr_fmt")
        if not isinstance(minr_cfg, dict):
            raise KeyError("model.minr_fmt must be defined for legacy runs")
        required = [
            "use_adapter",
            "adapter_strategy",
            "adapter_mode",
            "adapter_cond_dim",
            "adapter_r_ratio",
            "norm_type",
            "implicit_field_hidden_dim",
            "implicit_field_d_x",
            "implicit_field_d_f",
            "view_weight_embed_dim",
            "view_weight_hidden_dim",
            "multiscale",
            "background",
        ]
        ConfigExtractor._require_keys(minr_cfg, required, "minr_fmt")

        # Core
        out = {key: minr_cfg[key] for key in required if key not in {"multiscale", "background"}}
        out["adapter_cond_dim"] = int(out["adapter_cond_dim"])

        # Multi-scale fusion config (required)
        multiscale = minr_cfg.get("multiscale")
        if not isinstance(multiscale, dict):
            raise KeyError("model.minr_fmt.multiscale must be a mapping")
        ConfigExtractor._require_keys(
            multiscale,
            [
                "view_embed_dim",
                "depth_dim",
                "cross_view_hidden_dim",
                "scale_hidden_dim",
                "tau",
                "depth_max",
            ],
            "minr_fmt.multiscale",
        )
        out["multiscale"] = {
            "view_embed_dim": int(multiscale["view_embed_dim"]),
            "depth_dim": int(multiscale["depth_dim"]),
            "cross_view_hidden_dim": int(multiscale["cross_view_hidden_dim"]),
            "scale_hidden_dim": int(multiscale["scale_hidden_dim"]),
            "tau": float(multiscale["tau"]),
            "depth_max": float(multiscale["depth_max"]),
        }

        # Background branch config (required mapping; can be disabled)
        bg = minr_cfg.get("background")
        if not isinstance(bg, dict):
            raise KeyError("model.minr_fmt.background must be a mapping")
        ConfigExtractor._require_keys(
            bg,
            ["enable_background", "head", "guidance"],
            "minr_fmt.background",
        )
        head = bg.get("head")
        guidance = bg.get("guidance")
        if not isinstance(head, dict):
            raise KeyError("model.minr_fmt.background.head must be a mapping")
        if not isinstance(guidance, dict):
            raise KeyError("model.minr_fmt.background.guidance must be a mapping")

        ConfigExtractor._require_keys(
            head,
            ["hidden_dim", "d_x", "d_f"],
            "minr_fmt.background.head",
        )
        ConfigExtractor._require_keys(
            guidance,
            ["enable", "dim", "mode", "hidden_dim", "scale", "gate_init_bias"],
            "minr_fmt.background.guidance",
        )

        out["background"] = {
            "enable_background": bool(bg["enable_background"]),
            "head": {
                "hidden_dim": int(head["hidden_dim"]),
                "d_x": int(head["d_x"]),
                "d_f": int(head["d_f"]),
            },
            "guidance": {
                "enable": bool(guidance["enable"]),
                "dim": int(guidance["dim"]),
                "mode": str(guidance["mode"]),
                "hidden_dim": int(guidance["hidden_dim"]),
                "scale": float(guidance["scale"]),
                "gate_init_bias": float(guidance["gate_init_bias"]),
            },
        }

        return out

    @staticmethod
    def extract_gisc_fmt_config(config: Any) -> Dict[str, Any]:
        """Extract config for GISC-FMT.

        Preferred section: model.gisc_fmt
        Backward-compatible fallback: model.minr_fmt
        """
        model_cfg = ConfigExtractor._model_cfg(config)

        section_key = "gisc_fmt"
        gisc_cfg = model_cfg.get(section_key)
        if not isinstance(gisc_cfg, dict):
            # Backward compatibility
            section_key = "minr_fmt"
            gisc_cfg = model_cfg.get(section_key)
            if not isinstance(gisc_cfg, dict):
                raise KeyError("model.gisc_fmt must be defined for GISC-FMT runs")

        required = [
            "use_adapter",
            "adapter_strategy",
            "adapter_mode",
            "adapter_cond_dim",
            "adapter_r_ratio",
            "norm_type",
            "implicit_field_hidden_dim",
            "implicit_field_d_x",
            "implicit_field_d_f",
            "view_weight_embed_dim",
            "view_weight_hidden_dim",
            "multiscale",
            "background",
        ]
        ConfigExtractor._require_keys(gisc_cfg, required, section_key)

        # Core
        out = {key: gisc_cfg[key] for key in required if key not in {"multiscale", "background"}}
        out["adapter_cond_dim"] = int(out["adapter_cond_dim"])

        # Multi-scale fusion config (required)
        multiscale = gisc_cfg.get("multiscale")
        if not isinstance(multiscale, dict):
            raise KeyError(f"model.{section_key}.multiscale must be a mapping")
        ConfigExtractor._require_keys(
            multiscale,
            [
                "view_embed_dim",
                "depth_dim",
                "cross_view_hidden_dim",
                "scale_hidden_dim",
                "tau",
                "depth_max",
            ],
            f"{section_key}.multiscale",
        )
        out["multiscale"] = {
            "view_embed_dim": int(multiscale["view_embed_dim"]),
            "depth_dim": int(multiscale["depth_dim"]),
            "cross_view_hidden_dim": int(multiscale["cross_view_hidden_dim"]),
            "scale_hidden_dim": int(multiscale["scale_hidden_dim"]),
            "tau": float(multiscale["tau"]),
            "depth_max": float(multiscale["depth_max"]),
        }

        # Background branch config (required mapping; can be disabled)
        bg = gisc_cfg.get("background")
        if not isinstance(bg, dict):
            raise KeyError(f"model.{section_key}.background must be a mapping")
        ConfigExtractor._require_keys(
            bg,
            ["enable_background", "head", "guidance"],
            f"{section_key}.background",
        )

        head = bg.get("head")
        guidance = bg.get("guidance")
        if not isinstance(head, dict):
            raise KeyError(f"model.{section_key}.background.head must be a mapping")
        if not isinstance(guidance, dict):
            raise KeyError(f"model.{section_key}.background.guidance must be a mapping")

        ConfigExtractor._require_keys(
            head,
            ["hidden_dim", "d_x", "d_f"],
            f"{section_key}.background.head",
        )
        ConfigExtractor._require_keys(
            guidance,
            ["enable", "dim", "mode", "hidden_dim", "scale", "gate_init_bias"],
            f"{section_key}.background.guidance",
        )

        out["background"] = {
            "enable_background": bool(bg["enable_background"]),
            "head": {
                "hidden_dim": int(head["hidden_dim"]),
                "d_x": int(head["d_x"]),
                "d_f": int(head["d_f"]),
            },
            "guidance": {
                "enable": bool(guidance["enable"]),
                "dim": int(guidance["dim"]),
                "mode": str(guidance["mode"]),
                "hidden_dim": int(guidance["hidden_dim"]),
                "scale": float(guidance["scale"]),
                "gate_init_bias": float(guidance["gate_init_bias"]),
            },
        }

        return out

    @staticmethod
    def extract_uhr_deepfmt_config(config: Any) -> Dict[str, Any]:
        model_cfg = ConfigExtractor._model_cfg(config)
        uhr_cfg = model_cfg.get("uhr_deepfmt")
        if not isinstance(uhr_cfg, dict):
            raise KeyError("model.uhr_deepfmt must be defined for UHR runs")
        required = ["base_channels", "num_levels", "se_reduction"]
        ConfigExtractor._require_keys(uhr_cfg, required, "uhr_deepfmt")
        return {key: uhr_cfg[key] for key in required}

    @staticmethod
    def extract_vox_dmrn_config(config: Any) -> Dict[str, Any]:
        model_cfg = ConfigExtractor._model_cfg(config)
        vox_cfg = model_cfg.get("vox_dmrn")
        if not isinstance(vox_cfg, dict):
            raise KeyError("model.vox_dmrn must be defined for VoxDMRN runs")
        required = ["base_channels", "num_blocks_per_stage", "use_bn"]
        ConfigExtractor._require_keys(vox_cfg, required, "vox_dmrn")

        # Prefer ROI-sized dense prediction to avoid an enormous output head.
        # ROI is defined by data.voxel_ranges (x/y/z are half-open: [start, end)).
        output_dim = None
        try:
            data_cfg = ConfigExtractor._data_cfg(config)
            vr = data_cfg.get("voxel_ranges")
            if isinstance(vr, dict):
                dx = int(vr["x"][1] - vr["x"][0])
                dy = int(vr["y"][1] - vr["y"][0])
                dz = int(vr["z"][1] - vr["z"][0])
                output_dim = dx * dy * dz
        except Exception:
            output_dim = None

        if output_dim is None:
            geometry = ConfigExtractor.extract_geometry_config(config)
            voxel_shape = geometry["global_voxel_shape"]
            output_dim = int(voxel_shape[0] * voxel_shape[1] * voxel_shape[2])

        return {
            "base_channels": vox_cfg["base_channels"],
            "num_blocks_per_stage": vox_cfg["num_blocks_per_stage"],
            "use_bn": vox_cfg["use_bn"],
            "output_dim": int(output_dim),
        }

    @staticmethod
    def to_dict(config: Any) -> Dict[str, Any]:
        return ConfigExtractor._to_dict(config)
