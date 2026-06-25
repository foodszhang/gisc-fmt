"""Baseline fidelity registry and benchmark preflight checks.

The registry separates official/paper-guided methods from adapted mechanisms and
controlled architectures. Exact literature names are not allowed in the TMI
main table when the implementation does not preserve the paper's defining
mechanisms and training objective.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BaselineSpec:
    method_id: str
    display_name: str
    fidelity: str
    main_table_allowed: bool
    reason: str
    reference: str | None = None


_SPECS = {
    "ssq_fmt": BaselineSpec(
        "ssq_fmt", "SSQ-FMT", "proposed", True, "Current proposed method."
    ),
    "tikhonov_fem": BaselineSpec(
        "tikhonov_fem",
        "Tikhonov-FEM",
        "classical",
        True,
        "Deterministic classical inverse solver.",
    ),
    "l1_fem": BaselineSpec(
        "l1_fem",
        "L1-FEM",
        "classical",
        True,
        "Deterministic sparse inverse solver; use only when its regularization is tuned on validation data.",
    ),
    "elasticnet_fem": BaselineSpec(
        "elasticnet_fem",
        "ElasticNet-FEM",
        "classical",
        False,
        "Valid classical control, but redundant with L1/FISTA for the compact main table.",
    ),
    "fista_fem": BaselineSpec(
        "fista_fem",
        "FISTA-L1-FEM",
        "classical",
        True,
        "Deterministic sparse inverse solver.",
    ),
    "stomp_fem": BaselineSpec(
        "stomp_fem",
        "StOMP-FEM",
        "classical",
        True,
        "Deterministic sparse inverse solver.",
    ),
    "fem_coarse": BaselineSpec(
        "fem_coarse",
        "Coarse FEM reconstruction",
        "physical_control",
        True,
        "Stage-1 physical reconstruction control; not an independent learned competitor.",
    ),
    "fem_to_voxel": BaselineSpec(
        "fem_to_voxel",
        "FEM-to-voxel interpolation",
        "physical_control",
        True,
        "Deterministic mapping of the coarse FEM result to the common voxel grid.",
    ),
    "stage1_fem": BaselineSpec(
        "stage1_fem",
        "Coarse FEM reconstruction",
        "physical_control",
        True,
        "Alias of the stage-1 physical reconstruction control.",
    ),
    "stage1_to_voxel": BaselineSpec(
        "stage1_to_voxel",
        "FEM-to-voxel interpolation",
        "physical_control",
        True,
        "Alias of the deterministic FEM-to-voxel control.",
    ),
    "fem2vox_unet": BaselineSpec(
        "fem2vox_unet",
        "FEM-prior residual 3D CNN",
        "controlled",
        True,
        "Controlled refinement baseline that explicitly consumes the same coarse FEM prior.",
    ),
    "stage1_unet": BaselineSpec(
        "stage1_unet",
        "FEM-prior residual 3D CNN",
        "controlled",
        True,
        "Alias of the FEM-prior refinement control.",
    ),
    "stage1_interpolation": BaselineSpec(
        "stage1_interpolation",
        "FEM-to-voxel interpolation",
        "physical_control",
        True,
        "Deterministic interpolation baseline.",
    ),
    "cnn3d_baseline": BaselineSpec(
        "cnn3d_baseline",
        "3D CNN (capacity control)",
        "controlled",
        True,
        "Controlled architecture baseline; not a literature-method reimplementation.",
    ),
    "transunet3d_baseline": BaselineSpec(
        "transunet3d_baseline",
        "3D TransUNet (capacity control)",
        "controlled",
        True,
        "Controlled architecture baseline; not a literature-method reimplementation.",
    ),
    "uhr_deepfmt": BaselineSpec(
        "uhr_deepfmt",
        "UHR-DeepFMT (adapted)",
        "mechanism_preserving_adaptation",
        True,
        "Retains a 3-D encoder-decoder and SE-based skip fusion, but the original data formation and exact dual-sampling implementation are unavailable.",
        "10.1109/TMI.2021.3071556",
    ),
    "pah2t_former": BaselineSpec(
        "pah2t_former",
        "PAH2T-Former (adapted)",
        "mechanism_preserving_adaptation",
        False,
        "Current code keeps paired spatial/channel attention modules but maps view depth to reconstruction depth heuristically; appendix only until paper-faithful input formation is verified.",
        "10.1109/TCI.2025.3559431",
    ),
    "map_pgan": BaselineSpec(
        "map_pgan",
        "MAP-PGAN-inspired adaptation",
        "architecture_proxy",
        False,
        "Current training does not implement the defining WGAN, gradient penalty, parameterized skip connections, and attention-prior loss.",
        "10.1364/BOE.469505",
    ),
    "d2_recst": BaselineSpec(
        "d2_recst",
        "D2-RecST-inspired adaptation",
        "architecture_proxy",
        False,
        "Current perceptual-domain objective is inactive and the defining image-domain adversarial training is absent.",
        "10.1016/j.cmpb.2022.107293",
    ),
    "dspgn": BaselineSpec(
        "dspgn",
        "DSPGN-inspired adaptation",
        "architecture_proxy",
        False,
        "Current graph branch does not embed the FEM imaging-system prior or reconstruct on the paper's mesh graph.",
        "10.1016/j.cmpb.2025.108948",
    ),
    "fmt_reconnet": BaselineSpec(
        "fmt_reconnet",
        "Template-STN reconstruction control",
        "architecture_proxy",
        False,
        "The available description is insufficient for an exact implementation; current code shares a generic template-STN-VNet proxy.",
        "10.1109/EMBC53108.2024.10781645",
    ),
    "pgdpnn": BaselineSpec(
        "pgdpnn",
        "Template-STN reconstruction control",
        "architecture_proxy",
        False,
        "Current implementation is identical to the FMT-ReconNet proxy and must not be reported as a separate literature method.",
    ),
    "two_stage_deepfmt": BaselineSpec(
        "two_stage_deepfmt",
        "Two-stage projection-to-volume control",
        "architecture_proxy",
        False,
        "The current learned profile-to-slice mapping is not a verified inverse-Radon implementation.",
    ),
    "vox_dmrn": BaselineSpec(
        "vox_dmrn",
        "Vox-DMRN (single-view adaptation)",
        "architecture_proxy",
        False,
        "Single-view input and a fully connected voxel head are not directly comparable with the seven-view protocol and become unsafe at the reference grid size.",
    ),
    "gaicn": BaselineSpec(
        "gaicn",
        "Graph-unrolled FEM control",
        "architecture_proxy",
        False,
        "Repository class is explicitly GAICN-like rather than a verified paper-faithful implementation.",
    ),
}

_ALIASES = {
    "pgd_pnn": "pgdpnn",
    "pgd-pnn": "pgdpnn",
}


def _canonical_method_id(method_id: str) -> str:
    method_id = str(method_id).lower()
    return _ALIASES.get(method_id, method_id)


def get_baseline_spec(method_id: str) -> BaselineSpec:
    method_id = _canonical_method_id(method_id)
    return _SPECS.get(
        method_id,
        BaselineSpec(method_id, method_id, "unclassified", False, "No fidelity audit is registered."),
    )


def _as_tuple3(value: Any, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if value is None:
        return default
    values = tuple(int(v) for v in value)
    if len(values) != 3 or any(v <= 0 for v in values):
        raise ValueError(f"Expected a positive 3-D shape, got {values}")
    return values


def _as_float_tuple3(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if value is None:
        return default
    values = tuple(float(v) for v in value)
    if len(values) != 3 or any(v <= 0.0 for v in values):
        raise ValueError(f"Expected a positive 3-D physical extent, got {values}")
    return values


def _get_nested(obj: Any, *keys: str, default=None):
    current = obj
    for key in keys:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return current


def _roi_shape_from_data(cfg: Any) -> tuple[int, int, int] | None:
    ranges = _get_nested(cfg, "data", "voxel_ranges", default=None)
    if ranges is None:
        return None
    try:
        return (
            int(_get_nested(ranges, "x")[1] - _get_nested(ranges, "x")[0]),
            int(_get_nested(ranges, "y")[1] - _get_nested(ranges, "y")[0]),
            int(_get_nested(ranges, "z")[1] - _get_nested(ranges, "z")[0]),
        )
    except (TypeError, IndexError, KeyError):
        return None


def validate_baseline_protocol(cfg: Any) -> BaselineSpec:
    """Validate reporting tier, common grid, and unsafe full-grid architectures."""

    raw_method_id = str(_get_nested(cfg, "model", "name", default="")).lower()
    method_id = _canonical_method_id(raw_method_id)
    spec = get_baseline_spec(method_id)
    protocol = _get_nested(cfg, "benchmark_protocol", default={}) or {}
    table_tier = str(_get_nested(protocol, "table_tier", default="development")).lower()
    strict = bool(_get_nested(protocol, "strict_fidelity", default=False))
    mode = str(_get_nested(protocol, "mode", default="native_to_reference")).lower()
    if mode not in {"native_to_reference", "resolution_matched", "development"}:
        raise ValueError(f"Unknown benchmark protocol mode: {mode}")

    reference_shape = _as_tuple3(
        _get_nested(cfg, "model", "geometry", "global_voxel_shape", default=None),
        (190, 200, 104),
    )
    roi_shape = _roi_shape_from_data(cfg)
    if roi_shape is not None and reference_shape != roi_shape:
        raise ValueError(
            "The common evaluation grid must match data.voxel_ranges. "
            f"reference_shape={reference_shape}, roi_shape={roi_shape}."
        )

    native_shape = _get_nested(cfg, "model", raw_method_id, "native_output_shape", default=None)
    if native_shape is None:
        native_shape = _get_nested(cfg, "model", "benchmark", "native_output_shape", default=None)
    if native_shape is not None:
        native_shape = _as_tuple3(native_shape, reference_shape)
        physical_extent = _get_nested(
            cfg,
            "model",
            "benchmark",
            "physical_extent_mm",
            default=None,
        )
        if physical_extent is None:
            raise ValueError(
                f"{raw_method_id} declares native_output_shape={native_shape} but does not declare "
                "model.benchmark.physical_extent_mm. Native and reference grids must explicitly share "
                "the same physical reconstruction region."
            )
        _as_float_tuple3(physical_extent, (38.0, 40.0, 20.8))

    if table_tier == "main" and not spec.main_table_allowed:
        raise ValueError(
            f"{raw_method_id} is not approved for the TMI main table: {spec.reason} "
            "Use benchmark_protocol.table_tier=appendix/development or implement the missing defining mechanisms."
        )
    if strict and spec.fidelity in {"architecture_proxy", "unclassified"}:
        raise ValueError(f"Strict fidelity check failed for {raw_method_id}: {spec.reason}")

    reference_voxels = math.prod(reference_shape)
    if method_id == "vox_dmrn" and reference_voxels > 1_000_000:
        allow = bool(_get_nested(protocol, "allow_unsafe_fully_connected_head", default=False))
        if not allow:
            raise ValueError(
                "vox_dmrn is blocked on the common reference grid because its fully connected output head "
                f"would emit {reference_voxels:,} voxels. Keep its native grid and evaluate after fixed resampling, "
                "or use it only in a single-view appendix experiment."
            )

    if method_id in {"fmt_reconnet", "pgdpnn"} and reference_voxels > 1_000_000:
        allow = bool(_get_nested(protocol, "allow_unsafe_template_full_grid", default=False))
        if not allow:
            raise ValueError(
                f"{raw_method_id} is blocked at reference_shape={reference_shape}: the current template-STN-VNet "
                "proxy performs full-grid 3-D warping and refinement. A native-grid template library and explicit "
                "physical-space mapping are required before large-grid training."
            )

    return spec


def write_baseline_manifest(cfg: Any, output_dir: str | Path) -> Path:
    """Write an auditable method-fidelity record next to experiment outputs."""

    spec = validate_baseline_protocol(cfg)
    raw_method_id = str(_get_nested(cfg, "model", "name", default="")).lower()
    reference_shape = _as_tuple3(
        _get_nested(cfg, "model", "geometry", "global_voxel_shape", default=None),
        (190, 200, 104),
    )
    native_shape = _get_nested(cfg, "model", raw_method_id, "native_output_shape", default=None)
    if native_shape is None:
        native_shape = _get_nested(cfg, "model", "benchmark", "native_output_shape", default=None)
    physical_extent = _get_nested(
        cfg,
        "model",
        "benchmark",
        "physical_extent_mm",
        default=None,
    )
    protocol = _get_nested(cfg, "benchmark_protocol", default={}) or {}
    payload = {
        "baseline": asdict(spec),
        "requested_method_id": raw_method_id,
        "protocol_mode": str(_get_nested(protocol, "mode", default="native_to_reference")),
        "table_tier": str(_get_nested(protocol, "table_tier", default="development")),
        "strict_fidelity": bool(_get_nested(protocol, "strict_fidelity", default=False)),
        "reference_grid_shape": list(reference_shape),
        "native_grid_shape": list(native_shape) if native_shape is not None else None,
        "physical_extent_mm": list(physical_extent) if physical_extent is not None else None,
        "fixed_resampling_to_reference": native_shape is not None and tuple(native_shape) != reference_shape,
    }
    path = Path(output_dir) / "baseline_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
