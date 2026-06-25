"""Baseline fidelity registry and TMI comparison preflight checks."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_DEFAULT_REFERENCE_SHAPE = (190, 200, 104)
_DEFAULT_PHYSICAL_EXTENT_MM = (38.0, 40.0, 20.8)


@dataclass(frozen=True)
class BaselineSpec:
    method_id: str
    display_name: str
    fidelity: str
    main_table_allowed: bool
    reason: str
    reference: str | None = None


def _spec(
    method_id: str,
    display_name: str,
    fidelity: str,
    main_table_allowed: bool,
    reason: str,
    reference: str | None = None,
) -> BaselineSpec:
    return BaselineSpec(
        method_id,
        display_name,
        fidelity,
        main_table_allowed,
        reason,
        reference,
    )


_SPECS = {
    "ssq_fmt": _spec("ssq_fmt", "SSQ-FMT", "proposed", True, "Current proposed method."),
    "tikhonov_fem": _spec(
        "tikhonov_fem", "Tikhonov-FEM", "classical", True,
        "Deterministic classical inverse solver.",
    ),
    "l1_fem": _spec(
        "l1_fem", "L1-FEM", "classical", True,
        "Sparse inverse solver; regularization must be selected on validation data.",
    ),
    "elasticnet_fem": _spec(
        "elasticnet_fem", "ElasticNet-FEM", "classical", False,
        "Valid classical control but redundant with L1/FISTA in the compact main table.",
    ),
    "fista_fem": _spec(
        "fista_fem", "FISTA-L1-FEM", "classical", True,
        "Deterministic sparse inverse solver.",
    ),
    "stomp_fem": _spec(
        "stomp_fem", "StOMP-FEM", "classical", True,
        "Deterministic sparse inverse solver.",
    ),
    "fem_coarse": _spec(
        "fem_coarse", "Coarse FEM reconstruction", "physical_control", True,
        "Stage-1 physical control, not an independent learned competitor.",
    ),
    "fem_to_voxel": _spec(
        "fem_to_voxel", "FEM-to-voxel interpolation", "physical_control", True,
        "Deterministic mapping of the coarse FEM result to the common voxel grid.",
    ),
    "stage1_fem": _spec(
        "stage1_fem", "Coarse FEM reconstruction", "physical_control", True,
        "Alias of the stage-1 physical control.",
    ),
    "stage1_to_voxel": _spec(
        "stage1_to_voxel", "FEM-to-voxel interpolation", "physical_control", True,
        "Alias of the deterministic FEM-to-voxel control.",
    ),
    "stage1_interpolation": _spec(
        "stage1_interpolation", "FEM-to-voxel interpolation", "physical_control", True,
        "Deterministic interpolation baseline.",
    ),
    "fem2vox_unet": _spec(
        "fem2vox_unet", "FEM-prior residual 3D CNN", "controlled", True,
        "Controlled refinement baseline consuming the same coarse FEM prior.",
    ),
    "stage1_unet": _spec(
        "stage1_unet", "FEM-prior residual 3D CNN", "controlled", True,
        "Alias of the FEM-prior refinement control.",
    ),
    "cnn3d_baseline": _spec(
        "cnn3d_baseline", "3D CNN (capacity control)", "controlled", True,
        "Controlled architecture baseline, not a literature-method reproduction.",
    ),
    "transunet3d_baseline": _spec(
        "transunet3d_baseline", "3D TransUNet (capacity control)", "controlled", True,
        "Controlled architecture baseline, not a literature-method reproduction.",
    ),
    "uhr_deepfmt": _spec(
        "uhr_deepfmt", "UHR-DeepFMT (adapted)", "mechanism_preserving_adaptation", True,
        "Retains a 3-D encoder-decoder and SE skip fusion, but exact dual-sampling input formation is unavailable.",
        "10.1109/TMI.2021.3071556",
    ),
    "pah2t_former": _spec(
        "pah2t_former", "PAH2T-Former (adapted)", "mechanism_preserving_adaptation", False,
        "Paired attention is retained, but view-to-volume input formation is heuristic.",
        "10.1109/TCI.2025.3559431",
    ),
    "map_pgan": _spec(
        "map_pgan", "MAP-PGAN-inspired adaptation", "architecture_proxy", False,
        "WGAN optimization, gradient penalty, parameterized skips, and attention-prior loss are absent.",
        "10.1364/BOE.469505",
    ),
    "d2_recst": _spec(
        "d2_recst", "D2-RecST-inspired adaptation", "architecture_proxy", False,
        "Perceptual transfer is inactive and image-domain adversarial training is absent.",
        "10.1016/j.cmpb.2022.107293",
    ),
    "dspgn": _spec(
        "dspgn", "DSPGN-inspired adaptation", "architecture_proxy", False,
        "The FEM imaging-system prior and paper-specific mesh graph are absent.",
        "10.1016/j.cmpb.2025.108948",
    ),
    "fmt_reconnet": _spec(
        "fmt_reconnet", "Template-STN reconstruction control", "architecture_proxy", False,
        "Available details are insufficient for exact reproduction; code uses a generic template-STN-VNet proxy.",
        "10.1109/EMBC53108.2024.10781645",
    ),
    "pgdpnn": _spec(
        "pgdpnn", "Template-STN reconstruction control", "architecture_proxy", False,
        "Implementation is identical to the FMT-ReconNet proxy and is not an independent method.",
    ),
    "two_stage_deepfmt": _spec(
        "two_stage_deepfmt", "Two-stage projection-to-volume control", "architecture_proxy", False,
        "The learned profile-to-slice mapping is not a verified inverse-Radon implementation.",
    ),
    "vox_dmrn": _spec(
        "vox_dmrn", "Vox-DMRN (single-view adaptation)", "architecture_proxy", False,
        "Single-view input and a fully connected voxel head do not match the seven-view protocol.",
    ),
    "gaicn": _spec(
        "gaicn", "Graph-unrolled FEM control", "architecture_proxy", False,
        "The repository class is explicitly GAICN-like rather than paper-faithful.",
    ),
}

_ALIASES = {"pgd_pnn": "pgdpnn", "pgd-pnn": "pgdpnn"}


def _canonical_method_id(method_id: str) -> str:
    method_id = str(method_id).lower()
    return _ALIASES.get(method_id, method_id)


def get_baseline_spec(method_id: str) -> BaselineSpec:
    method_id = _canonical_method_id(method_id)
    return _SPECS.get(
        method_id,
        _spec(method_id, method_id, "unclassified", False, "No fidelity audit is registered."),
    )


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


def _tuple3(value: Any, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if value is None:
        return default
    values = tuple(int(v) for v in value)
    if len(values) != 3 or any(v <= 0 for v in values):
        raise ValueError(f"Expected a positive 3-D shape, got {values}")
    return values


def _float_tuple3(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if value is None:
        return default
    values = tuple(float(v) for v in value)
    if len(values) != 3 or any(v <= 0.0 for v in values):
        raise ValueError(f"Expected a positive 3-D physical extent, got {values}")
    return values


def _roi_shape(cfg: Any) -> tuple[int, int, int] | None:
    ranges = _get_nested(cfg, "data", "voxel_ranges", default=None)
    if ranges is None:
        return None
    try:
        axes = [_get_nested(ranges, axis) for axis in ("x", "y", "z")]
        return tuple(int(axis[1] - axis[0]) for axis in axes)
    except (TypeError, IndexError, KeyError):
        return None


def _reference_shape(cfg: Any) -> tuple[int, int, int]:
    roi = _roi_shape(cfg)
    declared = _get_nested(cfg, "model", "geometry", "global_voxel_shape", default=None)
    return _tuple3(declared, roi or _DEFAULT_REFERENCE_SHAPE)


def _native_shape(cfg: Any, raw_method_id: str) -> tuple[int, int, int] | None:
    section_shape = _get_nested(
        cfg, "model", raw_method_id, "native_output_shape", default=None
    )
    if section_shape is None:
        section_shape = _get_nested(cfg, "model", raw_method_id, "internal_shape", default=None)
    if section_shape is None:
        section_shape = _get_nested(cfg, "model", "benchmark", "native_output_shape", default=None)
    return None if section_shape is None else _tuple3(section_shape, _reference_shape(cfg))


def validate_baseline_protocol(cfg: Any) -> BaselineSpec:
    """Validate reporting tier, common physical grid, and unsafe architectures."""

    raw_method_id = str(_get_nested(cfg, "model", "name", default="")).lower()
    method_id = _canonical_method_id(raw_method_id)
    spec = get_baseline_spec(method_id)
    protocol = _get_nested(cfg, "benchmark_protocol", default={}) or {}
    table_tier = str(_get_nested(protocol, "table_tier", default="development")).lower()
    strict = bool(_get_nested(protocol, "strict_fidelity", default=False))
    mode = str(_get_nested(protocol, "mode", default="native_to_reference")).lower()
    if mode not in {"native_to_reference", "resolution_matched", "development"}:
        raise ValueError(f"Unknown benchmark protocol mode: {mode}")

    roi = _roi_shape(cfg)
    reference_shape = _reference_shape(cfg)
    if roi is not None and reference_shape != roi:
        raise ValueError(
            "The common evaluation grid must match data.voxel_ranges. "
            f"reference_shape={reference_shape}, roi_shape={roi}."
        )

    native_shape = _native_shape(cfg, raw_method_id)
    if native_shape is not None:
        physical_extent = _get_nested(
            cfg, "model", "benchmark", "physical_extent_mm", default=None
        )
        if physical_extent is None:
            raise ValueError(
                f"{raw_method_id} declares native/internal shape {native_shape} but does not declare "
                "model.benchmark.physical_extent_mm. Native and reference grids must explicitly "
                "cover the same physical reconstruction region."
            )
        _float_tuple3(physical_extent, _DEFAULT_PHYSICAL_EXTENT_MM)

    if table_tier == "main" and not spec.main_table_allowed:
        raise ValueError(
            f"{raw_method_id} is not approved for the TMI main table: {spec.reason} "
            "Use appendix/development tier or implement the missing defining mechanisms."
        )
    if strict and spec.fidelity in {"architecture_proxy", "unclassified"}:
        raise ValueError(f"Strict fidelity check failed for {raw_method_id}: {spec.reason}")

    reference_voxels = math.prod(reference_shape)
    if method_id == "vox_dmrn" and reference_voxels > 1_000_000:
        if not bool(_get_nested(protocol, "allow_unsafe_fully_connected_head", default=False)):
            raise ValueError(
                "vox_dmrn is blocked on the common reference grid because its fully connected "
                f"output head would emit {reference_voxels:,} voxels."
            )

    if method_id in {"fmt_reconnet", "pgdpnn"} and reference_voxels > 1_000_000:
        if not bool(_get_nested(protocol, "allow_unsafe_template_full_grid", default=False)):
            raise ValueError(
                f"{raw_method_id} is blocked at reference_shape={reference_shape}: the current "
                "template-STN-VNet proxy performs full-grid 3-D warping and refinement."
            )

    return spec


def write_baseline_manifest(cfg: Any, output_dir: str | Path) -> Path:
    """Write an auditable method-fidelity and grid record for one run."""

    spec = validate_baseline_protocol(cfg)
    raw_method_id = str(_get_nested(cfg, "model", "name", default="")).lower()
    reference_shape = _reference_shape(cfg)
    native_shape = _native_shape(cfg, raw_method_id)
    physical_extent = _get_nested(
        cfg, "model", "benchmark", "physical_extent_mm", default=None
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
        "fixed_resampling_to_reference": native_shape is not None and native_shape != reference_shape,
    }
    path = Path(output_dir) / "baseline_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
