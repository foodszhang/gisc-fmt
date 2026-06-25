"""Baseline fidelity registry and benchmark preflight checks.

The registry separates official/paper-guided methods from adapted mechanisms and
controlled architectures.  Exact literature names are not allowed in the TMI
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
        "tikhonov_fem", "Tikhonov-FEM", "classical", True, "Deterministic classical inverse solver."
    ),
    "fista_fem": BaselineSpec(
        "fista_fem", "FISTA-L1-FEM", "classical", True, "Deterministic sparse inverse solver."
    ),
    "stomp_fem": BaselineSpec(
        "stomp_fem", "StOMP-FEM", "classical", True, "Deterministic sparse inverse solver."
    ),
    "cnn3d_baseline": BaselineSpec(
        "cnn3d_baseline", "3D CNN (capacity control)", "controlled", True,
        "Controlled architecture baseline; not a literature-method reimplementation.",
    ),
    "transunet3d_baseline": BaselineSpec(
        "transunet3d_baseline", "3D TransUNet (capacity control)", "controlled", True,
        "Controlled architecture baseline; not a literature-method reimplementation.",
    ),
    "uhr_deepfmt": BaselineSpec(
        "uhr_deepfmt", "UHR-DeepFMT (adapted)", "mechanism_preserving_adaptation", True,
        "Retains a 3-D encoder-decoder and SE-based skip fusion, but the original data formation and exact dual-sampling implementation are unavailable.",
        "10.1109/TMI.2021.3071556",
    ),
    "pah2t_former": BaselineSpec(
        "pah2t_former", "PAH2T-Former (adapted)", "mechanism_preserving_adaptation", False,
        "Current code keeps paired spatial/channel attention modules but maps view depth to reconstruction depth heuristically; appendix only until paper-faithful input formation is verified.",
        "10.1109/TCI.2025.3559431",
    ),
    "map_pgan": BaselineSpec(
        "map_pgan", "MAP-PGAN-inspired adaptation", "architecture_proxy", False,
        "Current training does not implement the defining WGAN, gradient penalty, parameterized skip connections, and attention-prior loss.",
        "10.1364/BOE.469505",
    ),
    "d2_recst": BaselineSpec(
        "d2_recst", "D2-RecST-inspired adaptation", "architecture_proxy", False,
        "Current perceptual-domain objective is inactive and the defining image-domain adversarial training is absent.",
        "10.1016/j.cmpb.2022.107293",
    ),
    "dspgn": BaselineSpec(
        "dspgn", "DSPGN-inspired adaptation", "architecture_proxy", False,
        "Current graph branch does not embed the FEM imaging-system prior or reconstruct on the paper's mesh graph.",
        "10.1016/j.cmpb.2025.108948",
    ),
    "fmt_reconnet": BaselineSpec(
        "fmt_reconnet", "Template-STN reconstruction control", "architecture_proxy", False,
        "The available description is insufficient for an exact implementation; current code shares a generic template-STN-VNet proxy.",
        "10.1109/EMBC53108.2024.10781645",
    ),
    "pgdpnn": BaselineSpec(
        "pgdpnn", "Template-STN reconstruction control", "architecture_proxy", False,
        "Current implementation is identical to the FMT-ReconNet proxy and must not be reported as a separate literature method.",
    ),
    "two_stage_deepfmt": BaselineSpec(
        "two_stage_deepfmt", "Two-stage projection-to-volume control", "architecture_proxy", False,
        "The current learned profile-to-slice mapping is not a verified inverse-Radon implementation.",
    ),
    "vox_dmrn": BaselineSpec(
        "vox_dmrn", "Vox-DMRN (single-view adaptation)", "architecture_proxy", False,
        "Single-view input and a fully connected voxel head are not directly comparable with the seven-view protocol and become unsafe at the reference grid size.",
    ),
    "gaicn": BaselineSpec(
        "gaicn", "Graph-unrolled FEM control", "architecture_proxy", False,
        "Repository class is explicitly GAICN-like rather than a verified paper-faithful implementation.",
    ),
}


def get_baseline_spec(method_id: str) -> BaselineSpec:
    method_id = str(method_id).lower()
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


def validate_baseline_protocol(cfg: Any) -> BaselineSpec:
    """Validate reporting tier, reference grid, and unsafe full-grid heads."""

    method_id = str(_get_nested(cfg, "model", "name", default="")).lower()
    spec = get_baseline_spec(method_id)
    protocol = _get_nested(cfg, "benchmark_protocol", default={}) or {}
    table_tier = str(_get_nested(protocol, "table_tier", default="development")).lower()
    strict = bool(_get_nested(protocol, "strict_fidelity", default=False))

    reference_shape = _as_tuple3(
        _get_nested(cfg, "model", "geometry", "global_voxel_shape", default=None),
        (190, 200, 104),
    )
    native_shape = _get_nested(cfg, "model", method_id, "native_output_shape", default=None)
    if native_shape is None:
        native_shape = _get_nested(cfg, "model", "benchmark", "native_output_shape", default=None)
    if native_shape is not None:
        native_shape = _as_tuple3(native_shape, reference_shape)

    if table_tier == "main" and not spec.main_table_allowed:
        raise ValueError(
            f"{method_id} is not approved for the TMI main table: {spec.reason} "
            "Use benchmark_protocol.table_tier=appendix/development or implement the missing defining mechanisms."
        )
    if strict and spec.fidelity in {"architecture_proxy", "unclassified"}:
        raise ValueError(f"Strict fidelity check failed for {method_id}: {spec.reason}")

    reference_voxels = math.prod(reference_shape)
    if method_id == "vox_dmrn" and reference_voxels > 1_000_000:
        allow = bool(_get_nested(protocol, "allow_unsafe_fully_connected_head", default=False))
        if not allow:
            raise ValueError(
                "vox_dmrn is blocked on the common reference grid because its fully connected output head "
                f"would emit {reference_voxels:,} voxels. Keep its native grid and evaluate after fixed resampling, "
                "or use it only in a single-view appendix experiment."
            )

    return spec


def write_baseline_manifest(cfg: Any, output_dir: str | Path) -> Path:
    """Write an auditable method-fidelity record next to experiment outputs."""

    spec = validate_baseline_protocol(cfg)
    method_id = str(_get_nested(cfg, "model", "name", default="")).lower()
    reference_shape = _as_tuple3(
        _get_nested(cfg, "model", "geometry", "global_voxel_shape", default=None),
        (190, 200, 104),
    )
    native_shape = _get_nested(cfg, "model", method_id, "native_output_shape", default=None)
    if native_shape is None:
        native_shape = _get_nested(cfg, "model", "benchmark", "native_output_shape", default=None)
    payload = {
        "baseline": asdict(spec),
        "reference_grid_shape": list(reference_shape),
        "native_grid_shape": list(native_shape) if native_shape is not None else None,
        "fixed_resampling_to_reference": native_shape is not None and tuple(native_shape) != reference_shape,
    }
    path = Path(output_dir) / "baseline_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
