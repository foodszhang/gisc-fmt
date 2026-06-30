"""Sample-level source-hypothesis construction for PHSA training.

The released view-complementary model historically reused the current density-query
set to construct source hypotheses. Full-volume inference instead constructs the
hypotheses once from a deterministic proposal set and reuses them across decoder
chunks. This module aligns training/validation with that sample-level definition
without enabling the rejected fixed-grid or bounded-routing variants.
"""

from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Iterator
from typing import Any, Callable

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

_ACTIVATED = False


def _plain_config(config: Any) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        value = OmegaConf.to_container(config, resolve=True)
        return value if isinstance(value, dict) else {}
    return config if isinstance(config, dict) else {}


def _stable_sample_seed(base_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return (int(base_seed) + int.from_bytes(digest[:8], "little")) % (2**63 - 1)


def _deterministic_points_mm(
    sample_id: str,
    shape: tuple[int, int, int],
    voxel_size_mm: float,
    count: int,
    base_seed: int,
) -> torch.Tensor:
    total = int(np.prod(shape))
    actual = min(max(int(count), 1), total)
    rng = np.random.default_rng(_stable_sample_seed(base_seed, sample_id))
    indices = np.sort(rng.choice(total, size=actual, replace=False).astype(np.int64))
    ijk = np.stack(np.unravel_index(indices, shape), axis=-1).astype(np.float32)
    return torch.from_numpy((ijk + 0.5) * float(voxel_size_mm))


@contextlib.contextmanager
def _replace_forward(module: torch.nn.Module, replacement: Callable[..., Any]) -> Iterator[None]:
    original = module.forward
    module.forward = replacement  # type: ignore[method-assign]
    try:
        yield
    finally:
        module.forward = original  # type: ignore[method-assign]


def activate_phsa_sample_level_hypotheses() -> None:
    """Patch the instantiated Patch-2 class before model construction.

    The patch is parameter-free and therefore preserves state-dict compatibility.
    It changes only how proposal evidence is constructed during dataloader-driven
    training/validation when ``batch.sample_id`` is available.
    """

    global _ACTIVATED
    if _ACTIVATED:
        return

    from minr_fmt.model_factory import SSQFMTPatch2

    if getattr(SSQFMTPatch2, "_phsa_sample_level_patch", False):
        _ACTIVATED = True
        return

    original_init = SSQFMTPatch2.__init__
    original_forward_view = SSQFMTPatch2._forward_view_complementary
    original_evidence_loss = SSQFMTPatch2._view_evidence_heatmap_loss

    def patched_init(self, config: Any) -> None:
        original_init(self, config)
        root = _plain_config(config)
        model_cfg = root.get("model", {}) if isinstance(root, dict) else {}
        ssq_cfg = model_cfg.get("ssq_fmt", {}) if isinstance(model_cfg, dict) else {}
        view_cfg = ssq_cfg.get("view_complementary", {}) if isinstance(ssq_cfg, dict) else {}
        sample_cfg = view_cfg.get("sample_level_hypotheses", {}) or {}
        geometry_cfg = model_cfg.get("geometry", {}) if isinstance(model_cfg, dict) else {}
        data_cfg = root.get("data", {}) if isinstance(root, dict) else {}

        self.phsa_sample_level_hypotheses_enabled = bool(sample_cfg.get("enabled", False))
        self.phsa_sample_level_hypothesis_count = int(sample_cfg.get("count", 4096))
        self.phsa_sample_level_hypothesis_seed = int(sample_cfg.get("seed", 42))
        shape = geometry_cfg.get("global_voxel_shape", [190, 200, 104])
        self.phsa_sample_level_voxel_shape = tuple(int(v) for v in shape)
        self.phsa_sample_level_voxel_size_mm = float(data_cfg.get("voxel_size_mm", 0.2))
        self._phsa_hypothesis_point_cache: dict[str, torch.Tensor] = {}
        self._phsa_active_hypothesis_points: torch.Tensor | None = None
        self._phsa_active_hypothesis_valid: torch.Tensor | None = None

    def sample_level_points(self, batch: dict[str, Any], reference: torch.Tensor) -> torch.Tensor | None:
        if not getattr(self, "phsa_sample_level_hypotheses_enabled", False):
            return None
        sample_ids = batch.get("sample_id")
        if sample_ids is None:
            return None
        if isinstance(sample_ids, str):
            sample_ids = [sample_ids]
        sample_ids = [str(value) for value in sample_ids]
        if len(sample_ids) != reference.shape[0]:
            raise RuntimeError(
                "sample_id batch size does not match density-query batch size: "
                f"ids={len(sample_ids)} batch={reference.shape[0]}"
            )

        points = []
        for sample_id in sample_ids:
            cache_key = (
                f"{sample_id}|{self.phsa_sample_level_hypothesis_count}|"
                f"{self.phsa_sample_level_hypothesis_seed}|"
                f"{self.phsa_sample_level_voxel_shape}|"
                f"{self.phsa_sample_level_voxel_size_mm}"
            )
            cached = self._phsa_hypothesis_point_cache.get(cache_key)
            if cached is None:
                cached = _deterministic_points_mm(
                    sample_id,
                    self.phsa_sample_level_voxel_shape,
                    self.phsa_sample_level_voxel_size_mm,
                    self.phsa_sample_level_hypothesis_count,
                    self.phsa_sample_level_hypothesis_seed,
                )
                self._phsa_hypothesis_point_cache[cache_key] = cached
            points.append(cached)
        return torch.stack(points, dim=0).to(
            device=reference.device,
            dtype=reference.dtype,
            non_blocking=True,
        )

    def patched_evidence_loss(
        self,
        evidence: torch.Tensor,
        points_mm: torch.Tensor,
        query_valid: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        active_points = getattr(self, "_phsa_active_hypothesis_points", None)
        active_valid = getattr(self, "_phsa_active_hypothesis_valid", None)
        if torch.is_tensor(active_points) and torch.is_tensor(active_valid):
            points_mm = active_points
            query_valid = active_valid
        return original_evidence_loss(
            self,
            evidence,
            points_mm,
            query_valid,
            *args,
            **kwargs,
        )

    def patched_forward_view(
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
        if batch is None or self.hypothesis_grid_enabled:
            return original_forward_view(
                self,
                features,
                measurements,
                samples,
                mapped,
                points_mm,
                detector_valid_mask,
                depth_maps,
                norm_scale,
                return_diagnostics,
                batch,
            )

        hypothesis_points = sample_level_points(self, batch, points_mm)
        if hypothesis_points is None:
            return original_forward_view(
                self,
                features,
                measurements,
                samples,
                mapped,
                points_mm,
                detector_valid_mask,
                depth_maps,
                norm_scale,
                return_diagnostics,
                batch,
            )

        from minr_fmt.network.ssq_geometry import infer_detector_margin_map

        hypothesis_mapped = self.geometry_mapper(
            hypothesis_points,
            depth_maps=depth_maps,
            detector_margin_map=infer_detector_margin_map(batch),
            detector_valid_mask=detector_valid_mask,
        )
        hypothesis_samples = self.surface_sampler(
            features,
            measurements,
            hypothesis_mapped,
            detector_valid_mask=detector_valid_mask,
            depth_maps=depth_maps,
        )
        evidence_per_view = (
            hypothesis_samples["sample_features"] * hypothesis_samples["A"][..., None]
        ).sum(dim=3)
        evidence_geometry = torch.stack(
            [
                hypothesis_mapped["detector_side_path_proxy"],
                hypothesis_mapped["boundary_distance"]
                .div(max(self.surface_sampler.boundary_margin_radius_px, 1.0e-6))
                .clamp(0.0, 1.0),
                hypothesis_samples["sigma_f"].div(
                    max(self.surface_sampler.sigma_max_px, 1.0e-6)
                ),
                hypothesis_mapped["grid"].square().sum(dim=-1).sqrt().div(2.0**0.5),
            ],
            dim=-1,
        )
        hypothesis_valid = hypothesis_samples["query_view_valid"]
        proposal = self.view_candidate_evidence(
            evidence_per_view,
            evidence_geometry,
            hypothesis_points,
            hypothesis_valid,
            grid_shape=None,
        )
        candidates = self.diverse_candidate_constructor(proposal)

        self._phsa_active_hypothesis_points = hypothesis_points
        self._phsa_active_hypothesis_valid = hypothesis_valid
        try:
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    _replace_forward(self.view_candidate_evidence, lambda *_a, **_k: proposal)
                )
                stack.enter_context(
                    _replace_forward(self.diverse_candidate_constructor, lambda *_a, **_k: candidates)
                )
                return original_forward_view(
                    self,
                    features,
                    measurements,
                    samples,
                    mapped,
                    points_mm,
                    detector_valid_mask,
                    depth_maps,
                    norm_scale,
                    return_diagnostics,
                    batch,
                )
        finally:
            self._phsa_active_hypothesis_points = None
            self._phsa_active_hypothesis_valid = None

    SSQFMTPatch2.__init__ = patched_init  # type: ignore[method-assign]
    SSQFMTPatch2._view_evidence_heatmap_loss = patched_evidence_loss  # type: ignore[method-assign]
    SSQFMTPatch2._forward_view_complementary = patched_forward_view  # type: ignore[method-assign]
    SSQFMTPatch2._phsa_sample_level_patch = True
    _ACTIVATED = True
