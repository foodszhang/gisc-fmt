"""Non-GT query sampling for FMT-SimGen datasets.

# NO-GT-LEAKAGE GUARANTEE:
# Forbidden inputs for allocation:
#   - gt_voxels.npy
#   - gt_nodes.npy
#   - tumor_params.json
#   - any GT-derived bbox / mask
# Forbidden in this task version:
#   - body_mask from any source
# Allowed:
#   - proj.npz (multi-view measurement)
#   - proposal heatmap (precomputed from proj.npz only)
#   - trunk bounding box
#   - FMT-SimGen geometry
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf


class NonGTQuerySampler:
    """Fixed-budget sampler using trunk-uniform and measurement-proposal branches."""

    def __init__(self, config: Any):
        cfg = self._to_plain_config(config)
        qs_cfg = cfg.get("query_sampling", {}) or {}

        if bool(qs_cfg.get("use_body_mask", False)):
            raise ValueError("body_mask is disabled in this task (unreliable source).")
        if bool(qs_cfg.get("use_depth_silhouette_mask", False)):
            raise NotImplementedError(
                "depth-silhouette mask is reserved for a later ablation task."
            )
        if bool(qs_cfg.get("use_gt_bbox", False)) or bool(
            qs_cfg.get("use_gt_foreground_oversampling", False)
        ):
            raise ValueError("GT-derived query allocation is forbidden for NonGTQuerySampler.")

        self.trunk_uniform_ratio = float(qs_cfg.get("trunk_uniform_ratio", 0.5))
        self.meas_proposal_ratio = float(qs_cfg.get("meas_proposal_ratio", 0.5))
        if self.trunk_uniform_ratio < 0 or self.meas_proposal_ratio < 0:
            raise ValueError("query_sampling ratios must be non-negative.")
        if self.trunk_uniform_ratio + self.meas_proposal_ratio <= 0:
            raise ValueError("At least one query_sampling ratio must be positive.")

        self.proposal_subdir = str(cfg.get("proposal_subdir", "proposal"))
        self.proposal_filename = str(cfg.get("proposal_filename", "meas_backproj_heatmap.npy"))
        self.proposal_meta_filename = str(
            cfg.get("proposal_meta_filename", "meas_backproj_meta.json")
        )
        self.voxel_size_mm = float(cfg.get("voxel_size_mm", 0.2))
        self.trunk_size_mm = np.asarray(
            cfg.get("trunk_size_mm", [38.0, 40.0, 20.8]), dtype=np.float32
        )
        if self.trunk_size_mm.shape != (3,):
            raise ValueError(f"trunk_size_mm must have 3 values, got {self.trunk_size_mm}")

    @staticmethod
    def _to_plain_config(config: Any) -> dict:
        if config is None:
            return {}
        if isinstance(config, DictConfig):
            cfg = config.data if "data" in config else config
            return OmegaConf.to_container(cfg, resolve=True)
        if isinstance(config, dict):
            return config.get("data", config)
        return {}

    def sample(
        self,
        sample_dir: Path,
        gt_shape: tuple[int, int, int],
        n_queries: int,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        """Sample query voxel indices without using GT values for allocation."""
        n_trunk, n_proposal = self._split_budget(int(n_queries))
        parts = []
        if n_trunk > 0:
            parts.append(self._sample_trunk_uniform(gt_shape, n_trunk, rng, src_tag=0))
        if n_proposal > 0:
            parts.append(self._sample_proposal(sample_dir, gt_shape, n_proposal, rng))
        if not parts:
            raise ValueError("n_queries must be positive.")
        return self._concat(parts)

    def _split_budget(self, n_queries: int) -> tuple[int, int]:
        ratio_sum = self.trunk_uniform_ratio + self.meas_proposal_ratio
        n_proposal = int(round(n_queries * self.meas_proposal_ratio / ratio_sum))
        n_proposal = max(0, min(n_queries, n_proposal))
        n_trunk = n_queries - n_proposal
        return n_trunk, n_proposal

    def _sample_trunk_uniform(
        self,
        gt_shape: tuple[int, int, int],
        n: int,
        rng: np.random.Generator,
        src_tag: int,
    ) -> dict[str, np.ndarray]:
        ix = rng.integers(0, int(gt_shape[0]), size=n, dtype=np.int64)
        iy = rng.integers(0, int(gt_shape[1]), size=n, dtype=np.int64)
        iz = rng.integers(0, int(gt_shape[2]), size=n, dtype=np.int64)
        ijk = np.stack([ix, iy, iz], axis=-1).astype(np.float32)
        points_mm = (ijk + 0.5) * self.voxel_size_mm
        return {
            "ix": ix,
            "iy": iy,
            "iz": iz,
            "points_mm": points_mm.astype(np.float32),
            "src_tag": np.full(n, src_tag, dtype=np.int64),
        }

    def _sample_proposal(
        self,
        sample_dir: Path,
        gt_shape: tuple[int, int, int],
        n: int,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        proposal_dir = sample_dir / self.proposal_subdir
        heatmap_path = proposal_dir / self.proposal_filename
        meta_path = proposal_dir / self.proposal_meta_filename
        if not heatmap_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Proposal heatmap missing for {sample_dir}. "
                "Run scripts/precompute_measurement_proposal.py first."
            )

        heatmap = np.load(heatmap_path).astype(np.float64)
        meta = json.loads(meta_path.read_text())
        grid_size = tuple(int(v) for v in meta.get("grid_size", heatmap.shape))
        if heatmap.shape != grid_size:
            raise ValueError(f"{heatmap_path} shape {heatmap.shape} != meta grid_size {grid_size}")

        p = np.clip(heatmap.reshape(-1), 0.0, None)
        total = float(p.sum())
        if total <= 0 or not np.isfinite(total):
            warnings.warn(
                f"Proposal heatmap is empty for {sample_dir}; falling back to trunk-uniform.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._sample_trunk_uniform(gt_shape, n, rng, src_tag=0)

        cells = rng.choice(p.size, size=n, replace=True, p=p / total)
        cell_ijk = np.stack(np.unravel_index(cells, grid_size), axis=-1).astype(np.float32)
        jitter = rng.uniform(low=-0.5, high=0.5, size=(n, 3)).astype(np.float32)

        trunk_size_mm = np.asarray(meta.get("trunk_size_mm", self.trunk_size_mm), dtype=np.float32)
        cell_size_mm = np.asarray(meta.get("cell_size_mm"), dtype=np.float32)
        if cell_size_mm.shape != (3,):
            cell_size_mm = trunk_size_mm / np.asarray(grid_size, dtype=np.float32)

        points_mm = (cell_ijk + 0.5 + jitter) * cell_size_mm
        points_mm = np.clip(points_mm, 0.0, trunk_size_mm - 1.0e-6)

        voxel_size_mm = float(meta.get("voxel_size_mm", self.voxel_size_mm))
        ijk_int = np.floor(points_mm / voxel_size_mm).astype(np.int64)
        ijk_int[:, 0] = np.clip(ijk_int[:, 0], 0, int(gt_shape[0]) - 1)
        ijk_int[:, 1] = np.clip(ijk_int[:, 1], 0, int(gt_shape[1]) - 1)
        ijk_int[:, 2] = np.clip(ijk_int[:, 2], 0, int(gt_shape[2]) - 1)

        return {
            "ix": ijk_int[:, 0],
            "iy": ijk_int[:, 1],
            "iz": ijk_int[:, 2],
            "points_mm": points_mm.astype(np.float32),
            "src_tag": np.full(n, 1, dtype=np.int64),
        }

    @staticmethod
    def _concat(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        return {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]}
