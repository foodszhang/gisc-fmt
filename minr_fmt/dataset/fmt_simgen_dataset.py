"""FMT-SimGen projection dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from scipy import ndimage
from torch.utils.data import Dataset

from .query_sampler import NonGTQuerySampler


class FmtSimGenProjDataset(Dataset):
    """Read FMT-SimGen sample folders without depending on legacy MCX JSON fields."""

    def __init__(
        self,
        data_dir: str,
        config: Any = None,
        split: str = "train",
        is_training: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.data_dir = Path(data_dir).expanduser()
        self.split = str(split)
        self.is_training = bool(is_training)
        self.device = device
        self.config = self._extract_config(config)

        default_angles = [-90, -60, -30, 0, 30, 60, 90]
        self.view_angles = [int(v) for v in self.config.get("view_angles", default_angles)]
        default_sample_num = int(
            self.config.get("sample_num", self.config.get("num_queries", 16384)) or 16384
        )
        if self.is_training:
            self.sample_num = default_sample_num
        else:
            eval_sample_num = self.config.get(
                f"{self.split}_sample_num",
                self.config.get("eval_sample_num", self.config.get("eval_num_queries")),
            )
            self.sample_num = int(eval_sample_num or default_sample_num)
        self.voxel_size_mm = float(self.config.get("voxel_size_mm", 0.2) or 0.2)
        self.use_nongt_sampler = bool(self.config.get("use_nongt_sampler", False))
        self.projection_norm = str(self.config.get("projection_norm", "per_view_max"))
        if str(self.config.get("model_name", "")).lower() == "ssq_fmt":
            self.projection_norm = str(self.config.get("projection_norm", "raw"))
        self.projection_eps = float(self.config.get("projection_eps", 1e-8) or 1e-8)
        target_files = self.config.get(
            "descatter_target_files", self.config.get("descatter_target_file")
        )
        if target_files is None:
            target_files = ["proj_noscatter.npz", "no_proj.npz"]
        elif isinstance(target_files, str):
            target_files = [target_files]
        self.descatter_target_files = [str(p) for p in target_files]
        self.descatter_target_scale = self.config.get("descatter_target_scale")
        self.descatter_target_scale_mode = "raw_per_view_max"
        if isinstance(self.descatter_target_scale, str):
            self.descatter_target_scale_mode = self.descatter_target_scale
            self.descatter_target_scale = None
        elif self.descatter_target_scale is not None:
            self.descatter_target_scale = float(self.descatter_target_scale)
        self.load_stage1_prior = bool(self.config.get("load_stage1_prior", True))
        self.load_stage1_mesh = bool(self.config.get("load_stage1_mesh", True))
        stage1_files = self.config.get(
            "stage1_prior_files",
            ["stage1_recon.npy", "fem_recon.npy", "coarse_prior.npy", "stage1_voxel.npy"],
        )
        if isinstance(stage1_files, str):
            stage1_files = [stage1_files]
        self.stage1_prior_files = [str(p) for p in stage1_files]
        stage1_mesh_files = self.config.get(
            "stage1_mesh_files",
            ["coarse_d.npy", "stage1_mesh.npy", "fem_nodes.npy"],
        )
        if isinstance(stage1_mesh_files, str):
            stage1_mesh_files = [stage1_mesh_files]
        self.stage1_mesh_files = [str(p) for p in stage1_mesh_files]
        self.query_sampler = NonGTQuerySampler(self.config) if self.use_nongt_sampler else None
        self.center_distance_targets_enabled = bool(
            self.config.get("center_distance_targets", False)
        )
        self.center_distance_subdir = str(
            self.config.get("center_distance_subdir", "center_distance")
        )
        self.center_distance_require_precomputed = bool(
            self.config.get("center_distance_require_precomputed", False)
        )
        self.center_distance_filenames = {
            "center_target": str(self.config.get("center_target_filename", "center_target.npy")),
            "distance_target": str(
                self.config.get("distance_target_filename", "distance_target.npy")
            ),
            "fg_mask": str(self.config.get("center_distance_fg_filename", "fg_mask.npy")),
        }
        self._center_distance_cache: dict[str, dict[str, np.ndarray]] = {}
        source_hyp_cfg = self.config.get("source_hypothesis", {}) or {}
        self.source_hypothesis_enabled = bool(source_hyp_cfg.get("enabled", False))
        self.source_hypothesis_top_m = int(source_hyp_cfg.get("top_m", 5))
        self.source_hypothesis_blur_sigma = float(source_hyp_cfg.get("blur_sigma", 1.0))
        self.source_hypothesis_min_distance_cells = int(source_hyp_cfg.get("min_distance_cells", 3))
        self.source_hypothesis_min_value_ratio = float(source_hyp_cfg.get("min_value_ratio", 0.1))
        ssq_candidate_cfg = self.config.get("ssq_candidates", {}) or {}
        self.ssq_candidates_enabled = bool(ssq_candidate_cfg.get("enabled", False))
        if self.ssq_candidates_enabled:
            self.source_hypothesis_top_m = int(
                ssq_candidate_cfg.get("top_m", self.source_hypothesis_top_m)
            )
            self.source_hypothesis_blur_sigma = float(
                ssq_candidate_cfg.get("blur_sigma", self.source_hypothesis_blur_sigma)
            )
            self.source_hypothesis_min_distance_cells = int(
                ssq_candidate_cfg.get(
                    "min_distance_cells", self.source_hypothesis_min_distance_cells
                )
            )
            self.source_hypothesis_min_value_ratio = float(
                ssq_candidate_cfg.get("min_value_ratio", self.source_hypothesis_min_value_ratio)
            )
        self.proposal_subdir = str(self.config.get("proposal_subdir", "proposal"))
        self.proposal_filename = str(
            self.config.get("proposal_filename", "meas_backproj_heatmap.npy")
        )
        self.proposal_meta_filename = str(
            self.config.get("proposal_meta_filename", "meas_backproj_meta.json")
        )
        self.resample_queries_each_epoch = bool(
            self.config.get("resample_queries_each_epoch", False)
        )
        self.query_epoch_seed_stride = int(
            self.config.get("query_epoch_seed_stride", 1_000_003) or 1_000_003
        )
        self.current_epoch = 0
        self._base_seed = int(self.config.get("subset_seed", 0) or 0) + {
            "train": 0,
            "val": 1000,
            "test": 2000,
        }.get(self.split, 0)

        all_samples = self._scan_samples()
        self.dirs = self._select_split(all_samples)
        self.dirs = self._apply_subset(self.dirs)
        print(f"FmtSimGenProjDataset split={self.split}: {len(self.dirs)} samples")

    def _extract_config(self, config: Any) -> dict:
        if config is None:
            return {}
        if isinstance(config, DictConfig):
            cfg = config.data if "data" in config else config
            return OmegaConf.to_container(cfg, resolve=True)
        if isinstance(config, dict):
            return config.get("data", config)
        return {}

    def _sample_root(self) -> Path:
        direct = sorted(
            p for p in self.data_dir.iterdir() if p.is_dir() and p.name.startswith("sample_")
        )
        if direct:
            return self.data_dir
        nested = self.data_dir / "samples"
        if nested.exists():
            return nested
        return self.data_dir

    def _scan_samples(self) -> list[Path]:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"FMT-SimGen data_dir does not exist: {self.data_dir}")

        sample_root = self._sample_root()
        samples = []
        for p in sorted(sample_root.iterdir(), key=lambda x: x.name):
            if not (p.is_dir() and p.name.startswith("sample_")):
                continue
            if (p / "proj.npz").exists() and (p / "gt_voxels.npy").exists():
                samples.append(p)
        if not samples:
            raise FileNotFoundError(
                f"No valid sample_XXXX folders with proj.npz and gt_voxels.npy under {sample_root}"
            )
        return samples

    def _select_split(self, all_samples: list[Path]) -> list[Path]:
        split_file = self.data_dir / "splits" / f"{self.split}.txt"
        if split_file.exists():
            names = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
            by_name = {p.name: p for p in all_samples}
            selected = [by_name[name] for name in names if name in by_name]
            if selected:
                return selected

        n = len(all_samples)
        if n == 1000:
            train_end, val_end = 700, 900
        else:
            train_end = int(round(n * 0.7))
            val_end = train_end + int(round(n * 0.2))

        if self.split == "train":
            return all_samples[:train_end]
        if self.split == "val":
            return all_samples[train_end:val_end]
        if self.split == "test":
            return all_samples[val_end:]
        return all_samples

    def _apply_subset(self, dirs: list[Path]) -> list[Path]:
        max_key = f"{self.split}_max_samples"
        max_samples = self.config.get(max_key, self.config.get("max_samples"))
        if max_samples is None:
            return dirs
        max_samples = int(max_samples)
        if max_samples <= 0 or max_samples >= len(dirs):
            return dirs
        if str(self.config.get("subset_policy", "first")) == "random":
            rng = np.random.default_rng(self._base_seed)
            idx = sorted(int(i) for i in rng.permutation(len(dirs))[:max_samples])
            return [dirs[i] for i in idx]
        return dirs[:max_samples]

    def __len__(self) -> int:
        return len(self.dirs)

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for optional train-time query resampling.

        This changes only the RNG seed used for query allocation. It does not use GT
        information and is disabled by default for validation/test datasets.
        """
        self.current_epoch = int(epoch)

    def _query_seed(self, index: int) -> int:
        seed = self._base_seed + int(index)
        if self.resample_queries_each_epoch and self.is_training:
            seed += int(self.current_epoch) * self.query_epoch_seed_stride
        return seed

    def _load_projection(
        self, sample_dir: Path
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor] | None,
    ]:
        z = np.load(sample_dir / "proj.npz")
        descatter_zip = None
        descatter_path = None
        for candidate in self.descatter_target_files:
            candidate_path = sample_dir / candidate
            if candidate_path.exists():
                descatter_path = candidate_path
                descatter_zip = np.load(candidate_path)
                break
        projections = {}
        descatter_targets = {} if descatter_zip is not None else None
        packed = []
        scales = []
        depth_maps = []

        for angle in self.view_angles:
            key = str(angle)
            if key not in z.files:
                raise KeyError(f"{sample_dir / 'proj.npz'} missing projection key {key!r}")
            proj = np.nan_to_num(z[key].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            if self.projection_norm == "per_view_max":
                scale = max(float(np.abs(proj).max()), self.projection_eps)
                proj = proj / scale
            else:
                scale = 1.0
            t = torch.tensor(proj, dtype=torch.float32, device=self.device)
            projections[key] = t
            packed.append(t)
            scales.append(scale)

            if descatter_zip is not None:
                if key not in descatter_zip.files:
                    raise KeyError(f"{descatter_path} missing descatter target key {key!r}")
                descatter = np.nan_to_num(
                    descatter_zip[key].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
                )
                target_scale = self.descatter_target_scale
                if target_scale is None:
                    if self.descatter_target_scale_mode == "descatter_per_view_max":
                        target_scale = max(float(np.abs(descatter).max()), self.projection_eps)
                    elif self.descatter_target_scale_mode == "raw_per_view_max":
                        target_scale = scale
                    else:
                        raise ValueError(
                            "data.descatter_target_scale must be numeric, "
                            "'raw_per_view_max', or 'descatter_per_view_max', got "
                            f"{self.descatter_target_scale_mode!r}"
                        )
                descatter = descatter / max(float(target_scale), self.projection_eps)
                descatter_targets[key] = torch.tensor(
                    descatter, dtype=torch.float32, device=self.device
                )

            depth_key = f"depth_{angle}"
            if depth_key in z.files:
                depth = z[depth_key].astype(np.float32)
                depth = np.nan_to_num(depth, nan=np.inf, posinf=np.inf, neginf=np.inf)
            else:
                # Missing depth maps are marked invalid with +inf; exit-depth PTFA then
                # clamps sigma to sigma_min for valid projections rather than fabricating geometry.
                depth = np.full_like(proj, np.inf, dtype=np.float32)
            depth_maps.append(torch.tensor(depth, dtype=torch.float32, device=self.device))

        projections_packed = torch.stack(packed, dim=0).unsqueeze(1)
        projection_scales = torch.tensor(scales, dtype=torch.float32, device=self.device)
        depth_maps_tensor = torch.stack(depth_maps, dim=0)
        return (
            projections,
            projections_packed,
            projection_scales,
            depth_maps_tensor,
            descatter_targets,
        )

    def _load_gt(self, sample_dir: Path) -> np.ndarray:
        gt = np.load(sample_dir / "gt_voxels.npy").astype(np.float32)
        gt = np.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0)
        gt = np.clip(gt, 0.0, None)
        gt_max = float(gt.max())
        if gt_max > 0:
            gt = gt / gt_max
        return gt

    def _load_stage1_prior(self, sample_dir: Path, target_shape: tuple[int, ...]):
        for rel_path in self.stage1_prior_files:
            path = sample_dir / rel_path
            if not path.exists():
                continue
            prior = np.load(path).astype(np.float32)
            prior = np.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0)
            prior = np.clip(prior, 0.0, None)
            prior_max = float(prior.max())
            if prior_max > 0:
                prior = prior / prior_max
            if prior.shape != target_shape:
                t = torch.tensor(prior, dtype=torch.float32).view(1, 1, *prior.shape)
                t = torch.nn.functional.interpolate(
                    t,
                    size=target_shape,
                    mode="trilinear",
                    align_corners=False,
                )
                prior = t[0, 0].numpy()
            return prior, path.name
        return None, None

    def _load_stage1_mesh(self, sample_dir: Path):
        for rel_path in self.stage1_mesh_files:
            path = sample_dir / rel_path
            if not path.exists():
                continue
            values = np.load(path).astype(np.float32)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
            values = np.clip(values, 0.0, None)
            vmax = float(values.max())
            if vmax > 0:
                values = values / vmax
            return values, path.name
        return None, None

    def _load_measurement_b(self, sample_dir: Path):
        path = sample_dir / "measurement_b.npy"
        if not path.exists():
            return None
        b = np.load(path).astype(np.float32).reshape(-1)
        b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
        b = np.clip(b, 0.0, None)
        bmax = float(b.max())
        if bmax > 0:
            b = b / bmax
        return b

    def _load_source_hypotheses(self, sample_dir: Path) -> dict[str, np.ndarray]:
        """Extract measurement-derived source hypotheses from proposal heatmap only."""
        top_m = max(1, int(self.source_hypothesis_top_m))
        centers = np.zeros((top_m, 3), dtype=np.float32)
        peak_scores = np.zeros((top_m,), dtype=np.float32)
        scales = np.zeros((top_m,), dtype=np.float32)
        valid = np.zeros((top_m,), dtype=np.float32)

        proposal_dir = sample_dir / self.proposal_subdir
        heatmap_path = proposal_dir / self.proposal_filename
        meta_path = proposal_dir / self.proposal_meta_filename
        if not heatmap_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Source hypothesis proposal heatmap missing for {sample_dir}. "
                "Run scripts/precompute_measurement_proposal.py first."
            )

        heatmap = np.load(heatmap_path).astype(np.float64)
        meta = json.loads(meta_path.read_text())
        grid_size = tuple(int(v) for v in meta.get("grid_size", heatmap.shape))
        if heatmap.shape != grid_size:
            raise ValueError(f"{heatmap_path} shape {heatmap.shape} != meta grid_size {grid_size}")

        heatmap = np.clip(np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        if float(heatmap.max()) <= 0.0:
            return {
                "centers": centers,
                "peak_scores": peak_scores,
                "scales": scales,
                "valid": valid,
            }

        smooth = ndimage.gaussian_filter(heatmap, sigma=self.source_hypothesis_blur_sigma)
        max_value = float(np.max(smooth))
        if max_value <= 0.0 or not np.isfinite(max_value):
            return {
                "centers": centers,
                "peak_scores": peak_scores,
                "scales": scales,
                "valid": valid,
            }

        min_distance = max(1, int(self.source_hypothesis_min_distance_cells))
        size = 2 * min_distance + 1
        local_max = smooth == ndimage.maximum_filter(smooth, size=size, mode="nearest")
        local_max &= smooth >= (self.source_hypothesis_min_value_ratio * max_value)
        coords = np.argwhere(local_max)
        if coords.size == 0:
            return {
                "centers": centers,
                "peak_scores": peak_scores,
                "scales": scales,
                "valid": valid,
            }

        values = smooth[tuple(coords.T)]
        order = np.argsort(-values)
        selected: list[tuple[int, int, int]] = []
        selected_values: list[float] = []
        for idx in order:
            cand = tuple(int(v) for v in coords[idx])
            far_enough = all(
                np.linalg.norm(np.asarray(cand) - np.asarray(prev)) >= float(min_distance)
                for prev in selected
            )
            if far_enough:
                selected.append(cand)
                selected_values.append(float(values[idx]))
            if len(selected) >= top_m:
                break

        trunk_size_mm = np.asarray(meta.get("trunk_size_mm", [38.0, 40.0, 20.8]), dtype=np.float32)
        cell_size_mm = np.asarray(meta.get("cell_size_mm"), dtype=np.float32)
        if cell_size_mm.shape != (3,):
            cell_size_mm = trunk_size_mm / np.asarray(grid_size, dtype=np.float32)
        radius_cells = max(1, min_distance)

        for i, (coord, value) in enumerate(zip(selected, selected_values)):
            coord_arr = np.asarray(coord, dtype=np.int64)
            centers[i] = (coord_arr.astype(np.float32) + 0.5) * cell_size_mm
            peak_scores[i] = float(value / max_value)
            lo = np.maximum(coord_arr - radius_cells, 0)
            hi = np.minimum(coord_arr + radius_cells + 1, np.asarray(grid_size, dtype=np.int64))
            patch = smooth[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
            patch_sum = float(np.sum(patch))
            if patch_sum > 0.0 and np.isfinite(patch_sum):
                gx, gy, gz = np.meshgrid(
                    np.arange(lo[0], hi[0], dtype=np.float32),
                    np.arange(lo[1], hi[1], dtype=np.float32),
                    np.arange(lo[2], hi[2], dtype=np.float32),
                    indexing="ij",
                )
                coords_mm = np.stack([gx + 0.5, gy + 0.5, gz + 0.5], axis=-1) * cell_size_mm
                center_mm = centers[i][None, None, None, :]
                second_moment = float(
                    np.sum(patch[..., None] * np.square(coords_mm - center_mm)) /
                    (3.0 * patch_sum + 1e-8)
                )
                scale_mm = float(np.sqrt(max(second_moment, 0.0)))
            else:
                scale_mm = float(np.linalg.norm(cell_size_mm) * radius_cells)
            scales[i] = float(np.clip(scale_mm, 1e-6, np.inf))
            valid[i] = 1.0
        return {
            "centers": centers,
            "peak_scores": peak_scores,
            "scales": scales,
            "valid": valid,
        }

    def _build_center_distance_targets(self, gt: np.ndarray) -> dict[str, np.ndarray]:
        structure = ndimage.generate_binary_structure(3, 1)
        labeled, num = ndimage.label(gt > 0.0, structure=structure)
        center_target = np.zeros_like(gt, dtype=np.float32)
        distance_target = np.zeros_like(gt, dtype=np.float32)
        fg_mask = np.zeros_like(gt, dtype=np.float32)
        if num <= 0:
            return {
                "center_target": center_target,
                "distance_target": distance_target,
                "fg_mask": fg_mask,
            }

        xs = np.arange(gt.shape[0], dtype=np.float32)[:, None, None]
        ys = np.arange(gt.shape[1], dtype=np.float32)[None, :, None]
        zs = np.arange(gt.shape[2], dtype=np.float32)[None, None, :]
        sigma_vox = max(
            float(self.config.get("center_sigma_mm", 0.5)) / max(self.voxel_size_mm, 1e-6),
            1e-6,
        )

        for label_id in range(1, num + 1):
            mask = labeled == label_id
            if not mask.any():
                continue
            coords = np.argwhere(mask).astype(np.float32)
            centroid = coords.mean(axis=0)
            dist_sq = (xs - centroid[0]) ** 2 + (ys - centroid[1]) ** 2 + (zs - centroid[2]) ** 2
            center = np.exp(-dist_sq / (2.0 * sigma_vox**2)).astype(np.float32)
            center_target = np.maximum(center_target, center)
            dist_map = ndimage.distance_transform_edt(mask).astype(np.float32)
            radius = max(float(dist_map[mask].max()), 1.0)
            fg = mask.astype(np.float32)
            fg_mask = np.maximum(fg_mask, fg)
            distance = np.where(mask, dist_map / radius, 0.0).astype(np.float32)
            distance_target = np.maximum(distance_target, distance)

        return {
            "center_target": center_target,
            "distance_target": distance_target,
            "fg_mask": fg_mask,
        }

    def _center_distance_targets(self, sample_dir: Path, gt: np.ndarray) -> dict[str, np.ndarray]:
        if not self.center_distance_targets_enabled:
            return {}
        precomputed = self._load_precomputed_center_distance_targets(sample_dir)
        if precomputed is not None:
            return precomputed
        if self.center_distance_require_precomputed:
            target_dir = sample_dir / self.center_distance_subdir
            raise FileNotFoundError(
                f"Missing precomputed center-distance targets under {target_dir}. "
                "Run scripts/precompute_center_distance_targets.py first, or set "
                "data.center_distance_require_precomputed=false to use slow online generation."
            )
        key = sample_dir.name
        cached = self._center_distance_cache.get(key)
        if cached is None:
            cached = self._build_center_distance_targets(gt)
            self._center_distance_cache[key] = cached
        return cached

    def _load_precomputed_center_distance_targets(
        self, sample_dir: Path
    ) -> dict[str, np.ndarray] | None:
        target_dir = sample_dir / self.center_distance_subdir
        paths = {
            key: target_dir / filename for key, filename in self.center_distance_filenames.items()
        }
        if not all(path.exists() for path in paths.values()):
            return None
        return {
            key: np.load(path, mmap_mode="r").astype(np.float32, copy=False)
            for key, path in paths.items()
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_dir = self.dirs[index]
        (
            projections,
            projections_packed,
            projection_scales,
            depth_maps,
            descatter_targets,
        ) = self._load_projection(sample_dir)
        gt = self._load_gt(sample_dir)
        if self.load_stage1_prior:
            stage1_prior, stage1_source = self._load_stage1_prior(sample_dir, gt.shape)
        else:
            stage1_prior, stage1_source = None, None
        if self.load_stage1_mesh:
            stage1_mesh, stage1_mesh_source = self._load_stage1_mesh(sample_dir)
        else:
            stage1_mesh, stage1_mesh_source = None, None
        measurement_b = self._load_measurement_b(sample_dir)
        gt_nodes_path = sample_dir / "gt_nodes.npy"
        gt_nodes = None
        if gt_nodes_path.exists():
            gt_nodes = np.load(gt_nodes_path).astype(np.float32).reshape(-1)
            gt_nodes = np.nan_to_num(gt_nodes, nan=0.0, posinf=0.0, neginf=0.0)
            gt_nodes = np.clip(gt_nodes, 0.0, None)
            gt_nodes_max = float(gt_nodes.max())
            if gt_nodes_max > 0:
                gt_nodes = gt_nodes / gt_nodes_max
        rng = np.random.default_rng(self._query_seed(index))

        if self.query_sampler is not None:
            sampled = self.query_sampler.sample(sample_dir, gt.shape, self.sample_num, rng)
            ix = sampled["ix"].astype(np.int64)
            iy = sampled["iy"].astype(np.int64)
            iz = sampled["iz"].astype(np.int64)
            points_ijk = np.stack([ix, iy, iz], axis=-1).astype(np.float32)
            points_mm = sampled.get("points_mm")
            if points_mm is None:
                points_mm = (points_ijk + 0.5) * self.voxel_size_mm
            query_src_tag = sampled["src_tag"].astype(np.int64)
        else:
            total_voxels = int(np.prod(gt.shape))
            replace = self.sample_num > total_voxels
            choice = rng.choice(total_voxels, size=self.sample_num, replace=replace)
            points_ijk = np.stack(np.unravel_index(choice, gt.shape), axis=-1).astype(np.float32)
            # points_mm is trunk-local mm aligned to FMT-SimGen physical projection.
            points_mm = (points_ijk + 0.5) * self.voxel_size_mm
            query_src_tag = np.full(self.sample_num, -1, dtype=np.int64)

        denom = np.maximum(np.asarray(gt.shape, dtype=np.float32) - 1.0, 1.0)
        points_norm = points_ijk / denom
        point_densities = gt[
            points_ijk[:, 0].astype(np.int64),
            points_ijk[:, 1].astype(np.int64),
            points_ijk[:, 2].astype(np.int64),
        ]
        center_distance_targets = self._center_distance_targets(sample_dir, gt)

        num_foci = -1
        tumor_path = sample_dir / "tumor_params.json"
        if tumor_path.exists():
            try:
                num_foci = int(json.loads(tumor_path.read_text()).get("num_foci", -1))
            except Exception:
                num_foci = -1

        item = {
            "sample_id": sample_dir.name,
            "projections": projections,
            "projections_packed": projections_packed,
            "surface_measurements": projections,
            "surface_measurements_packed": projections_packed,
            "detector_valid_mask": torch.isfinite(depth_maps),
            "depth_maps": depth_maps,
            "gt_voxels": torch.tensor(gt, dtype=torch.float32, device=self.device),
            "points": torch.tensor(points_norm, dtype=torch.float32, device=self.device),
            "point_densities": torch.tensor(
                point_densities, dtype=torch.float32, device=self.device
            ),
            "points_ijk": torch.tensor(points_ijk, dtype=torch.float32, device=self.device),
            "points_mm": torch.tensor(points_mm, dtype=torch.float32, device=self.device),
            "query_coordinates_mm": torch.tensor(
                points_mm, dtype=torch.float32, device=self.device
            ),
            "query_src_tag": torch.tensor(query_src_tag, dtype=torch.long, device=self.device),
            "global_voxel_shape": tuple(int(v) for v in gt.shape),
            "feasible_voxel_shape": tuple(int(v) for v in gt.shape),
            "range_x": (0, int(gt.shape[0])),
            "range_y": (0, int(gt.shape[1])),
            "range_z": (0, int(gt.shape[2])),
            "projection_scales": projection_scales,
            "num_foci": num_foci,
        }
        if center_distance_targets:
            ix = points_ijk[:, 0].astype(np.int64)
            iy = points_ijk[:, 1].astype(np.int64)
            iz = points_ijk[:, 2].astype(np.int64)
            item["center_target"] = torch.tensor(
                center_distance_targets["center_target"][ix, iy, iz],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(-1)
            item["distance_target"] = torch.tensor(
                center_distance_targets["distance_target"][ix, iy, iz],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(-1)
            item["center_distance_fg_mask"] = torch.tensor(
                center_distance_targets["fg_mask"][ix, iy, iz],
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(-1)
        if descatter_targets is not None:
            item["descatter_targets"] = descatter_targets
        if stage1_prior is not None:
            item["stage1_voxel"] = torch.tensor(
                stage1_prior, dtype=torch.float32, device=self.device
            )
            item["stage1_source"] = stage1_source
        if stage1_mesh is not None:
            item["stage1_mesh"] = torch.tensor(stage1_mesh, dtype=torch.float32, device=self.device)
            item["stage1_mesh_source"] = stage1_mesh_source
        if measurement_b is not None:
            item["measurement_b"] = torch.tensor(
                measurement_b, dtype=torch.float32, device=self.device
            )
        if gt_nodes is not None:
            item["gt_nodes"] = torch.tensor(gt_nodes, dtype=torch.float32, device=self.device)
        if self.source_hypothesis_enabled or self.ssq_candidates_enabled:
            source_hyp = self._load_source_hypotheses(sample_dir)
            if self.ssq_candidates_enabled:
                item["candidate_centers_mm"] = torch.tensor(
                    source_hyp["centers"], dtype=torch.float32, device=self.device
                )
                item["candidate_scores"] = torch.tensor(
                    source_hyp["peak_scores"], dtype=torch.float32, device=self.device
                )
                item["candidate_scales_mm"] = torch.tensor(
                    source_hyp["scales"], dtype=torch.float32, device=self.device
                )
                item["candidate_valid_mask"] = torch.tensor(
                    source_hyp["valid"] > 0.0, dtype=torch.bool, device=self.device
                )
        if self.source_hypothesis_enabled:
            source_hyp = self._load_source_hypotheses(sample_dir)
            item["source_hypothesis_centers"] = torch.tensor(
                source_hyp["centers"], dtype=torch.float32, device=self.device
            )
            item["source_hypothesis_peak_scores"] = torch.tensor(
                source_hyp["peak_scores"], dtype=torch.float32, device=self.device
            )
            item["source_hypothesis_scales"] = torch.tensor(
                source_hyp["scales"], dtype=torch.float32, device=self.device
            )
            item["source_hypothesis_valid"] = torch.tensor(
                source_hyp["valid"], dtype=torch.float32, device=self.device
            )
        return item
