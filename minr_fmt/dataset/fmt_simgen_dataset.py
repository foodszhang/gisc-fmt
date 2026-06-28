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

from minr_fmt.utils.ssq_candidate_extraction import (
    CANDIDATE_CACHE_VERSION,
    candidate_support_covariances_mm,
    extract_candidate_anchors,
    load_candidate_cache,
)

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
        self.ssq_anisotropic_support = bool(
            ssq_candidate_cfg.get("anisotropic_support", False)
        )
        self.ssq_support_moment_radius_mm = float(
            ssq_candidate_cfg.get("support_moment_radius_mm", 4.0)
        )
        self.ssq_reextract_candidates = bool(
            ssq_candidate_cfg.get("reextract_candidates", False)
        )
        self.ssq_candidate_smoothing_sigma_cells = float(
            ssq_candidate_cfg.get("smoothing_sigma_cells", 1.0)
        )
        self.ssq_candidate_nms_radius_mm = float(
            ssq_candidate_cfg.get("nms_radius_mm", 3.0)
        )
        self.ssq_candidate_min_value_ratio = float(
            ssq_candidate_cfg.get("min_value_ratio", 0.1)
        )
        self.ssq_candidate_filename = str(
            ssq_candidate_cfg.get("candidate_filename", "candidate_anchors.npz")
        )
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
        morph_cfg = self.config.get("morphology", {}) or {}
        self.morphology_enabled = bool(morph_cfg.get("enabled", False))
        self.morphology_require_precomputed = bool(morph_cfg.get("require_precomputed", False))
        self.morphology_subdir = str(morph_cfg.get("subdir", "morphology"))
        self.morphology_target_filename = str(morph_cfg.get("target_filename", "sdf_target.npy"))
        self.morphology_meta_filename = str(morph_cfg.get("meta_filename", "sdf_meta.json"))
        self.morphology_target_version = str(
            morph_cfg.get("target_version", "ssq_candidate_conditioned_sdf_v2")
        )
        self.resample_queries_each_epoch = bool(
            self.config.get("resample_queries_each_epoch", False)
        )
        self.query_epoch_seed_stride = int(
            self.config.get("query_epoch_seed_stride", 1_000_003) or 1_000_003
        )
        self.current_epoch = 0
        component_cfg = self.config.get("component_supervision", {}) or {}
        self.component_supervision_enabled = bool(component_cfg.get("enabled", False))
        self.component_supervision_max_components = int(
            component_cfg.get("max_components", 5) or 5
        )
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

    def _component_supervision(
        self,
        sample_dir: Path,
        gt: np.ndarray,
        points_mm: np.ndarray,
        point_densities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        max_components = self.component_supervision_max_components
        centers = np.zeros((max_components, 3), dtype=np.float32)
        valid = np.zeros(max_components, dtype=np.bool_)
        tumor_path = sample_dir / "tumor_params.json"
        foci = []
        if tumor_path.exists():
            foci = json.loads(tumor_path.read_text()).get("foci", [])
        for component_index, focus in enumerate(foci[:max_components]):
            center = focus.get("center")
            if center is not None and len(center) == 3:
                centers[component_index] = np.asarray(center, dtype=np.float32)
                valid[component_index] = True
        if not valid.any():
            labels, count = ndimage.label(
                gt > 0.5, ndimage.generate_binary_structure(3, 1)
            )
            components = []
            for label_id in range(1, count + 1):
                coordinates = np.argwhere(labels == label_id)
                if len(coordinates):
                    components.append((len(coordinates), coordinates))
            components.sort(key=lambda item: item[0], reverse=True)
            for component_index, (_, coordinates) in enumerate(
                components[:max_components]
            ):
                centers[component_index] = (
                    coordinates.mean(axis=0) + 0.5
                ) * self.voxel_size_mm
                valid[component_index] = True
        query_ids = np.zeros(len(points_mm), dtype=np.int64)
        positive = point_densities > 0.0
        if positive.any() and valid.any():
            valid_indices = np.flatnonzero(valid)
            distance = np.linalg.norm(
                points_mm[positive, None] - centers[valid_indices][None], axis=-1
            )
            query_ids[positive] = valid_indices[distance.argmin(axis=1)] + 1
        return query_ids, centers, valid

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
        """Load measurement-derived source hypotheses from the shared candidate cache."""
        top_m = max(1, int(self.source_hypothesis_top_m))
        centers = np.zeros((top_m, 3), dtype=np.float32)
        peak_scores = np.zeros((top_m,), dtype=np.float32)
        scales = np.zeros((top_m,), dtype=np.float32)
        valid = np.zeros((top_m,), dtype=np.float32)
        cache = load_candidate_cache(
            sample_dir,
            expected_version=CANDIDATE_CACHE_VERSION,
            filename=self.ssq_candidate_filename,
        )
        if self.ssq_reextract_candidates or top_m > int(cache["centers_mm"].shape[0]):
            heatmap_path = sample_dir / self.proposal_subdir / self.proposal_filename
            metadata_path = sample_dir / self.proposal_subdir / self.proposal_meta_filename
            heatmap = np.load(heatmap_path).astype(np.float32)
            metadata = json.loads(metadata_path.read_text())
            anchors = extract_candidate_anchors(
                heatmap,
                metadata,
                top_m=top_m,
                smoothing_sigma_cells=self.ssq_candidate_smoothing_sigma_cells,
                nms_radius_mm=self.ssq_candidate_nms_radius_mm,
                support_moment_radius_mm=self.ssq_support_moment_radius_mm,
                min_value_ratio=self.ssq_candidate_min_value_ratio,
            )
            cache = {
                "centers_mm": anchors["centers_mm"],
                "scores": anchors["scores"],
                "raw_support_scales_mm": anchors["raw_support_scales_mm"],
                "valid": anchors["valid"],
            }
        n = min(top_m, int(cache["centers_mm"].shape[0]))
        centers[:n] = cache["centers_mm"][:n]
        peak_scores[:n] = cache["scores"][:n]
        scales[:n] = cache["raw_support_scales_mm"][:n]
        valid[:n] = cache["valid"][:n].astype(np.float32)
        result = {
            "centers": centers,
            "peak_scores": peak_scores,
            "scales": scales,
            "valid": valid,
        }
        if self.ssq_anisotropic_support:
            heatmap_path = sample_dir / self.proposal_subdir / self.proposal_filename
            metadata_path = sample_dir / self.proposal_subdir / self.proposal_meta_filename
            if not heatmap_path.exists() or not metadata_path.exists():
                raise FileNotFoundError(
                    f"Anisotropic SSQ support requires {heatmap_path} and {metadata_path}"
                )
            heatmap = np.load(heatmap_path).astype(np.float32)
            metadata = json.loads(metadata_path.read_text())
            result["support_covariances_mm"] = candidate_support_covariances_mm(
                heatmap,
                centers,
                valid > 0.0,
                np.asarray(metadata["cell_size_mm"], dtype=np.float32),
                self.ssq_support_moment_radius_mm,
            )
        return result

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
        sdf_targets = None
        sdf_path = sample_dir / self.morphology_subdir / self.morphology_target_filename
        sdf_meta_path = sample_dir / self.morphology_subdir / self.morphology_meta_filename
        if self.morphology_enabled and sdf_path.exists():
            if not sdf_meta_path.exists():
                raise FileNotFoundError(f"SDF metadata missing for {sdf_path}")
            sdf_meta = json.loads(sdf_meta_path.read_text())
            if sdf_meta.get("version") != self.morphology_target_version:
                raise ValueError(
                    f"{sdf_meta_path} version {sdf_meta.get('version')!r} != "
                    f"{self.morphology_target_version!r}"
                )
            sdf_grid = np.load(sdf_path).astype(np.float32)
            if tuple(sdf_meta.get("target_shape", sdf_grid.shape)) != tuple(sdf_grid.shape):
                raise ValueError(f"{sdf_meta_path} target_shape does not match {sdf_path}")
            ix = points_ijk[:, 0].astype(np.int64)
            iy = points_ijk[:, 1].astype(np.int64)
            iz = points_ijk[:, 2].astype(np.int64)
            ix = np.clip(ix, 0, sdf_grid.shape[0] - 1)
            iy = np.clip(iy, 0, sdf_grid.shape[1] - 1)
            iz = np.clip(iz, 0, sdf_grid.shape[2] - 1)
            sdf_targets = sdf_grid[ix, iy, iz]
        elif self.morphology_enabled and self.morphology_require_precomputed:
            raise FileNotFoundError(f"Required SDF target missing for {sample_dir}: {sdf_path}")

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
        if self.component_supervision_enabled and self.is_training:
            component_ids, component_centers, component_valid = self._component_supervision(
                sample_dir, gt, points_mm, point_densities
            )
            item["query_component_ids"] = torch.tensor(
                component_ids, dtype=torch.long, device=self.device
            )
            item["gt_component_centers_mm"] = torch.tensor(
                component_centers, dtype=torch.float32, device=self.device
            )
            item["gt_component_valid_mask"] = torch.tensor(
                component_valid, dtype=torch.bool, device=self.device
            )
        if sdf_targets is not None:
            item["sdf_targets"] = torch.tensor(
                sdf_targets, dtype=torch.float32, device=self.device
            ).unsqueeze(-1)
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
                support_scales = torch.tensor(
                    source_hyp["scales"], dtype=torch.float32, device=self.device
                )
                item["candidate_support_scales_mm"] = support_scales
                item["candidate_scales_mm"] = support_scales
                item["candidate_valid_mask"] = torch.tensor(
                    source_hyp["valid"] > 0.0, dtype=torch.bool, device=self.device
                )
                if "support_covariances_mm" in source_hyp:
                    item["candidate_support_covariances_mm"] = torch.tensor(
                        source_hyp["support_covariances_mm"],
                        dtype=torch.float32,
                        device=self.device,
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
