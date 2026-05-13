"""FMT-SimGen projection dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
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
        self.sample_num = int(
            self.config.get("sample_num", self.config.get("num_queries", 16384)) or 16384
        )
        self.voxel_size_mm = float(self.config.get("voxel_size_mm", 0.2) or 0.2)
        self.use_nongt_sampler = bool(self.config.get("use_nongt_sampler", False))
        self.projection_norm = str(self.config.get("projection_norm", "per_view_max"))
        self.projection_eps = float(self.config.get("projection_eps", 1e-8) or 1e-8)
        self.query_sampler = NonGTQuerySampler(self.config) if self.use_nongt_sampler else None
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

    def _load_projection(
        self, sample_dir: Path
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        z = np.load(sample_dir / "proj.npz")
        projections = {}
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
        return projections, projections_packed, projection_scales, depth_maps_tensor

    def _load_gt(self, sample_dir: Path) -> np.ndarray:
        gt = np.load(sample_dir / "gt_voxels.npy").astype(np.float32)
        gt = np.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0)
        gt = np.clip(gt, 0.0, None)
        gt_max = float(gt.max())
        if gt_max > 0:
            gt = gt / gt_max
        return gt

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_dir = self.dirs[index]
        projections, projections_packed, projection_scales, depth_maps = self._load_projection(
            sample_dir
        )
        gt = self._load_gt(sample_dir)
        rng = np.random.default_rng(self._base_seed + int(index))

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

        num_foci = -1
        tumor_path = sample_dir / "tumor_params.json"
        if tumor_path.exists():
            try:
                num_foci = int(json.loads(tumor_path.read_text()).get("num_foci", -1))
            except Exception:
                num_foci = -1

        return {
            "sample_id": sample_dir.name,
            "projections": projections,
            "projections_packed": projections_packed,
            "depth_maps": depth_maps,
            "points": torch.tensor(points_norm, dtype=torch.float32, device=self.device),
            "point_densities": torch.tensor(
                point_densities, dtype=torch.float32, device=self.device
            ),
            "points_ijk": torch.tensor(points_ijk, dtype=torch.float32, device=self.device),
            "points_mm": torch.tensor(points_mm, dtype=torch.float32, device=self.device),
            "query_src_tag": torch.tensor(query_src_tag, dtype=torch.long, device=self.device),
            "global_voxel_shape": tuple(int(v) for v in gt.shape),
            "feasible_voxel_shape": tuple(int(v) for v in gt.shape),
            "projection_scales": projection_scales,
            "num_foci": num_foci,
        }
