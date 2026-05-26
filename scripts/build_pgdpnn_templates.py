#!/usr/bin/env python
"""Build PGDPNN/FMT-ReconNet surface/source templates from FMT-SimGen training data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.models.voxel_baselines import (  # noqa: E402
    SurfaceVolumeBuilder,
    TemplateFeatureExtractor,
)


def sample_root(data_dir: Path) -> Path:
    direct = sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("sample_"))
    return data_dir if direct else data_dir / "samples"


def sample_dirs(data_dir: Path, split: str) -> list[Path]:
    root = sample_root(data_dir)
    all_samples = sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and p.name.startswith("sample_") and (p / "proj.npz").exists()
    )
    split_file = data_dir / "splits" / f"{split}.txt"
    if split_file.exists():
        names = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        by_name = {p.name: p for p in all_samples}
        return [by_name[name] for name in names if name in by_name]
    return all_samples


def load_projection(sample_dir: Path, view_angles: list[int]) -> dict[str, torch.Tensor]:
    z = np.load(sample_dir / "proj.npz")
    out = {}
    for angle in view_angles:
        key = str(int(angle))
        arr = np.nan_to_num(z[key].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        out[key] = torch.from_numpy(arr).unsqueeze(0)
    return out


def load_gt(sample_dir: Path) -> np.ndarray:
    gt = np.load(sample_dir / "gt_voxels.npy").astype(np.float32)
    gt = np.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0)
    gt = np.clip(gt, 0.0, None)
    gt_max = float(gt.max())
    if gt_max > 0:
        gt = gt / gt_max
    return gt


def kmeans(features: np.ndarray, k: int, seed: int, iters: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    if k > n:
        raise ValueError(f"K={k} exceeds available samples n={n}")
    centers = features[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        dist = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        labels = dist.argmin(axis=1)
        new_centers = centers.copy()
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                new_centers[i] = features[mask].mean(axis=0)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return labels, centers


def medoid_indices(features: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> list[int]:
    out = []
    for i in range(centers.shape[0]):
        idx = np.nonzero(labels == i)[0]
        if idx.size == 0:
            idx = np.arange(features.shape[0])
        dist = ((features[idx] - centers[i]) ** 2).sum(axis=-1)
        out.append(int(idx[int(dist.argmin())]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--roi_shape", nargs=3, type=int, default=[190, 200, 104])
    parser.add_argument(
        "--view_angles",
        nargs="+",
        type=int,
        default=[-90, -60, -30, 0, 30, 60, 90],
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    dirs = sample_dirs(data_dir, args.split)
    if args.max_samples:
        dirs = dirs[: args.max_samples]
    if not dirs:
        raise FileNotFoundError(f"No samples found for split={args.split} under {data_dir}")

    builder = SurfaceVolumeBuilder(tuple(args.roi_shape), args.view_angles)
    extractor = TemplateFeatureExtractor()
    features, case_ids = [], []
    with torch.no_grad():
        for i, sample_dir in enumerate(dirs, 1):
            surface = builder(load_projection(sample_dir, args.view_angles))
            feat = extractor(surface).cpu().numpy()[0]
            features.append(feat)
            case_ids.append(sample_dir.name)
            if i == 1 or i % 100 == 0 or i == len(dirs):
                print(f"[{i}/{len(dirs)}] {sample_dir.name}")

    feature_arr = np.asarray(features, dtype=np.float32)
    mean = feature_arr.mean(axis=0, keepdims=True)
    std = feature_arr.std(axis=0, keepdims=True) + 1.0e-6
    norm_features = (feature_arr - mean) / std
    labels, centers = kmeans(norm_features, args.k, args.seed, args.iters)
    selected = medoid_indices(norm_features, labels, centers)

    surface_templates, source_templates = [], []
    with torch.no_grad():
        for idx in selected:
            sample_dir = dirs[idx]
            surface = builder(load_projection(sample_dir, args.view_angles)).cpu().numpy()[0]
            source = load_gt(sample_dir).reshape(1, *tuple(args.roi_shape))
            surface_templates.append(surface)
            source_templates.append(source.astype(np.float32))

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        surface_templates=np.asarray(surface_templates, dtype=np.float32),
        source_templates=np.asarray(source_templates, dtype=np.float32),
        template_case_ids=np.asarray([case_ids[i] for i in selected]),
        template_features=np.asarray([feature_arr[i] for i in selected], dtype=np.float32),
        selected_indices=np.asarray(selected, dtype=np.int64),
        k=np.int64(args.k),
        seed=np.int64(args.seed),
    )
    meta = {
        "data_dir": str(data_dir),
        "split": args.split,
        "num_samples": len(dirs),
        "k": args.k,
        "seed": args.seed,
        "roi_shape": args.roi_shape,
        "view_angles": args.view_angles,
        "template_case_ids": [case_ids[i] for i in selected],
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
