"""Generate DU2Vox Stage 1 FEM coarse outputs for FMT-SimGen samples.

This script reuses DU2Vox's trained Stage 1 GCAIN implementation and writes
per-case mesh and full-voxel priors:

  sample_xxxx/coarse_d.npy
  sample_xxxx/stage1_mesh.npy
  sample_xxxx/stage1_voxel.npy
  sample_xxxx/stage1_meta.json

It does not train a new model and does not fabricate FEM assets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DU2VOX_ROOT = Path("/home/foods/pro/DU2Vox")
if str(DU2VOX_ROOT) not in sys.path:
    sys.path.insert(0, str(DU2VOX_ROOT))

from du2vox.bridge.stage1_inference import _load_shared_assets  # noqa: E402
from du2vox.models.stage1.gcain import GCAIN_full  # noqa: E402
from minr_fmt.utils.fem_assets import (  # noqa: E402
    apply_mesh_to_voxel_barycentric,
    build_mesh_to_voxel_barycentric_map,
    load_mesh,
)


def _sample_root(data_dir: Path) -> Path:
    return data_dir / "samples" if (data_dir / "samples").exists() else data_dir


def _read_split_ids(data_dir: Path, split: str) -> list[str]:
    split_file = data_dir / "splits" / f"{split}.txt"
    if split_file.exists():
        return [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    root = _sample_root(data_dir)
    return sorted(p.name for p in root.glob("sample_*") if p.is_dir())


def _activation_fn(name: str, slope: float):
    if name == "sigmoid":
        return torch.sigmoid
    if name == "leaky_relu":
        return lambda x: F.leaky_relu(x, negative_slope=slope).clamp(max=1.0)
    if name == "clamp":
        return lambda x: x.clamp(0.0, 1.0)
    raise ValueError(f"Unsupported Stage1 activation: {name}")


def _load_model(
    checkpoint: Path,
    config_path: Path,
    shared_dir: Path,
    device: torch.device,
) -> tuple[GCAIN_full, dict, dict]:
    cfg = yaml.safe_load(config_path.read_text())
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    assets = _load_shared_assets(
        shared_dir,
        str(device),
        use_visible_mask=bool(data_cfg.get("use_visible_mask", False)),
    )
    model = GCAIN_full(
        L=assets["L"],
        A=assets["A"],
        LTL=assets["LTL"],
        ATA=assets["ATA"],
        L0=assets["L0"],
        L1=assets["L1"],
        L2=assets["L2"],
        L3=assets["L3"],
        knn_idx=assets["knn_idx"],
        sens_w=assets["sens_w"],
        num_layer=int(model_cfg.get("num_layer", 6)),
        feat_dim=int(model_cfg.get("feat_dim", 6)),
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    return model, cfg, assets


def _load_measurements(
    sample_dirs: list[Path],
    n_surface: int,
    normalize: bool,
    visible_mask: np.ndarray | None,
) -> torch.Tensor:
    rows = []
    for sample_dir in sample_dirs:
        b = np.load(sample_dir / "measurement_b.npy").astype(np.float32).reshape(-1)
        if visible_mask is not None and b.shape[0] != n_surface:
            b = b[visible_mask]
        if b.shape[0] != n_surface:
            raise ValueError(f"{sample_dir.name}: measurement_b rows {b.shape[0]} != {n_surface}")
        if normalize:
            bmax = float(np.max(b))
            if bmax > 1e-8:
                b = b / bmax
        rows.append(torch.from_numpy(b).unsqueeze(-1))
    return torch.stack(rows, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k")
    parser.add_argument("--shared-dir", default="/home/foods/pro/FMT-SimGen/output/shared_mesh_20k")
    parser.add_argument(
        "--checkpoint",
        default="/home/foods/pro/DU2Vox/runs/stage1_uniform_1000_20k_balanced_dice05/checkpoints/best_dice05.pth",
    )
    parser.add_argument(
        "--config",
        default="/home/foods/pro/DU2Vox/configs/stage1/uniform_1000_20k_balanced_dice05.yaml",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--voxel-shape", nargs=3, type=int, default=[190, 200, 104])
    parser.add_argument("--voxel-size-mm", type=float, default=0.2)
    parser.add_argument("--n-candidates", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-voxel", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    sample_root = _sample_root(data_dir)
    shared_dir = Path(args.shared_dir).expanduser()
    checkpoint = Path(args.checkpoint).expanduser()
    config_path = Path(args.config).expanduser()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model, cfg, assets = _load_model(checkpoint, config_path, shared_dir, device)
    normalize_b = bool(cfg.get("data", {}).get("normalize_b", True))
    activation = _activation_fn(
        str(cfg.get("training", {}).get("activation", "sigmoid")),
        float(cfg.get("training", {}).get("leaky_relu_slope", 0.01)),
    )
    visible_mask = assets.get("visible_mask")
    n_surface = int(assets["A"].shape[0])
    n_nodes = int(assets["nodes"].shape[0])

    mesh = load_mesh(shared_dir)
    elements = mesh["elements"].astype(np.int64)
    mapping = None
    if not args.skip_voxel:
        cache_path = (
            ROOT
            / "outputs"
            / "fem_cache"
            / f"mesh_to_voxel_{n_nodes}_{args.voxel_shape[0]}x{args.voxel_shape[1]}x{args.voxel_shape[2]}_center_k{args.n_candidates}.npz"
        )
        print(f"[Stage1] building/loading mesh_to_voxel map: {cache_path}")
        mapping = build_mesh_to_voxel_barycentric_map(
            shared_dir,
            tuple(args.voxel_shape),
            args.voxel_size_mm,
            n_candidates=args.n_candidates,
            cache_path=cache_path,
        )
        valid_ratio = float(np.mean(mapping["tet_ids"] >= 0))
        print(f"[Stage1] mesh_to_voxel valid_ratio={valid_ratio:.4f}")

    sample_ids: list[str] = []
    for split in args.splits:
        sample_ids.extend(_read_split_ids(data_dir, split))
    sample_ids = list(dict.fromkeys(sample_ids))
    if args.max_samples is not None:
        sample_ids = sample_ids[: int(args.max_samples)]
    sample_dirs = [sample_root / sid for sid in sample_ids if (sample_root / sid).exists()]
    print(f"[Stage1] generating {len(sample_dirs)} samples on {device}")

    done = 0
    for start in range(0, len(sample_dirs), int(args.batch_size)):
        batch_dirs = sample_dirs[start : start + int(args.batch_size)]
        pending = [
            p
            for p in batch_dirs
            if args.overwrite or not (p / "stage1_voxel.npy").exists() or args.skip_voxel
        ]
        if not pending:
            done += len(batch_dirs)
            continue

        b = _load_measurements(pending, n_surface, normalize_b, visible_mask).to(device)
        with torch.no_grad():
            x0 = torch.zeros(b.shape[0], n_nodes, 1, device=device)
            pred = activation(model(x0, b)).clamp(0.0, 1.0)
        pred_np = pred.squeeze(-1).cpu().numpy().astype(np.float32)

        for sample_dir, coarse in zip(pending, pred_np, strict=True):
            np.save(sample_dir / "coarse_d.npy", coarse)
            np.save(sample_dir / "stage1_mesh.npy", coarse)
            if mapping is not None:
                voxel = apply_mesh_to_voxel_barycentric(
                    coarse,
                    elements,
                    mapping,
                    tuple(args.voxel_shape),
                )
                vmax = float(voxel.max())
                if vmax > 0:
                    voxel = voxel / vmax
                np.save(sample_dir / "stage1_voxel.npy", voxel.astype(np.float32))
            meta = {
                "source": "DU2Vox Stage1 GCAIN",
                "checkpoint": str(checkpoint),
                "config": str(config_path),
                "shared_dir": str(shared_dir),
                "mesh_nodes": n_nodes,
                "measurements": n_surface,
                "voxel_shape": list(args.voxel_shape),
                "voxel_size_mm": args.voxel_size_mm,
                "mesh_to_voxel": "barycentric" if mapping is not None else "not_saved",
            }
            (sample_dir / "stage1_meta.json").write_text(json.dumps(meta, indent=2))

        done += len(batch_dirs)
        print(f"[Stage1] {done}/{len(sample_dirs)}")


if __name__ == "__main__":
    main()
