"""Diagnose DU2Vox/FMT-SimGen Stage 1 FEM assets for the active Hydra config."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.utils.fem_assets import (  # noqa: E402
    fem_shared_dir_from_config,
    summarize_fem_assets,
)


def _sample_root(data_dir: Path) -> Path:
    if (data_dir / "samples").exists():
        return data_dir / "samples"
    return data_dir


def _first_sample(data_dir: Path) -> Path | None:
    root = _sample_root(data_dir)
    samples = sorted(p for p in root.glob("sample_*") if p.is_dir())
    return samples[0] if samples else None


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    shared_dir = fem_shared_dir_from_config(cfg)
    summary = summarize_fem_assets(shared_dir)
    data_dir = Path(str(cfg.data.get("data_dir", cfg.data.get("train_dir", "")))).expanduser()
    sample = _first_sample(data_dir)

    has_measurement = bool(sample and (sample / "measurement_b.npy").exists())
    has_stage1_prediction = bool(
        sample
        and any(
            (sample / name).exists()
            for name in (
                "coarse_d.npy",
                "stage1_mesh.npy",
                "stage1_recon.npy",
                "fem_recon.npy",
                "coarse_prior.npy",
                "stage1_voxel.npy",
            )
        )
    )
    has_gt_nodes = bool(sample and (sample / "gt_nodes.npy").exists())
    gt_nodes_count = None
    if has_gt_nodes:
        gt_nodes_count = int(np.load(sample / "gt_nodes.npy").reshape(-1).shape[0])

    has_mesh_to_voxel = summary.has_mesh_nodes
    has_voxel_to_mesh = has_gt_nodes and summary.num_nodes == gt_nodes_count

    print(f"shared_dir: {shared_dir}")
    print(f"sample_checked: {sample if sample else 'none'}")
    print(f"num_mesh_nodes: {summary.num_nodes}")
    print(f"num_measurements_A: {summary.num_measurements}")
    print(f"num_gt_nodes_sample: {gt_nodes_count}")
    print(f"asset_dataset_node_count_match: {summary.num_nodes == gt_nodes_count}")
    print(f"has_system_matrix: {str(summary.has_system_matrix).lower()}")
    print(f"has_measurement_vector: {str(has_measurement).lower()}")
    print(f"has_mesh_nodes: {str(summary.has_mesh_nodes).lower()}")
    print(f"has_tetrahedra: {str(summary.has_tetrahedra).lower()}")
    print(f"has_edges: {str(summary.has_edges).lower()}")
    print(f"has_stage1_prediction: {str(has_stage1_prediction).lower()}")
    print(f"has_mesh_to_voxel: {str(has_mesh_to_voxel).lower()}")
    print(f"has_voxel_to_mesh: {str(has_voxel_to_mesh).lower()}")


if __name__ == "__main__":
    main()
