#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${FMT_SIMGEN_DATA_DIR:-/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k}"
SHARED_DIR="${FEM_SHARED_DIR:-/home/foods/pro/FMT-SimGen/output/shared_mesh_20k}"
LOG_DIR="$ROOT_DIR/outputs/fmt_simgen_v2_3k_20k/logs"
STATUS_JSON="$LOG_DIR/fem_assets_status.json"

mkdir -p "$LOG_DIR"

echo "shared FEM files:"
find "$SHARED_DIR" -maxdepth 2 -type f | head -100 || true

ROOT_DIR="$ROOT_DIR" DATA_DIR="$DATA_DIR" SHARED_DIR="$SHARED_DIR" STATUS_JSON="$STATUS_JSON" \
  uv run python - <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(os.environ["ROOT_DIR"])
sys.path.insert(0, str(root))
from minr_fmt.utils.fem_assets import summarize_fem_assets

data_dir = Path(os.environ["DATA_DIR"])
shared_dir = Path(os.environ["SHARED_DIR"])
status_path = Path(os.environ["STATUS_JSON"])
sample_root = data_dir / "samples" if (data_dir / "samples").is_dir() else data_dir
samples = sorted(path for path in sample_root.glob("sample_*") if path.is_dir())

def count_any(names):
    return sum(any((sample / name).exists() for name in names) for sample in samples)

def stage1_cache_metadata():
    metadata = []
    for sample in samples:
        path = sample / "stage1_meta.json"
        if not path.exists():
            continue
        try:
            metadata.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    expected_shape = [190, 200, 104]
    compatible = [
        item
        for item in metadata
        if item.get("source") == "DU2Vox Stage1 GCAIN"
        and item.get("mesh_to_voxel") == "barycentric"
        and item.get("voxel_shape") == expected_shape
    ]
    return len(metadata), len(compatible)

summary = summarize_fem_assets(shared_dir)
num_samples = len(samples)
stage1_meta_count, du2vox_stage1_cache_count = stage1_cache_metadata()
counts = {
    "measurement_b": count_any(("measurement_b.npy", "phi.npy", "measurement_vector.npy")),
    "fem_coarse_voxel": count_any(("fem_prior_voxel.npy", "stage1_voxel.npy", "coarse_prior.npy")),
    "fem_coarse_mesh": count_any(
        ("fem_prior_mesh.npy", "stage1_mesh.npy", "coarse_d.npy", "fem_nodes.npy")
    ),
}
all_samples = num_samples > 0
status = {
    "data_dir": str(data_dir),
    "shared_dir": str(shared_dir),
    "num_samples": num_samples,
    "sample_counts": counts,
    "stage1_meta_count": stage1_meta_count,
    "du2vox_stage1_gcain_barycentric_cache_count": du2vox_stage1_cache_count,
    "has_du2vox_stage1_gcain_barycentric_cache": (
        all_samples and du2vox_stage1_cache_count == num_samples
    ),
    "has_fem_coarse_voxel": all_samples and counts["fem_coarse_voxel"] == num_samples,
    "has_fem_coarse_mesh": all_samples and counts["fem_coarse_mesh"] == num_samples,
    "has_measurement_b": all_samples and counts["measurement_b"] == num_samples,
    "has_forward_matrix": summary.has_system_matrix,
    "has_mesh_nodes": summary.has_mesh_nodes,
    "has_tetrahedra": summary.has_tetrahedra,
    "has_edges": summary.has_edges,
    "has_mesh_to_voxel": summary.has_mesh_nodes and summary.has_tetrahedra,
}
status["can_run_fem_coarse"] = status["has_fem_coarse_voxel"] or (
    status["has_fem_coarse_mesh"] and status["has_mesh_to_voxel"]
)
status["can_run_traditional_fem"] = all(
    status[key] for key in ("has_measurement_b", "has_forward_matrix", "has_mesh_to_voxel")
)
status["can_run_gaicn"] = all(
    status[key]
    for key in ("has_measurement_b", "has_forward_matrix", "has_mesh_nodes", "has_mesh_to_voxel")
) and (status["has_tetrahedra"] or status["has_edges"])
status_path.write_text(json.dumps(status, indent=2) + "\n")
print(json.dumps(status, indent=2))
if not status["can_run_fem_coarse"]:
    print("FEM coarse prediction missing. fem_coarse / fem_to_voxel / fem2vox_unet cannot run.")
    print("Generate real assets with scripts/generate_stage1_fem_assets.py; zero priors are invalid.")
if not status["can_run_traditional_fem"]:
    print("Traditional FEM blocked: system matrix A and per-sample measurement_b/phi are required.")
if not status["can_run_gaicn"]:
    print("GAICN skipped: missing one or more FEM graph/A/y/mesh_to_voxel assets.")
PY
