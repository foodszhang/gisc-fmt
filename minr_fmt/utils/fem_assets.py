"""FEM asset loading and lightweight mesh-to-voxel helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import sparse
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class FEMAssetSummary:
    shared_dir: str
    has_system_matrix: bool
    has_mesh_nodes: bool
    has_tetrahedra: bool
    has_edges: bool
    has_laplacian: bool
    num_nodes: int | None
    num_measurements: int | None


def _cfg_value(config, *keys, default=None):
    cur = config
    for key in keys:
        if cur is None:
            return default
        cur = getattr(cur, key, None)
    return default if cur is None else cur


def fem_shared_dir_from_config(config) -> Path:
    value = _cfg_value(config, "model", "fem", "shared_dir")
    if value is None:
        value = _cfg_value(config, "data", "fem_shared_dir")
    if value is None:
        value = "/home/foods/pro/FMT-SimGen/output/shared_mesh_20k"
    return Path(str(value)).expanduser()


def load_forward_matrix(shared_dir: str | Path) -> torch.Tensor:
    path = Path(shared_dir) / "system_matrix.A.npz"
    if not path.exists():
        raise FileNotFoundError(f"FEM system matrix missing: {path}")
    z = np.load(path, allow_pickle=True)
    if "forward_matrix" in z.files:
        arr = z["forward_matrix"].astype(np.float32)
    else:
        arr = sparse.load_npz(path).toarray().astype(np.float32)
    return torch.from_numpy(arr)


def load_mesh(shared_dir: str | Path) -> dict[str, np.ndarray]:
    path = Path(shared_dir) / "mesh.npz"
    if not path.exists():
        raise FileNotFoundError(f"FEM mesh missing: {path}")
    z = np.load(path)
    out = {"nodes": z["nodes"].astype(np.float32)}
    if "elements" in z.files:
        out["elements"] = z["elements"].astype(np.int64)
    if "surface_node_indices" in z.files:
        out["surface_node_indices"] = z["surface_node_indices"].astype(np.int64)
    return out


def build_edges_from_tetrahedra(elements: np.ndarray) -> np.ndarray:
    pairs = set()
    for tet in elements.astype(np.int64):
        a, b, c, d = [int(v) for v in tet]
        for u, v in ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d)):
            if u > v:
                u, v = v, u
            pairs.add((u, v))
    return np.asarray(sorted(pairs), dtype=np.int64)


def summarize_fem_assets(shared_dir: str | Path) -> FEMAssetSummary:
    shared = Path(shared_dir)
    mesh_path = shared / "mesh.npz"
    matrix_path = shared / "system_matrix.A.npz"
    has_mesh = mesh_path.exists()
    has_matrix = matrix_path.exists()
    num_nodes = None
    has_tets = False
    has_edges = False
    if has_mesh:
        mesh = np.load(mesh_path)
        num_nodes = int(mesh["nodes"].shape[0]) if "nodes" in mesh.files else None
        has_tets = "elements" in mesh.files
        has_edges = "edges" in mesh.files or has_tets
    num_meas = None
    if has_matrix:
        try:
            A = load_forward_matrix(shared)
            num_meas = int(A.shape[0])
            if num_nodes is None:
                num_nodes = int(A.shape[1])
        except Exception:
            num_meas = None
    has_lap = (shared / "graph_laplacian_full.Lap.npz").exists()
    return FEMAssetSummary(
        shared_dir=str(shared),
        has_system_matrix=has_matrix,
        has_mesh_nodes=has_mesh and num_nodes is not None,
        has_tetrahedra=has_tets,
        has_edges=has_edges,
        has_laplacian=has_lap,
        num_nodes=num_nodes,
        num_measurements=num_meas,
    )


@lru_cache(maxsize=8)
def _nearest_node_index(
    shared_dir: str,
    voxel_shape: tuple[int, int, int],
    voxel_size_mm: float,
) -> np.ndarray:
    mesh = load_mesh(shared_dir)
    nodes = mesh["nodes"]
    axes = [(np.arange(n, dtype=np.float32) + 0.5) * float(voxel_size_mm) for n in voxel_shape]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    _, idx = cKDTree(nodes).query(grid, k=1, workers=-1)
    return idx.astype(np.int64).reshape(voxel_shape)


def voxel_centers_mm(
    voxel_shape: tuple[int, int, int],
    voxel_size_mm: float,
    offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Return full voxel-center coordinates in trunk-local millimeters."""
    axes = [
        float(offset_mm[i]) + (np.arange(int(n), dtype=np.float32) + 0.5) * float(voxel_size_mm)
        for i, n in enumerate(voxel_shape)
    ]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def build_mesh_to_voxel_barycentric_map(
    shared_dir: str | Path,
    voxel_shape: tuple[int, int, int],
    voxel_size_mm: float,
    *,
    offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    n_candidates: int = 32,
    chunk_size: int = 131_072,
    cache_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Build/cache a barycentric FEM mesh -> voxel interpolation map.

    The returned map contains one tetrahedron id and four barycentric weights per
    voxel center. Voxels outside the tetrahedral mesh have tet_id=-1 and zero
    weights. This follows DU2Vox's FEMBridge interpolation semantics but builds a
    reusable dense full-volume lookup table.
    """
    if cache_path is not None:
        cache = Path(cache_path)
        if cache.exists():
            z = np.load(cache)
            if tuple(int(v) for v in z["voxel_shape"]) == tuple(voxel_shape):
                return {"tet_ids": z["tet_ids"], "bary": z["bary"]}

    mesh = load_mesh(shared_dir)
    nodes = mesh["nodes"].astype(np.float64)
    if "elements" not in mesh:
        raise ValueError("FEM mesh_to_voxel requires tetrahedra/elements in mesh.npz")
    elements = mesh["elements"].astype(np.int64)
    verts = nodes[elements]
    centroids = verts.mean(axis=1)
    tree = cKDTree(centroids)
    coords = voxel_centers_mm(voxel_shape, voxel_size_mm, offset_mm).astype(np.float64)

    n_voxels = coords.shape[0]
    tet_ids = np.full(n_voxels, -1, dtype=np.int32)
    bary = np.zeros((n_voxels, 4), dtype=np.float32)
    k = min(int(n_candidates), len(elements))

    for start in range(0, n_voxels, int(chunk_size)):
        end = min(start + int(chunk_size), n_voxels)
        pts = coords[start:end]
        _, cand = tree.query(pts, k=k, workers=-1)
        if k == 1:
            cand = cand[:, None]
        unresolved = np.ones(len(pts), dtype=bool)
        local_tet = np.full(len(pts), -1, dtype=np.int32)
        local_bary = np.zeros((len(pts), 4), dtype=np.float32)

        for j in range(k):
            idx = np.nonzero(unresolved)[0]
            if idx.size == 0:
                break
            tet_idx = cand[idx, j]
            tv = verts[tet_idx]
            mat = np.stack(
                [tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0], tv[:, 3] - tv[:, 0]],
                axis=-1,
            )
            rhs = (pts[idx] - tv[:, 0])[..., None]
            try:
                lam123 = np.linalg.solve(mat, rhs)[..., 0]
            except np.linalg.LinAlgError:
                # Degenerate tets are rare; fall back to pointwise solves for this candidate.
                lam123 = np.full((idx.size, 3), -1.0, dtype=np.float64)
                for row in range(idx.size):
                    try:
                        lam123[row] = np.linalg.solve(mat[row], rhs[row, :, 0])
                    except np.linalg.LinAlgError:
                        pass
            lam0 = 1.0 - lam123.sum(axis=1, keepdims=True)
            cand_bary = np.concatenate([lam0, lam123], axis=1)
            inside = np.all(cand_bary >= -1e-6, axis=1)
            if np.any(inside):
                hit = idx[inside]
                local_tet[hit] = tet_idx[inside].astype(np.int32)
                local_bary[hit] = cand_bary[inside].astype(np.float32)
                unresolved[hit] = False

        tet_ids[start:end] = local_tet
        bary[start:end] = local_bary

    if cache_path is not None:
        cache = Path(cache_path)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache,
            tet_ids=tet_ids,
            bary=bary,
            voxel_shape=np.asarray(voxel_shape, dtype=np.int32),
            voxel_size_mm=np.asarray([voxel_size_mm], dtype=np.float32),
            offset_mm=np.asarray(offset_mm, dtype=np.float32),
        )
    return {"tet_ids": tet_ids, "bary": bary}


def apply_mesh_to_voxel_barycentric(
    x_mesh: np.ndarray,
    elements: np.ndarray,
    mapping: dict[str, np.ndarray],
    voxel_shape: tuple[int, int, int],
) -> np.ndarray:
    """Apply a barycentric mesh-to-voxel map to one mesh-node prediction."""
    values = np.asarray(x_mesh, dtype=np.float32).reshape(-1)
    tet_ids = mapping["tet_ids"].reshape(-1)
    bary = mapping["bary"].reshape(-1, 4)
    out = np.zeros(tet_ids.shape[0], dtype=np.float32)
    valid = tet_ids >= 0
    if np.any(valid):
        vertex_ids = elements[tet_ids[valid].astype(np.int64)].astype(np.int64)
        out[valid] = np.sum(values[vertex_ids] * bary[valid], axis=1)
    return out.reshape(tuple(int(v) for v in voxel_shape))


class NearestNodeMeshToVoxel(nn.Module):
    """Map mesh node values to voxel grid by nearest FEM node.

    This is a deterministic fallback used when no DU2Vox barycentric mesh-to-voxel
    lookup table is available. It preserves the FEM node space and fails if node
    counts do not match, but it is not a substitute for barycentric interpolation.
    """

    def __init__(
        self,
        shared_dir: str | Path,
        voxel_shape: tuple[int, int, int],
        voxel_size_mm: float,
    ):
        super().__init__()
        idx = _nearest_node_index(
            str(Path(shared_dir).expanduser()),
            tuple(voxel_shape),
            float(voxel_size_mm),
        )
        self.register_buffer("node_index", torch.from_numpy(idx.reshape(-1)), persistent=False)
        self.voxel_shape = tuple(int(v) for v in voxel_shape)

    def forward(self, x_mesh: torch.Tensor) -> torch.Tensor:
        if x_mesh.dim() == 3 and x_mesh.size(-1) == 1:
            x_mesh = x_mesh[..., 0]
        if x_mesh.dim() != 2:
            raise ValueError(f"x_mesh must be [B,N] or [B,N,1], got {tuple(x_mesh.shape)}")
        values = x_mesh[:, self.node_index.to(device=x_mesh.device)]
        return values.view(x_mesh.shape[0], 1, *self.voxel_shape)


class BarycentricMeshToVoxel(nn.Module):
    """Map FEM mesh-node values to voxel grid with tetrahedral barycentric weights."""

    def __init__(
        self,
        shared_dir: str | Path,
        voxel_shape: tuple[int, int, int],
        voxel_size_mm: float,
        *,
        cache_path: str | Path | None = None,
        n_candidates: int = 32,
    ):
        super().__init__()
        shared = Path(shared_dir).expanduser()
        if cache_path is None:
            cache_path = (
                Path("outputs")
                / "fem_cache"
                / (
                    f"mesh_to_voxel_{tuple(voxel_shape)[0]}x{tuple(voxel_shape)[1]}x"
                    f"{tuple(voxel_shape)[2]}_center_k{int(n_candidates)}.npz"
                )
            )
        mapping = build_mesh_to_voxel_barycentric_map(
            shared,
            tuple(voxel_shape),
            float(voxel_size_mm),
            n_candidates=int(n_candidates),
            cache_path=cache_path,
        )
        mesh = load_mesh(shared)
        elements = mesh["elements"].astype(np.int64)
        tet_ids = mapping["tet_ids"].reshape(-1).astype(np.int64)
        valid = tet_ids >= 0
        vertex_ids = np.zeros((tet_ids.shape[0], 4), dtype=np.int64)
        vertex_ids[valid] = elements[tet_ids[valid]]
        self.register_buffer("vertex_ids", torch.from_numpy(vertex_ids), persistent=False)
        self.register_buffer("bary", torch.from_numpy(mapping["bary"].reshape(-1, 4)), persistent=False)
        self.register_buffer("valid", torch.from_numpy(valid), persistent=False)
        self.voxel_shape = tuple(int(v) for v in voxel_shape)

    def forward(self, x_mesh: torch.Tensor) -> torch.Tensor:
        if x_mesh.dim() == 3 and x_mesh.size(-1) == 1:
            x_mesh = x_mesh[..., 0]
        if x_mesh.dim() != 2:
            raise ValueError(f"x_mesh must be [B,N] or [B,N,1], got {tuple(x_mesh.shape)}")
        vertex_ids = self.vertex_ids.to(device=x_mesh.device)
        bary = self.bary.to(device=x_mesh.device, dtype=x_mesh.dtype)
        gathered = x_mesh[:, vertex_ids.reshape(-1)].view(x_mesh.shape[0], -1, 4)
        values = (gathered * bary.unsqueeze(0)).sum(dim=-1)
        values = values.masked_fill(~self.valid.to(device=x_mesh.device).unsqueeze(0), 0.0)
        return values.view(x_mesh.shape[0], 1, *self.voxel_shape)
