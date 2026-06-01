"""FEM-domain baselines adapted from FEM coarse/prior assets."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from minr_fmt.utils.fem_assets import (
    BarycentricMeshToVoxel,
    NearestNodeMeshToVoxel,
    build_edges_from_tetrahedra,
    fem_shared_dir_from_config,
    load_forward_matrix,
    load_mesh,
)


def _roi_shape(config) -> tuple[int, int, int]:
    vr = config.data.voxel_ranges
    return (int(vr.x[1] - vr.x[0]), int(vr.y[1] - vr.y[0]), int(vr.z[1] - vr.z[0]))


def _model_section(config, key: str):
    return getattr(config.model, key, {})


def _first_tensor(batch: dict, keys: tuple[str, ...]) -> torch.Tensor | None:
    for key in keys:
        value = batch.get(key)
        if torch.is_tensor(value):
            return value
    return None


def _soft_threshold(x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * F.relu(x.abs() - threshold)


def _probability_to_logits(x: torch.Tensor) -> torch.Tensor:
    return torch.logit(x.clamp(1e-6, 1.0 - 1e-6))


def _normalize_probability(x: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, x.dim()))
    return x / x.amax(dim=dims, keepdim=True).clamp_min(1e-8)


class FEMBase(nn.Module):
    output_type = "voxel"

    def __init__(self, config, section: str):
        super().__init__()
        self.config = config
        self.section = section
        self.shared_dir = fem_shared_dir_from_config(config)
        self.roi_shape = _roi_shape(config)
        self.voxel_size_mm = float(getattr(config.data, "voxel_size_mm", 0.2))
        params = _model_section(config, section)
        self.mapper_name = str(getattr(params, "mesh_to_voxel", "barycentric"))
        self.mapper_n_candidates = int(getattr(params, "n_candidates", 32))
        self.mapper: nn.Module | None = None

    def _validate_mesh_node_count(self):
        mesh = load_mesh(self.shared_dir)
        if "nodes" not in mesh:
            raise ValueError(
                f"{self.section} requires mesh assets with nodes, "
                f"but no nodes were found in {self.shared_dir}."
            )
        num_nodes = int(mesh["nodes"].shape[0])
        if hasattr(self, "A") and self.A.shape[1] != num_nodes:
            raise ValueError(f"FEM A node count {self.A.shape[1]} != mesh node count {num_nodes}")

    def mesh_to_voxel(self, x_mesh: torch.Tensor) -> torch.Tensor:
        if self.mapper is None:
            if self.mapper_name == "nearest_node_fallback":
                self.mapper = NearestNodeMeshToVoxel(
                    self.shared_dir, self.roi_shape, self.voxel_size_mm
                )
            else:
                self.mapper = BarycentricMeshToVoxel(
                    self.shared_dir,
                    self.roi_shape,
                    self.voxel_size_mm,
                    n_candidates=self.mapper_n_candidates,
                )
            self.mapper.to(x_mesh.device)
        return self.mapper(x_mesh)

    def _measurement(self, batch: dict) -> torch.Tensor:
        b = _first_tensor(batch, ("measurement_b", "phi", "measurement_vector"))
        if b is None:
            raise ValueError(
                f"{self.section} requires measurement_b.npy/phi in the FMT-SimGen sample."
            )
        if b.dim() == 3 and b.size(-1) == 1:
            b = b[..., 0]
        return b.float()

    def _stage1_mesh(self, batch: dict) -> torch.Tensor:
        x = _first_tensor(batch, ("stage1_mesh", "coarse_d", "fem_nodes"))
        if x is None:
            raise ValueError(
                f"{self.section} requires a real FEM coarse mesh prediction in the batch"
            )
        return x.float()


class FEMCoarseBaseline(FEMBase):
    def __init__(self, config):
        super().__init__(config, "fem_coarse")

    def forward(self, projections, *args, **kwargs):
        batch = kwargs.get("batch") or {}
        fem_prior_voxel = _first_tensor(
            batch,
            ("fem_prior_voxel", "stage1_voxel", "fem_prior", "coarse_prior"),
        )
        if fem_prior_voxel is not None:
            if fem_prior_voxel.dim() == 4:
                fem_prior_voxel = fem_prior_voxel.unsqueeze(1)
            return {
                "pred_voxel": _probability_to_logits(fem_prior_voxel.float()),
                "aux_outputs": {"output_space": "full_voxel", "alignment_mode": "full"},
            }
        x_mesh = self._stage1_mesh(batch)
        pred = _probability_to_logits(_normalize_probability(self.mesh_to_voxel(x_mesh)))
        return {
            "pred_voxel": pred,
            "aux_outputs": {"output_space": "mesh", "alignment_mode": "mesh_to_voxel"},
        }


class FEMToVoxelBaseline(FEMCoarseBaseline):
    def __init__(self, config):
        FEMBase.__init__(self, config, "fem_to_voxel")

    def forward(self, projections, *args, **kwargs):
        batch = kwargs.get("batch") or {}
        x_mesh = self._stage1_mesh(batch)
        pred = _probability_to_logits(_normalize_probability(self.mesh_to_voxel(x_mesh)))
        return {
            "pred_voxel": pred,
            "aux_outputs": {"output_space": "mesh", "alignment_mode": "mesh_to_voxel"},
        }


Stage1FEMBaseline = FEMCoarseBaseline
Stage1ToVoxelBaseline = FEMToVoxelBaseline


class IterativeFEMBase(FEMBase):
    def __init__(self, config, section: str):
        super().__init__(config, section)
        params = _model_section(config, section)
        self.register_buffer("A", load_forward_matrix(self.shared_dir), persistent=False)
        self.num_iters = int(getattr(params, "num_iters", 100))
        self.step_size = float(getattr(params, "step_size", 0.0))
        self.nonnegative = bool(getattr(params, "nonnegative", True))
        self.normalize_columns = bool(getattr(params, "normalize_columns", True))
        self.normalize_measurement = bool(getattr(params, "normalize_measurement", True))
        self.relative_regularization = bool(getattr(params, "relative_regularization", True))
        self.power_iters = int(getattr(params, "power_iters", 12))
        self._cached_step: float | None = None
        self._validate_mesh_node_count()

    def _step(self, A: torch.Tensor) -> float:
        if self.step_size > 0:
            return self.step_size
        if self._cached_step is not None:
            return self._cached_step
        # Power iteration estimates ||A||_2^2. Frobenius norm is much too
        # conservative for the dense FEM matrix and stalls first-order solvers.
        with torch.no_grad():
            v = torch.ones(A.shape[1], device=A.device, dtype=A.dtype)
            v = v / v.norm().clamp_min(1e-8)
            for _ in range(self.power_iters):
                v = A.t().mv(A.mv(v))
                v = v / v.norm().clamp_min(1e-8)
            lipschitz = A.mv(v).square().sum().clamp_min(1e-8)
        self._cached_step = float(1.0 / lipschitz)
        return self._cached_step

    def _solver_inputs(self, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        A = self.A.to(phi.device, dtype=phi.dtype)
        phi = self._normalize_phi(phi)
        column_norm = A.norm(dim=0).clamp_min(1e-8)
        if self.normalize_columns:
            A = A / column_norm
        else:
            column_norm = torch.ones_like(column_norm)
        return A, phi, column_norm

    def _normalize_phi(self, phi: torch.Tensor) -> torch.Tensor:
        if self.normalize_measurement:
            return phi / phi.amax(dim=1, keepdim=True).clamp_min(1e-8)
        return phi

    def _regularization_scale(self, A: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        if not self.relative_regularization:
            return torch.ones(phi.shape[0], 1, device=phi.device, dtype=phi.dtype)
        return (phi @ A).abs().amax(dim=1, keepdim=True).clamp_min(1e-8)

    def _solve(self, phi: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, projections, *args, **kwargs):
        phi = self._measurement(kwargs.get("batch") or {})
        x_mesh = self._solve(phi.to(self.A.device, dtype=self.A.dtype))
        pred = _probability_to_logits(
            _normalize_probability(self.mesh_to_voxel(x_mesh.to(phi.device)))
        )
        target_phi = self._normalize_phi(phi.to(self.A.device, dtype=self.A.dtype))
        residual = torch.mean((target_phi - x_mesh @ self.A.t()).square()).detach()
        return {
            "pred_voxel": pred,
            "aux_outputs": {
                "x_mesh": x_mesh.detach(),
                "data_residual": residual,
                "output_space": "mesh",
                "alignment_mode": "mesh_to_voxel",
            },
        }


class TikhonovFEM(IterativeFEMBase):
    def __init__(self, config):
        super().__init__(config, "tikhonov_fem")
        params = _model_section(config, "tikhonov_fem")
        self.lambda_l2 = float(getattr(params, "lambda_l2", 1e-3))

    def _solve(self, phi: torch.Tensor) -> torch.Tensor:
        A, phi, column_norm = self._solver_inputs(phi)
        x = torch.zeros(phi.shape[0], A.shape[1], device=phi.device, dtype=phi.dtype)
        step = self._step(A)
        lambda_l2 = self.lambda_l2 * self._regularization_scale(A, phi)
        for _ in range(self.num_iters):
            grad = (x @ A.t() - phi) @ A + lambda_l2 * x
            x = x - step * grad
            if self.nonnegative:
                x = x.clamp_min(0.0)
        return x / column_norm


class L1FEM(IterativeFEMBase):
    def __init__(self, config):
        super().__init__(config, "l1_fem")
        params = _model_section(config, "l1_fem")
        self.lambda_l1 = float(getattr(params, "lambda_l1", 1e-4))

    def _solve(self, phi: torch.Tensor) -> torch.Tensor:
        A, phi, column_norm = self._solver_inputs(phi)
        x = torch.zeros(phi.shape[0], A.shape[1], device=phi.device, dtype=phi.dtype)
        step = self._step(A)
        lambda_l1 = self.lambda_l1 * self._regularization_scale(A, phi)
        for _ in range(self.num_iters):
            grad = (x @ A.t() - phi) @ A
            threshold = step * lambda_l1
            x = _soft_threshold(x - step * grad, threshold)
            if self.nonnegative:
                x = x.clamp_min(0.0)
        return x / column_norm


class ElasticNetFEM(IterativeFEMBase):
    def __init__(self, config):
        super().__init__(config, "elasticnet_fem")
        params = _model_section(config, "elasticnet_fem")
        self.lambda_l1 = float(getattr(params, "lambda_l1", 1e-4))
        self.lambda_l2 = float(getattr(params, "lambda_l2", 1e-3))

    def _solve(self, phi: torch.Tensor) -> torch.Tensor:
        A, phi, column_norm = self._solver_inputs(phi)
        x = torch.zeros(phi.shape[0], A.shape[1], device=phi.device, dtype=phi.dtype)
        step = self._step(A)
        lambda_l1 = self.lambda_l1 * self._regularization_scale(A, phi)
        lambda_l2 = self.lambda_l2 * self._regularization_scale(A, phi)
        for _ in range(self.num_iters):
            grad = (x @ A.t() - phi) @ A + lambda_l2 * x
            threshold = step * lambda_l1
            x = _soft_threshold(x - step * grad, threshold)
            if self.nonnegative:
                x = x.clamp_min(0.0)
        return x / column_norm


class FISTAFEM(L1FEM):
    def __init__(self, config):
        super().__init__(config)
        self.section = "fista_fem"
        params = _model_section(config, "fista_fem")
        self.num_iters = int(getattr(params, "num_iters", self.num_iters))
        self.lambda_l1 = float(getattr(params, "lambda_l1", self.lambda_l1))

    def _solve(self, phi: torch.Tensor) -> torch.Tensor:
        A, phi, column_norm = self._solver_inputs(phi)
        x = torch.zeros(phi.shape[0], A.shape[1], device=phi.device, dtype=phi.dtype)
        z = x.clone()
        t = 1.0
        step = self._step(A)
        threshold = step * self.lambda_l1 * self._regularization_scale(A, phi)
        for _ in range(self.num_iters):
            grad = (z @ A.t() - phi) @ A
            x_next = _soft_threshold(z - step * grad, threshold)
            if self.nonnegative:
                x_next = x_next.clamp_min(0.0)
            t_next = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
            z = x_next + ((t - 1.0) / t_next) * (x_next - x)
            x, t = x_next, t_next
        return x / column_norm


class StOMPFEM(IterativeFEMBase):
    """Stagewise Orthogonal Matching Pursuit FEM baseline when A/y are available."""

    def __init__(self, config):
        super().__init__(config, "stomp_fem")
        params = _model_section(config, "stomp_fem")
        self.max_support = int(getattr(params, "max_support", 128))
        self.threshold_sigma = float(getattr(params, "threshold_sigma", 2.5))
        self.num_iters = int(getattr(params, "num_iters", 20))

    def _solve(self, phi: torch.Tensor) -> torch.Tensor:
        A_raw = self.A.to(phi.device, dtype=phi.dtype)
        phi = self._normalize_phi(phi)
        column_norm = A_raw.norm(dim=0).clamp_min(1e-8)
        A = A_raw / column_norm
        out = torch.zeros(phi.shape[0], A.shape[1], device=phi.device, dtype=phi.dtype)
        for b in range(phi.shape[0]):
            y = phi[b]
            residual = y.clone()
            support = torch.zeros(A.shape[1], dtype=torch.bool, device=phi.device)
            coef = None
            for _ in range(self.num_iters):
                corr = A.t() @ residual
                sigma = corr.std().clamp_min(1e-8)
                new = corr.abs() >= self.threshold_sigma * sigma
                support |= new
                if int(support.sum()) > self.max_support:
                    top = torch.topk(corr.abs(), k=self.max_support).indices
                    support.zero_()
                    support[top] = True
                idx = torch.nonzero(support, as_tuple=False).flatten()
                if idx.numel() == 0:
                    break
                sol = torch.linalg.lstsq(A[:, idx], y).solution
                residual = y - A[:, idx] @ sol
                coef = (idx, sol)
                if float(residual.norm()) < 1e-6:
                    break
            if coef is not None:
                idx, sol = coef
                out[b, idx] = sol
            if self.nonnegative:
                out[b] = out[b].clamp_min(0.0)
        return out / column_norm


class GraphConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.lin = nn.Linear(in_ch, out_ch)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        msg = self.lin(x[:, src])
        out = msg.new_zeros(x.shape[0], x.shape[1], msg.shape[-1])
        out.index_add_(1, dst, msg)
        deg = torch.bincount(dst, minlength=x.shape[1]).to(x.device, x.dtype).clamp_min(1.0)
        return F.silu(out / deg.view(1, -1, 1))


class GAICNLikeFEM(FEMBase):
    """GAICN-like graph unrolling baseline using FEM matrix and mesh graph."""

    def __init__(self, config):
        super().__init__(config, "gaicn")
        params = _model_section(config, "gaicn")
        self.register_buffer("A", load_forward_matrix(self.shared_dir), persistent=False)
        mesh = load_mesh(self.shared_dir)
        if "elements" not in mesh:
            raise ValueError(
                "GAICN requires FEM tetrahedra/mesh graph assets, but they are missing."
            )
        edges = build_edges_from_tetrahedra(mesh["elements"])
        edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
        self.register_buffer(
            "edge_index",
            torch.from_numpy(edges.T.astype("int64")),
            persistent=False,
        )
        self.num_phases = int(getattr(params, "num_phases", 6))
        self.train_in_mesh_space = bool(getattr(params, "train_in_mesh_space", True))
        hidden = int(getattr(params, "hidden_channels", 8))
        rho_init = float(getattr(params, "rho_init", 1e-3))
        threshold_init = float(getattr(params, "threshold_init", 1e-3))
        self.rho = nn.Parameter(torch.full((self.num_phases,), rho_init))
        self.threshold = nn.Parameter(torch.full((self.num_phases,), threshold_init))
        self.gcn1 = nn.ModuleList([GraphConv(1, hidden) for _ in range(self.num_phases)])
        self.gcn2 = nn.ModuleList([GraphConv(hidden, hidden) for _ in range(self.num_phases)])
        self.proj = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(self.num_phases)])

    def forward(self, projections, *args, **kwargs):
        batch = kwargs.get("batch") or {}
        phi = self._measurement(batch).to(self.A.device, dtype=self.A.dtype)
        init = _first_tensor(batch, ("stage1_mesh", "coarse_d"))
        if init is None:
            x = torch.zeros(phi.shape[0], self.A.shape[1], device=phi.device, dtype=phi.dtype)
        else:
            x = init.to(phi.device, dtype=phi.dtype).reshape(phi.shape[0], -1)
            if x.shape[1] != self.A.shape[1]:
                raise ValueError(
                    f"GAICN init node count {x.shape[1]} != FEM A node count {self.A.shape[1]}"
                )
        z = x
        x_prev = x
        phase_debug = []
        A = self.A.to(phi.device, dtype=phi.dtype)
        measurement_backprojection = phi @ A
        edge_index = self.edge_index.to(phi.device)
        for k in range(self.num_phases):
            rho = F.softplus(self.rho[k])
            # Keep the fixed measurement backprojection outside the unrolled loop.
            # Computing A^T A explicitly is larger and slower for this mesh.
            data_gradient = (z @ A.t()) @ A - measurement_backprojection
            r = z - rho * data_gradient
            h = self.gcn1[k](r.unsqueeze(-1), edge_index)
            h = self.gcn2[k](h, edge_index)
            delta = self.proj[k](h).squeeze(-1)
            delta = _soft_threshold(delta, F.softplus(self.threshold[k]))
            x_next = (r + delta).clamp_min(0.0)
            momentum = 0.0 if k == 0 else float(k) / float(k + 3)
            z = x_next + momentum * (x_next - x_prev)
            x_prev = x
            x = x_next
            phase_debug.append(x.detach())
        training_space = "mesh" if self.training and self.train_in_mesh_space else "voxel"
        if training_space == "mesh":
            pred = _probability_to_logits(_normalize_probability(x))
        else:
            pred = _probability_to_logits(_normalize_probability(self.mesh_to_voxel(x)))
        residual = torch.mean((x @ A.t() - phi).square())
        return {
            "pred_voxel": pred,
            "aux_outputs": {
                "x_mesh": x.detach(),
                "phase_mesh": phase_debug,
                "data_residual": residual.detach(),
                "distribution_loss": 1e-4 * x.abs().mean() + 1e-3 * residual,
                "learned_rho": F.softplus(self.rho).detach(),
                "learned_threshold": F.softplus(self.threshold).detach(),
                "output_space": "mesh",
                "alignment_mode": "mesh_to_voxel",
                "cached_measurement_backprojection": True,
                "training_space": training_space,
            },
        }
