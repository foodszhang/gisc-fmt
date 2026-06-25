"""Candidate-specific per-view representation, fusion, and assignment."""

from __future__ import annotations

import torch
import torch.nn as nn

from minr_fmt.network.ssq_diagnostics import (
    make_residual_mlp,
    masked_softmax,
    zero_init_last_linear,
)


class CandidateViewEncoder(nn.Module):
    def __init__(
        self, feature_dim: int, hidden_dim: int = 128, blocks: int = 3, dropout: float = 0.0
    ):
        super().__init__()
        self.representation_net = make_residual_mlp(
            feature_dim + 9,
            hidden_dim,
            hidden_dim,
            blocks=blocks,
            dropout=dropout,
        )

    def forward(
        self,
        samples: dict[str, torch.Tensor],
        router: dict[str, torch.Tensor],
        mapped: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        f = samples["sample_features"]
        zeta = router["zeta"]
        a_kernel = samples["A"]
        b, v, n, k, c = f.shape
        mb = zeta.shape[-1]
        offsets = samples["offsets_template"].to(device=f.device, dtype=f.dtype)
        offsets = offsets / offsets.abs().amax().clamp_min(1.0)
        local = torch.cat(
            [
                f[:, :, :, :, None].expand(b, v, n, k, mb, c),
                offsets[None, None, None, :, None].expand(b, v, n, k, mb, 2),
            ],
            dim=-1,
        )
        weighted = (a_kernel[..., None, None] * zeta[..., None] * local).sum(dim=3)
        pooled = weighted / router["a"].clamp_min(1e-8)[..., None]
        geom = torch.stack(
            [
                mapped["detector_side_path_proxy"],
                mapped["boundary_distance"],
            ],
            dim=-1,
        )[:, :, :, None].expand(b, v, n, mb, 2)
        stats = torch.stack([router["a"], router["nu"]], dim=-1)
        rel = router["relative_query"][:, None].expand(b, v, n, mb, 3)
        rep = self.representation_net(torch.cat([pooled, stats, rel, geom], dim=-1))
        return rep.permute(0, 2, 1, 3, 4).contiguous()


class CandidateSpecificViewFusion(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 128, mode: str = "candidate_specific"):
        super().__init__()
        self.transform = make_residual_mlp(
            feature_dim, hidden_dim, feature_dim, blocks=2, final_activation=True
        )
        self.cross_view_fusion = make_residual_mlp(feature_dim * 4, hidden_dim, 1, blocks=1)
        zero_init_last_linear(self.cross_view_fusion)
        self.mode = str(mode)

    def forward(
        self,
        per_view: torch.Tensor,
        view_support: torch.Tensor,
        view_valid: torch.Tensor,
        branch_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, v, mb, c = per_view.shape
        h = self.transform(per_view)
        support = view_support.permute(0, 2, 1, 3).contiguous()
        valid = view_valid.permute(0, 2, 1)[:, :, :, None] & branch_valid[:, None, None]
        support_valid = torch.where(valid, support, torch.zeros_like(support))
        sum_support = support_valid.sum(dim=2, keepdim=True)
        sum_h = (support_valid[..., None] * h).sum(dim=2, keepdim=True)
        loo_denom = (sum_support - support_valid).clamp_min(1e-8)
        loo = (sum_h - support_valid[..., None] * h) / loo_denom[..., None]
        loo = torch.where(
            (sum_support - support_valid)[..., None] > 0.0, loo, torch.zeros_like(loo)
        )
        correction_in = torch.cat([h, loo, h - loo, h * loo], dim=-1)
        logits = torch.log(support.clamp_min(1e-8)) + torch.tanh(
            self.cross_view_fusion(correction_in).squeeze(-1)
        )
        weights = masked_softmax(logits, valid, dim=2)
        z = (weights[..., None] * h).sum(dim=2)
        no_views = ~valid.any(dim=2)
        z = torch.where(no_views[..., None], torch.zeros_like(z), z)
        valid_view_count = view_valid.permute(0, 2, 1).sum(dim=2, keepdim=True).clamp_min(1)
        lam = (support_valid.sum(dim=2) / valid_view_count).clamp(0.0, 1.0)
        if self.mode == "shared":
            branch_support = lam * branch_valid[:, None].to(dtype=lam.dtype)
            denom = branch_support.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            z_shared = (branch_support[..., None] * z).sum(dim=2, keepdim=True) / denom[..., None]
            z = z_shared.expand_as(z)
        return z, weights, lam


class CandidateAssignmentHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        p_min: float = 1e-4,
        support_center: float = 0.10,
        support_temperature: float = 0.10,
        support_logit_weight: float = 1.0,
        tau0: float = 0.25,
        tau_p: float = 1.0,
    ):
        super().__init__()
        self.compensation_assignment_head = make_residual_mlp(
            feature_dim + 2, hidden_dim, 1, blocks=1
        )
        self.candidate_assignment_head = make_residual_mlp(feature_dim + 4, hidden_dim, 1, blocks=1)
        zero_init_last_linear(self.compensation_assignment_head)
        zero_init_last_linear(self.candidate_assignment_head)
        self.p_min = float(p_min)
        self.support_center = float(support_center)
        self.support_temperature = float(support_temperature)
        self.support_logit_weight = float(support_logit_weight)
        self.tau0 = float(tau0)
        self.tau_p = float(tau_p)

    def forward(
        self,
        z: torch.Tensor,
        points_mm: torch.Tensor,
        router: dict[str, torch.Tensor],
        candidates: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        b, n, mb, _c = z.shape
        m = mb - 1
        branch_exists = router["branch_valid"][:, None].expand(b, n, mb).clone()
        branch_exists[:, :, 0] = True
        p_all = router["p_all"]
        lam = router["Lambda"]
        no_valid_views = router["valid_view_count"] <= 0
        p0 = p_all[..., :1].clamp(self.p_min, 1.0 - self.p_min)
        comp_delta = torch.tanh(
            self.compensation_assignment_head(
                torch.cat([z[:, :, :1], 1.0 - p0[..., None], lam[:, :, :1, None]], dim=-1)
            )
        ).squeeze(-1)
        pi0 = torch.sigmoid(self.tau0 * torch.logit(p0) + comp_delta)
        if m == 0:
            return torch.ones((b, n, 1), dtype=z.dtype, device=z.device)
        rel = (points_mm[:, :, None] - candidates["candidate_centers_mm"][:, None]) / candidates[
            "candidate_support_scales_mm"
        ][:, None, :, None].clamp_min(1e-6)
        cand_visible = router.get("candidate_geometric_visible")
        if cand_visible is None:
            cand_visible = lam[:, :, 1:] > 0.0
        cand_active = branch_exists[:, :, 1:] & cand_visible
        support_gate = torch.sigmoid(
            (lam[:, :, 1:] - self.support_center) / max(self.support_temperature, 1e-6)
        )
        cand_delta = torch.tanh(
            self.candidate_assignment_head(
                torch.cat([z[:, :, 1:], rel, lam[:, :, 1:, None]], dim=-1)
            )
        ).squeeze(-1)
        cand_logits = (
            self.tau_p * torch.log(p_all[..., 1:].clamp(self.p_min, 1.0))
            + self.support_logit_weight * torch.log(support_gate.clamp_min(self.p_min))
            + cand_delta
        )
        cand_pi = masked_softmax(cand_logits, cand_active, dim=-1)
        has_cand = cand_active.any(dim=-1, keepdim=True) & ~no_valid_views[..., None]
        pi0 = torch.where(has_cand, pi0, torch.ones_like(pi0))
        cand_pi = torch.where(has_cand, (1.0 - pi0) * cand_pi, torch.zeros_like(cand_pi))
        pi = torch.cat([pi0, cand_pi], dim=-1)
        return pi / pi.sum(dim=-1, keepdim=True).clamp_min(1e-8)
