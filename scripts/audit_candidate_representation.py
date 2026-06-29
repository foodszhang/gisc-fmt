#!/usr/bin/env python
"""Audit frozen SHQ candidate geometry, representations, routing, and gradients."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data._utils.collate import default_collate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from quick_audit_view_separability import (  # noqa: E402
    formal_density_loss,
    infer_checkpoint_overrides,
    pairwise_separability,
)

from minr_fmt.datamodule import TrainingDataModule  # noqa: E402
from minr_fmt.module import TrainingLightningModule  # noqa: E402

EPS = 1.0e-8


def hungarian_candidate_matching(
    candidate_centers: torch.Tensor,
    candidate_valid: torch.Tensor,
    component_centers: torch.Tensor,
    component_valid: torch.Tensor,
    threshold_mm: float,
) -> dict[str, float]:
    """Match candidates to GT components and report thresholded geometry rates."""
    candidates = candidate_centers[candidate_valid].float()
    components = component_centers[component_valid].float()
    nc, ng = len(candidates), len(components)
    if nc == 0 or ng == 0:
        return {
            "component_coverage_rate": 0.0,
            "candidate_matching_rate": 0.0,
            "mean_matched_center_error_mm": float("nan"),
            "duplicate_candidate_rate": 0.0,
            "unmatched_candidate_rate": float(nc > 0),
            "one_candidate_multiple_components_rate": 0.0,
            "valid_candidate_count": float(nc),
            "component_count": float(ng),
        }
    distance = torch.cdist(candidates, components).cpu().numpy()
    rows, cols = linear_sum_assignment(distance)
    accepted = distance[rows, cols] <= threshold_mm
    matched_rows = set(rows[accepted].tolist())
    matched_cols = set(cols[accepted].tolist())
    close = distance <= threshold_mm
    duplicate = np.asarray(
        [i not in matched_rows and bool(close[i].any()) for i in range(nc)], dtype=bool
    )
    multi = (close.sum(axis=1) > 1).sum()
    errors = distance[rows[accepted], cols[accepted]]
    return {
        "component_coverage_rate": len(matched_cols) / ng,
        "candidate_matching_rate": len(matched_rows) / nc,
        "mean_matched_center_error_mm": float(errors.mean()) if len(errors) else float("nan"),
        "duplicate_candidate_rate": float(duplicate.mean()),
        "unmatched_candidate_rate": (nc - len(matched_rows)) / nc,
        "one_candidate_multiple_components_rate": float(multi / nc),
        "valid_candidate_count": float(nc),
        "component_count": float(ng),
    }


def paired_feature_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return row-wise cosine, L2 distance, and relative L2 distance."""
    left, right = left.float(), right.float()
    l2 = (left - right).norm(dim=-1)
    return {
        "cosine": F.cosine_similarity(left, right, dim=-1, eps=EPS),
        "l2": l2,
        "relative_l2": l2 / right.norm(dim=-1).clamp_min(EPS),
    }


def routing_entropy(allocation: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Normalized entropy across the candidate dimension."""
    weight = torch.where(valid, allocation.float().clamp_min(0), 0.0)
    count = valid.sum(dim=-1)
    probability = weight / weight.sum(dim=-1, keepdim=True).clamp_min(EPS)
    entropy = -(probability * probability.clamp_min(EPS).log()).sum(dim=-1)
    normalizer = count.clamp_min(2).float().log()
    return torch.where(count > 1, entropy / normalizer, torch.zeros_like(entropy))


def top_bottom_score_gap(score: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    masked_high = score.masked_fill(~valid, -torch.inf).topk(2, dim=-1).values
    masked_low = score.masked_fill(~valid, torch.inf).topk(2, dim=-1, largest=False).values
    usable = valid.sum(dim=-1) >= 4
    gap = masked_high.mean(-1) - masked_low.mean(-1)
    return torch.where(usable, gap, torch.full_like(gap, torch.nan))


def ranking_perturbation_stability(
    score: torch.Tensor,
    valid: torch.Tensor,
    *,
    noise_std: float = 0.01,
    repeats: int = 20,
    seed: int = 0,
) -> torch.Tensor:
    """Mean top-2 set overlap after small Gaussian score perturbations."""
    generator = torch.Generator(device=score.device).manual_seed(seed)
    base = score.masked_fill(~valid, -torch.inf).topk(2, dim=-1).indices
    overlaps = []
    for _ in range(repeats):
        noise = torch.randn(score.shape, generator=generator, device=score.device)
        changed = (score + noise_std * noise).masked_fill(~valid, -torch.inf)
        perturbed = changed.topk(2, dim=-1).indices
        overlap = (base[..., :, None] == perturbed[..., None, :]).any(dim=-1).float().mean(-1)
        overlaps.append(overlap)
    result = torch.stack(overlaps).mean(0)
    return torch.where(valid.sum(-1) >= 2, result, torch.full_like(result, torch.nan))


def collect_gradient_norm(module: nn.Module) -> float:
    return (
        float(
            sum(
                parameter.grad.detach().float().square().sum()
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            .sqrt()
            .item()
        )
        if any(parameter.grad is not None for parameter in module.parameters())
        else 0.0
    )


def compare_gated_ungated_backward(
    decoder: nn.Module,
    context: torch.Tensor,
    encoded: torch.Tensor,
    shared_logit: torch.Tensor,
    gate: torch.Tensor,
    alpha: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """Small reusable gradient diagnostic used by the audit and unit tests."""
    norms = {}
    for name in ("gated", "ungated", "unweighted"):
        decoder.zero_grad(set_to_none=True)
        delta = decoder(context, encoded)
        if name == "gated":
            logit = shared_logit.detach() + gate[..., None] * (alpha[..., None] * delta).sum(2)
        elif name == "ungated":
            logit = shared_logit.detach() + (alpha.detach()[..., None] * delta).sum(2)
        else:
            logit = shared_logit.detach() + delta.mean(2)
        F.binary_cross_entropy_with_logits(logit, target).backward()
        norms[name] = collect_gradient_norm(decoder)
    norms["ungated_over_gated"] = norms["ungated"] / max(norms["gated"], EPS)
    norms["unweighted_over_gated"] = norms["unweighted"] / max(norms["gated"], EPS)
    return norms


class AuditHooks:
    def __init__(self, model: nn.Module):
        self.values: dict[str, Any] = {}
        self.handles = [
            model.candidate_builder.register_forward_hook(self._set("candidate")),
            model.candidate_router.register_forward_hook(self._set("router")),
            model.view_encoder.register_forward_hook(self._set("per_view")),
            model.quotient_aggregator.register_forward_hook(self._append("quotient")),
        ]

    def _set(self, key: str):
        def hook(_module, _inputs, output):
            self.values[key] = output

        return hook

    def _append(self, key: str):
        def hook(_module, _inputs, output):
            self.values.setdefault(key, []).append(output)

        return hook

    def clear(self) -> None:
        self.values.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def tensor_summary(values: Iterable[torch.Tensor | float]) -> dict[str, float]:
    parts = [torch.as_tensor(value).detach().float().flatten().cpu() for value in values]
    finite = torch.cat(parts) if parts else torch.empty(0)
    finite = finite[torch.isfinite(finite)]
    if not len(finite):
        return {key: float("nan") for key in ("mean", "median", "p10", "p50", "p90")}
    return {
        "mean": float(finite.mean()),
        "median": float(finite.median()),
        "p10": float(torch.quantile(finite, 0.1)),
        "p50": float(torch.quantile(finite, 0.5)),
        "p90": float(torch.quantile(finite, 0.9)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        rows = [{"status": "no_data"}]
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sample_indices(dataset: Any, count: int, preferred_ids: list[str]) -> list[int]:
    preferred = set(preferred_ids)
    chosen, fallback = [], []
    for index in range(len(dataset)):
        item = dataset[index]
        if int(torch.as_tensor(item["num_foci"]).item()) not in (2, 3):
            continue
        sample_id = str(item["sample_id"])
        (chosen if sample_id in preferred else fallback).append(index)
        if len(chosen) == len(preferred) and len(chosen) + len(fallback) >= count:
            break
    result = chosen + fallback
    if len(result) < count:
        raise RuntimeError(f"Need {count} multi-source samples, found {len(result)}")
    return result[:count]


def component_centers(
    batch: Mapping[str, Any], voxel_size_mm: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if "gt_component_centers_mm" in batch:
        return batch["gt_component_centers_mm"], batch["gt_component_valid_mask"]
    from scipy import ndimage

    volume = batch["gt_voxels"][0].detach().cpu().numpy() > 0.5
    labels, count = ndimage.label(volume, ndimage.generate_binary_structure(3, 1))
    components = []
    for label_id in range(1, count + 1):
        coordinates = np.argwhere(labels == label_id)
        if len(coordinates):
            components.append((len(coordinates), (coordinates.mean(0) + 0.5) * voxel_size_mm))
    components.sort(reverse=True, key=lambda item: item[0])
    centers = torch.tensor(np.asarray([item[1] for item in components]), dtype=torch.float32)[None]
    valid = torch.ones(centers.shape[:2], dtype=torch.bool)
    return centers, valid


def off_diagonal_pairs(
    value: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten ordered off-diagonal feature pairs from ``[..., M, C]``."""
    m = value.shape[-2]
    pair_valid = valid[..., :, None] & valid[..., None, :]
    pair_valid &= ~torch.eye(m, dtype=torch.bool, device=value.device)
    left = value[..., :, None, :].expand(*value.shape[:-2], m, m, value.shape[-1])
    right = value[..., None, :, :].expand_as(left)
    return left[pair_valid], right[pair_valid]


def analyze_sample(
    hooks: AuditHooks,
    output: Mapping[str, Any],
    batch: Mapping[str, Any],
    sample_id: str,
    args: argparse.Namespace,
) -> dict[str, list[dict[str, Any]]]:
    candidate = hooks.values["candidate"]
    router = hooks.values["router"]
    per_view = hooks.values["per_view"][..., 1:, :].float()  # B,N,V,M,C
    shared_q = hooks.values["quotient"][-2]["quotient"].squeeze(2).float()
    candidate_q = hooks.values["quotient"][-1]["quotient"].float()
    aux = output["aux_outputs"]
    valid_candidate = candidate["candidate_valid_mask"]
    b, n, v, m, _ = per_view.shape
    detector_valid = candidate["candidate_detector_valid_mask"][:, None]
    routed = router["r"].permute(0, 2, 1, 3)[..., 1:]
    view_valid = detector_valid & valid_candidate[:, None, None] & (routed > 0)
    query_valid = aux["measurement_supported"][:, :, None, None]
    view_valid &= query_valid
    foreground = batch["point_densities"] > 0
    shared_error = (torch.sigmoid(aux["shared_logit"]).squeeze(-1) - batch["point_densities"]).abs()
    error_cut = torch.quantile(shared_error.float(), 0.8, dim=1, keepdim=True)
    masks = {
        "all_queries": torch.ones_like(foreground, dtype=torch.bool),
        "foreground_queries": foreground,
        "shared_error_top20": shared_error >= error_cut,
        f"{int(torch.as_tensor(batch['num_foci']).item())}_source": torch.ones_like(
            foreground, dtype=torch.bool
        ),
    }

    feature_rows = []
    for slice_name, query_mask in masks.items():
        uv_valid = view_valid & query_mask[:, :, None, None]
        left, right = off_diagonal_pairs(per_view, uv_valid)
        uv_metrics = paired_feature_metrics(left, right) if len(left) else {}
        q_valid = valid_candidate[:, None].expand(b, n, m) & query_mask[..., None]
        q_left, q_right = off_diagonal_pairs(candidate_q, q_valid)
        q_metrics = paired_feature_metrics(q_left, q_right) if len(q_left) else {}
        shared_expanded = shared_q[:, :, None].expand_as(candidate_q)
        qs_metrics = paired_feature_metrics(candidate_q[q_valid], shared_expanded[q_valid])
        # Same candidate across view: compare each view with the next valid view.
        rolled = per_view.roll(-1, dims=2)
        cross_valid = uv_valid & uv_valid.roll(-1, dims=2)
        cross_metrics = paired_feature_metrics(per_view[cross_valid], rolled[cross_valid])
        row = {"sample_id": sample_id, "slice": slice_name}
        for prefix, values in (
            ("u_inter_candidate", uv_metrics),
            ("q_inter_candidate", q_metrics),
            ("q_to_shared", qs_metrics),
            ("same_candidate_cross_view", cross_metrics),
        ):
            for metric, tensor in values.items():
                row[f"{prefix}_{metric}_median"] = (
                    float(tensor.median()) if len(tensor) else float("nan")
                )
        feature_rows.append(row)

    allocation = router["a"].permute(0, 2, 1, 3)[..., 1:].float()
    allocation_valid = valid_candidate[:, None, None].expand_as(allocation)
    entropy = routing_entropy(allocation, allocation_valid)
    alloc_pairs_l, alloc_pairs_r = off_diagonal_pairs(
        allocation.permute(0, 1, 3, 2), valid_candidate[:, None].expand(b, n, m)
    )
    routing_cos = paired_feature_metrics(alloc_pairs_l, alloc_pairs_r)["cosine"]
    utilization = (allocation.sum(dim=(1, 2)) > args.allocation_zero_threshold) & valid_candidate
    uniform = (
        allocation
        - allocation.sum(-1, keepdim=True)
        / valid_candidate.sum(-1, keepdim=True).clamp_min(1)[:, None]
    ).abs().amax(-1) < args.uniform_tolerance
    routing_row = {
        "sample_id": sample_id,
        "routing_entropy_mean": float(entropy.mean()),
        "candidate_utilization_rate": float(utilization[valid_candidate].float().mean()),
        "mean_allocation_difference": float((alloc_pairs_l - alloc_pairs_r).abs().mean()),
        "nearly_uniform_fraction": float(uniform.float().mean()),
        "almost_zero_candidate_fraction": float((~utilization[valid_candidate]).float().mean()),
        "routing_profile_cosine_median": float(routing_cos.median()),
    }

    q_valid_all = valid_candidate[:, None].expand(b, n, m)
    ql, qr = off_diagonal_pairs(candidate_q, q_valid_all)
    q_inter = paired_feature_metrics(ql, qr)
    q_shared = paired_feature_metrics(
        candidate_q[q_valid_all], shared_q[:, :, None].expand_as(candidate_q)[q_valid_all]
    )
    quotient_row = {
        "sample_id": sample_id,
        "dispersion_mean": float(aux["quotient_dispersion"][q_valid_all].float().mean()),
        "inter_candidate_l2_median": float(q_inter["l2"].median()),
        "candidate_shared_l2_median": float(q_shared["l2"].median()),
        "candidate_shared_cosine_median": float(q_shared["cosine"].median()),
        "candidate_shared_relative_l2_median": float(q_shared["relative_l2"].median()),
        "candidate_quotient_variance_queries": float(candidate_q[q_valid_all].var(dim=0).mean()),
        "candidate_quotient_variance_views": float(per_view[view_valid].var(dim=0).mean()),
    }

    centers = candidate["candidate_uv_px"][:, None].expand(-1, n, -1, -1, -1)
    scales = candidate["candidate_detector_support_scales_px"][:, None].expand(-1, n, -1, -1)
    geo, meas, combined, usable = pairwise_separability(per_view, centers, scales, view_valid)
    sep_rows = []
    for name, score in (("geometry", geo), ("measurement", meas), ("combined", combined)):
        ordered_score = score.permute(0, 1, 3, 2)
        ordered_valid = (view_valid & usable).permute(0, 1, 3, 2)
        gap = top_bottom_score_gap(ordered_score, ordered_valid)
        stability = ranking_perturbation_stability(
            ordered_score,
            ordered_valid,
            noise_std=args.perturbation_std,
            repeats=args.perturbation_repeats,
            seed=args.seed,
        )
        sep_rows.append(
            {
                "sample_id": sample_id,
                "score": name,
                "top_bottom_gap_median": float(torch.nanmedian(gap)),
                "top_bottom_gap_p10": float(torch.nanquantile(gap, 0.1)),
                "top_bottom_gap_p90": float(torch.nanquantile(gap, 0.9)),
                "gap_below_0_05_fraction": float((gap[torch.isfinite(gap)] < 0.05).float().mean()),
                "top2_overlap_mean": float(torch.nanmean(stability)),
                "unstable_top2_fraction": float(
                    (stability[torch.isfinite(stability)] < 0.6).float().mean()
                ),
            }
        )

    gt_centers, gt_valid = component_centers(batch, args.voxel_size_mm)
    matching_rows = []
    for threshold in args.match_thresholds:
        values = hungarian_candidate_matching(
            candidate["candidate_centers_mm"][0].cpu(),
            valid_candidate[0].cpu(),
            gt_centers[0].cpu(),
            gt_valid[0].cpu(),
            threshold,
        )
        matching_rows.append({"sample_id": sample_id, "threshold_mm": threshold, **values})
    return {
        "matching": matching_rows,
        "feature": feature_rows,
        "routing": [routing_row],
        "quotient": [quotient_row],
        "separability": sep_rows,
    }


def formal_loss_cfg(cfg: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(cfg.loss.get(key, default))
        for key, default in {
            "pos_weight": 1.0,
            "dice_weight": 0.0,
            "sparse_weight": 0.0,
            "density_bce_weight": 0.0,
            "tversky_weight": 0.0,
            "tversky_alpha": 0.6,
            "tversky_beta": 0.4,
            "tversky_gamma": 1.33,
        }.items()
    }


def gradient_audit(
    model: nn.Module,
    dataset: Any,
    indices: list[int],
    device: torch.device,
    loss_cfg: Mapping[str, float],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    modules = {
        "router": model.candidate_router,
        "candidate_view_encoder": model.view_encoder,
        "quotient": model.quotient_aggregator,
        "residual_decoder": model.source_hypothesis_residual_decoder,
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in modules.values():
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    rows = []
    captured: dict[str, torch.Tensor] = {}

    def capture_residual(_module, _inputs, value):
        captured["raw_residual"] = value

    residual_handle = model.source_hypothesis_residual_decoder.register_forward_hook(
        capture_residual
    )
    for batch_index, index in enumerate(indices[: args.gradient_batches]):
        batch = default_collate([dataset[index]])
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        for variant in ("gated", "ungated", "unweighted"):
            model.zero_grad(set_to_none=True)
            captured.clear()
            output = model(
                batch["surface_measurements_packed"],
                batch["query_coordinates_mm"],
                detector_valid_mask=batch.get("detector_valid_mask"),
                depth_maps=batch.get("depth_maps"),
                batch=batch,
                return_diagnostics=True,
            )
            aux = output["aux_outputs"]
            gate = aux["proposal_gate"]
            alpha = aux["alpha"]
            delta_logit = model.delta_l_max * torch.tanh(captured["raw_residual"])
            if variant == "gated":
                logit = aux["final_logit"]
            elif variant == "ungated":
                logit = aux["shared_logit"].detach() + (
                    alpha.detach()[..., None] * delta_logit
                ).sum(2)
            else:
                logit = aux["shared_logit"].detach() + delta_logit.mean(2)
            prediction = torch.sigmoid(logit)
            support = aux["measurement_supported"][..., None]
            sample_ids = torch.zeros(prediction.shape[1], dtype=torch.long, device=device)
            loss = formal_density_loss(
                prediction.squeeze(0),
                batch["point_densities"].reshape(-1, 1),
                support.reshape(-1, 1),
                sample_ids,
                loss_cfg,
            )
            loss.backward()
            scale = gate[..., None] * alpha
            rows.append(
                {
                    "batch": batch_index,
                    "sample_id": str(batch["sample_id"][0]),
                    "variant": variant,
                    "loss": float(loss.detach()),
                    "mean_g_p": float(gate.detach().mean()),
                    "mean_alpha": float(alpha.detach().mean()) if alpha.numel() else 0.0,
                    "mean_g_p_alpha": float(scale.detach().mean()) if scale.numel() else 0.0,
                    "median_g_p_alpha": float(scale.detach().median()) if scale.numel() else 0.0,
                    "p10_g_p_alpha": float(torch.quantile(scale.detach().float(), 0.1))
                    if scale.numel()
                    else 0.0,
                    "p90_g_p_alpha": float(torch.quantile(scale.detach().float(), 0.9))
                    if scale.numel()
                    else 0.0,
                    **{
                        f"{name}_gradient_norm": collect_gradient_norm(module)
                        for name, module in modules.items()
                    },
                    "shared_decoder_gradient_norm": collect_gradient_norm(
                        model.shared_density_logit_decoder
                    ),
                }
            )
    residual_handle.remove()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="outputs/shq_fmt/q8192_bs2_cosine10_from_e3/checkpoints/epoch=07-val_dice=0.5995.ckpt",
    )
    parser.add_argument("--config", default="configs/exp/fmt_simgen_v2_shq_quotient_residual.yaml")
    parser.add_argument("--output-dir", default="outputs/shq_candidate_representation_audit_v1")
    parser.add_argument(
        "--preferred-audit-dir", default="outputs/shq_audit/epoch7_q8192_100x100_v6_final_nested"
    )
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--queries-per-sample", type=int, default=8192)
    parser.add_argument("--gradient-batches", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--voxel-size-mm", type=float, default=0.2)
    parser.add_argument("--match-thresholds", type=float, nargs="+", default=[6.0, 8.0, 10.0])
    parser.add_argument("--collapse-cosine-threshold", type=float, default=0.95)
    parser.add_argument("--collapse-relative-distance-threshold", type=float, default=0.05)
    parser.add_argument("--routing-entropy-threshold", type=float, default=0.95)
    parser.add_argument("--routing-cosine-threshold", type=float, default=0.95)
    parser.add_argument("--allocation-zero-threshold", type=float, default=1e-3)
    parser.add_argument("--uniform-tolerance", type=float, default=0.02)
    parser.add_argument("--perturbation-std", type=float, default=0.01)
    parser.add_argument("--perturbation-repeats", type=int, default=20)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                f"exp={Path(args.config).stem}",
                "data.dataset_type=fmt_simgen",
                f"data.sample_num={args.queries_per_sample}",
                f"data.num_queries={args.queries_per_sample}",
                f"data.eval_sample_num={args.queries_per_sample}",
                "data.batch_size=1",
                "data.eval_batch_size=1",
                "data.num_workers=0",
                "data.persistent_workers=false",
                *infer_checkpoint_overrides(args.checkpoint),
                *args.overrides,
            ],
        )
    module = TrainingLightningModule.load_from_checkpoint(
        args.checkpoint, cfg=cfg, map_location="cpu"
    )
    model = module.net.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dm = TrainingDataModule(cfg)
    dm.setup("fit")
    preferred_ids = []
    for path in (
        Path(args.preferred_audit_dir) / "val_sample_ids.json",
        Path(args.preferred_audit_dir) / "fixed_sample_ids.json",
        Path("outputs/shq_separability_quick_v1/fixed_sample_ids.json"),
    ):
        if path.exists():
            payload = json.loads(path.read_text())
            preferred_ids = (
                payload.get("validation", payload) if isinstance(payload, dict) else payload
            )
            break
    val_indices = sample_indices(dm.val_dataset, args.samples, preferred_ids)
    rows = {key: [] for key in ("matching", "feature", "routing", "quotient", "separability")}
    hooks = AuditHooks(model)
    selected_ids = []
    try:
        for audit_index, index in enumerate(val_indices):
            seed = args.seed + index
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            batch = default_collate([dm.val_dataset[index]])
            sample_id = str(batch["sample_id"][0])
            selected_ids.append(sample_id)
            device_batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            hooks.clear()
            with (
                torch.inference_mode(),
                torch.autocast(
                    device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda" and args.precision == "bf16",
                ),
            ):
                result = model(
                    device_batch["surface_measurements_packed"],
                    device_batch["query_coordinates_mm"],
                    detector_valid_mask=device_batch.get("detector_valid_mask"),
                    depth_maps=device_batch.get("depth_maps"),
                    batch=device_batch,
                    return_diagnostics=True,
                )
            analyzed = analyze_sample(hooks, result, device_batch, sample_id, args)
            for key in rows:
                rows[key].extend(analyzed[key])
    finally:
        hooks.close()
    (output_dir / "fixed_sample_ids.json").write_text(json.dumps(selected_ids, indent=2))
    gradients = gradient_audit(
        model,
        dm.train_dataset,
        sample_indices(dm.train_dataset, args.gradient_batches, []),
        device,
        formal_loss_cfg(cfg),
        args,
    )
    file_map = {
        "matching": "candidate_matching.csv",
        "feature": "feature_collapse_metrics.csv",
        "routing": "routing_metrics.csv",
        "quotient": "quotient_metrics.csv",
        "separability": "separability_stability.csv",
    }
    for key, filename in file_map.items():
        write_csv(output_dir / filename, rows[key])
    write_csv(output_dir / "gradient_flow_metrics.csv", gradients)

    matching8 = [r for r in rows["matching"] if r["threshold_mm"] == 8.0]
    aggregate = {
        "coverage": float(np.mean([r["component_coverage_rate"] for r in matching8])),
        "duplicate": float(np.mean([r["duplicate_candidate_rate"] for r in matching8])),
        "unmatched": float(np.mean([r["unmatched_candidate_rate"] for r in matching8])),
    }
    all_features = [r for r in rows["feature"] if r["slice"] == "all_queries"]
    feature = {
        key: float(np.nanmedian([r[key] for r in all_features]))
        for key in (
            "u_inter_candidate_cosine_median",
            "q_to_shared_cosine_median",
            "q_to_shared_relative_l2_median",
        )
    }
    routing = {
        key: float(np.nanmean([r[key] for r in rows["routing"]]))
        for key in (
            "routing_entropy_mean",
            "candidate_utilization_rate",
            "routing_profile_cosine_median",
        )
    }
    combined = [r for r in rows["separability"] if r["score"] == "combined"]
    separability = {
        "top_bottom_gap_median": float(
            np.nanmedian([r["top_bottom_gap_median"] for r in combined])
        ),
        "top2_overlap_mean": float(np.nanmean([r["top2_overlap_mean"] for r in combined])),
    }
    gated = [r for r in gradients if r["variant"] == "gated"]
    ungated = [r for r in gradients if r["variant"] == "ungated"]
    grad_ratio = float(
        np.mean(
            [
                u["residual_decoder_gradient_norm"] / max(g["residual_decoder_gradient_norm"], EPS)
                for g, u in zip(gated, ungated, strict=True)
            ]
        )
    )
    scale_mean = float(np.mean([r["mean_g_p_alpha"] for r in gated]))
    scale_median = float(np.median([r["median_g_p_alpha"] for r in gated]))
    cases = []
    if aggregate["coverage"] < 0.70 or aggregate["duplicate"] > 0.40:
        cases.append(
            (
                "candidate_construction",
                "Candidate construction does not provide sufficiently distinct source-region "
                "references; downstream routing and quotient results cannot yet be interpreted.",
            )
        )
    if (
        aggregate["coverage"] >= 0.70
        and feature["q_to_shared_cosine_median"] > args.collapse_cosine_threshold
        and feature["u_inter_candidate_cosine_median"] > args.collapse_cosine_threshold
    ):
        cases.append(
            (
                "candidate_representation_collapse",
                "The candidate supports are spatially relevant, but candidate-specific measurement "
                "representations have collapsed toward the shared representation.",
            )
        )
    if (
        routing["routing_entropy_mean"] > args.routing_entropy_threshold
        or routing["routing_profile_cosine_median"] > args.routing_cosine_threshold
    ):
        cases.append(
            (
                "routing_collapse",
                "The current routing does not establish meaningful candidate-specific competition "
                "before cross-view aggregation.",
            )
        )
    if scale_median < 0.02 and grad_ratio >= 5.0:
        cases.append(
            (
                "gradient_starvation",
                "The candidate path is substantially under-trained because proposal gating and "
                "candidate weighting suppress its task gradient.",
            )
        )
    if separability["top_bottom_gap_median"] < 0.05 or separability["top2_overlap_mean"] < 0.6:
        cases.append(
            (
                "separability_ranking_unreliable",
                "The current high-/low-separability view partition is not stable enough "
                "to evaluate view complementarity.",
            )
        )
    summary = {
        "candidate_geometry_status": cases[0][1]
        if cases and cases[0][0] == "candidate_construction"
        else "acceptable",
        "candidate_feature_status": next(
            (text for case, text in cases if case == "candidate_representation_collapse"),
            "no automatic collapse flag",
        ),
        "routing_status": next(
            (text for case, text in cases if case == "routing_collapse"),
            "competitive routing detected",
        ),
        "quotient_status": "shared-like collapse"
        if feature["q_to_shared_cosine_median"] > args.collapse_cosine_threshold
        or feature["q_to_shared_relative_l2_median"] < args.collapse_relative_distance_threshold
        else "candidate-specific distance detected",
        "separability_ranking_status": next(
            (text for case, text in cases if case == "separability_ranking_unreliable"), "stable"
        ),
        "gradient_flow_status": next(
            (text for case, text in cases if case == "gradient_starvation"),
            "no automatic starvation flag",
        ),
        "primary_failure_point": cases[0][0] if cases else "none_detected",
        "recommended_next_step": cases[0][1]
        if cases
        else "No formal method change is justified by this audit alone.",
        "metrics": {
            "candidate_matching_8mm": aggregate,
            "features": feature,
            "routing": routing,
            "separability": separability,
            "gradient": {
                "mean_g_p_alpha": scale_mean,
                "median_g_p_alpha": scale_median,
                "ungated_over_gated_candidate_gradient": grad_ratio,
            },
        },
        "active_cases": [{"case": case, "message": text} for case, text in cases],
        "settings": vars(args),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
