#!/usr/bin/env python
"""Strictly nested S0--S2 frozen-feature audit for SHQ-FMT.

The three variants share one architecture and differ only by deterministic input masks.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
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
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.datamodule import TrainingDataModule  # noqa: E402
from minr_fmt.module import TrainingLightningModule  # noqa: E402

EPS = 1.0e-8
CACHE_VERSION = "shq_nested_frozen_features_v6"
VARIANTS = ("S0", "S1", "S2")
FEATURE_GROUPS = (
    "shared_quotient",
    "query_coordinate_encoding",
    "quotient_difference",
    "relative_query",
    "candidate_score",
    "spatial_prior",
    "quotient_support",
    "quotient_dispersion",
    "covariance_eigen_scales",
)
FEATURE_MASKS = {
    "S0": {
        name: name in {"shared_quotient", "query_coordinate_encoding"} for name in FEATURE_GROUPS
    },
    "S1": {name: name != "quotient_difference" for name in FEATURE_GROUPS},
    "S2": {name: True for name in FEATURE_GROUPS},
}
LOSS_COMPONENTS = ("density", "density_bce", "dice", "tversky", "sparse")


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def feature_slices(dimensions: Mapping[str, int]) -> dict[str, slice]:
    start = 0
    result = {}
    for name in FEATURE_GROUPS:
        width = int(dimensions[name])
        result[name] = slice(start, start + width)
        start += width
    return result


def apply_feature_mask(
    features: torch.Tensor,
    variant: str,
    dimensions: Mapping[str, int],
) -> torch.Tensor:
    if variant not in FEATURE_MASKS:
        raise ValueError(f"Unknown nested variant: {variant}")
    result = features.clone()
    for name, feature_slice in feature_slices(dimensions).items():
        if not FEATURE_MASKS[variant][name]:
            result[..., feature_slice] = 0
    return result


class NestedCandidateProbe(nn.Module):
    """The single branch-wise, alpha-pooled architecture used by S0, S1, and S2."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, delta_l_max: float = 3.0):
        super().__init__()
        self.delta_l_max = float(delta_l_max)
        self.shared_candidate_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.branch_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        candidate_feature: torch.Tensor,
        alpha: torch.Tensor,
        candidate_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if candidate_feature.shape[-2] == 0:
            return candidate_feature.new_zeros((*candidate_feature.shape[:-2], 1))
        valid = candidate_valid_mask.bool()
        encoded = self.shared_candidate_encoder(candidate_feature)
        branch = self.delta_l_max * torch.tanh(self.branch_head(encoded))
        weights = alpha.detach().to(branch.dtype) * valid.to(branch.dtype)
        residual = (weights[..., None] * branch).sum(dim=-2)
        return torch.where(valid.any(dim=-1, keepdim=True), residual, torch.zeros_like(residual))


def build_candidate_features(
    data: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, int]]:
    m = data["q_m"].shape[-2]
    q_s = data["q_s"].unsqueeze(-2).expand(*data["q_s"].shape[:-1], m, -1)
    query_encoding = (
        data["query_coordinate_encoding"]
        .unsqueeze(-2)
        .expand(*data["query_coordinate_encoding"].shape[:-1], m, -1)
    )
    groups = {
        "shared_quotient": q_s,
        "query_coordinate_encoding": query_encoding,
        "quotient_difference": data["q_m_minus_q_s"],
        "relative_query": data["relative_query"],
        "candidate_score": data["candidate_score"].unsqueeze(-1),
        "spatial_prior": data["spatial_prior"].unsqueeze(-1),
        "quotient_support": data["quotient_support"].unsqueeze(-1),
        "quotient_dispersion": data["quotient_dispersion"].unsqueeze(-1),
        "covariance_eigen_scales": data["candidate_covariance_eigen_scales"],
    }
    dimensions = {name: int(groups[name].shape[-1]) for name in FEATURE_GROUPS}
    return torch.cat([groups[name] for name in FEATURE_GROUPS], dim=-1), dimensions


def formal_density_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
    sample_id: torch.Tensor,
    loss_config: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Final-density-only terms from MorphologyAwareDensityLoss, grouped by sample."""
    pred = prediction.clamp(0.0, 1.0)
    target = target.to(pred).clamp(0.0, 1.0)
    support = support.to(pred)
    weight = (1.0 + target * (float(loss_config["pos_weight"]) - 1.0)) * support
    density = (F.smooth_l1_loss(pred, target, reduction="none") * weight).sum()
    density = density / weight.sum().clamp_min(EPS)
    density_bce = pred.sum() * 0.0
    if float(loss_config["density_bce_weight"]) > 0:
        bounded = pred.float().clamp(1.0e-6, 1.0 - 1.0e-6)
        bce = -(target.float() * bounded.log() + (1.0 - target.float()) * (1.0 - bounded).log())
        density_bce = (bce.to(pred) * weight).sum() / weight.sum().clamp_min(EPS)
    dices, tverskys = [], []
    for identifier in sample_id.unique(sorted=True):
        mask = sample_id == identifier
        p, t, s = pred[mask], target[mask], support[mask]
        if not bool((s.sum() > 0).item()):
            continue
        intersection = (p * t * s).sum()
        denom = (p * s).sum() + (t * s).sum()
        dices.append(1.0 - (2.0 * intersection + 1.0e-6) / (denom + 1.0e-6))
        fp = (p * (1.0 - t) * s).sum()
        fn = ((1.0 - p) * t * s).sum()
        ti = (intersection + 1.0e-6) / (
            intersection
            + float(loss_config["tversky_alpha"]) * fp
            + float(loss_config["tversky_beta"]) * fn
            + 1.0e-6
        )
        tverskys.append(torch.clamp(1.0 - ti, 0.0, 1.0).pow(float(loss_config["tversky_gamma"])))
    zero = pred.sum() * 0.0
    dice = torch.stack(dices).mean() if dices else zero
    tversky = torch.stack(tverskys).mean() if tverskys else zero
    sparse = (pred * (1.0 - target) * support).sum() / support.sum().clamp_min(EPS)
    total = (
        density
        + float(loss_config["density_bce_weight"]) * density_bce
        + float(loss_config["dice_weight"]) * dice
        + float(loss_config["tversky_weight"]) * tversky
        + float(loss_config["sparse_weight"]) * sparse
    )
    return total, {
        "density": density,
        "density_bce": density_bce,
        "dice": dice,
        "tversky": tversky,
        "sparse": sparse,
    }


def task_descent_direction(
    shared_logit: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
    sample_id: torch.Tensor,
    loss_config: Mapping[str, float],
) -> torch.Tensor:
    logits = shared_logit.detach().clone().requires_grad_(True)
    loss, _ = formal_density_loss(torch.sigmoid(logits), target, support, sample_id, loss_config)
    return -torch.autograd.grad(loss, logits)[0].detach()


def cosine_alignment(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    x, y = x[mask].float().flatten(), y[mask].float().flatten()
    if not x.numel():
        return float("nan")
    return float(torch.dot(x, y) / (x.norm() * y.norm()).clamp_min(EPS))


def query_dice(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred = prediction >= threshold
    truth = target > 0
    return float((2 * (pred & truth).sum() + 1.0e-6) / (pred.sum() + truth.sum() + 1.0e-6))


def metric_values(
    prediction: torch.Tensor,
    residual: torch.Tensor,
    data: Mapping[str, torch.Tensor],
    mask: torch.Tensor,
    threshold: float,
    loss_config: Mapping[str, float],
    descent: torch.Tensor,
) -> dict[str, float]:
    if not mask.any():
        return {}
    pred, target = prediction[mask], data["ground_truth_density"][mask]
    support = data["measurement_supported"][mask]
    ids = data["sample_id"][mask]
    loss, components = formal_density_loss(pred, target, support, ids, loss_config)
    centered_pred = pred.float().flatten() - pred.float().mean()
    centered_target = target.float().flatten() - target.float().mean()
    ncc = torch.dot(centered_pred, centered_target) / (
        centered_pred.norm() * centered_target.norm()
    ).clamp_min(EPS)
    nrmse = torch.sqrt(F.mse_loss(pred.float(), target.float())) / (
        target.float().max() - target.float().min()
    ).clamp_min(EPS)
    fp = (pred * (1.0 - target) * support).sum() / support.sum().clamp_min(EPS)
    fn = ((1.0 - pred) * target * support).sum() / support.sum().clamp_min(EPS)
    values = residual[mask].float().flatten()
    result = {
        "formal_density_loss": float(loss),
        **{f"loss_{name}": float(value) for name, value in components.items()},
        "query_dice": query_dice(pred, target, threshold),
        "ncc": float(ncc),
        "nrmse": float(nrmse),
        "prediction_positive_ratio": float((pred >= threshold).float().mean()),
        "false_positive_mass": float(fp),
        "false_negative_mass": float(fn),
        "mean_absolute_residual": float(values.abs().mean()),
        "residual_p10": float(torch.quantile(values, 0.1)),
        "residual_p50": float(torch.quantile(values, 0.5)),
        "residual_p90": float(torch.quantile(values, 0.9)),
        "descent_alignment": cosine_alignment(residual, descent, mask),
    }
    return result


def subset_masks(data: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    target = data["ground_truth_density"].squeeze(-1)
    error = (data["shared_density"] - data["ground_truth_density"]).abs().squeeze(-1)
    top = error >= torch.quantile(error, 0.8)
    masks = {
        "all_samples": torch.ones_like(target, dtype=torch.bool),
        "1_source": data["source_count"] == 1,
        "2_source": data["source_count"] == 2,
        "3_source": data["source_count"] == 3,
        "foreground_queries": target > 0,
        "background_queries": target <= 0,
        "shared_error_top20_queries": top,
    }
    for name in ("weak_source_flag", "deep_source_flag", "adjacent_source_flag"):
        if name in data:
            masks[name.removesuffix("_flag")] = data[name].bool()
    return masks


def sample_metrics(
    prediction: torch.Tensor,
    residual: torch.Tensor,
    data: Mapping[str, torch.Tensor],
    threshold: float,
    loss_config: Mapping[str, float],
) -> list[dict[str, float | int]]:
    rows = []
    descent = task_descent_direction(
        data["shared_logit"],
        data["ground_truth_density"],
        data["measurement_supported"],
        data["sample_id"],
        loss_config,
    )
    for identifier in data["sample_id"].unique(sorted=True):
        mask = data["sample_id"] == identifier
        row: dict[str, float | int] = {"sample_id": int(identifier)}
        row.update(metric_values(prediction, residual, data, mask, threshold, loss_config, descent))
        rows.append(row)
    return rows


def paired_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    repetitions: int = 5000,
    seed: int = 20260629,
) -> dict[str, float]:
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("paired_bootstrap requires equal one-dimensional sample arrays")
    rng = np.random.default_rng(seed)
    differences = right - left
    indices = rng.integers(0, len(differences), size=(repetitions, len(differences)))
    distribution = differences[indices].mean(axis=1)
    return {
        "mean_difference": float(differences.mean()),
        "ci_lower": float(np.quantile(distribution, 0.025)),
        "ci_upper": float(np.quantile(distribution, 0.975)),
    }


def seed_statistics(
    values: Iterable[float], baseline: Iterable[float] | None = None
) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    result: dict[str, float | int] = {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }
    if baseline is not None:
        result["positive_seed_count"] = int((array > np.asarray(list(baseline))).sum())
    return result


def automatic_conclusion(
    comparisons: Mapping[str, Any], variants: Mapping[str, Any], args: argparse.Namespace
) -> tuple[str, str]:
    m10 = comparisons["S1_minus_S0"]["dice"]["mean"]
    m21 = comparisons["S2_minus_S1"]["dice"]["mean"]
    ci10 = comparisons["S1_minus_S0"]["bootstrap_dice"]
    ci21 = comparisons["S2_minus_S1"]["bootstrap_dice"]
    ci20 = comparisons["S2_minus_S0"]["bootstrap_dice"]
    positive10 = comparisons["S1_minus_S0"]["dice"]["positive_seed_count"]
    positive21 = comparisons["S2_minus_S1"]["dice"]["positive_seed_count"]
    loss21 = comparisons["S2_minus_S1"]["formal_density_loss"]["mean"]
    align2 = variants["S2"]["descent_alignment"]["mean"]
    if (
        m21 >= args.quotient_dice_gain
        and positive21 >= args.required_positive_seeds
        and ci21["ci_lower"] > 0
        and loss21 < 0
        and align2 > 0
    ):
        return (
            "The source-hypothesis quotient difference provides reproducible task-relevant "
            "information beyond shared features and hypothesis metadata.",
            "Preserve the formal network and use this locked audit result as the sole evidence "
            "for the quotient-difference claim.",
        )
    metadata_signal = (
        m10 >= args.metadata_dice_gain
        and positive10 >= args.required_positive_seeds
        and ci10["ci_lower"] > 0
    )
    quotient_null = abs(m21) < args.negligible_gain or ci21["ci_lower"] <= 0 <= ci21["ci_upper"]
    if metadata_signal and quotient_null:
        return (
            "Source-hypothesis geometry and support metadata provide incremental value, but the "
            "quotient difference has not demonstrated an independent contribution.",
            "Keep the formal network and manuscript claim unchanged; retain metadata and make "
            "no independent quotient-difference claim.",
        )
    overlap = all(item["ci_lower"] <= 0 <= item["ci_upper"] for item in (ci10, ci21, ci20))
    if abs(m10) < args.negligible_gain and abs(m21) < args.negligible_gain and overlap:
        return (
            "The observed correction gain is primarily attributable to additional decoder "
            "capacity or optimization variance rather than source-hypothesis-specific information.",
            "Make no formal-network change and attribute no incremental value to metadata or "
            "quotient difference.",
        )
    contradictions = (
        positive10 < args.required_positive_seeds
        or positive21 < args.required_positive_seeds
        or ci10["ci_lower"] <= 0 <= ci10["ci_upper"]
        or ci21["ci_lower"] <= 0 <= ci21["ci_upper"]
        or (m21 * -loss21 < 0)
        or align2 <= 0
    )
    if contradictions:
        return (
            "The current frozen-feature evidence is inconclusive. Do not modify the formal SHQ "
            "architecture or manuscript claim based on this audit.",
            "Stop probing; keep the formal SHQ architecture and manuscript claim unchanged.",
        )
    return (
        "The current frozen-feature evidence is inconclusive. Do not modify the formal SHQ "
        "architecture or manuscript claim based on this audit.",
        "Stop probing; keep the formal SHQ architecture and manuscript claim unchanged.",
    )


def protocol_hash(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


class AuditHooks:
    def __init__(self, model: nn.Module):
        self.values: dict[str, Any] = {}
        self.handles = [
            model.candidate_builder.register_forward_hook(self._set("candidate")),
            model.candidate_router.register_forward_hook(self._set("router")),
            model.quotient_aggregator.register_forward_hook(self._append("quotient")),
        ]

    def _set(self, name: str):
        def hook(_module, _inputs, output):
            self.values[name] = output

        return hook

    def _append(self, name: str):
        def hook(_module, _inputs, output):
            self.values.setdefault(name, []).append(output)

        return hook

    def clear(self) -> None:
        self.values.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()
    }


def _flatten_queries(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().cpu().reshape(-1, *value.shape[2:])


def extract_frozen_sample(
    model: nn.Module,
    hooks: AuditHooks,
    batch: Mapping[str, Any],
    device: torch.device,
    use_amp: bool,
    audit_sample_id: int,
) -> dict[str, torch.Tensor]:
    hooks.clear()
    batch = _batch_to_device(batch, device)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp),
    ):
        output = model(
            batch["surface_measurements_packed"],
            batch["query_coordinates_mm"],
            detector_valid_mask=batch.get("detector_valid_mask"),
            depth_maps=batch.get("depth_maps"),
            batch=batch,
            return_diagnostics=True,
        )
    aux = output["aux_outputs"]
    candidate = hooks.values["candidate"]
    router = hooks.values["router"]
    shared_q, candidate_q = hooks.values["quotient"][-2:]
    q_s = shared_q["quotient"].squeeze(2)
    q_m = candidate_q["quotient"]
    relative = router["relative_query"][:, :, 1:]
    m = q_m.shape[2]
    scores = candidate["candidate_scores"][:, None].expand(-1, q_s.shape[1], -1)
    covariance = candidate["candidate_support_covariances_mm"].float()
    eig = torch.linalg.eigvalsh(covariance).clamp_min(EPS).sqrt()
    eig = eig / max(float(model._candidate_cfg.get("scale_max_mm", 12.0)), EPS)
    eig = eig[:, None].expand(-1, q_s.shape[1], -1, -1)
    spatial = torch.exp(-0.5 * relative.square().sum(dim=-1))
    query_norm = (
        (batch["query_coordinates_mm"] / torch.as_tensor(model.trunk_size_mm, device=device))
        .mul(2)
        .sub(1)
    )
    target = batch["point_densities"].float().unsqueeze(-1)
    sample_id = torch.full(target.shape[:2], audit_sample_id, dtype=torch.long, device=device)
    points_ijk = batch["points_ijk"].round().long()
    shape = batch["gt_voxels"].shape[1:]
    query_id = (
        points_ijk[..., 0] * shape[1] * shape[2]
        + points_ijk[..., 1] * shape[2]
        + points_ijk[..., 2]
    )
    result = {
        "shared_logit": _flatten_queries(aux["shared_logit"]),
        "shared_density": _flatten_queries(aux["shared_density"]),
        "ground_truth_density": _flatten_queries(target),
        "query_coordinate_encoding": _flatten_queries(model.position_encoding(query_norm)),
        "q_s": _flatten_queries(q_s),
        "q_m": _flatten_queries(q_m),
        "q_m_minus_q_s": _flatten_queries(q_m - q_s[:, :, None]),
        "relative_query": _flatten_queries(relative),
        "candidate_score": _flatten_queries(scores),
        "spatial_prior": _flatten_queries(spatial),
        "quotient_support": _flatten_queries(candidate_q["support"]),
        "quotient_dispersion": _flatten_queries(candidate_q["dispersion"]),
        "candidate_covariance_eigen_scales": _flatten_queries(eig),
        "candidate_valid_mask": candidate["candidate_valid_mask"]
        .detach()
        .cpu()
        .repeat_interleave(target.shape[1], 0),
        "alpha": _flatten_queries(aux["alpha"]),
        "proposal_gate": _flatten_queries(aux["proposal_gate"][..., None]),
        "measurement_supported": _flatten_queries(aux["measurement_supported"][..., None]).bool(),
        "sample_id": sample_id.flatten().cpu(),
        "query_id": query_id.flatten().cpu(),
        "source_count": torch.as_tensor(batch["num_foci"])
        .flatten()
        .cpu()
        .repeat_interleave(target.shape[1]),
    }
    for name in (
        "weak_source_flag",
        "deep_source_flag",
        "adjacent_source_flag",
        "minimum_separation",
        "intensity_ratio",
    ):
        if name in batch:
            value = torch.as_tensor(batch[name]).flatten().cpu()
            result[name] = value.repeat_interleave(target.shape[1])
    if result["q_m"].shape[1] != m:
        raise RuntimeError("Candidate dimension changed while flattening frozen features")
    return result


def concatenate(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {name: torch.cat([part[name] for part in parts]) for name in parts[0]}


def select_fixed_queries(
    data: Mapping[str, torch.Tensor], indices: torch.Tensor
) -> dict[str, torch.Tensor]:
    return {name: value[indices] for name, value in data.items()}


def save_cache(path: Path, data: Mapping[str, torch.Tensor]) -> None:
    torch.save({"cache_version": CACHE_VERSION, "tensors": dict(data)}, path)


def load_cache(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("cache_version") != CACHE_VERSION:
        raise RuntimeError(f"Feature cache version mismatch in {path}")
    return payload["tensors"]


def train_probe(
    variant: str,
    seed: int,
    train: Mapping[str, torch.Tensor],
    validation: Mapping[str, torch.Tensor],
    dimensions: Mapping[str, int],
    device: torch.device,
    args: argparse.Namespace,
    loss_config: Mapping[str, float],
    checkpoint_dir: Path,
) -> tuple[NestedCandidateProbe, dict[str, torch.Tensor], list[dict[str, float]]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_raw, actual_dimensions = build_candidate_features(train)
    if dict(actual_dimensions) != dict(dimensions):
        raise RuntimeError("Frozen feature dimensions differ across splits")
    train_features = apply_feature_mask(train_raw, variant, dimensions)
    del train_raw
    probe = NestedCandidateProbe(train_features.shape[-1], args.hidden_dim, args.delta_l_max).to(
        device
    )
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_loss, stale, best_state, best_epoch = float("inf"), 0, None, -1
    history = []
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(args.max_epochs):
        probe.train()
        order = torch.randperm(len(train["sample_id"]), generator=generator)
        for start in range(0, len(order), args.batch_size):
            index = order[start : start + args.batch_size]
            residual = probe(
                train_features[index].to(device),
                train["alpha"][index].to(device),
                train["candidate_valid_mask"][index].to(device),
            )
            corrected = torch.sigmoid(train["shared_logit"][index].to(device).detach() + residual)
            loss, _ = formal_density_loss(
                corrected,
                train["ground_truth_density"][index].to(device),
                train["measurement_supported"][index].to(device),
                train["sample_id"][index].to(device),
                loss_config,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        probe.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(validation["sample_id"]), args.batch_size):
                stop = min(start + args.batch_size, len(validation["sample_id"]))
                mini = {name: value[start:stop] for name, value in validation.items()}
                raw, validation_dimensions = build_candidate_features(mini)
                if dict(validation_dimensions) != dict(dimensions):
                    raise RuntimeError("Frozen feature dimensions differ across splits")
                chunks.append(
                    probe(
                        apply_feature_mask(raw, variant, dimensions).to(device),
                        validation["alpha"][start:stop].to(device),
                        validation["candidate_valid_mask"][start:stop].to(device),
                    ).cpu()
                )
        residual = torch.cat(chunks)
        corrected = torch.sigmoid(validation["shared_logit"].detach() + residual)
        val_loss, _ = formal_density_loss(
            corrected,
            validation["ground_truth_density"],
            validation["measurement_supported"],
            validation["sample_id"],
            loss_config,
        )
        dice = query_dice(corrected, validation["ground_truth_density"], args.threshold)
        history.append(
            {
                "epoch": epoch,
                "validation_formal_density_loss": float(val_loss),
                "validation_query_dice": dice,
            }
        )
        if float(val_loss) < best_loss - args.early_stopping_min_delta:
            best_loss, stale, best_epoch = float(val_loss), 0, epoch
            best_state = copy.deepcopy(probe.state_dict())
        else:
            stale += 1
            if stale >= args.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("Probe training did not produce a checkpoint")
    probe.load_state_dict(best_state)
    checkpoint_path = checkpoint_dir / f"{variant}_seed{seed}_best_density_loss.pt"
    torch.save(
        {
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "best_validation_density_loss": best_loss,
        },
        checkpoint_path,
    )
    predictions = {}
    probe.eval()
    with torch.no_grad():
        for split, data in (("train", train), ("validation", validation)):
            chunks = []
            for start in range(0, len(data["sample_id"]), args.batch_size):
                stop = min(start + args.batch_size, len(data["sample_id"]))
                if split == "train":
                    mini_features = train_features[start:stop]
                else:
                    mini = {name: value[start:stop] for name, value in data.items()}
                    raw, validation_dimensions = build_candidate_features(mini)
                    if dict(validation_dimensions) != dict(dimensions):
                        raise RuntimeError("Frozen feature dimensions differ across splits")
                    mini_features = apply_feature_mask(raw, variant, dimensions)
                chunks.append(
                    probe(
                        mini_features.to(device),
                        data["alpha"][start:stop].to(device),
                        data["candidate_valid_mask"][start:stop].to(device),
                    ).cpu()
                )
            predictions[split] = torch.cat(chunks)
    return probe, predictions, history


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_figures(output: Path, summary: Mapping[str, Any]) -> None:
    import matplotlib.pyplot as plt

    variants = list(VARIANTS)
    figure, axis = plt.subplots(figsize=(5.2, 3.6))
    for index, variant in enumerate(variants):
        values = [
            seed["subsets"]["all_samples"]["query_dice"]
            for seed in summary[variant]["seeds"].values()
        ]
        axis.scatter([index] * len(values), values, s=22, zorder=3)
        axis.errorbar(
            index,
            summary[variant]["query_dice"]["mean"],
            yerr=summary[variant]["query_dice"]["std"],
            color="black",
            capsize=4,
            marker="_",
        )
    axis.set_xticks(range(len(variants)), variants)
    axis.set_ylabel("Validation query Dice")
    figure.tight_layout()
    figure.savefig(output / "figures" / "nested_probe_seed_dice.png", dpi=180)
    plt.close(figure)

    names = list(summary["comparisons"])
    means = [summary["comparisons"][name]["bootstrap_dice"]["mean_difference"] for name in names]
    lower = [summary["comparisons"][name]["bootstrap_dice"]["ci_lower"] for name in names]
    upper = [summary["comparisons"][name]["bootstrap_dice"]["ci_upper"] for name in names]
    figure, axis = plt.subplots(figsize=(6.0, 3.6))
    axis.errorbar(
        range(len(names)),
        means,
        yerr=[np.asarray(means) - np.asarray(lower), np.asarray(upper) - np.asarray(means)],
        fmt="o",
        capsize=4,
    )
    axis.axhline(0.0, color="black", linewidth=1, linestyle="--")
    axis.set_xticks(range(len(names)), [name.replace("_minus_", " − ") for name in names])
    axis.set_ylabel("Paired sample Dice difference (95% CI)")
    figure.tight_layout()
    figure.savefig(output / "figures" / "paired_bootstrap_dice.png", dpi=180)
    plt.close(figure)


def infer_checkpoint_overrides(checkpoint: str) -> list[str]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    stem = state.get("net.surface_encoder.stem.net.0.weight")
    profiles = {16: "small", 24: "efficient", 32: "base"}
    return (
        []
        if not torch.is_tensor(stem) or int(stem.shape[0]) not in profiles
        else [f"model.ssq_fmt.encoder.profile={profiles[int(stem.shape[0])]}"]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/exp/fmt_simgen_v2_shq_quotient_residual.yaml")
    parser.add_argument(
        "--output-dir", default="outputs/shq_audit/epoch7_q8192_100x100_v6_final_nested"
    )
    parser.add_argument("--train-samples", type=int, default=100)
    parser.add_argument("--val-samples", type=int, default=100)
    parser.add_argument("--queries-per-sample", type=int, default=8192)
    parser.add_argument("--train-query-limit", type=int, default=50000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--delta-l-max", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1.0e-6)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260629)
    parser.add_argument("--data-seed", type=int, default=20260629)
    parser.add_argument("--quotient-dice-gain", type=float, default=0.005)
    parser.add_argument("--metadata-dice-gain", type=float, default=0.005)
    parser.add_argument("--negligible-gain", type=float, default=0.003)
    parser.add_argument("--required-positive-seeds", type=int, default=4)
    parser.add_argument("--reuse-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    cache_dir, checkpoint_dir = output / "cache", output / "checkpoints"
    for path in (output / "figures", cache_dir, checkpoint_dir):
        path.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    exp = Path(args.config).stem
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                f"exp={exp}",
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
    if str(cfg.model.ssq_fmt.composition.mode) != "quotient_residual":
        raise ValueError("Audit requires quotient_residual composition")
    loss_config = {
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
    cache_paths = {
        split: cache_dir / f"{split}_frozen_features.pt" for split in ("train", "validation")
    }
    expected_queries = {
        "train": args.train_samples * args.queries_per_sample,
        "validation": args.val_samples * args.queries_per_sample,
    }
    split_data = {}
    if args.reuse_cache:
        for split, path in cache_paths.items():
            if path.exists():
                cached = load_cache(path)
                if len(cached["sample_id"]) == expected_queries[split]:
                    split_data[split] = cached
                else:
                    del cached
    missing_splits = set(cache_paths) - set(split_data)
    if missing_splits:
        module = TrainingLightningModule.load_from_checkpoint(
            args.checkpoint, cfg=cfg, map_location="cpu"
        )
        model = module.net.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        dm = TrainingDataModule(cfg)
        dm.setup("fit")
        hooks = AuditHooks(model)
        sample_ids_json = {
            split: json.loads(
                (
                    output
                    / ("train_sample_ids.json" if split == "train" else "val_sample_ids.json")
                ).read_text()
            )
            for split in split_data
        }
        try:
            for split, dataset, count in (
                ("train", dm.train_dataset, args.train_samples),
                ("validation", dm.val_dataset, args.val_samples),
            ):
                if split not in missing_splits:
                    continue
                if hasattr(dataset, "set_epoch"):
                    dataset.set_epoch(0)
                parts, names = [], []
                for index in range(min(count, len(dataset))):
                    fixed_seed = args.data_seed + index + (0 if split == "train" else 100000)
                    torch.manual_seed(fixed_seed)
                    np.random.seed(fixed_seed)
                    random.seed(fixed_seed)
                    batch = default_collate([dataset[index]])
                    names.append(str(batch["sample_id"][0]))
                    parts.append(
                        extract_frozen_sample(
                            model,
                            hooks,
                            batch,
                            device,
                            device.type == "cuda" and args.precision == "bf16",
                            index,
                        )
                    )
                split_data[split] = concatenate(parts)
                sample_ids_json[split] = names
                save_cache(cache_paths[split], split_data[split])
        finally:
            hooks.close()
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("Frozen SHQ model received gradients")
        (output / "train_sample_ids.json").write_text(
            json.dumps(sample_ids_json["train"], indent=2)
        )
        (output / "val_sample_ids.json").write_text(
            json.dumps(sample_ids_json["validation"], indent=2)
        )
    train_full, validation = split_data["train"], split_data["validation"]
    generator = torch.Generator().manual_seed(args.data_seed)
    train_index = torch.randperm(len(train_full["sample_id"]), generator=generator)[
        : min(args.train_query_limit, len(train_full["sample_id"]))
    ]
    validation_index = torch.arange(len(validation["sample_id"]))
    train = select_fixed_queries(train_full, train_index)
    np.savez(
        output / "train_query_indices.npz",
        indices=train_index.numpy(),
        sample_id=train["sample_id"].numpy(),
        query_id=train["query_id"].numpy(),
    )
    np.savez(
        output / "val_query_indices.npz",
        indices=validation_index.numpy(),
        sample_id=validation["sample_id"].numpy(),
        query_id=validation["query_id"].numpy(),
    )
    np.savez(
        output / "fixed_query_indices.npz",
        train_indices=train_index.numpy(),
        train_sample_id=train["sample_id"].numpy(),
        train_query_id=train["query_id"].numpy(),
        val_indices=validation_index.numpy(),
        val_sample_id=validation["sample_id"].numpy(),
        val_query_id=validation["query_id"].numpy(),
    )
    fixed_ids = {
        "train": json.loads((output / "train_sample_ids.json").read_text()),
        "validation": json.loads((output / "val_sample_ids.json").read_text()),
    }
    (output / "fixed_sample_ids.json").write_text(json.dumps(fixed_ids, indent=2))
    _, dimensions = build_candidate_features(train)
    mask_definition = {
        "feature_order": list(FEATURE_GROUPS),
        "feature_dimensions": dimensions,
        "variants": FEATURE_MASKS,
    }
    (output / "feature_mask_definition.json").write_text(json.dumps(mask_definition, indent=2))
    architecture = {
        "class": "NestedCandidateProbe",
        "hidden_dim": args.hidden_dim,
        "delta_l_max": args.delta_l_max,
        "candidate_pooling": "detached_alpha_weighted_sum",
        "candidate_identity_embedding": False,
    }
    training_config = {
        key: getattr(args, key)
        for key in (
            "learning_rate",
            "weight_decay",
            "max_epochs",
            "early_stopping_patience",
            "batch_size",
            "hidden_dim",
            "delta_l_max",
            "seeds",
        )
    }
    metric_config = {
        "threshold": args.threshold,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.bootstrap_seed,
    }
    protocol_payload = {
        "sample_ids": fixed_ids,
        "query_ids": {
            "train": train["query_id"].tolist(),
            "validation": validation["query_id"].tolist(),
        },
        "feature_cache_version": CACHE_VERSION,
        "probe_architecture": architecture,
        "loss_config": loss_config,
        "training_config": training_config,
        "metric_config": metric_config,
    }
    digest = protocol_hash(protocol_payload)
    seed_rows, sample_rows = [], []
    variant_results: dict[str, Any] = {variant: {"seeds": {}} for variant in VARIANTS}
    sample_by_variant: dict[str, dict[int, list[dict[str, Any]]]] = {
        variant: {} for variant in VARIANTS
    }
    parameter_counts = set()
    for variant in VARIANTS:
        for seed in args.seeds:
            probe, predictions, history = train_probe(
                variant,
                seed,
                train,
                validation,
                dimensions,
                device,
                args,
                loss_config,
                checkpoint_dir,
            )
            parameter_counts.add(parameter_count(probe))
            if any(value.requires_grad for data in (train, validation) for value in data.values()):
                raise RuntimeError("Frozen cache unexpectedly requires gradients")
            residual = predictions["validation"]
            corrected = torch.sigmoid(validation["shared_logit"].detach() + residual)
            descent = task_descent_direction(
                validation["shared_logit"],
                validation["ground_truth_density"],
                validation["measurement_supported"],
                validation["sample_id"],
                loss_config,
            )
            subsets = {
                name: metric_values(
                    corrected, residual, validation, mask, args.threshold, loss_config, descent
                )
                for name, mask in subset_masks(validation).items()
            }
            best_dice_epoch = max(history, key=lambda row: row["validation_query_dice"])["epoch"]
            variant_results[variant]["seeds"][str(seed)] = {
                "subsets": subsets,
                "selected_epoch": int(
                    min(history, key=lambda row: row["validation_formal_density_loss"])["epoch"]
                ),
                "best_dice_epoch": int(best_dice_epoch),
            }
            for subset, metrics in subsets.items():
                seed_rows.append({"variant": variant, "seed": seed, "subset": subset, **metrics})
            rows = sample_metrics(corrected, residual, validation, args.threshold, loss_config)
            sample_by_variant[variant][seed] = rows
            sample_rows.extend({"variant": variant, "seed": seed, **row} for row in rows)
    if len(parameter_counts) != 1:
        raise RuntimeError(f"S0/S1/S2 parameter counts differ: {sorted(parameter_counts)}")
    for variant in VARIANTS:
        for metric in ("query_dice", "formal_density_loss", "descent_alignment"):
            values = [
                variant_results[variant]["seeds"][str(seed)]["subsets"]["all_samples"][metric]
                for seed in args.seeds
            ]
            variant_results[variant][metric] = seed_statistics(values)
        variant_results[variant]["parameter_count"] = next(iter(parameter_counts))
    comparisons = {}
    bootstrap_rows = []
    for left, right in (("S0", "S1"), ("S1", "S2"), ("S0", "S2")):
        name = f"{right}_minus_{left}"
        comparison = {}
        for metric in ("query_dice", "formal_density_loss", "descent_alignment"):
            lv = [
                variant_results[left]["seeds"][str(seed)]["subsets"]["all_samples"][metric]
                for seed in args.seeds
            ]
            rv = [
                variant_results[right]["seeds"][str(seed)]["subsets"]["all_samples"][metric]
                for seed in args.seeds
            ]
            comparison["dice" if metric == "query_dice" else metric] = seed_statistics(
                np.asarray(rv) - np.asarray(lv), np.zeros(len(lv))
            )
        for metric in ("query_dice", "formal_density_loss"):
            per_seed = []
            for seed in args.seeds:
                left_values = np.asarray([row[metric] for row in sample_by_variant[left][seed]])
                right_values = np.asarray([row[metric] for row in sample_by_variant[right][seed]])
                per_seed.append(right_values - left_values)
            averaged = np.stack(per_seed).mean(axis=0)
            boot = paired_bootstrap(
                np.zeros_like(averaged), averaged, args.bootstrap_repetitions, args.bootstrap_seed
            )
            comparison[f"bootstrap_{'dice' if metric == 'query_dice' else metric}"] = boot
            bootstrap_rows.append({"comparison": name, "metric": metric, **boot})
        comparisons[name] = comparison
    conclusion, next_step = automatic_conclusion(comparisons, variant_results, args)
    gate_review = None
    stable_variants = []
    for variant, reference in (("S1", "S0"), ("S2", "S1")):
        comparison = comparisons[f"{variant}_minus_{reference}"]
        if (
            comparison["dice"]["mean"] >= args.negligible_gain
            and comparison["dice"]["positive_seed_count"] >= args.required_positive_seeds
            and comparison["bootstrap_dice"]["ci_lower"] > 0
            and comparison["formal_density_loss"]["mean"] < 0
            and variant_results[variant]["descent_alignment"]["mean"] > 0
        ):
            stable_variants.append(variant)
    if stable_variants:
        selected = max(
            stable_variants, key=lambda item: variant_results[item]["query_dice"]["mean"]
        )
        best_seed = min(
            args.seeds,
            key=lambda seed: variant_results[selected]["seeds"][str(seed)]["subsets"][
                "all_samples"
            ]["formal_density_loss"],
        )
        residual = torch.load(
            checkpoint_dir / f"{selected}_seed{best_seed}_best_density_loss.pt",
            map_location="cpu",
            weights_only=False,
        )
        probe = NestedCandidateProbe(sum(dimensions.values()), args.hidden_dim, args.delta_l_max)
        probe.load_state_dict(residual["state_dict"])
        chunks = []
        with torch.no_grad():
            for start in range(0, len(validation["sample_id"]), args.batch_size):
                stop = min(start + args.batch_size, len(validation["sample_id"]))
                mini = {name: value[start:stop] for name, value in validation.items()}
                raw, validation_dimensions = build_candidate_features(mini)
                if dict(validation_dimensions) != dict(dimensions):
                    raise RuntimeError("Frozen feature dimensions differ across splits")
                chunks.append(
                    probe(
                        apply_feature_mask(raw, selected, dimensions),
                        validation["alpha"][start:stop],
                        validation["candidate_valid_mask"][start:stop],
                    )
                )
        r_pred = torch.cat(chunks)
        gate_review = {"variant": selected, "seed": best_seed, "gamma": {}}
        for gamma in (1, 2, 4, 8):
            density = torch.sigmoid(
                validation["shared_logit"] + gamma * validation["proposal_gate"] * r_pred
            )
            loss, _ = formal_density_loss(
                density,
                validation["ground_truth_density"],
                validation["measurement_supported"],
                validation["sample_id"],
                loss_config,
            )
            gate_review["gamma"][str(gamma)] = {
                "query_dice": query_dice(
                    density, validation["ground_truth_density"], args.threshold
                ),
                "formal_density_loss": float(loss),
            }
        ungated = torch.sigmoid(validation["shared_logit"] + r_pred)
        gate_review["ungated"] = {
            "query_dice": query_dice(ungated, validation["ground_truth_density"], args.threshold)
        }
    summary = {
        "protocol_hash": digest,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
        "fixed_data": {
            "train_samples": len(fixed_ids["train"]),
            "validation_samples": len(fixed_ids["validation"]),
            "train_queries": len(train["sample_id"]),
            "validation_queries": len(validation["sample_id"]),
            "queries_per_sample": args.queries_per_sample,
            "cache_version": CACHE_VERSION,
        },
        **variant_results,
        "comparisons": comparisons,
        "gate_review": gate_review,
        "automatic_conclusion": conclusion,
        "recommended_next_step": next_step,
    }
    write_csv(output / "nested_probe_seed_metrics.csv", seed_rows)
    write_csv(output / "nested_probe_sample_metrics.csv", sample_rows)
    write_csv(output / "paired_bootstrap_results.csv", bootstrap_rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    save_figures(output, summary)
    print(
        json.dumps(
            {
                "protocol_hash": digest,
                "comparisons": comparisons,
                "automatic_conclusion": conclusion,
                "recommended_next_step": next_step,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
