#!/usr/bin/env python
"""Quick frozen-feature audit of candidate view separability in SHQ-FMT."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.datamodule import TrainingDataModule  # noqa: E402
from minr_fmt.module import TrainingLightningModule  # noqa: E402

EPS = 1.0e-8
CACHE_VERSION = "shq_quick_view_separability_v1"
VARIANTS = (
    "high",
    "low",
    "random",
    "uniform",
    "geometry_sep_aware",
    "measurement_sep_aware",
    "combined_sep_aware",
    "shuffled_sep_aware",
)


def pairwise_separability(
    evidence: torch.Tensor,
    detector_centers: torch.Tensor,
    detector_scales: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return candidate-wise min pair scores for tensors shaped ``[..., V, M, C]``."""
    if evidence.dim() < 3 or detector_centers.shape[-1] != 2:
        raise ValueError("Invalid evidence or detector-center shape")
    m = evidence.shape[-2]
    center_delta = detector_centers[..., :, None, :] - detector_centers[..., None, :, :]
    scale2 = detector_scales[..., :, None].square() + detector_scales[..., None, :].square()
    d2 = center_delta.square().sum(dim=-1) / scale2.clamp_min(EPS)
    geo_pair = 1.0 - torch.exp(-0.5 * d2)
    normalized = F.normalize(evidence.float(), dim=-1, eps=EPS)
    meas_pair = 1.0 - torch.einsum("...vmc,...vnc->...vmn", normalized, normalized)
    meas_pair = meas_pair.clamp(0.0, 1.0)
    geo_pair = geo_pair.clamp(0.0, 1.0)
    pair_valid = valid[..., :, None] & valid[..., None, :]
    eye = torch.eye(m, dtype=torch.bool, device=evidence.device)
    pair_valid = pair_valid & ~eye

    def reduce_pair(score: torch.Tensor) -> torch.Tensor:
        reduced = score.masked_fill(~pair_valid, float("inf")).amin(dim=-1)
        return torch.where(torch.isfinite(reduced), reduced, torch.zeros_like(reduced))

    geo = reduce_pair(geo_pair)
    meas = reduce_pair(meas_pair)
    usable = pair_valid.any(dim=-1)
    combined = (0.5 * geo + 0.5 * meas).clamp(0.0, 1.0)
    return geo, meas, combined, usable


def select_view_weights(
    score: torch.Tensor,
    valid: torch.Tensor,
    strategy: str,
    *,
    tau: float = 0.25,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Build normalized view weights over the penultimate (view) dimension."""
    valid = valid.bool()
    if strategy in {"high", "low", "random"}:
        if strategy == "random":
            keys = torch.rand(score.shape, generator=generator, device=score.device)
        else:
            keys = score if strategy == "high" else -score
        keys = keys.masked_fill(~valid, -torch.inf)
        k = min(2, score.shape[-2])
        chosen = torch.zeros_like(valid)
        chosen.scatter_(-2, keys.topk(k, dim=-2).indices, True)
        chosen &= valid
        weights = chosen.to(score.dtype)
    elif strategy == "uniform":
        weights = valid.to(score.dtype)
    elif strategy == "sep_aware":
        logits = (score / max(float(tau), EPS)).masked_fill(~valid, -torch.inf)
        weights = torch.softmax(logits, dim=-2)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
    else:
        raise ValueError(f"Unknown view strategy: {strategy}")
    return weights / weights.sum(dim=-2, keepdim=True).clamp_min(EPS)


def shuffle_view_scores(
    score: torch.Tensor, valid: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Shuffle valid view scores independently within every sample/query/candidate."""
    keys = torch.rand(score.shape, generator=generator, device=score.device)
    permutation = keys.masked_fill(~valid, torch.inf).argsort(dim=-2)
    compact = score.gather(-2, permutation)
    valid_rank = valid.long().cumsum(dim=-2).sub(1).clamp_min(0)
    shuffled = compact.gather(-2, valid_rank)
    return torch.where(valid, shuffled, score)


def aggregate_candidate_evidence(
    evidence: torch.Tensor,
    score: torch.Tensor,
    valid: torch.Tensor,
    candidate_valid: torch.Tensor,
    strategy: str,
    *,
    tau: float = 0.25,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    weights = select_view_weights(score, valid, strategy, tau=tau, generator=generator)
    per_candidate = (weights[..., None] * evidence).sum(dim=-3)
    usable = valid.any(dim=-2) & candidate_valid
    pooled = (per_candidate * usable[..., None]).sum(dim=-2)
    return pooled / usable.sum(dim=-1, keepdim=True).clamp_min(1).to(pooled.dtype)


class DensityProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(inputs))


def formal_density_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
    sample_id: torch.Tensor,
    cfg: Mapping[str, float],
) -> torch.Tensor:
    pred = prediction.clamp(0, 1)
    target = target.to(pred).clamp(0, 1)
    support = support.to(pred)
    weight = (1 + target * (cfg["pos_weight"] - 1)) * support
    density = (F.smooth_l1_loss(pred, target, reduction="none") * weight).sum()
    density = density / weight.sum().clamp_min(EPS)
    bounded = pred.float().clamp(1e-6, 1 - 1e-6)
    bce_raw = -(target.float() * bounded.log() + (1 - target.float()) * (1 - bounded).log())
    bce = (bce_raw.to(pred) * weight).sum() / weight.sum().clamp_min(EPS)
    _, inverse = sample_id.unique(sorted=True, return_inverse=True)
    groups = int(inverse.max().item()) + 1

    def grouped_sum(value: torch.Tensor) -> torch.Tensor:
        return torch.zeros(groups, device=pred.device, dtype=pred.dtype).scatter_add_(
            0, inverse, value.flatten()
        )

    intersection = grouped_sum(pred * target * support)
    pred_mass = grouped_sum(pred * support)
    target_mass = grouped_sum(target * support)
    fp = grouped_sum(pred * (1 - target) * support)
    fn = grouped_sum((1 - pred) * target * support)
    dice = (1 - (2 * intersection + 1e-6) / (pred_mass + target_mass + 1e-6)).mean()
    ti = (intersection + 1e-6) / (
        intersection + cfg["tversky_alpha"] * fp + cfg["tversky_beta"] * fn + 1e-6
    )
    tversky = (1 - ti).clamp(0, 1).pow(cfg["tversky_gamma"]).mean()
    sparse = (pred * (1 - target) * support).sum() / support.sum().clamp_min(EPS)
    return (
        density
        + cfg["density_bce_weight"] * bce
        + cfg["dice_weight"] * dice
        + cfg["tversky_weight"] * tversky
        + cfg["sparse_weight"] * sparse
    )


def query_dice(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred, truth = prediction >= threshold, target > 0
    return float((2 * (pred & truth).sum() + 1e-6) / (pred.sum() + truth.sum() + 1e-6))


class FeatureHooks:
    def __init__(self, model: nn.Module):
        self.values: dict[str, Any] = {}
        self.handles = [
            model.candidate_builder.register_forward_hook(self._set("candidate")),
            model.candidate_router.register_forward_hook(self._set("router")),
            model.view_encoder.register_forward_hook(self._set("per_view")),
            model.geometry_mapper.register_forward_hook(self._append("mapped")),
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


def _flatten(value: torch.Tensor) -> torch.Tensor:
    return value.detach().float().cpu().reshape(-1, *value.shape[2:])


def extract_sample(
    model: nn.Module,
    hooks: FeatureHooks,
    batch: Mapping[str, Any],
    device: torch.device,
    amp: bool,
    audit_id: int,
) -> dict[str, torch.Tensor]:
    hooks.clear()
    batch = {
        key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()
    }
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        output = model(
            batch["surface_measurements_packed"],
            batch["query_coordinates_mm"],
            detector_valid_mask=batch.get("detector_valid_mask"),
            depth_maps=batch.get("depth_maps"),
            batch=batch,
            return_diagnostics=True,
        )
    candidate = hooks.values["candidate"]
    router = hooks.values["router"]
    query_mapped = hooks.values["mapped"][0]
    per_view = hooks.values["per_view"][:, :, :, 1:]
    shared_q = hooks.values["quotient"][-2]["quotient"].squeeze(2)
    n, m = per_view.shape[1], per_view.shape[3]
    candidate_valid = candidate["candidate_valid_mask"]
    detector_valid = candidate["candidate_detector_valid_mask"]
    query_valid = query_mapped["valid_mask"].permute(0, 2, 1)[..., None]
    routed = router["r"].permute(0, 2, 1, 3)[..., 1:]
    valid = query_valid & detector_valid[:, None] & candidate_valid[:, None, None] & (routed > 0)
    centers = candidate["candidate_uv_px"][:, None].expand(-1, n, -1, -1, -1)
    scales = candidate["candidate_detector_support_scales_px"][:, None]
    scales = scales.expand(-1, n, -1, -1)
    trunk = torch.as_tensor(
        model.trunk_size_mm, device=device, dtype=batch["query_coordinates_mm"].dtype
    )
    query_norm = (batch["query_coordinates_mm"] / trunk).mul(2).sub(1)
    target = batch["point_densities"].float().unsqueeze(-1)
    result = {
        "u_m_v": _flatten(per_view),
        "detector_centers": _flatten(centers),
        "detector_scales": _flatten(scales),
        "view_valid": _flatten(valid).bool(),
        "candidate_valid": candidate_valid.detach().cpu().repeat_interleave(n, 0),
        "router_allocation": _flatten(router["a"].permute(0, 2, 1, 3)[..., 1:]),
        "conditional_quality": _flatten(router["nu"].permute(0, 2, 1, 3)[..., 1:]),
        "routed_support": _flatten(routed),
        "support_covariance": candidate["candidate_support_covariances_mm"]
        .detach()
        .float()
        .cpu()
        .repeat_interleave(n, 0),
        "q_s": _flatten(shared_q),
        "shared_logit": _flatten(output["aux_outputs"]["shared_logit"]),
        "ground_truth_density": _flatten(target),
        "query_coordinate_encoding": _flatten(model.position_encoding(query_norm)),
        "measurement_supported": _flatten(
            output["aux_outputs"]["measurement_supported"][..., None]
        ).bool(),
        "sample_id": torch.full((n,), audit_id, dtype=torch.long),
        "source_count": torch.full(
            (n,), int(torch.as_tensor(batch["num_foci"]).item()), dtype=torch.long
        ),
    }
    if result["u_m_v"].shape[-2] != m:
        raise RuntimeError("Candidate dimension changed during feature extraction")
    return result


def concatenate(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}


def fixed_multisource_indices(dataset: Any, count: int) -> list[int]:
    indices = []
    for index in range(len(dataset)):
        item = dataset[index]
        if int(torch.as_tensor(item["num_foci"]).item()) in (2, 3):
            indices.append(index)
        if len(indices) == count:
            return indices
    raise RuntimeError(f"Dataset contains fewer than {count} two-/three-source samples")


def variant_features(
    data: Mapping[str, torch.Tensor], variant: str, seed: int, tau: float
) -> torch.Tensor:
    evidence = data["u_m_v"]
    geo, meas, combined, usable = pairwise_separability(
        evidence, data["detector_centers"], data["detector_scales"], data["view_valid"]
    )
    valid = data["view_valid"] & usable
    generator = torch.Generator().manual_seed(10000 + seed)
    if variant in {"high", "low", "random", "uniform"}:
        score, strategy = combined, variant
    elif variant == "geometry_sep_aware":
        score, strategy = geo, "sep_aware"
    elif variant == "measurement_sep_aware":
        score, strategy = meas, "sep_aware"
    elif variant == "combined_sep_aware":
        score, strategy = combined, "sep_aware"
    elif variant == "shuffled_sep_aware":
        score = shuffle_view_scores(combined, valid, generator)
        strategy = "sep_aware"
    else:
        raise ValueError(variant)
    aggregated = aggregate_candidate_evidence(
        evidence, score, valid, data["candidate_valid"], strategy, tau=tau, generator=generator
    )
    return torch.cat([data["q_s"], aggregated, data["query_coordinate_encoding"]], dim=-1)


def train_probe(
    variant: str,
    seed: int,
    train: Mapping[str, torch.Tensor],
    validation: Mapping[str, torch.Tensor],
    device: torch.device,
    args: argparse.Namespace,
    loss_cfg: Mapping[str, float],
) -> tuple[DensityProbe, torch.Tensor]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_x = variant_features(train, variant, seed, args.tau)
    val_x = variant_features(validation, variant, seed, args.tau)
    probe = DensityProbe(train_x.shape[-1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    best_loss, stale, best_state = float("inf"), 0, None
    for _epoch in range(args.max_epochs):
        probe.train()
        order = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            prediction = probe(train_x[idx].to(device))
            loss = formal_density_loss(
                prediction,
                train["ground_truth_density"][idx].to(device),
                train["measurement_supported"][idx].to(device),
                train["sample_id"][idx].to(device),
                loss_cfg,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        probe.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(val_x), args.batch_size):
                chunks.append(probe(val_x[start : start + args.batch_size].to(device)).cpu())
        prediction = torch.cat(chunks)
        val_loss = formal_density_loss(
            prediction,
            validation["ground_truth_density"],
            validation["measurement_supported"],
            validation["sample_id"],
            loss_cfg,
        )
        if float(val_loss) < best_loss - args.early_stopping_min_delta:
            best_loss, stale, best_state = float(val_loss), 0, copy.deepcopy(probe.state_dict())
        else:
            stale += 1
            if stale >= args.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("Probe training produced no checkpoint")
    probe.load_state_dict(best_state)
    probe.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(val_x), args.batch_size):
            chunks.append(probe(val_x[start : start + args.batch_size].to(device)).cpu())
    return probe, torch.cat(chunks)


def metrics(
    prediction: torch.Tensor, data: Mapping[str, torch.Tensor], loss_cfg: Mapping[str, float]
) -> dict[str, float]:
    result = {
        "query_dice": query_dice(prediction, data["ground_truth_density"]),
        "formal_density_loss": float(
            formal_density_loss(
                prediction,
                data["ground_truth_density"],
                data["measurement_supported"],
                data["sample_id"],
                loss_cfg,
            )
        ),
    }
    for count in (2, 3):
        mask = data["source_count"] == count
        result[f"{count}_source_dice"] = query_dice(
            prediction[mask], data["ground_truth_density"][mask]
        )
    return result


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="outputs/shq_fmt/q8192_bs2_cosine10_from_e3/checkpoints/epoch=07-val_dice=0.5995.ckpt",
    )
    parser.add_argument("--config", default="configs/exp/fmt_simgen_v2_shq_quotient_residual.yaml")
    parser.add_argument("--output-dir", default="outputs/shq_separability_quick_v1")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--queries-per-sample", type=int, default=8192)
    parser.add_argument("--train-query-limit", type=int, default=50000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--tau", type=float, default=0.25)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--data-seed", type=int, default=20260629)
    parser.add_argument("--reuse-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
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
    loss_cfg = {
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
        split: output / f"{split}_frozen_features.pt" for split in ("train", "validation")
    }
    split_data: dict[str, dict[str, torch.Tensor]] = {}
    if args.reuse_cache:
        for split, path in cache_paths.items():
            if path.exists():
                payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
                if payload.get("cache_version") == CACHE_VERSION:
                    split_data[split] = payload["tensors"]
    if len(split_data) != 2:
        module = TrainingLightningModule.load_from_checkpoint(
            args.checkpoint, cfg=cfg, map_location="cpu"
        )
        model = module.net.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        dm = TrainingDataModule(cfg)
        dm.setup("fit")
        hooks = FeatureHooks(model)
        fixed_ids: dict[str, list[str]] = {}
        try:
            for split, dataset in (("train", dm.train_dataset), ("validation", dm.val_dataset)):
                if split in split_data:
                    continue
                if hasattr(dataset, "set_epoch"):
                    dataset.set_epoch(0)
                indices = fixed_multisource_indices(dataset, args.samples)
                parts, names = [], []
                for audit_id, index in enumerate(indices):
                    seed = args.data_seed + index + (100000 if split == "validation" else 0)
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    batch = default_collate([dataset[index]])
                    names.append(str(batch["sample_id"][0]))
                    parts.append(
                        extract_sample(
                            model,
                            hooks,
                            batch,
                            device,
                            device.type == "cuda" and args.precision == "bf16",
                            audit_id,
                        )
                    )
                split_data[split] = concatenate(parts)
                torch.save(
                    {"cache_version": CACHE_VERSION, "tensors": split_data[split]},
                    cache_paths[split],
                )
                fixed_ids[split] = names
        finally:
            hooks.close()
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("Frozen main model received gradients")
        (output / "fixed_sample_ids.json").write_text(json.dumps(fixed_ids, indent=2))

    train_generator = torch.Generator().manual_seed(args.data_seed)
    train_indices = torch.randperm(
        len(split_data["train"]["sample_id"]), generator=train_generator
    )[: args.train_query_limit]
    train_data = {
        key: value[train_indices] for key, value in split_data["train"].items()
    }
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for seed in args.seeds:
            probe, prediction = train_probe(
                variant, seed, train_data, split_data["validation"], device, args, loss_cfg
            )
            values = metrics(prediction, split_data["validation"], loss_cfg)
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "parameter_count": sum(p.numel() for p in probe.parameters()),
                    **values,
                }
            )
    write_csv(output / "quick_separability_metrics.csv", rows)
    summary: dict[str, Any] = {"settings": vars(args), "variants": {}, "comparisons": {}}
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        summary["variants"][variant] = {}
        for metric in ("query_dice", "formal_density_loss", "2_source_dice", "3_source_dice"):
            values = np.asarray([row[metric] for row in selected])
            summary["variants"][variant][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
            }
    comparisons = {
        "high_minus_low": ("high", "low"),
        "high_minus_random": ("high", "random"),
        "combined_sep_aware_minus_uniform": ("combined_sep_aware", "uniform"),
        "combined_sep_aware_minus_shuffled": ("combined_sep_aware", "shuffled_sep_aware"),
        "geometry_sep_aware_minus_uniform": ("geometry_sep_aware", "uniform"),
        "measurement_sep_aware_minus_uniform": ("measurement_sep_aware", "uniform"),
    }
    for name, (left, right) in comparisons.items():
        left_rows = [row for row in rows if row["variant"] == left]
        right_rows = [row for row in rows if row["variant"] == right]
        dice_delta = np.asarray(
            [a["query_dice"] - b["query_dice"] for a, b in zip(left_rows, right_rows, strict=True)]
        )
        loss_delta = np.asarray(
            [
                a["formal_density_loss"] - b["formal_density_loss"]
                for a, b in zip(left_rows, right_rows, strict=True)
            ]
        )
        summary["comparisons"][name] = {
            "query_dice_mean": float(dice_delta.mean()),
            "positive_seed_count": int((dice_delta > 0).sum()),
            "formal_density_loss_mean": float(loss_delta.mean()),
            "loss_positive_direction_count": int((loss_delta < 0).sum()),
        }
    c = summary["comparisons"]
    worth = (
        c["high_minus_low"]["query_dice_mean"] > 0
        and c["high_minus_low"]["positive_seed_count"] == 3
        and c["high_minus_random"]["positive_seed_count"] >= 2
        and c["combined_sep_aware_minus_uniform"]["query_dice_mean"] > 0
        and c["combined_sep_aware_minus_uniform"]["positive_seed_count"] >= 2
        and c["combined_sep_aware_minus_shuffled"]["query_dice_mean"] > 0
        and c["combined_sep_aware_minus_shuffled"]["positive_seed_count"] >= 2
        and c["high_minus_low"]["formal_density_loss_mean"] < 0
        and c["combined_sep_aware_minus_uniform"]["formal_density_loss_mean"] < 0
        and (
            c["high_minus_low"]["query_dice_mean"] >= 0.005
            or c["combined_sep_aware_minus_uniform"]["query_dice_mean"] >= 0.003
        )
    )
    summary["conclusion"] = (
        "View-specific separability is detectable in the current frozen features and is worth "
        "deeper investigation."
        if worth
        else "The current frozen representation does not show a sufficiently strong view-specific "
        "separability signal. Do not introduce a separability module yet."
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
