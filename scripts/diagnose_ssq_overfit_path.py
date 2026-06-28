#!/usr/bin/env python
"""Run staged SSQ-FMT single-sample overfit diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import hydra
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.dataset.fmt_simgen_dataset import FmtSimGenProjDataset  # noqa: E402
from minr_fmt.model_factory import ModelFactory  # noqa: E402


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _dice(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_bin = (pred >= 0.5).float()
    target_bin = (target > 0.0).float()
    inter = (pred_bin * target_bin).sum()
    return (2.0 * inter + 1e-8) / (pred_bin.sum() + target_bin.sum() + 1e-8)


def _training_like_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float,
    dice_weight: float,
    sparse_weight: float,
) -> torch.Tensor:
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    weight = 1.0 + target * (float(pos_weight) - 1.0)
    density_loss = (torch.nn.functional.smooth_l1_loss(pred, target, reduction="none") * weight)
    density_loss = density_loss.sum() / weight.sum().clamp_min(1e-8)
    pred_flat = pred.squeeze(-1)
    target_flat = target.squeeze(-1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    dice = (2.0 * intersection + 1e-6) / (
        pred_flat.sum(dim=1) + target_flat.sum(dim=1) + 1e-6
    )
    dice_loss = 1.0 - dice.mean()
    sparse_loss = (pred * (1.0 - target)).mean()
    return density_loss + float(dice_weight) * dice_loss + float(sparse_weight) * sparse_loss


def _metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    diag: dict[str, torch.Tensor],
) -> dict[str, float]:
    pred_bin = (pred >= 0.5).float()
    target_bin = (target > 0.0).float()
    tp = (pred_bin * target_bin).sum()
    precision = (tp + 1e-8) / (pred_bin.sum() + 1e-8)
    recall = (tp + 1e-8) / (target_bin.sum() + 1e-8)
    candidate_valid = diag["candidate_valid_mask"].float()
    p_all = diag["p_all"]
    gt_mask = target.squeeze(-1) > 0.0
    coverage = (
        (p_all[..., 1:].amax(dim=-1)[gt_mask] > 0.05).float().mean()
        if p_all.shape[-1] > 1 and gt_mask.any()
        else torch.zeros((), device=target.device)
    )
    branch_density = diag["branch_density"]
    bce = torch.nn.functional.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), target)
    return {
        "loss": float(bce.detach().cpu()),
        "sampled_dice": float(_dice(pred, target).detach().cpu()),
        "precision": float(precision.detach().cpu()),
        "recall": float(recall.detach().cpu()),
        "predicted_foreground_ratio": float(pred_bin.mean().detach().cpu()),
        "candidate_coverage": float(coverage.detach().cpu()),
        "pi0": float(diag["pi"][..., 0].mean().detach().cpu()),
        "candidate_mass": float(diag["pi"][..., 1:].sum(dim=-1).mean().detach().cpu())
        if diag["pi"].shape[-1] > 1
        else 0.0,
        "branch_density_mean": float(branch_density.mean().detach().cpu()),
        "branch_density_std": float(branch_density.std(unbiased=False).detach().cpu()),
        "measurement_supported_positive_ratio": float(
            (diag["measurement_supported"] & (target.squeeze(-1) > 0.0))
            .float()
            .mean()
            .detach()
            .cpu()
        ),
        "valid_candidate_count": float(candidate_valid.sum(dim=1).mean().detach().cpu()),
    }


def _mode_density(mode: str, out: dict[str, Any], target: torch.Tensor) -> torch.Tensor:
    diag = out["diagnostics"]
    branch_density = diag["branch_density"]
    if mode == "D0":
        return branch_density[:, :, 0].clamp(0.0, 1.0)
    if mode == "D1":
        p_all = diag["p_all"]
        weights = p_all / p_all.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return (weights[..., None] * branch_density).sum(dim=2).clamp(0.0, 1.0)
    if mode == "D2":
        return out["density"].clamp(0.0, 1.0)
    if mode == "D3":
        oracle = torch.where(target > 0.0, branch_density.amax(dim=2), out["density"])
        return oracle.clamp(0.0, 1.0)
    raise ValueError(f"unknown mode {mode}")


def _run_mode(cfg, batch: dict[str, Any], device: torch.device, mode: str, steps: int, lr: float):
    model = ModelFactory.create_model(cfg.model.name, config=cfg).to(device)
    model.train(mode != "D3")
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(cfg.optim.get("weight_decay", 0.0)),
    )
    rows = []
    target = batch["point_densities"].unsqueeze(-1)
    for step in range(steps if mode != "D3" else 1):
        opt.zero_grad(set_to_none=True)
        out = model(
            batch.get("surface_measurements_packed", batch["projections_packed"]),
            batch.get("query_coordinates_mm", batch["points_mm"]),
            detector_valid_mask=batch.get("detector_valid_mask"),
            depth_maps=batch.get("depth_maps"),
            batch=batch,
            return_diagnostics=True,
        )
        pred = _mode_density(mode, out, target)
        total = _training_like_loss(
            pred,
            target,
            pos_weight=float(cfg.loss.get("pos_weight", 50.0)),
            dice_weight=float(cfg.loss.get("dice_weight", 0.5)),
            sparse_weight=float(cfg.loss.get("sparse_weight", 0.1)),
        )
        if mode != "D3":
            total.backward()
            opt.step()
        if step == 0 or step == steps - 1 or (step + 1) % max(1, steps // 10) == 0:
            row = {"mode": mode, "step": step, **_metrics(pred, target, out["diagnostics"])}
            rows.append(row)
            print(json.dumps(row, sort_keys=True))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="fmt_simgen_v2_ssq_gate_32gb")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num-queries", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", default="D0,D1,D2,D3")
    parser.add_argument("--out", default="outputs/ssq_fmt_overfit_path")
    args, extra = parser.parse_known_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    overrides = [
        f"exp={args.exp}",
        "data.batch_size=1",
        "data.eval_batch_size=1",
        f"data.sample_num={args.num_queries}",
        f"data.num_queries={args.num_queries}",
        f"data.eval_sample_num={args.num_queries}",
        f"seed={args.seed}",
        "trainer.precision=32-true",
    ] + extra
    if args.data_dir is not None:
        overrides.extend(
            [
                f"data.data_dir={args.data_dir}",
                f"data.train_dir={args.data_dir}",
                f"data.val_dir={args.data_dir}",
                f"data.test_dir={args.data_dir}",
            ]
        )
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = hydra.compose(config_name="config", overrides=overrides)
    ds = FmtSimGenProjDataset(str(cfg.data.train_dir), config=cfg, split="train", is_training=True)
    if hasattr(ds, "set_epoch"):
        ds.set_epoch(0)
    batch = _move_batch(next(iter(DataLoader(ds, batch_size=1, num_workers=0))), device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for mode in [item.strip() for item in args.modes.split(",") if item.strip()]:
        mode_cfg = deepcopy(cfg)
        rows = _run_mode(mode_cfg, batch, device, mode, args.steps, args.lr)
        all_rows.extend(rows)
        best_dice = max(float(row["sampled_dice"]) for row in rows)
        if mode == "D0" and best_dice < 0.9:
            print(
                "D0 did not overfit; stop before D1-D3 and inspect "
                "encoder/decoder/coordinate/loss."
            )
            break
    csv_path = out_dir / "ssq_overfit_path.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    (out_dir / "ssq_overfit_path.json").write_text(json.dumps(all_rows, indent=2))
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
