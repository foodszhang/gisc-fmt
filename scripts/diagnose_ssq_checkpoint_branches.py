#!/usr/bin/env python
"""Diagnose branch-level behavior of a trained SSQ-FMT checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import hydra
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.datamodule import TrainingDataModule  # noqa: E402
from minr_fmt.model_factory import ModelFactory  # noqa: E402


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _load_net(cfg, ckpt_path: Path, device: torch.device):
    net = ModelFactory.create_model(cfg.model.name, config=cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw_state = ckpt.get("state_dict", ckpt)
    state = {
        key.removeprefix("net."): value
        for key, value in raw_state.items()
        if key.startswith("net.") or not key.startswith(("loss_func.", "ssq_loss_func."))
    }
    missing, unexpected = net.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    net.eval()
    return net


def _binary_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = pred.clamp(0.0, 1.0)
    if pred.dim() == 2:
        pred = pred.unsqueeze(-1)
    target = target.clamp(0.0, 1.0)
    pred_bin = pred >= 0.5
    target_bin = target > 0.0
    tp = (pred_bin & target_bin).float().sum()
    fp = (pred_bin & ~target_bin).float().sum()
    fn = (~pred_bin & target_bin).float().sum()
    dice = (2.0 * tp + 1.0e-8) / (2.0 * tp + fp + fn + 1.0e-8)
    precision = (tp + 1.0e-8) / (tp + fp + 1.0e-8)
    recall = (tp + 1.0e-8) / (tp + fn + 1.0e-8)
    return {
        "dice": float(dice.detach().cpu()),
        "precision": float(precision.detach().cpu()),
        "recall": float(recall.detach().cpu()),
        "pred_fg_ratio": float(pred_bin.float().mean().detach().cpu()),
        "mean_density": float(pred.mean().detach().cpu()),
    }


def _row_mean(rows: list[dict[str, float]], key: str) -> float:
    vals = [float(row[key]) for row in rows if key in row]
    return sum(vals) / max(len(vals), 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--exp", default="fmt_simgen_v2_ssq_train_32gb")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--num-queries", type=int, default=12288)
    parser.add_argument("--out", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    overrides = [
        f"exp={args.exp}",
        "model=ssq_fmt",
        "data.dataset_type=fmt_simgen",
        f"data.sample_num={args.num_queries}",
        f"data.num_queries={args.num_queries}",
        f"data.eval_sample_num={args.num_queries}",
        f"data.query_sampling.num_queries={args.num_queries}",
        "data.batch_size=1",
        "data.eval_batch_size=1",
        "data.num_workers=0",
        "model.ssq_fmt.routing.query_chunk_size=12288",
        "model.ssq_fmt.representation.query_chunk_size=12288",
        "model.ssq_fmt.fusion.query_chunk_size=12288",
        "model.ssq_fmt.decoder.query_chunk_size=12288",
        *args.overrides,
    ]
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = hydra.compose(config_name="config", overrides=overrides)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = TrainingDataModule(cfg)
    stage = "test" if args.split == "test" else "fit"
    dm.setup(stage=stage)
    dataset = {
        "train": dm.train_dataset,
        "val": dm.val_dataset,
        "test": dm.test_dataset,
    }[args.split]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    net = _load_net(cfg, Path(args.ckpt), device)

    rows: list[dict[str, float | str | int]] = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= args.max_samples:
                break
            batch = _move_batch(batch, device)
            sample_id = batch.get("sample_id", [f"idx_{idx}"])
            if isinstance(sample_id, (list, tuple)):
                sample_id = sample_id[0]
            target = batch["point_densities"].unsqueeze(-1).float()
            out = net(
                batch["surface_measurements_packed"],
                batch["query_coordinates_mm"],
                detector_valid_mask=batch.get("detector_valid_mask"),
                depth_maps=batch.get("depth_maps"),
                batch=batch,
                return_diagnostics=True,
            )
            diag = out["diagnostics"]
            branch_density = diag["branch_density"].clamp(0.0, 1.0)
            branch_density_pre_envelope = diag["branch_density_pre_envelope"].clamp(0.0, 1.0)
            final = out["density"].clamp(0.0, 1.0)
            comp = branch_density[:, :, 0]
            p_all = diag["p_all"].clamp_min(0.0)
            fixed_prior = (p_all[..., None] * branch_density).sum(dim=2) / p_all.sum(
                dim=-1, keepdim=True
            ).clamp_min(1.0e-8)
            branch_oracle = branch_density.amax(dim=2)
            no_envelope = (
                diag["pi"][..., None] * branch_density_pre_envelope
            ).sum(dim=2) * diag["measurement_supported"][..., None].to(
                dtype=branch_density_pre_envelope.dtype
            )
            fixed_prior_no_envelope = (
                p_all[..., None] * branch_density_pre_envelope
            ).sum(dim=2) / p_all.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            branch_oracle_no_envelope = branch_density_pre_envelope.amax(dim=2)
            positive_oracle = torch.where(target > 0.0, branch_oracle, final)
            cand_contrib = diag["branch_contributions"][:, :, 1:].sum(dim=2).clamp(0.0, 1.0)
            gt_pos = target.squeeze(-1) > 0.0
            pi = diag["pi"]
            candidate_pi = pi[..., 1:]
            candidate_prior = p_all[..., 1:]
            best_candidate_index = candidate_prior.argmax(dim=-1, keepdim=True)
            selected_candidate_pi = candidate_pi.gather(
                dim=-1, index=best_candidate_index
            ).squeeze(-1)
            positive_candidate_mass = (
                candidate_pi.sum(dim=-1)[gt_pos].mean()
                if candidate_pi.shape[-1] > 0 and gt_pos.any()
                else torch.zeros((), device=device)
            )
            positive_selected_candidate_pi = (
                selected_candidate_pi[gt_pos].mean()
                if selected_candidate_pi.numel() > 0 and gt_pos.any()
                else torch.zeros((), device=device)
            )
            positive_candidate_density = (
                branch_density[:, :, 1:].amax(dim=2).squeeze(-1)[gt_pos].mean()
                if branch_density.shape[2] > 1 and gt_pos.any()
                else torch.zeros((), device=device)
            )
            candidate_coverage = (
                (p_all[..., 1:].amax(dim=-1)[gt_pos] > 0.05).float().mean()
                if p_all.shape[-1] > 1 and gt_pos.any()
                else torch.zeros((), device=device)
            )
            row: dict[str, float | str | int] = {
                "sample_index": idx,
                "sample_id": str(sample_id),
                "target_pos_ratio": float(gt_pos.float().mean().detach().cpu()),
                "measurement_supported_ratio": float(
                    diag["measurement_supported"].float().mean().detach().cpu()
                ),
                "measurement_supported_pos_ratio": float(
                    (diag["measurement_supported"] & gt_pos).float().mean().detach().cpu()
                ),
                "candidate_coverage": float(candidate_coverage.detach().cpu()),
                "pi0": float(diag["pi"][..., 0].mean().detach().cpu()),
                "positive_pi0": float(
                    pi[..., 0][gt_pos].mean().detach().cpu()
                    if gt_pos.any()
                    else torch.zeros((), device=device).cpu()
                ),
                "candidate_mass": float(diag["pi"][..., 1:].sum(dim=-1).mean().detach().cpu())
                if diag["pi"].shape[-1] > 1
                else 0.0,
                "positive_candidate_mass": float(positive_candidate_mass.detach().cpu()),
                "positive_selected_candidate_pi": float(
                    positive_selected_candidate_pi.detach().cpu()
                ),
                "positive_candidate_density": float(positive_candidate_density.detach().cpu()),
                "candidate_utilization": float(diag["candidate_utilization"].detach().cpu()),
                "unused_candidate_ratio": float(diag["unused_candidate_ratio"].detach().cpu()),
                "comp_contrib_ratio": float(
                    diag["compensation_contribution_ratio"].detach().cpu()
                ),
                "cand_contrib_ratio": float(diag["candidate_contribution_ratio"].detach().cpu()),
                "branch_density_mean": float(branch_density.mean().detach().cpu()),
                "branch_density_std": float(branch_density.std(unbiased=False).detach().cpu()),
            }
            for name, pred in [
                ("final", final),
                ("comp", comp),
                ("fixed_prior", fixed_prior),
                ("cand_contrib", cand_contrib),
                ("branch_oracle", branch_oracle),
                ("no_envelope", no_envelope),
                ("fixed_prior_no_envelope", fixed_prior_no_envelope),
                ("branch_oracle_no_envelope", branch_oracle_no_envelope),
                ("positive_oracle", positive_oracle),
            ]:
                for metric, value in _binary_metrics(pred, target).items():
                    row[f"{name}_{metric}"] = value
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    if not rows:
        raise RuntimeError("no samples evaluated")
    numeric_keys = [key for key, value in rows[0].items() if isinstance(value, (int, float))]
    summary = {key: _row_mean(rows, key) for key in numeric_keys}
    summary["num_samples"] = len(rows)
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "branch_diagnostics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_path / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print("SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print(f"wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
