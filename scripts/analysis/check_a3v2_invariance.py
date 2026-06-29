#!/usr/bin/env python3
"""Lightweight per-sample fixed-grid and analysis-threshold invariance checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from minr_fmt.datamodule import TrainingDataModule  # noqa: E402
from minr_fmt.module import TrainingLightningModule  # noqa: E402


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    return value


def load_model(config_path: Path, checkpoint: Path, query_count: int):
    cfg = OmegaConf.load(config_path)
    cfg.data.num_queries = query_count
    cfg.data.eval_sample_num = query_count
    cfg.data.query_sampling.num_queries = query_count
    cfg.data.val_max_samples = 32
    cfg.data.num_workers = 0
    module = TrainingLightningModule(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    module.load_state_dict(state, strict=True)
    module.eval().cuda()
    return cfg, module


def collect(config_path: Path, checkpoint: Path, query_count: int):
    cfg, module = load_model(config_path, checkpoint, query_count)
    data = TrainingDataModule(cfg)
    data.setup("validate")
    records = {}
    densities = {}
    with torch.no_grad():
        for index, batch in enumerate(data.val_dataloader()):
            if index >= 32:
                break
            batch = to_device(batch, "cuda")
            out = module._call_ssq_model(batch, return_diagnostics=True)
            diag = out["diagnostics"]
            sample_id = str(batch["sample_id"][0])
            records[sample_id] = {
                "centers": diag["candidate_centers_mm"][0].float().cpu(),
                "existence": diag["candidate_existence_probability"][0].float().cpu(),
                "covariance": diag["candidate_covariances_mm"][0].float().cpu(),
                "support": diag["candidate_view_support"][0].float().cpu(),
                "slots": diag["candidate_slot_valid_mask"][0].cpu(),
                "analysis": diag["candidate_analysis_valid_mask"][0].cpu(),
                "gt_centers": batch["gt_component_centers_mm"][0].float().cpu(),
                "gt_valid": batch["gt_component_valid_mask"][0].cpu(),
            }
            densities[sample_id] = out["density"][0].float().cpu()
    return module, records, densities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    config = args.run_dir / "config/config.yaml"
    all_records = {}
    for count in (2048, 4096, 8192):
        _, records, _ = collect(config, args.checkpoint, count)
        all_records[count] = records
    reference = all_records[4096]
    query_rows = []
    for count, records in all_records.items():
        for sample_id in sorted(reference.keys() & records.keys()):
            left, right = reference[sample_id], records[sample_id]
            predicted = right["centers"][right["analysis"]]
            target = right["gt_centers"][right["gt_valid"]]
            distance = (
                torch.cdist(predicted, target) if predicted.numel() and target.numel() else None
            )
            nearest_target = (
                distance.amin(dim=0)
                if distance is not None
                else torch.full((len(target),), torch.inf)
            )
            nearest_candidate = (
                distance.amin(dim=1)
                if distance is not None
                else torch.full((len(predicted),), torch.inf)
            )
            assigned = (
                distance.argmin(dim=1) if distance is not None else torch.empty(0, dtype=torch.long)
            )
            duplicate = torch.zeros(len(predicted), dtype=torch.bool)
            if distance is not None:
                for target_index in range(len(target)):
                    members = torch.where((assigned == target_index) & (nearest_candidate <= 8))[0]
                    if len(members) > 1:
                        duplicate[members] = True
                        duplicate[members[distance[members, target_index].argmin()]] = False
            query_rows.append(
                {
                    "sample_id": sample_id,
                    "query_count": count,
                    "center_drift_max_mm": float(
                        (left["centers"] - right["centers"]).norm(dim=-1).max()
                    ),
                    "existence_drift_max": float(
                        (left["existence"] - right["existence"]).abs().max()
                    ),
                    "covariance_drift_max": float(
                        (left["covariance"] - right["covariance"]).abs().max()
                    ),
                    "support_drift_max": float((left["support"] - right["support"]).abs().max()),
                    "candidate_count_drift": int(left["slots"].sum() - right["slots"].sum()),
                    "candidate_count": int(right["analysis"].sum()),
                    "coverage_6mm": float((nearest_target <= 6).float().mean())
                    if len(target)
                    else 0.0,
                    "coverage_8mm": float((nearest_target <= 8).float().mean())
                    if len(target)
                    else 0.0,
                    "coverage_10mm": float((nearest_target <= 10).float().mean())
                    if len(target)
                    else 0.0,
                    "duplicate_rate": float(duplicate.float().mean()) if len(predicted) else 0.0,
                    "unmatched_rate": float((nearest_candidate > 8).float().mean())
                    if len(predicted)
                    else 1.0,
                    "matched_center_error_mm": float(
                        nearest_candidate[nearest_candidate <= 8].mean()
                    )
                    if (nearest_candidate <= 8).any()
                    else None,
                }
            )

    cfg, module = load_model(config, args.checkpoint, 4096)
    data = TrainingDataModule(cfg)
    data.setup("validate")
    threshold_rows = []
    baseline_density = {}
    with torch.no_grad():
        batches = list(data.val_dataloader())[:32]
        for threshold in (0.2, 0.3, 0.4, 0.5):
            module.net.diverse_candidate_constructor.candidate_conf_threshold = threshold
            max_diff, dices, counts = 0.0, [], []
            for batch in batches:
                batch = to_device(batch, "cuda")
                out = module._call_ssq_model(batch, return_diagnostics=True)
                sample_id = str(batch["sample_id"][0])
                density = out["density"].float()
                if threshold == 0.2:
                    baseline_density[sample_id] = density.cpu()
                else:
                    max_diff = max(
                        max_diff, float((density.cpu() - baseline_density[sample_id]).abs().max())
                    )
                target = batch["point_densities"].float().unsqueeze(-1)
                pred_bin = density >= 0.5
                target_bin = target > 0
                intersection = (pred_bin & target_bin).float().sum()
                dices.append(
                    float(
                        (2 * intersection + 1.0e-6)
                        / (pred_bin.float().sum() + target_bin.float().sum() + 1.0e-6)
                    )
                )
                counts.append(float(out["diagnostics"]["candidate_analysis_valid_mask"].sum()))
            threshold_rows.append(
                {
                    "threshold": threshold,
                    "density_max_abs_diff": max_diff,
                    "val_dice_32": sum(dices) / len(dices),
                    "candidate_count_mean": sum(counts) / len(counts),
                }
            )
    payload = {
        "query_count_invariance": query_rows,
        "analysis_threshold_invariance": threshold_rows,
    }
    output = args.run_dir / "invariance_results.json"
    output.write_text(json.dumps(payload, indent=2))
    print(output)


if __name__ == "__main__":
    main()
