#!/usr/bin/env python3
"""Evaluate view-complementary checkpoints on one aligned validation loader."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from minr_fmt.datamodule import TrainingDataModule  # noqa: E402
from minr_fmt.module import TrainingLightningModule  # noqa: E402

MODELS = {
    "A2-U": "long_1k_continue_a2u_seed42",
    "A3-old": "long_1k_continue_a3_seed42",
    "Geometry-only": "a3v2_fast_geometry_only_seed42_corrected",
    "A3-v2": "a3v2_fast_bounded_routing_seed42_corrected",
}


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    return value


def best_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted((run_dir / "checkpoints").glob("epoch=*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No best checkpoint under {run_dir}")
    return checkpoints[0]


def sample_dice(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = prediction.squeeze(-1) >= 0.5
    truth = target > 0
    dims = tuple(range(1, pred.ndim))
    intersection = (pred & truth).sum(dim=dims).float()
    return (2 * intersection + 1e-8) / (pred.sum(dim=dims) + truth.sum(dim=dims) + 1e-8)


def candidate_stats(diagnostics: dict, batch: dict, index: int) -> dict[str, float]:
    valid = diagnostics.get(
        "candidate_analysis_valid_mask", diagnostics["candidate_valid_mask"]
    )[index].bool()
    centers = diagnostics["candidate_centers_mm"][index, valid].float()
    gt_valid = batch["gt_component_valid_mask"][index].bool()
    targets = batch["gt_component_centers_mm"][index, gt_valid].float()
    result = {"candidate_count": float(valid.sum())}
    if not centers.numel() or not targets.numel():
        result.update(coverage_8mm=0.0, duplicate_rate=0.0, unmatched_rate=1.0)
        return result
    distance = torch.cdist(centers, targets)
    nearest_target = distance.amin(dim=0)
    nearest_candidate = distance.amin(dim=1)
    assigned = distance.argmin(dim=1)
    duplicate = torch.zeros(len(centers), dtype=torch.bool, device=centers.device)
    for target_index in range(len(targets)):
        members = torch.where((assigned == target_index) & (nearest_candidate <= 8))[0]
        if len(members) > 1:
            duplicate[members] = True
            duplicate[members[distance[members, target_index].argmin()]] = False
    result.update(
        coverage_8mm=float((nearest_target <= 8).float().mean()),
        duplicate_rate=float(duplicate.float().mean()),
        unmatched_rate=float((nearest_candidate > 8).float().mean()),
    )
    return result


def evaluate(name: str, run_dir: Path, query_count: int, max_samples: int) -> list[dict]:
    cfg = OmegaConf.load(run_dir / "config/config.yaml")
    cfg.data.num_queries = query_count
    cfg.data.eval_sample_num = query_count
    cfg.data.query_sampling.num_queries = query_count
    cfg.data.val_max_samples = max_samples
    cfg.data.num_workers = 0
    # Canonical loader is the incumbent A2-U validation loader.  Some newer
    # resolved configs used `first`, which silently selected different cases.
    cfg.data.subset_policy = "random"
    cfg.data.subset_seed = 42
    module = TrainingLightningModule(cfg)
    checkpoint = best_checkpoint(run_dir)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    module.load_state_dict(state, strict=True)
    module.eval().cuda()
    data = TrainingDataModule(cfg)
    data.setup("validate")
    rows = []
    with torch.inference_mode():
        for batch in data.val_dataloader():
            batch = to_device(batch, "cuda")
            out = module._call_ssq_model(batch, return_diagnostics=True)
            diagnostics = out["diagnostics"]
            final_dice = sample_dice(out["density"], batch["point_densities"])
            shared_dice = sample_dice(diagnostics["shared_density"], batch["point_densities"])
            for index, sample_id in enumerate(batch["sample_id"]):
                row = {
                    "sample_id": str(sample_id),
                    "model": name,
                    "dice": float(final_dice[index]),
                    "shared_dice": float(shared_dice[index]),
                    "final_minus_shared_dice": float(final_dice[index] - shared_dice[index]),
                    "checkpoint": str(checkpoint),
                }
                row.update(candidate_stats(diagnostics, batch, index))
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, directory in MODELS.items():
        rows.extend(
            evaluate(
                name,
                ROOT / "outputs/view_complementary" / directory,
                args.queries,
                args.samples,
            )
        )
    ids = {name: {row["sample_id"] for row in rows if row["model"] == name} for name in MODELS}
    if len({tuple(sorted(value)) for value in ids.values()}) != 1:
        raise RuntimeError("Validation sample IDs are not aligned across models")
    with (args.output_dir / "paired_per_sample.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for name in MODELS:
        selected = [row for row in rows if row["model"] == name]
        metric_names = (
            "dice",
            "shared_dice",
            "final_minus_shared_dice",
            "candidate_count",
            "coverage_8mm",
            "duplicate_rate",
            "unmatched_rate",
        )
        summary[name] = {
            key: sum(float(row[key]) for row in selected) / len(selected)
            for key in metric_names
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
