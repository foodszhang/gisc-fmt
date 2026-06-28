#!/usr/bin/env python
"""Train and evaluate a measurement-only SSQ candidate heatmap refiner."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.network.ssq_candidate_refiner import MeasurementCandidateRefiner  # noqa: E402
from minr_fmt.utils.ssq_candidate_extraction import (  # noqa: E402
    extract_candidate_anchors,
    write_candidate_cache,
)


class CandidateRefinerDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, max_samples: int | None):
        names = [
            line.strip()
            for line in (data_dir / "splits" / f"{split}.txt").read_text().splitlines()
            if line.strip()
        ]
        if max_samples is not None:
            names = names[: int(max_samples)]
        self.samples = [data_dir / "samples" / name for name in names]

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _target(gt: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
        labels, count = ndimage.label(gt / max(float(gt.max()), 1.0e-8) > 0.5)
        target = np.zeros(shape, dtype=np.float32)
        scale = np.asarray(shape, dtype=np.float32) / np.asarray(gt.shape, dtype=np.float32)
        for index in range(1, count + 1):
            coordinates = np.argwhere(labels == index)
            if not len(coordinates):
                continue
            center = (coordinates.mean(axis=0) + 0.5) * scale - 0.5
            impulse = np.zeros(shape, dtype=np.float32)
            center_index = np.clip(np.round(center).astype(np.int64), 0, np.asarray(shape) - 1)
            impulse[tuple(center_index)] = 1.0
            target = np.maximum(target, ndimage.gaussian_filter(impulse, sigma=1.25))
        if target.max() > 0:
            target /= target.max()
        return target

    def __getitem__(self, index: int):
        sample = self.samples[index]
        heatmap = np.load(sample / "proposal" / "meas_backproj_heatmap.npy").astype(np.float32)
        gt = np.load(sample / "gt_voxels.npy").astype(np.float32)
        target = self._target(gt, heatmap.shape)
        return {
            "heatmap": torch.from_numpy(heatmap[None]),
            "target": torch.from_numpy(target[None]),
            "sample_dir": str(sample),
        }


def heatmap_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(
        logits.float(), target.float(), pos_weight=torch.tensor(50.0, device=logits.device)
    )
    probability = torch.sigmoid(logits.float())
    intersection = (probability * target).sum(dim=(1, 2, 3, 4))
    dice = (2.0 * intersection + 1.0e-6) / (
        probability.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4)) + 1.0e-6
    )
    return bce + 0.5 * (1.0 - dice.mean())


def component_centers(gt: np.ndarray, voxel_size_mm: float = 0.2) -> np.ndarray:
    labels, count = ndimage.label(gt / max(float(gt.max()), 1.0e-8) > 0.5)
    centers = []
    for index in range(1, count + 1):
        coordinates = np.argwhere(labels == index)
        if len(coordinates):
            centers.append((coordinates.mean(axis=0) + 0.5) * voxel_size_mm)
    return np.asarray(centers, dtype=np.float32)


@torch.no_grad()
def evaluate(model, loader, device, top_m: int) -> dict[str, float]:
    model.eval()
    recall, full_coverage, precision = [], [], []
    for batch in loader:
        prediction = torch.sigmoid(model(batch["heatmap"].to(device))).cpu().numpy()
        for item, sample_dir in zip(prediction[:, 0], batch["sample_dir"], strict=True):
            sample = Path(sample_dir)
            metadata = json.loads((sample / "proposal" / "meas_backproj_meta.json").read_text())
            anchors = extract_candidate_anchors(
                item,
                metadata,
                top_m=top_m,
                smoothing_sigma_cells=0.5,
                nms_radius_mm=3.0,
                support_moment_radius_mm=4.0,
                min_value_ratio=0.05,
            )
            candidates = anchors["centers_mm"][anchors["valid"]]
            gt = np.load(sample / "gt_voxels.npy").astype(np.float32)
            components = component_centers(gt)
            distance = np.linalg.norm(components[:, None] - candidates[None], axis=-1)
            component_distance = distance.min(axis=1)
            candidate_distance = distance.min(axis=0)
            recall.append(float((component_distance <= 3.0).mean()))
            full_coverage.append(float((component_distance <= 3.0).all()))
            precision.append(float((candidate_distance <= 3.0).mean()))
    return {
        "component_recall_3mm": float(np.mean(recall)),
        "full_coverage_3mm": float(np.mean(full_coverage)),
        "candidate_precision_3mm": float(np.mean(precision)),
    }


@torch.no_grad()
def export_candidates(
    model, dataset, device, top_m: int, filename: str, num_workers: int
) -> None:
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    for batch in loader:
        prediction = torch.sigmoid(model(batch["heatmap"].to(device)))[0, 0].cpu().numpy()
        sample = Path(batch["sample_dir"][0])
        metadata = json.loads((sample / "proposal" / "meas_backproj_meta.json").read_text())
        anchors = extract_candidate_anchors(
            prediction,
            metadata,
            top_m=top_m,
            smoothing_sigma_cells=0.5,
            nms_radius_mm=3.0,
            support_moment_radius_mm=4.0,
            min_value_ratio=0.05,
        )
        metadata = {
            **metadata,
            "proposal_version": "measurement_candidate_refiner_v1",
            "uses_gt": False,
            "refiner_training_uses_gt_centers": True,
        }
        write_candidate_cache(sample, anchors, metadata, filename=filename)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", default="/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k"
    )
    parser.add_argument("--train-samples", type=int, default=200)
    parser.add_argument("--val-samples", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--export-num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--top-m", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--export-filename")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_dir = Path(args.data_dir)
    train_dataset = CandidateRefinerDataset(data_dir, "train", args.train_samples)
    val_dataset = CandidateRefinerDataset(data_dir, "val", args.val_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
    )
    device = torch.device("cuda")
    model = MeasurementCandidateRefiner().to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-4)
    scaler = torch.amp.GradScaler("cuda")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    best_recall = -1.0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            heatmap = batch["heatmap"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = heatmap_loss(model(heatmap), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate(model, val_loader, device, args.top_m)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics}
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if metrics["component_recall_3mm"] > best_recall:
            best_recall = metrics["component_recall_3mm"]
            torch.save({"state_dict": model.state_dict(), "metrics": metrics}, out_dir / "best.pt")
    (out_dir / "metrics.json").write_text(json.dumps(rows, indent=2))
    if args.export_filename:
        best_path = out_dir / "best.pt"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
        export_candidates(
            model,
            train_dataset,
            device,
            args.top_m,
            args.export_filename,
            args.export_num_workers,
        )
        export_candidates(
            model,
            val_dataset,
            device,
            args.top_m,
            args.export_filename,
            args.export_num_workers,
        )


if __name__ == "__main__":
    main()
