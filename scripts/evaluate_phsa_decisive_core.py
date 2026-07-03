#!/usr/bin/env python
"""Paired sampled-query evaluation for the frozen-Phase-A PHSA comparison."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.datamodule import TrainingDataModule  # noqa: E402
from minr_fmt.module import TrainingLightningModule  # noqa: E402
from minr_fmt.phsa_sample_level import activate_phsa_sample_level_hypotheses  # noqa: E402

RUN_PATTERN = re.compile(r"(?P<source>gt|learned)_(?P<strategy>\w+)_seed(?P<seed>\d+)")


def best_checkpoint(run: Path) -> Path:
    checkpoints = [
        path for path in (run / "checkpoints").glob("*.ckpt") if path.name != "last.ckpt"
    ]
    if not checkpoints:
        raise FileNotFoundError(f"no selected checkpoint under {run}")

    def score(path: Path) -> float:
        match = re.search(r"val_dice=([0-9.]+)", path.name)
        return float(match.group(1).rstrip(".")) if match else -1.0

    return max(checkpoints, key=score)


def component_metrics(
    pred: torch.Tensor, target: torch.Tensor, ids: torch.Tensor
) -> dict[str, float]:
    recalls = []
    strengths = []
    missed = 0
    for component_id in ids.unique().tolist():
        if component_id < 0:
            continue
        mask = (ids == component_id) & (target > 0)
        if not mask.any():
            continue
        recall = float(pred[mask].float().mean())
        recalls.append(recall)
        strengths.append(float(target[mask].max()))
        missed += int(recall == 0.0)
    if not recalls:
        return {
            "component_recall": float("nan"),
            "weak_source_recall": float("nan"),
            "missed_component_rate": float("nan"),
        }
    weakest = int(np.argmin(strengths))
    return {
        "component_recall": float(np.mean(recalls)),
        "weak_source_recall": recalls[weakest],
        "missed_component_rate": missed / len(recalls),
    }


def evaluate(
    run: Path,
    device: torch.device,
    max_samples: int,
    strategy_override: str | None = None,
) -> list[dict[str, Any]]:
    match = RUN_PATTERN.fullmatch(run.name)
    if match is None:
        raise ValueError(f"invalid run name: {run.name}")
    info = match.groupdict()
    evaluated_strategy = strategy_override or info["strategy"]
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                "model=ssq_fmt",
                "exp=fmt_simgen_v2_phsa_decisive",
                "data.dataset_type=fmt_simgen",
                "seed=20260701",
                "data.subset_policy=first",
                f"data.val_max_samples={max_samples}",
                "data.eval_sample_num=4096",
                "data.eval_batch_size=2",
                "data.num_workers=0",
                f"model.ssq_fmt.view_complementary.hypothesis_source={info['source']}",
                f"model.ssq_fmt.view_complementary.aggregation_strategy={evaluated_strategy}",
            ],
        )
    module = (
        TrainingLightningModule.load_from_checkpoint(
            str(best_checkpoint(run)), cfg=cfg, map_location="cpu"
        )
        .to(device)
        .eval()
    )
    dm = TrainingDataModule(cfg)
    dm.setup("fit")
    rows = []
    with (
        torch.inference_mode(),
        torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"),
    ):
        for batch in dm.val_dataloader():
            sample_ids = [str(value) for value in batch["sample_id"]]
            batch = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            output = module.net(
                batch["surface_measurements_packed"],
                batch["query_coordinates_mm"],
                detector_valid_mask=batch.get("detector_valid_mask"),
                depth_maps=batch.get("depth_maps"),
                batch=batch,
            )
            result_strategy = (
                "oracle_from_uniform" if strategy_override == "oracle" else info["strategy"]
            )
            predictions = [(info["source"], result_strategy, output["density"])]
            if info["source"] == "gt" and info["strategy"] == "uniform" and not strategy_override:
                predictions.append(("shared", "only", output["aux_outputs"]["shared_density"]))
            for batch_index, sample_id in enumerate(sample_ids):
                target = batch["point_densities"][batch_index].float()
                supported = output["aux_outputs"]["measurement_supported"][batch_index].bool()
                for source, strategy, density in predictions:
                    probability = density[batch_index].squeeze(-1).float()
                    pred = (probability >= 0.5) & supported
                    truth = (target > 0) & supported
                    tp = int((pred & truth).sum())
                    fp = int((pred & ~truth).sum())
                    fn = int((~pred & truth).sum())
                    comp = component_metrics(
                        pred, target, batch["query_component_ids"][batch_index]
                    )
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "source": source,
                            "strategy": strategy,
                            "seed": int(info["seed"]),
                            "dice": (2 * tp + 1.0e-6) / (2 * tp + fp + fn + 1.0e-6),
                            "tp": tp,
                            "fp": fp,
                            "fn": fn,
                            **comp,
                        }
                    )
    return rows


def bootstrap(values: np.ndarray, seed: int = 20260701, draws: int = 10000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def summarize(rows: list[dict[str, Any]], strata_path: Path) -> dict[str, Any]:
    with strata_path.open() as handle:
        strata = {row["sample_id"]: row for row in csv.DictReader(handle)}
    grouped: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["source"], row["strategy"], row["seed"]), {})[row["sample_id"]] = (
            row
        )
    summary: dict[str, Any] = {"comparisons": {}}
    for seed in sorted({row["seed"] for row in rows}):
        uniform = grouped.get(("gt", "uniform", seed), {})
        for source, strategy in sorted({(row["source"], row["strategy"]) for row in rows}):
            current = grouped.get((source, strategy, seed), {})
            common = sorted(set(uniform) & set(current))
            if not common:
                continue
            name = f"{source}_{strategy}_minus_gt_uniform"
            entry = summary["comparisons"].setdefault(name, {})
            for stratum in [
                "all",
                "single_source",
                "multi_view_complementary",
                "multi_all_views_separable",
                "multi_intermediate",
            ]:
                ids = (
                    common
                    if stratum == "all"
                    else [sid for sid in common if strata[sid]["geometry_stratum"] == stratum]
                )
                if not ids:
                    continue
                seed_entry = entry.setdefault(stratum, {}).setdefault("per_seed", {})
                seed_entry[str(seed)] = {
                    metric: float(
                        np.nanmean([current[sid][metric] - uniform[sid][metric] for sid in ids])
                    )
                    for metric in (
                        "dice",
                        "component_recall",
                        "weak_source_recall",
                        "missed_component_rate",
                        "tp",
                        "fp",
                        "fn",
                    )
                } | {"n": len(ids)}
    # Pool sample-level paired deltas over seeds for confidence intervals.
    for name, strata_groups in summary["comparisons"].items():
        source_strategy = name.removesuffix("_minus_gt_uniform").split("_", 1)
        source, strategy = source_strategy
        for stratum, result in strata_groups.items():
            deltas = []
            for seed in sorted({row["seed"] for row in rows}):
                uniform = grouped.get(("gt", "uniform", seed), {})
                current = grouped.get((source, strategy, seed), {})
                ids = sorted(set(uniform) & set(current))
                if stratum != "all":
                    ids = [sid for sid in ids if strata[sid]["geometry_stratum"] == stratum]
                deltas.extend(current[sid]["dice"] - uniform[sid]["dice"] for sid in ids)
            if deltas:
                result["dice_bootstrap"] = bootstrap(np.asarray(deltas))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--strata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=300)
    args = parser.parse_args()
    activate_phsa_sample_level_hypotheses()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for run in sorted(
        path
        for path in args.runs.iterdir()
        if (path / "DONE").exists() and RUN_PATTERN.fullmatch(path.name)
    ):
        rows.extend(evaluate(run, device, args.max_samples))
        if run.name.startswith("gt_uniform_seed"):
            rows.extend(evaluate(run, device, args.max_samples, strategy_override="oracle"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "paired_per_sample.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summarize(rows, args.strata), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
