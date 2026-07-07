#!/usr/bin/env python
"""Serial SSQ-FMT tuning launcher with metric summaries and winner propagation."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

METRIC_PRIORITY = ("val_assd", "val_hd95", "val_dice", "val_component_recall")


def base_cmd(
    name: str,
    train_n: int,
    val_n: int,
    epochs: int,
    overrides: list[str],
    output_root: Path,
    num_queries: int,
    query_chunk_size: int,
) -> list[str]:
    out_dir = output_root / name
    return [
        "uv",
        "run",
        "python",
        "train.py",
        "fit",
        "model=ssq_fmt",
        "exp=fmt_simgen_v2_ssq_final",
        "data.dataset_type=fmt_simgen",
        "seed=42",
        f"name={name}",
        f"paths.output_dir={out_dir}",
        f"data.train_max_samples={train_n}",
        f"data.val_max_samples={val_n}",
        "data.batch_size=1",
        "data.eval_batch_size=1",
        f"data.num_queries={num_queries}",
        f"data.eval_sample_num={num_queries}",
        f"trainer.max_epochs={epochs}",
        "trainer.accumulate_grad_batches=4",
        f"model.ssq_fmt.routing.query_chunk_size={query_chunk_size}",
        f"model.ssq_fmt.representation.query_chunk_size={query_chunk_size}",
        *overrides,
    ]


def stage_plan(
    stage: str,
    output_root: Path,
    inherited: list[str],
    num_queries: int,
    query_chunk_size: int,
) -> list[dict[str, Any]]:
    if stage == "A":
        variants = {
            "E0_shallow_two_conv": ["model.ssq_fmt.encoder.type=shallow_encoder_ablation"],
            "E1_residual_unet": ["model.ssq_fmt.encoder.type=residual_unet_no_pyramid"],
            "E2_residual_unet_pyramid": [
                "model.ssq_fmt.encoder.type=residual_unet_pyramid_fusion"
            ],
        }
        return [
            {
                "variant": key,
                "overrides": inherited + value,
                "cmd": base_cmd(
                    f"ssq_tune_{key}",
                    400,
                    100,
                    15,
                    inherited + value,
                    output_root,
                    num_queries,
                    query_chunk_size,
                ),
            }
            for key, value in variants.items()
        ]
    if stage == "P0":
        variants = {
            "P0_lr3e4_pos30": [
                "optim.lr=0.0003",
                "loss.pos_weight=30",
                "loss.dice_weight=0.5",
            ],
            "P0_lr5e4_pos50": [
                "optim.lr=0.0005",
                "loss.pos_weight=50",
                "loss.dice_weight=0.5",
            ],
            "P0_lr5e4_dice1": [
                "optim.lr=0.0005",
                "loss.pos_weight=50",
                "loss.dice_weight=1.0",
            ],
        }
        return [
            {
                "variant": key,
                "overrides": inherited + value,
                "cmd": base_cmd(
                    f"ssq_tune_{key}",
                    400,
                    100,
                    15,
                    inherited + value,
                    output_root,
                    num_queries,
                    query_chunk_size,
                ),
            }
            for key, value in variants.items()
        ]
    if stage == "SDF_LAMBDA":
        return [
            {
                "variant": f"lambda_sdf_{value:g}",
                "overrides": inherited + [f"loss.lambda_sdf={value}", "loss.sdf_boundary_weight=1"],
                "cmd": base_cmd(
                    f"ssq_tune_sdf_lambda_{str(value).replace('.', 'p')}",
                    400,
                    100,
                    15,
                    inherited + [f"loss.lambda_sdf={value}", "loss.sdf_boundary_weight=1"],
                    output_root,
                    num_queries,
                    query_chunk_size,
                ),
            }
            for value in (0.0, 0.02, 0.05, 0.10)
        ]
    if stage == "SDF_BOUNDARY":
        return [
            {
                "variant": f"boundary_weight_{value:g}",
                "overrides": inherited + [f"loss.sdf_boundary_weight={value}"],
                "cmd": base_cmd(
                    f"ssq_tune_sdf_boundary_{value}",
                    400,
                    100,
                    15,
                    inherited + [f"loss.sdf_boundary_weight={value}"],
                    output_root,
                    num_queries,
                    query_chunk_size,
                ),
            }
            for value in (1, 2, 4)
        ]
    raise ValueError(f"unknown stage {stage!r}")


def _try_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_metrics(output_dir: Path) -> dict[str, float]:
    candidates = sorted(output_dir.glob("**/metrics.csv"))
    if not candidates:
        return {}
    rows = list(csv.DictReader(candidates[-1].open()))
    best: dict[str, float] = {}
    for row in rows:
        for key, value in row.items():
            val = _try_float(value)
            if val is None:
                continue
            if key not in best:
                best[key] = val
            elif key.endswith(("dice", "recall")):
                best[key] = max(best[key], val)
            elif key.endswith(("assd", "hd95", "loss")):
                best[key] = min(best[key], val)
            else:
                best[key] = val
    return best


def score_row(row: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = row["metrics"]
    assd = float(metrics.get("val_assd", metrics.get("val/assd", 1.0e9)))
    hd95 = float(metrics.get("val_hd95", metrics.get("val/hd95", 1.0e9)))
    dice = float(metrics.get("val_dice", metrics.get("val/dice", -1.0)))
    recall = float(metrics.get("val_component_recall", metrics.get("val/component_recall", -1.0)))
    return (assd, hd95, -dice, -recall)


def summarize(plan: list[dict[str, Any]], output_root: Path, stage: str) -> dict[str, Any]:
    rows = []
    for item in plan:
        name = next(v.split("=", 1)[1] for v in item["cmd"] if v.startswith("name="))
        metrics = read_metrics(output_root / name)
        rows.append(
            {
                "variant": item["variant"],
                "name": name,
                "overrides": item["overrides"],
                "metrics": metrics,
            }
        )
    winner = min(rows, key=score_row) if rows else None
    out_csv = output_root / f"stage_{stage}_tuning_summary.csv"
    out_md = output_root / f"stage_{stage}_tuning_summary.md"
    fieldnames = ["variant", "name", "winner", *METRIC_PRIORITY, "overrides"]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            metrics = row["metrics"]
            writer.writerow(
                {
                    "variant": row["variant"],
                    "name": row["name"],
                    "winner": bool(winner and row["variant"] == winner["variant"]),
                    **{
                        key: metrics.get(key, metrics.get(key.replace("_", "/"), ""))
                        for key in METRIC_PRIORITY
                    },
                    "overrides": " ".join(row["overrides"]),
                }
            )
    lines = [
        "| Variant | Winner | ASSD | HD95 | Dice | Component recall |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        metrics = row["metrics"]
        lines.append(
            "| {variant} | {winner} | {assd} | {hd95} | {dice} | {recall} |".format(
                variant=row["variant"],
                winner="yes" if winner and row["variant"] == winner["variant"] else "",
                assd=metrics.get("val_assd", metrics.get("val/assd", "")),
                hd95=metrics.get("val_hd95", metrics.get("val/hd95", "")),
                dice=metrics.get("val_dice", metrics.get("val/dice", "")),
                recall=metrics.get("val_component_recall", metrics.get("val/component_recall", "")),
            )
        )
    out_md.write_text("\n".join(lines) + "\n")
    result = {"stage": stage, "rows": rows, "winner": winner}
    (output_root / f"stage_{stage}_winner.json").write_text(json.dumps(result, indent=2))
    if winner is not None:
        (output_root / "winner_overrides.json").write_text(
            json.dumps(winner["overrides"], indent=2)
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["A", "P0", "SDF_LAMBDA", "SDF_BOUNDARY"],
        required=True,
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--out_dir", default="outputs/ssq_fmt_rebuild/tuning")
    parser.add_argument("--inherit_overrides_json")
    parser.add_argument("--num-queries", type=int, default=32768)
    parser.add_argument("--query-chunk-size", type=int, default=512)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inherited: list[str] = []
    if args.inherit_overrides_json:
        inherited = json.loads(Path(args.inherit_overrides_json).read_text())
    elif (out_dir / "winner_overrides.json").exists() and args.stage != "A":
        inherited = json.loads((out_dir / "winner_overrides.json").read_text())

    plan = stage_plan(
        args.stage,
        out_dir,
        inherited,
        args.num_queries,
        args.query_chunk_size,
    )
    plan_path = out_dir / f"stage_{args.stage}_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2))
    print(f"wrote {plan_path}")
    for item in plan:
        print(" ".join(item["cmd"]))
        if args.execute:
            subprocess.run(item["cmd"], check=True)
    if args.summarize or args.execute:
        result = summarize(plan, out_dir, args.stage)
        if result["winner"] is not None:
            print(f"winner: {result['winner']['variant']}")


if __name__ == "__main__":
    main()
