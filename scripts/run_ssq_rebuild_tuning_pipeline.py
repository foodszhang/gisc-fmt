#!/usr/bin/env python
"""Serial successive-halving style tuning launcher for rebuilt SSQ-FMT.

Default mode writes the exact command plan without launching training. Use
``--execute`` to run one stage serially.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def base_cmd(name: str, train_n: int, val_n: int, epochs: int, overrides: list[str]) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "train.py",
        "fit",
        "model=ssq_fmt",
        "exp=fmt_simgen_v2_ssq_rebuild_final",
        "data.dataset_type=fmt_simgen",
        "seed=42",
        f"name={name}",
        f"data.train_max_samples={train_n}",
        f"data.val_max_samples={val_n}",
        "data.num_queries=32768",
        "data.eval_sample_num=32768",
        f"trainer.max_epochs={epochs}",
        *overrides,
    ]


def stage_plan(stage: str) -> list[dict[str, object]]:
    if stage == "A":
        return [
            {
                "variant": "A0_shallow",
                "cmd": base_cmd(
                    "ssq_rebuild_A0_shallow",
                    400,
                    100,
                    15,
                    ["model.ssq_fmt.encoder.type=shallow_encoder_ablation"],
                ),
            },
            {
                "variant": "A1_fpn_geometry",
                "cmd": base_cmd("ssq_rebuild_A1_fpn_geometry", 400, 100, 15, []),
            },
            {
                "variant": "A2_fpn_zero_init",
                "cmd": base_cmd("ssq_rebuild_A2_fpn_zero_init", 400, 100, 15, []),
            },
        ]
    if stage == "B":
        variants = {
            "B1": [
                "optim.lr=0.0003",
                "loss.pos_weight=30",
                "loss.dice_weight=0.5",
                "loss.sparse_weight=0.1",
            ],
            "B2": [
                "optim.lr=0.0005",
                "loss.pos_weight=50",
                "loss.dice_weight=0.5",
                "loss.sparse_weight=0.1",
            ],
            "B3": [
                "optim.lr=0.0005",
                "loss.pos_weight=100",
                "loss.dice_weight=0.5",
                "loss.sparse_weight=0.1",
            ],
            "B4": [
                "optim.lr=0.0005",
                "loss.pos_weight=50",
                "loss.dice_weight=1.0",
                "loss.sparse_weight=0.05",
            ],
        }
        return [
            {"variant": k, "cmd": base_cmd(f"ssq_rebuild_{k}", 800, 150, 25, v)}
            for k, v in variants.items()
        ]
    if stage == "C":
        variants = {
            "C1_center005": [
                "model.ssq_fmt.routing.support_center=0.05",
                "model.ssq_fmt.routing.support_temperature=0.10",
                "model.ssq_fmt.routing.support_logit_weight=1.0",
            ],
            "C2_center010": [
                "model.ssq_fmt.routing.support_center=0.10",
                "model.ssq_fmt.routing.support_temperature=0.10",
                "model.ssq_fmt.routing.support_logit_weight=1.0",
            ],
            "C3_center020": [
                "model.ssq_fmt.routing.support_center=0.20",
                "model.ssq_fmt.routing.support_temperature=0.10",
                "model.ssq_fmt.routing.support_logit_weight=1.0",
            ],
            "C4_weight05": [
                "model.ssq_fmt.routing.support_center=0.10",
                "model.ssq_fmt.routing.support_temperature=0.10",
                "model.ssq_fmt.routing.support_logit_weight=0.5",
            ],
        }
        return [
            {"variant": k, "cmd": base_cmd(f"ssq_rebuild_{k}", 800, 150, 25, v)}
            for k, v in variants.items()
        ]
    if stage == "D":
        variants = {
            "D1": [
                "model.ssq_fmt.footprint.sigma_min_px=0.5",
                "model.ssq_fmt.footprint.sigma_max_px=4.0",
            ],
            "D2": [
                "model.ssq_fmt.footprint.sigma_min_px=1.0",
                "model.ssq_fmt.footprint.sigma_max_px=6.0",
            ],
            "D3": [
                "model.ssq_fmt.footprint.sigma_min_px=1.0",
                "model.ssq_fmt.footprint.sigma_max_px=8.0",
            ],
            "D4": [
                "model.ssq_fmt.footprint.sigma_min_px=2.0",
                "model.ssq_fmt.footprint.sigma_max_px=10.0",
            ],
        }
        return [
            {"variant": k, "cmd": base_cmd(f"ssq_rebuild_{k}", 800, 150, 25, v)}
            for k, v in variants.items()
        ]
    if stage == "E":
        variants = {
            "E1": [
                "data.query_sampling.trunk_uniform_ratio=0.6",
                "data.query_sampling.meas_proposal_ratio=0.4",
            ],
            "E2": [
                "data.query_sampling.trunk_uniform_ratio=0.5",
                "data.query_sampling.meas_proposal_ratio=0.5",
            ],
            "E3": [
                "data.query_sampling.trunk_uniform_ratio=0.4",
                "data.query_sampling.meas_proposal_ratio=0.6",
            ],
        }
        return [
            {"variant": k, "cmd": base_cmd(f"ssq_rebuild_{k}", 800, 150, 25, v)}
            for k, v in variants.items()
        ]
    if stage == "medium":
        return [
            {
                "variant": "medium_gate",
                "cmd": base_cmd("ssq_rebuild_medium_gate", 1200, 200, 40, []),
            }
        ]
    raise ValueError(f"unknown stage {stage!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["A", "B", "C", "D", "E", "medium"], required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--out_dir", default="outputs/ssq_fmt_rebuild/tuning")
    args = parser.parse_args()

    plan = stage_plan(args.stage)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / f"stage_{args.stage}_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2))
    print(f"wrote {plan_path}")
    for item in plan:
        print(" ".join(item["cmd"]))  # type: ignore[arg-type]
        if args.execute:
            subprocess.run(item["cmd"], check=True)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
