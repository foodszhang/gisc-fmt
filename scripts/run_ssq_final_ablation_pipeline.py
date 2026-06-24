#!/usr/bin/env python
"""Run final SSQ-FMT full/ablation training jobs serially."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

VARIANTS = {
    "full": "fmt_simgen_v2_ssq_final",
    "post_aggregation": "fmt_simgen_v2_ssq_ablate_post_aggregation",
    "shared_fusion": "fmt_simgen_v2_ssq_ablate_shared_fusion",
    "assignment_only": "fmt_simgen_v2_ssq_ablate_assignment_only",
    "fixed_footprint": "fmt_simgen_v2_ssq_ablate_fixed_footprint",
}


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["gate", "full"], required=True)
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=["full"])
    parser.add_argument("--output-root", default="outputs/ssq_fmt_final")
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.output_root)
    for variant in args.variants:
        exp = VARIANTS[variant]
        out_dir = root / variant
        if args.max_epochs is not None:
            epochs = args.max_epochs
        else:
            epochs = 5 if args.stage == "gate" else 80
        cmd = [
            "uv",
            "run",
            "python",
            "train.py",
            "fit",
            "model=ssq_fmt",
            f"exp={exp}",
            "data.dataset_type=fmt_simgen",
            f"paths.output_dir={out_dir}",
            f"trainer.max_epochs={epochs}",
        ]
        if args.stage == "gate":
            cmd.extend(
                [
                    "data.train_max_samples=200",
                    "data.val_max_samples=50",
                    "+trainer.limit_train_batches=1.0",
                    "+trainer.limit_val_batches=1.0",
                ]
            )
        _run(cmd)


if __name__ == "__main__":
    main()
