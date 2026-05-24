#!/usr/bin/env python
"""Run the FMT-SimGen v2 GISC multi-source improvement experiments."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs" / "fmt_simgen_v2_multisource"
RUN_ROOT = OUT_ROOT / "runs"
SELECTION_ROOT = OUT_ROOT / "selection"
TEST_ROOT = OUT_ROOT / "test"
LOG_ROOT = OUT_ROOT / "logs"
PYTHON_CMD = ["uv", "run", "python"] if shutil.which("uv") else [sys.executable]

EXPERIMENTS = [
    ("e12_v2", "fmt_simgen_v2_3k_20k_gisc_e12"),
    ("e12_mpb", "fmt_simgen_v2_e12_mpb"),
    ("e12_mpb_tversky", "fmt_simgen_v2_e12_mpb_tversky"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stage", choices=["all", "train", "select", "test"], default="all")
    parser.add_argument("--chunk_size", type=int, default=65536)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def clean_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def run_cmd(cmd: list[str], log_name: str, dry_run: bool = False) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{log_name}.log"
    print("[RUN]", " ".join(cmd), flush=True)
    if dry_run:
        return
    with log_path.open("a") as f:
        f.write(f"\n\n# {time.strftime('%F %T')} {' '.join(cmd)}\n")
        f.flush()
        subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
            env=clean_env(),
        )


def run_dir(name: str) -> Path:
    return RUN_ROOT / name


def selected_summary_path(name: str) -> Path:
    return SELECTION_ROOT / f"{name}_selection_summary.json"


def train(name: str, exp: str, args: argparse.Namespace) -> None:
    ckpt_dir = run_dir(name) / "checkpoints"
    if ckpt_dir.exists() and list(ckpt_dir.glob("*.ckpt")) and not args.force:
        print(f"[SKIP] train {name}: existing checkpoints", flush=True)
        return
    cmd = [
        *PYTHON_CMD,
        "train.py",
        "fit",
        "model=gisc_fmt",
        f"exp={exp}",
        "data.dataset_type=fmt_simgen",
        f"paths.output_dir={run_dir(name)}",
        "logger._target_=null",
        "callbacks.early_stopping=null",
        "data.batch_size=2",
        "data.eval_batch_size=1",
        "data.num_queries=32768",
        "data.sample_num=32768",
        "data.eval_sample_num=32768",
        "trainer.accumulate_grad_batches=2",
        "trainer.max_epochs=60",
    ]
    run_cmd(cmd, f"{name}_train", dry_run=args.dry_run)


def select(name: str, exp: str, args: argparse.Namespace) -> str:
    summary_path = selected_summary_path(name)
    if summary_path.exists() and not args.force:
        return json.loads(summary_path.read_text())["selected"]["ckpt"]
    cmd = [
        *PYTHON_CMD,
        "scripts/select_candidate_ckpt_fmt_simgen.py",
        "--run_dir",
        str(run_dir(name)),
        "--exp",
        exp,
        "--save_root",
        str(SELECTION_ROOT),
        "--val50_max_samples",
        "300",
        "--run_val200",
        "model=gisc_fmt",
        "data.dataset_type=fmt_simgen",
    ]
    run_cmd(cmd, f"{name}_select", dry_run=args.dry_run)
    if args.dry_run:
        return "<dry-run.ckpt>"
    produced = SELECTION_ROOT / f"{run_dir(name).name}_selection_summary.json"
    if produced.exists() and produced != summary_path:
        produced.replace(summary_path)
    return json.loads(summary_path.read_text())["selected"]["ckpt"]


def test(name: str, exp: str, ckpt: str, args: argparse.Namespace) -> None:
    test_dir = TEST_ROOT / name
    if (test_dir / "metrics.csv").exists() and not args.force:
        print(f"[SKIP] test {name}: existing metrics", flush=True)
        return
    cmd = [
        *PYTHON_CMD,
        "scripts/eval_full_volume_fmt_simgen.py",
        "model=gisc_fmt",
        f"exp={exp}",
        "data.dataset_type=fmt_simgen",
        "--ckpt_path",
        ckpt,
        "--split",
        "test",
        "--threshold",
        "0.5",
        "--chunk_size",
        str(args.chunk_size),
        "--save_dir",
        str(test_dir),
        "--save_predictions",
    ]
    run_cmd(cmd, f"{name}_test", dry_run=args.dry_run)
    component_cmd = [
        *PYTHON_CMD,
        "scripts/eval_components_fmt_simgen.py",
        "--eval_dir",
        str(test_dir),
        "--data_dir",
        "/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k",
        "--split",
        "test",
    ]
    run_cmd(component_cmd, f"{name}_components", dry_run=args.dry_run)


def main() -> None:
    args = parse_args()
    for path in [RUN_ROOT, SELECTION_ROOT, TEST_ROOT, LOG_ROOT]:
        path.mkdir(parents=True, exist_ok=True)
    for name, exp in EXPERIMENTS:
        print(f"==== {time.strftime('%F %T')} {name} {exp} ====", flush=True)
        if args.stage in {"all", "train"}:
            train(name, exp, args)
        ckpt = None
        if args.stage in {"all", "select", "test"}:
            ckpt = select(name, exp, args)
        if args.stage in {"all", "test"}:
            if ckpt is None:
                ckpt = select(name, exp, args)
            test(name, exp, ckpt, args)
        print(f"==== {time.strftime('%F %T')} done {name} ====", flush=True)


if __name__ == "__main__":
    main()
