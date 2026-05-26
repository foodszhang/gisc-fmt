#!/usr/bin/env python
"""Wait for E13-MSQ training, then run selection and test evaluation."""

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
RUN_DIR = OUT_ROOT / "runs" / "e13_msq_from_mpb"
SELECTION_ROOT = OUT_ROOT / "selection"
TEST_DIR = OUT_ROOT / "test" / "e13_msq_from_mpb"
LOG_ROOT = OUT_ROOT / "logs"
DATA_DIR = Path("/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k")
PYTHON_CMD = ["uv", "run", "python"] if shutil.which("uv") else [sys.executable]
NAME = "e13_msq_from_mpb"
EXP = "fmt_simgen_v2_e13_msq"
MAX_EPOCHS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll_seconds", type=int, default=300)
    parser.add_argument("--chunk_size", type=int, default=65536)
    return parser.parse_args()


def clean_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def log(message: str) -> None:
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def checkpoint_epoch(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    try:
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        log(f"WARN could not read checkpoint epoch from {path}: {exc}")
        return None
    epoch = ckpt.get("epoch")
    return int(epoch) if epoch is not None else None


def last_checkpoint() -> Path | None:
    ckpt_dir = RUN_DIR / "checkpoints"
    if not ckpt_dir.exists():
        return None
    candidates = list(ckpt_dir.glob("*.ckpt"))
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: (checkpoint_epoch(p) or -1, p.stat().st_mtime))[-1]


def run_cmd(cmd: list[str], log_name: str, attempts: int = 3) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{log_name}.log"
    log("RUN " + " ".join(cmd))
    for attempt in range(1, attempts + 1):
        try:
            with log_path.open("a") as f:
                f.write(
                    f"\n\n# {time.strftime('%F %T')} attempt={attempt}/{attempts} "
                    f"{' '.join(cmd)}\n"
                )
                f.flush()
                subprocess.run(
                    cmd,
                    cwd=ROOT,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=True,
                    env=clean_env(),
                )
            return
        except subprocess.CalledProcessError:
            if attempt >= attempts:
                raise
            time.sleep(120 * attempt)


def wait_for_training(args: argparse.Namespace) -> None:
    while True:
        ckpt = last_checkpoint()
        epoch = checkpoint_epoch(ckpt)
        log(f"e13 last_ckpt={ckpt} epoch={epoch}")
        if epoch is not None and epoch >= MAX_EPOCHS - 1:
            return
        time.sleep(args.poll_seconds)


def select_checkpoint() -> str:
    summary_path = SELECTION_ROOT / f"{NAME}_selection_summary.json"
    if summary_path.exists():
        selected = json.loads(summary_path.read_text())["selected"]["ckpt"]
        log(f"SKIP select {NAME}: {selected}")
        return selected
    cmd = [
        *PYTHON_CMD,
        "scripts/select_candidate_ckpt_fmt_simgen.py",
        "--run_dir",
        str(RUN_DIR),
        "--exp",
        EXP,
        "--save_root",
        str(SELECTION_ROOT),
        "--val50_max_samples",
        "300",
        "--run_val200",
        "model=gisc_fmt",
        "data.dataset_type=fmt_simgen",
    ]
    run_cmd(cmd, f"{NAME}_select")
    return json.loads(summary_path.read_text())["selected"]["ckpt"]


def evaluate(ckpt: str, args: argparse.Namespace) -> None:
    if (TEST_DIR / "metrics.csv").exists():
        log(f"SKIP test {NAME}: existing metrics")
    else:
        run_cmd(
            [
                *PYTHON_CMD,
                "scripts/eval_full_volume_fmt_simgen.py",
                "model=gisc_fmt",
                f"exp={EXP}",
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
                str(TEST_DIR),
                "--save_predictions",
            ],
            f"{NAME}_test",
        )
    if (TEST_DIR / "component_summary.json").exists():
        log(f"SKIP components {NAME}: existing summary")
        return
    run_cmd(
        [
            *PYTHON_CMD,
            "scripts/eval_components_fmt_simgen.py",
            "--eval_dir",
            str(TEST_DIR),
            "--data_dir",
            str(DATA_DIR),
            "--split",
            "test",
        ],
        f"{NAME}_components",
    )


def main() -> None:
    args = parse_args()
    wait_for_training(args)
    ckpt = select_checkpoint()
    evaluate(ckpt, args)
    run_cmd([*PYTHON_CMD, "scripts/summarize_fmt_simgen_v2_multisource.py"], "summary_after_e13")
    log("done")


if __name__ == "__main__":
    main()
