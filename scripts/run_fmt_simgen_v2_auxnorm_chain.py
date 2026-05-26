#!/usr/bin/env python
"""Run aux-normalized MPB projection warmup followed by full training."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs" / "fmt_simgen_v2_multisource"
RUN_DIR = OUT_ROOT / "runs" / "e12_mpb_auxnorm_fastwarmup"
FIXED_FULL_RUN_DIR = OUT_ROOT / "runs" / "e12_mpb_auxnorm_finetune"
LOG_DIR = OUT_ROOT / "logs"
PYTHON_CMD = ["uv", "run", "python"] if shutil.which("uv") else [sys.executable]


def clean_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def run(cmd: list[str], log_path: Path) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--skip_warmup",
        action="store_true",
        help="Start full MPB training from an existing warmup checkpoint as weights only.",
    )
    parser.add_argument(
        "--resume_full",
        type=Path,
        default=None,
        help="Resume full MPB training from a checkpoint with optimizer/scheduler state.",
    )
    return parser.parse_args()


def maybe_run(cmd: list[str], log_path: Path, dry_run: bool) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    if not dry_run:
        run(cmd, log_path)


def hydra_value(value: Path | str) -> str:
    return str(value).replace("=", r"\=")


def main() -> None:
    args = parse_args()
    log_path = LOG_DIR / "e12_mpb_auxnorm_chain.log"
    warmup_cmd = [
        *PYTHON_CMD,
        "train.py",
        "fit",
        "model=gisc_fmt",
        "exp=fmt_simgen_v2_e12_mpb_auxnorm_auxpretrain",
        "data.dataset_type=fmt_simgen",
        f"paths.output_dir={RUN_DIR}",
        "logger._target_=null",
        "callbacks.early_stopping=null",
        "data.batch_size=2",
        "data.eval_batch_size=1",
        "data.num_workers=1",
        "data.persistent_workers=false",
        "data.prefetch_factor=1",
        "data.num_queries=1",
        "data.sample_num=1",
        "data.eval_sample_num=1",
        "trainer.accumulate_grad_batches=2",
        "trainer.max_epochs=5",
        "+trainer.limit_val_batches=0",
        "callbacks.checkpoint.every_n_epochs=1",
        "callbacks.checkpoint.monitor=null",
        "callbacks.checkpoint.save_top_k=0",
        "callbacks.checkpoint.save_last=true",
        "callbacks.checkpoint.save_on_train_epoch_end=true",
    ]
    if not args.skip_warmup and args.resume_full is None:
        maybe_run(warmup_cmd, log_path, args.dry_run)

    last_ckpt = RUN_DIR / "checkpoints" / "last.ckpt"
    full_run_dir = FIXED_FULL_RUN_DIR if args.skip_warmup or args.resume_full else RUN_DIR
    full_cmd = [
        *PYTHON_CMD,
        "train.py",
        "fit",
        "model=gisc_fmt",
        "exp=fmt_simgen_v2_e12_mpb_auxnorm",
        "data.dataset_type=fmt_simgen",
        f"paths.output_dir={full_run_dir}",
        "logger._target_=null",
        "callbacks.early_stopping=null",
        "data.batch_size=2",
        "data.eval_batch_size=1",
        "data.num_queries=32768",
        "data.sample_num=32768",
        "data.eval_sample_num=32768",
        "trainer.accumulate_grad_batches=2",
        "trainer.max_epochs=60",
        "trainer.check_val_every_n_epoch=5",
        "callbacks.checkpoint.every_n_epochs=5",
        "callbacks.checkpoint.save_last=false",
        "+callbacks.last_checkpoint._target_=pytorch_lightning.callbacks.ModelCheckpoint",
        f"+callbacks.last_checkpoint.dirpath={full_run_dir / 'checkpoints'}",
        "+callbacks.last_checkpoint.save_top_k=0",
        "+callbacks.last_checkpoint.save_last=true",
        "+callbacks.last_checkpoint.every_n_epochs=1",
        "+callbacks.last_checkpoint.save_on_train_epoch_end=true",
    ]
    if args.resume_full is not None:
        full_cmd.append(f"ckpt_path={hydra_value(args.resume_full)}")
    else:
        full_cmd.extend(
            [
                "ckpt_path=null",
                f"+model.finetune.init_from_ckpt={last_ckpt}",
            ]
        )
    maybe_run(full_cmd, log_path, args.dry_run)


if __name__ == "__main__":
    main()
