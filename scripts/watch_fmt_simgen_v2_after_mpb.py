#!/usr/bin/env python
"""Wait for the active E12-MPB run, then continue follow-up experiments."""

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
DATA_DIR = Path("/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k")
PYTHON_CMD = ["uv", "run", "python"] if shutil.which("uv") else [sys.executable]

CURRENT_NAME = "e12_mpb_auxnorm"
CURRENT_EXP = "fmt_simgen_v2_e12_mpb_auxnorm"
CURRENT_RUN_DIR = RUN_ROOT / "e12_mpb_auxnorm_finetune"
TVERSKY_NAME = "e12_mpb_auxnorm_tversky"
TVERSKY_EXP = "fmt_simgen_v2_e12_mpb_auxnorm_tversky"
CURRENT_MAX_EPOCHS = 60
TVERSKY_MAX_EPOCHS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll_seconds", type=int, default=300)
    parser.add_argument("--chunk_size", type=int, default=65536)
    parser.add_argument("--skip_tversky", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def clean_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def log(message: str) -> None:
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def hydra_value(value: Path | str) -> str:
    return str(value).replace("=", r"\=")


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


def last_checkpoint(run_dir: Path) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    candidates = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: (checkpoint_epoch(p) or -1, p.stat().st_mtime))[-1]


def best_val_checkpoint(run_dir: Path) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    scored: list[tuple[float, Path]] = []
    for path in ckpt_dir.glob("*.ckpt"):
        if "val_dice=" not in path.name:
            continue
        try:
            score = float(path.name.split("val_dice=")[1].split(".ckpt")[0])
        except ValueError:
            continue
        scored.append((score, path))
    if scored:
        return max(scored, key=lambda item: item[0])[1]
    return last_checkpoint(run_dir)


def run_cmd(cmd: list[str], log_name: str, dry_run: bool = False, attempts: int = 3) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{log_name}.log"
    log("RUN " + " ".join(cmd))
    if dry_run:
        return
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
            sleep_s = 120 * attempt
            log(f"WARN command failed; retrying in {sleep_s}s")
            time.sleep(sleep_s)


def wait_for_current(args: argparse.Namespace) -> None:
    while True:
        ckpt = last_checkpoint(CURRENT_RUN_DIR)
        epoch = checkpoint_epoch(ckpt)
        log(f"current {CURRENT_NAME}: last_ckpt={ckpt} epoch={epoch}")
        if epoch is not None and epoch >= CURRENT_MAX_EPOCHS - 1:
            return
        time.sleep(args.poll_seconds)


def selected_summary_path(name: str) -> Path:
    return SELECTION_ROOT / f"{name}_selection_summary.json"


def select_checkpoint(name: str, exp: str, run_dir: Path, args: argparse.Namespace) -> str:
    summary_path = selected_summary_path(name)
    if summary_path.exists():
        selected = json.loads(summary_path.read_text())["selected"]["ckpt"]
        log(f"SKIP select {name}: {selected}")
        return selected
    cmd = [
        *PYTHON_CMD,
        "scripts/select_candidate_ckpt_fmt_simgen.py",
        "--run_dir",
        str(run_dir),
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
    run_cmd(cmd, f"{name}_select", args.dry_run)
    if args.dry_run:
        return "<dry-run.ckpt>"
    produced = SELECTION_ROOT / f"{run_dir.name}_selection_summary.json"
    if produced.exists() and produced != summary_path:
        produced.replace(summary_path)
    return json.loads(summary_path.read_text())["selected"]["ckpt"]


def evaluate(name: str, exp: str, ckpt: str, args: argparse.Namespace) -> None:
    test_dir = TEST_ROOT / name
    if (test_dir / "metrics.csv").exists():
        log(f"SKIP test {name}: existing metrics")
    else:
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
        run_cmd(cmd, f"{name}_test", args.dry_run)

    if (test_dir / "component_summary.json").exists():
        log(f"SKIP components {name}: existing summary")
        return
    component_cmd = [
        *PYTHON_CMD,
        "scripts/eval_components_fmt_simgen.py",
        "--eval_dir",
        str(test_dir),
        "--data_dir",
        str(DATA_DIR),
        "--split",
        "test",
    ]
    run_cmd(component_cmd, f"{name}_components", args.dry_run)


def train_tversky(init_ckpt: str, args: argparse.Namespace) -> Path:
    run_dir = RUN_ROOT / TVERSKY_NAME
    complete_ckpt = last_checkpoint(run_dir)
    if (checkpoint_epoch(complete_ckpt) or -1) >= TVERSKY_MAX_EPOCHS - 1:
        log(f"SKIP train {TVERSKY_NAME}: completed {complete_ckpt}")
        return run_dir
    cmd = [
        *PYTHON_CMD,
        "train.py",
        "fit",
        "model=gisc_fmt",
        f"exp={TVERSKY_EXP}",
        "data.dataset_type=fmt_simgen",
        f"paths.output_dir={run_dir}",
        "logger._target_=null",
        "callbacks.early_stopping=null",
        "data.batch_size=2",
        "data.eval_batch_size=1",
        "data.num_queries=32768",
        "data.sample_num=32768",
        "data.eval_sample_num=32768",
        "trainer.accumulate_grad_batches=2",
        f"trainer.max_epochs={TVERSKY_MAX_EPOCHS}",
        "trainer.check_val_every_n_epoch=5",
        "callbacks.checkpoint.every_n_epochs=5",
        "callbacks.checkpoint.save_last=false",
        "+callbacks.last_checkpoint._target_=pytorch_lightning.callbacks.ModelCheckpoint",
        f"+callbacks.last_checkpoint.dirpath={run_dir / 'checkpoints'}",
        "+callbacks.last_checkpoint.save_top_k=0",
        "+callbacks.last_checkpoint.save_last=true",
        "+callbacks.last_checkpoint.every_n_epochs=1",
        "+callbacks.last_checkpoint.save_on_train_epoch_end=true",
    ]
    if complete_ckpt is not None:
        cmd.append(f"ckpt_path={hydra_value(complete_ckpt)}")
    else:
        cmd.extend(["ckpt_path=null", f"+model.finetune.init_from_ckpt={hydra_value(init_ckpt)}"])
    run_cmd(cmd, f"{TVERSKY_NAME}_train", args.dry_run)
    return run_dir


def summarize(args: argparse.Namespace) -> None:
    cmd = [*PYTHON_CMD, "scripts/summarize_fmt_simgen_v2_multisource.py"]
    run_cmd(cmd, "summary_after_mpb", args.dry_run)


def main() -> None:
    args = parse_args()
    for path in [RUN_ROOT, SELECTION_ROOT, TEST_ROOT, LOG_ROOT]:
        path.mkdir(parents=True, exist_ok=True)
    wait_for_current(args)
    current_ckpt = select_checkpoint(CURRENT_NAME, CURRENT_EXP, CURRENT_RUN_DIR, args)
    evaluate(CURRENT_NAME, CURRENT_EXP, current_ckpt, args)
    if not args.skip_tversky:
        init_ckpt = current_ckpt or str(best_val_checkpoint(CURRENT_RUN_DIR))
        tversky_run_dir = train_tversky(init_ckpt, args)
        tversky_ckpt = select_checkpoint(TVERSKY_NAME, TVERSKY_EXP, tversky_run_dir, args)
        evaluate(TVERSKY_NAME, TVERSKY_EXP, tversky_ckpt, args)
    summarize(args)
    log("done")


if __name__ == "__main__":
    main()
