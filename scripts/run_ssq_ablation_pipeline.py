#!/usr/bin/env python
"""Serial train/eval runner for SSQ mainline ablations."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "outputs" / "fmt_simgen_v2_ssq_ablation" / "runs"
TEST_ROOT = ROOT / "outputs" / "fmt_simgen_v2_ssq_ablation" / "test300"
LOG_ROOT = ROOT / "logs" / "ssq_ablation"

EXPERIMENTS = [
    ("no_ptfa", "fmt_simgen_v2_ssq_no_ptfa"),
    ("no_canonical_reliability", "fmt_simgen_v2_ssq_no_canonical_reliability"),
    ("no_source_cue", "fmt_simgen_v2_ssq_no_source_cue"),
    ("no_center_aux", "fmt_simgen_v2_ssq_no_center_aux"),
    ("no_distance_aux", "fmt_simgen_v2_ssq_no_distance_aux"),
]

SKIP_EXPERIMENTS = {
    # Stopped by user request after reaching epoch 74 with best val_dice=0.7302.
    "no_source_cue",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--force_eval", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=65536)
    return parser.parse_args()


def run_cmd(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{time.strftime('%F %T')}] RUN {' '.join(cmd)}", flush=True)
    with log_path.open("a") as f:
        print(f"\n==== {time.strftime('%F %T')} RUN {' '.join(cmd)} ====", file=f, flush=True)
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)


def exp_max_epochs(exp: str) -> int:
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = compose(
            config_name="config",
            overrides=[f"exp={exp}", "data.dataset_type=fmt_simgen"],
        )
    return int(cfg.trainer.max_epochs)


def checkpoint_epoch(path: Path) -> int | None:
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    epoch = ckpt.get("epoch")
    return int(epoch) if epoch is not None else None


def best_checkpoint(run_dir: Path) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("epoch=*-val_dice=*.ckpt"))
    if not ckpts:
        last = ckpt_dir / "last.ckpt"
        if last.exists():
            return last
        raise FileNotFoundError(f"No checkpoints found under {ckpt_dir}")

    def score(path: Path) -> tuple[float, float]:
        match = re.search(r"val_dice=([0-9]+(?:\.[0-9]+)?)", path.name)
        val = float(match.group(1)) if match else -1.0
        return val, path.stat().st_mtime

    return max(ckpts, key=score)


def train(name: str, exp: str, force: bool) -> Path:
    run_dir = RUN_ROOT / name
    ckpt_dir = run_dir / "checkpoints"
    last_ckpt = ckpt_dir / "last.ckpt"
    resume_ckpt = None
    if ckpt_dir.exists() and any(ckpt_dir.glob("*.ckpt")) and not force:
        max_epochs = exp_max_epochs(exp)
        last_epoch = checkpoint_epoch(last_ckpt)
        if last_epoch is not None and last_epoch + 1 < max_epochs:
            resume_ckpt = last_ckpt
            print(
                f"[RESUME] train {name}: last epoch={last_epoch}, max_epochs={max_epochs}",
                flush=True,
            )
        else:
            print(f"[SKIP] train {name}: existing completed checkpoint", flush=True)
            return run_dir
    cmd = [
        sys.executable,
        "train.py",
        "fit",
        f"exp={exp}",
        "data.dataset_type=fmt_simgen",
        "data.num_workers=2",
        f"paths.output_dir={run_dir}",
        f"paths.checkpoint_dir={ckpt_dir}",
        f"paths.config_dir={run_dir / 'config'}",
    ]
    if resume_ckpt is not None:
        cmd.extend(
            [
                f"ckpt_path={resume_ckpt}",
                "ckpt_weights_only=false",
                "callbacks.early_stopping=null",
            ]
        )
    run_cmd(cmd, LOG_ROOT / f"{name}_train.log")
    return run_dir


def evaluate(name: str, exp: str, ckpt: Path, force: bool, chunk_size: int) -> None:
    test_dir = TEST_ROOT / name
    metrics_path = test_dir / "metrics_summary.json"
    components_path = test_dir / "components" / "component_summary.json"
    if metrics_path.exists() and components_path.exists() and not force:
        print(f"[SKIP] eval {name}: existing metrics and components", flush=True)
        return
    if not metrics_path.exists() or force:
        cmd = [
            sys.executable,
            "scripts/eval_full_volume_fmt_simgen.py",
            "--exp",
            exp,
            "--ckpt_path",
            str(ckpt),
            "--split",
            "test",
            "--threshold",
            "0.5",
            "--chunk_size",
            str(chunk_size),
            "--save_predictions",
            "--save_dir",
            str(test_dir),
        ]
        run_cmd(cmd, LOG_ROOT / f"{name}_test300.log")
    if not components_path.exists() or force:
        cmd = [
            sys.executable,
            "scripts/eval_components_fmt_simgen.py",
            "--eval_dir",
            str(test_dir),
            "--data_dir",
            "/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k",
            "--split",
            "test",
            "--save_dir",
            str(test_dir / "components"),
        ]
        run_cmd(cmd, LOG_ROOT / f"{name}_components.log")


def evaluation_complete(name: str) -> bool:
    test_dir = TEST_ROOT / name
    return (
        (test_dir / "metrics_summary.json").exists()
        and (test_dir / "components" / "component_summary.json").exists()
    )


def write_reuse_manifest() -> None:
    manifest = {
        "ssq_main": {
            "reused_from": "fmt_simgen_v2_e15_center_distance",
            "checkpoint": (
                "outputs/fmt_simgen_v2_e15_center_distance_precomputed/checkpoints/"
                "epoch=52-val_dice=0.7414.ckpt"
            ),
            "test300": (
                "outputs/fmt_simgen_v2_source_slots_eval/"
                "e15_center_distance_epoch52_test300"
            ),
            "reason": (
                "fmt_simgen_v2_ssq_main is E15 with explicit source_instance_decoder=false; "
                "E15 already uses measurement-derived source hypotheses, source_instance_cue, "
                "canonical reliability, and center/distance auxiliary supervision."
            ),
        },
        "source_slots_soft_union_ablation": {
            "checkpoint": (
                "outputs/gisc_fmt/fit/2026-06-12/12-37-02/checkpoints/"
                "epoch=10-val_dice=0.7232.ckpt"
            ),
            "test300": (
                "outputs/fmt_simgen_v2_source_slots_eval/"
                "source_slots_soft_union_epoch10_test300_corrected"
            ),
        },
    }
    out = ROOT / "outputs" / "fmt_simgen_v2_ssq_ablation" / "reuse_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))


def main() -> None:
    args = parse_args()
    for path in (RUN_ROOT, TEST_ROOT, LOG_ROOT):
        path.mkdir(parents=True, exist_ok=True)
    write_reuse_manifest()
    for name, exp in EXPERIMENTS:
        print(f"==== {time.strftime('%F %T')} {name} ({exp}) ====", flush=True)
        if name in SKIP_EXPERIMENTS:
            print(f"[SKIP] {name}: skipped by user request", flush=True)
            continue
        if evaluation_complete(name) and not args.force_train and not args.force_eval:
            print(f"[SKIP] {name}: existing test300 metrics and components", flush=True)
            continue
        run_dir = train(name, exp, force=args.force_train)
        ckpt = best_checkpoint(run_dir)
        print(f"[SELECT] {name}: {ckpt}", flush=True)
        evaluate(name, exp, ckpt, force=args.force_eval, chunk_size=args.chunk_size)
        print(f"==== {time.strftime('%F %T')} done {name} ====", flush=True)


if __name__ == "__main__":
    main()
