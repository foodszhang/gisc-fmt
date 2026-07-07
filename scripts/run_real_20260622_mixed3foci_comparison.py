#!/usr/bin/env python
"""Run non-FEM-prior baselines on the 20260622 real mixed3foci dataset.

The runner is sequential and resumable. FEM-prior methods are intentionally
excluded because this real dataset does not currently have the required FEM
stage1/prior assets.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs" / "real_20260622_mixed3foci_comparison"
RUN_ROOT = OUT_ROOT / "runs"
SMOKE_ROOT = OUT_ROOT / "smoke"
TEST_ROOT = OUT_ROOT / "test"
LOG_ROOT = OUT_ROOT / "logs"
STATUS_ROOT = OUT_ROOT / "status"
SUMMARY_ROOT = OUT_ROOT / "summary"
PYTHON_CMD = ["uv", "run", "python"] if shutil.which("uv") else [sys.executable]

EXP = "fmt_simgen_real_20260622_luoshu1_mixed3foci_main"
DATA_DIR = "/home/foods/pro/FMT-SimGen/data/real_20260622_luoshu1_manual_surface_mixed3foci_1k_20k"
REAL_SHAPE = "[318,614,136]"

METHODS: dict[str, dict[str, Any]] = {
    "pah2t_former": {
        "kind": "voxel",
        "overrides": [
            f"model.pah2t_former.output_shape={REAL_SHAPE}",
            f"model.pah2t_former.full_output_shape={REAL_SHAPE}",
        ],
    },
    "uhr_deepfmt": {
        "kind": "voxel",
        "overrides": [
            f"+model.uhr_deepfmt.full_output_shape={REAL_SHAPE}",
            f"+model.uhr_deepfmt.lowres_shape={REAL_SHAPE}",
        ],
    },
    "vox_dmrn": {"kind": "voxel", "overrides": []},
    "two_stage_deepfmt": {
        "kind": "voxel",
        "overrides": [f"model.two_stage_deepfmt.full_output_shape={REAL_SHAPE}"],
    },
    "fmt_reconnet": {
        "kind": "voxel",
        "overrides": [
            "model.fmt_reconnet.template_path=",
            "+model.fmt_reconnet.allow_template_fallback=true",
            "+model.fmt_reconnet.require_template_shape_match=false",
        ],
    },
    "pgdpnn": {
        "kind": "voxel",
        "overrides": [
            "model.pgdpnn.allow_template_fallback=true",
            "model.pgdpnn.require_template_shape_match=false",
        ],
    },
    "map_pgan": {"kind": "voxel", "overrides": []},
    "d2_recst": {"kind": "voxel", "overrides": []},
    "dspgn": {"kind": "voxel", "overrides": []},
}

EXCLUDED = {
    "fem2vox_unet": "Skipped by user request; also requires FEM/stage1 prior assets.",
    "stage1_unet": "FEM/stage1-prior method; prior assets are not present for this real dataset.",
    "stage1_fem": "FEM/stage1-prior method; prior assets are not present for this real dataset.",
    "stage1_to_voxel": "FEM/stage1-prior method; prior assets are not present for this real dataset.",
    "gaicn": "Requires FEM mesh/stage1 assets; excluded from this non-FEM-prior rerun.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "train", "test", "summary", "all"], default="smoke")
    parser.add_argument("--methods", nargs="*", choices=sorted(METHODS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--train_max_samples", type=int, default=800)
    parser.add_argument("--val_max_samples", type=int, default=100)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--chunk_size", type=int, default=65536)
    return parser.parse_args()


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def selected_methods(args: argparse.Namespace) -> list[str]:
    return args.methods or list(METHODS)


def run_dir(method: str) -> Path:
    return RUN_ROOT / method


def smoke_dir(method: str) -> Path:
    return SMOKE_ROOT / method


def log_path(name: str) -> Path:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    return LOG_ROOT / f"{name}.log"


def run_cmd(cmd: list[str], name: str, dry_run: bool = False) -> int:
    print("[RUN]", " ".join(cmd), flush=True)
    if dry_run:
        return 0
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    path = log_path(name)
    with path.open("a") as log:
        log.write(f"\n\n# {now()} {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        log.write(f"\n[returncode] {proc.returncode}\n")
        return int(proc.returncode)


def base_train_cmd(method: str, out_dir: Path, args: argparse.Namespace) -> list[str]:
    cfg = METHODS[method]
    return [
        *PYTHON_CMD,
        "train.py",
        "fit",
        f"model={method}",
        f"exp={EXP}",
        "data.dataset_type=fmt_simgen",
        f"paths.output_dir={out_dir}",
        "logger._target_=null",
        "data.batch_size=1",
        "data.eval_batch_size=1",
        f"data.num_workers={args.num_workers}",
        f"data.train_max_samples={args.train_max_samples}",
        f"data.val_max_samples={args.val_max_samples}",
        "trainer.accumulate_grad_batches=4",
        f"trainer.max_epochs={args.max_epochs}",
        f"trainer.check_val_every_n_epoch={args.check_val_every_n_epoch}",
        "callbacks.early_stopping=null",
        *cfg["overrides"],
    ]


def smoke(method: str, args: argparse.Namespace) -> int:
    cmd = base_train_cmd(method, smoke_dir(method), args)
    cmd.extend(
        [
            "data.train_max_samples=2",
            "data.val_max_samples=2",
            "data.num_workers=0",
            "+trainer.fast_dev_run=true",
            "callbacks.progress_bar=null",
        ]
    )
    return run_cmd(cmd, f"{method}_smoke", args.dry_run)


def latest_ckpt(method: str) -> Path | None:
    ckpt_dir = run_dir(method) / "checkpoints"
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        return None

    def key(path: Path) -> tuple[float, float]:
        match = re.search(r"val_dice=([0-9.]+)", path.name)
        score = float(match.group(1).rstrip(".")) if match else -1.0
        return score, path.stat().st_mtime

    return max(ckpts, key=key)


def train(method: str, args: argparse.Namespace) -> int:
    if latest_ckpt(method) is not None and not args.force:
        print(f"[SKIP] checkpoint exists for {method}: {latest_ckpt(method)}", flush=True)
        return 0
    return run_cmd(base_train_cmd(method, run_dir(method), args), f"{method}_train", args.dry_run)


def test(method: str, args: argparse.Namespace) -> int:
    ckpt = latest_ckpt(method)
    if ckpt is None:
        print(f"[SKIP] no checkpoint for {method}", flush=True)
        return 1
    save_dir = TEST_ROOT / method
    if (save_dir / "metrics.csv").exists() and not args.force:
        print(f"[SKIP] test exists for {method}: {save_dir}", flush=True)
        return 0
    cmd = [
        *PYTHON_CMD,
        "scripts/eval_full_volume_fmt_simgen.py",
        f"model={method}",
        f"exp={EXP}",
        "data.dataset_type=fmt_simgen",
        "--ckpt_path",
        str(ckpt),
        "--split",
        "val",
        "--threshold",
        "0.5",
        "--chunk_size",
        str(args.chunk_size),
        "--save_dir",
        str(save_dir),
        "--device",
        args.device,
    ]
    cmd.extend(METHODS[method]["overrides"])
    return run_cmd(cmd, f"{method}_test", args.dry_run)


def load_metric_csv(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        return {row["metric"]: row["value"] for row in csv.DictReader(f)}


def write_manifest(methods: list[str]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "updated_at": now(),
        "dataset": DATA_DIR,
        "exp": EXP,
        "real_shape": [318, 614, 136],
        "methods": [{"method": method, **METHODS[method]} for method in methods],
        "excluded": EXCLUDED,
    }
    (OUT_ROOT / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


def write_summary(methods: list[str]) -> None:
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in methods:
        metrics_path = TEST_ROOT / method / "metrics.csv"
        row: dict[str, Any] = {"method": method, "ckpt": str(latest_ckpt(method) or "")}
        if metrics_path.exists():
            row.update(load_metric_csv(metrics_path))
        rows.append(row)
    fields = sorted({key for row in rows for key in row})
    with (SUMMARY_ROOT / "val_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    methods = selected_methods(args)
    for path in (RUN_ROOT, SMOKE_ROOT, TEST_ROOT, LOG_ROOT, STATUS_ROOT, SUMMARY_ROOT):
        path.mkdir(parents=True, exist_ok=True)
    write_manifest(methods)

    if args.stage in {"smoke", "all"}:
        failed = []
        for method in methods:
            rc = smoke(method, args)
            (STATUS_ROOT / f"{method}_smoke.json").write_text(
                json.dumps({"method": method, "returncode": rc, "updated_at": now()}, indent=2)
            )
            if rc != 0:
                failed.append(method)
        if failed:
            raise SystemExit(f"Smoke failed for: {', '.join(failed)}")

    if args.stage in {"train", "all"}:
        for method in methods:
            rc = train(method, args)
            (STATUS_ROOT / f"{method}_train.json").write_text(
                json.dumps({"method": method, "returncode": rc, "updated_at": now()}, indent=2)
            )
            if rc != 0:
                raise SystemExit(f"Train failed for {method}")

    if args.stage in {"test", "all"}:
        for method in methods:
            rc = test(method, args)
            (STATUS_ROOT / f"{method}_test.json").write_text(
                json.dumps({"method": method, "returncode": rc, "updated_at": now()}, indent=2)
            )
            if rc != 0:
                raise SystemExit(f"Test failed for {method}")

    if args.stage in {"summary", "all"}:
        write_summary(methods)


if __name__ == "__main__":
    main()
