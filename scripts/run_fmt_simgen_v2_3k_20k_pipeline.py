#!/usr/bin/env python
"""Run the full FMT-SimGen v2 3k 20k experiment pipeline.

The script is intentionally resumable: existing checkpoints, selection summaries, and test
metrics are reused unless --force is set.
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
OUT_ROOT = ROOT / "outputs" / "fmt_simgen_v2_3k_20k"
LOG_ROOT = OUT_ROOT / "logs"
RUN_ROOT = OUT_ROOT / "runs"
SMOKE_ROOT = OUT_ROOT / "smoke"
SELECTION_ROOT = OUT_ROOT / "selection"
TEST_ROOT = OUT_ROOT / "test"
SUMMARY_ROOT = OUT_ROOT / "summary"
STATUS_ROOT = OUT_ROOT / "status"
PYTHON_CMD = ["uv", "run", "python"] if shutil.which("uv") else [sys.executable]

QUERY_METHODS = [
    ("gisc_fmt", "fmt_simgen_v2_3k_20k_gisc_e12"),
]
VOXEL_METHODS = [
    ("fem2vox_unet", "fmt_simgen_v2_3k_20k_common"),
    ("uhr_deepfmt", "fmt_simgen_v2_3k_20k_common"),
    ("vox_dmrn", "fmt_simgen_v2_3k_20k_common"),
    ("two_stage_deepfmt", "fmt_simgen_v2_3k_20k_common"),
    ("fmt_reconnet", "fmt_simgen_v2_3k_20k_common"),
    ("pgdpnn", "fmt_simgen_v2_3k_20k_common"),
    ("map_pgan", "fmt_simgen_v2_3k_20k_common"),
    ("d2_recst", "fmt_simgen_v2_3k_20k_common"),
    ("dspgn", "fmt_simgen_v2_3k_20k_common"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["all", "smoke", "train", "select", "test", "summary"],
        default="all",
    )
    parser.add_argument("--methods", nargs="*", help="Optional method subset")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip_smoke", action="store_true", default=True)
    parser.add_argument("--with_smoke", dest="skip_smoke", action="store_false")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--chunk_size", type=int, default=65536)
    parser.add_argument("--train_num_queries", type=int, default=32768)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--accumulate_grad_batches", type=int, default=2)
    parser.add_argument("--no_save_predictions", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def methods() -> list[tuple[str, str, str]]:
    return [(m, e, "query") for m, e in QUERY_METHODS] + [
        (m, e, "voxel") for m, e in VOXEL_METHODS
    ]


def run_dir(method: str) -> Path:
    return RUN_ROOT / method


def smoke_run_dir(method: str) -> Path:
    return SMOKE_ROOT / method


def log_path(tag: str) -> Path:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    return LOG_ROOT / f"{tag}.log"


def run_cmd(cmd: list[str], tag: str, dry_run: bool = False) -> None:
    print("[RUN]", " ".join(cmd))
    if dry_run:
        return
    log_file = log_path(tag)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with log_file.open("a") as f:
        f.write(f"\n\n# {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        f.flush()
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True, env=env)


def has_trained(method: str) -> bool:
    marker = STATUS_ROOT / f"{method}_trained.json"
    if not marker.exists():
        return False
    ckpt_dir = run_dir(method) / "checkpoints"
    if not ((ckpt_dir / "last.ckpt").exists() or bool(list(ckpt_dir.glob("*.ckpt")))):
        return False
    config_path = run_dir(method) / "config" / "config.yaml"
    return is_full_train_config(config_path)


def is_full_train_config(config_path: Path) -> bool:
    if not config_path.exists():
        return False
    text = config_path.read_text()
    required_markers = [
        "sample_num: 32768",
        "num_queries: 32768",
        "batch_size: 2",
        "max_epochs: 60",
        "accumulate_grad_batches: 2",
    ]
    return all(marker in text for marker in required_markers)


def latest_last_ckpt(method: str) -> Path | None:
    ckpts = list((run_dir(method) / "checkpoints").glob("last*.ckpt"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda path: path.stat().st_mtime)


def write_trained_marker(method: str, exp: str) -> None:
    STATUS_ROOT.mkdir(parents=True, exist_ok=True)
    last_ckpt = latest_last_ckpt(method) or (run_dir(method) / "checkpoints" / "last.ckpt")
    marker = {
        "method": method,
        "exp": exp,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir(method)),
        "last_ckpt": str(last_ckpt),
    }
    (STATUS_ROOT / f"{method}_trained.json").write_text(json.dumps(marker, indent=2))


def train(
    method: str,
    exp: str,
    args: argparse.Namespace,
    dry_run: bool = False,
    smoke: bool = False,
) -> None:
    out_dir = smoke_run_dir(method) if smoke else run_dir(method)
    cmd = [
        *PYTHON_CMD,
        "train.py",
        "fit",
        f"model={method}",
        f"exp={exp}",
        "data.dataset_type=fmt_simgen",
        f"paths.output_dir={out_dir}",
        "logger._target_=null",
        f"data.batch_size={args.train_batch_size}",
        f"data.num_queries={args.train_num_queries}",
        f"data.sample_num={args.train_num_queries}",
        f"data.eval_sample_num={args.train_num_queries}",
        f"trainer.accumulate_grad_batches={args.accumulate_grad_batches}",
        "callbacks.early_stopping=null",
    ]
    config_path = run_dir(method) / "config" / "config.yaml"
    last_ckpt = latest_last_ckpt(method) if is_full_train_config(config_path) else None
    if not smoke and last_ckpt is not None and not args.force:
        cmd.append(f"ckpt_path={last_ckpt}")
    if smoke:
        cmd.extend(
            [
                "+trainer.fast_dev_run=true",
                "data.train_max_samples=2",
                "data.val_max_samples=2",
                "data.num_queries=512",
                "data.sample_num=512",
                "data.eval_sample_num=512",
                "data.num_workers=0",
                "callbacks.progress_bar=null",
            ]
        )
    run_cmd(cmd, f"{method}_{'smoke' if smoke else 'train'}", dry_run=dry_run)
    if not smoke and not dry_run:
        write_trained_marker(method, exp)


def val_dice_from_name(path: Path) -> float | None:
    match = re.search(r"val_dice=([0-9]+(?:\.[0-9]+)?)", path.name)
    return float(match.group(1)) if match else None


def select_voxel_ckpt(method: str) -> dict[str, Any]:
    ckpt_dir = run_dir(method) / "checkpoints"
    ckpts = [p for p in ckpt_dir.glob("*.ckpt") if p.name != "last.ckpt"]
    scored = [(val_dice_from_name(p), p) for p in ckpts]
    scored = [(score, p) for score, p in scored if score is not None]
    if scored:
        score, path = max(scored, key=lambda item: item[0])
        return {"method": method, "ckpt": str(path), "selection_metric": "val_dice", "score": score}
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return {"method": method, "ckpt": str(last), "selection_metric": "last", "score": None}
    raise FileNotFoundError(f"No checkpoint for {method}: {ckpt_dir}")


def select_query_ckpt(method: str, exp: str, dry_run: bool = False, force: bool = False) -> dict:
    summary_path = SELECTION_ROOT / f"{method}_selection_summary.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text())
    cmd = [
        *PYTHON_CMD,
        "scripts/select_candidate_ckpt_fmt_simgen.py",
        "--run_dir",
        str(run_dir(method)),
        "--exp",
        exp,
        "--save_root",
        str(SELECTION_ROOT),
        "--val50_max_samples",
        "300",
        "--run_val200",
        "model=" + method,
        "data.dataset_type=fmt_simgen",
    ]
    run_cmd(cmd, f"{method}_select", dry_run=dry_run)
    produced = SELECTION_ROOT / f"{run_dir(method).name}_selection_summary.json"
    if produced.exists() and produced != summary_path:
        produced.replace(summary_path)
    return json.loads(summary_path.read_text()) if summary_path.exists() else {}


def selected_ckpt(method: str, exp: str, kind: str, dry_run: bool, force: bool) -> dict[str, Any]:
    if dry_run:
        return {
            "method": method,
            "ckpt": f"<selected-{method}.ckpt>",
            "selection_metric": "dry_run",
            "score": None,
        }
    if kind == "query":
        summary = select_query_ckpt(method, exp, dry_run=dry_run, force=force)
        selected = summary.get("selected", {})
        return {
            "method": method,
            "ckpt": selected.get("ckpt"),
            "selection_metric": "candidate_dice",
            "score": selected.get("candidate_dice"),
        }
    return select_voxel_ckpt(method)


def run_test(method: str, exp: str, ckpt: str, dry_run: bool, force: bool, args) -> None:
    save_dir = TEST_ROOT / method
    if (save_dir / "metrics.csv").exists() and not force:
        print(f"[SKIP] test exists for {method}: {save_dir}")
        return
    cmd = [
        *PYTHON_CMD,
        "scripts/eval_full_volume_fmt_simgen.py",
        f"model={method}",
        f"exp={exp}",
        "data.dataset_type=fmt_simgen",
        "--ckpt_path",
        str(ckpt),
        "--split",
        "test",
        "--threshold",
        "0.5",
        "--chunk_size",
        str(args.chunk_size),
        "--save_dir",
        str(save_dir),
        "--device",
        args.device,
    ]
    if not args.no_save_predictions:
        cmd.append("--save_predictions")
    run_cmd(cmd, f"{method}_test", dry_run=dry_run)


def load_metric_csv(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        return {row["metric"]: row["value"] for row in csv.DictReader(f)}


def write_summary(selection_rows: list[dict[str, Any]]) -> None:
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    by_num_rows = []
    by_depth_rows = []
    for row in selection_rows:
        method = row["method"]
        metrics_path = TEST_ROOT / method / "metrics.csv"
        grouped_path = TEST_ROOT / method / "metrics_grouped.csv"
        if metrics_path.exists():
            metrics = load_metric_csv(metrics_path)
            all_rows.append({"method": method, **row, **metrics})
        if grouped_path.exists():
            with grouped_path.open(newline="") as f:
                for grow in csv.DictReader(f):
                    out = {"method": method, **grow}
                    if grow.get("group_key") == "num_foci":
                        by_num_rows.append(out)
                    elif grow.get("group_key") == "depth_tier":
                        by_depth_rows.append(out)

    def write(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        fields = sorted({key for row in rows for key in row})
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write(SUMMARY_ROOT / "all_methods_metrics.csv", all_rows)
    write(SUMMARY_ROOT / "all_methods_by_num_foci.csv", by_num_rows)
    write(SUMMARY_ROOT / "all_methods_by_depth_tier.csv", by_depth_rows)
    write(SUMMARY_ROOT / "selection_summary.csv", selection_rows)


def write_manifest(selected: list[tuple[str, str, str]]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": "/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k",
        "split_manifest": (
            "/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k/"
            "splits/split_manifest_v2_3k_20k.json"
        ),
        "methods": [{"method": m, "exp": e, "kind": k} for m, e, k in selected],
    }
    (OUT_ROOT / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    args = parse_args()
    selected = methods()
    if args.methods:
        wanted = set(args.methods)
        selected = [item for item in selected if item[0] in wanted]
    write_manifest(selected)

    if args.stage in {"all", "smoke"} and not args.skip_smoke:
        for method, exp, _kind in selected:
            train(method, exp, args, dry_run=args.dry_run, smoke=True)

    if args.stage in {"all", "train"}:
        for method, exp, _kind in selected:
            if has_trained(method) and not args.force:
                print(f"[SKIP] checkpoints exist for {method}")
                continue
            train(method, exp, args, dry_run=args.dry_run, smoke=False)

    selection_rows: list[dict[str, Any]] = []
    if args.stage in {"all", "select", "test", "summary"}:
        for method, exp, kind in selected:
            row = selected_ckpt(method, exp, kind, dry_run=args.dry_run, force=args.force)
            selection_rows.append(row)

    if args.stage in {"all", "test"}:
        for method, exp, _kind in selected:
            row = next(r for r in selection_rows if r["method"] == method)
            if not row.get("ckpt"):
                raise RuntimeError(f"No selected checkpoint for {method}")
            run_test(method, exp, str(row["ckpt"]), args.dry_run, args.force, args)

    if args.stage in {"all", "summary"}:
        write_summary(selection_rows)


if __name__ == "__main__":
    main()
