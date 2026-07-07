"""Run current non-GISC comparison baselines on the E14 FMT-SimGen v2 protocol.

The runner is intentionally sequential and resumable. It records every command,
checkpoint, metric summary, and unavailable method in a manifest so later
verification does not depend on shell history.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path("/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k")
OUT_ROOT = ROOT / "outputs" / "fmt_simgen_v2_e14_comparison" / "baselines"
RUN_ROOT = OUT_ROOT / "runs"
TEST_ROOT = OUT_ROOT / "test"
LOG_ROOT = OUT_ROOT / "logs"
MANIFEST = OUT_ROOT / "checkpoint_manifest.json"
SUMMARY_CSV = OUT_ROOT / "summary.csv"

EXP = "fmt_simgen_v2_e14_source_field"
CURRENT_METHOD = "gisc_fmt"

TRAIN_EPOCHS = int(os.environ.get("BASELINE_EPOCHS", "20"))

TRAINABLE = [
    "stage1_unet",
    "cnn3d_baseline",
    "transunet3d_baseline",
    "pah2t_former",
    "two_stage_deepfmt",
    "pgdpnn",
    "fmt_reconnet",
    "map_pgan",
    "d2_recst",
    "dspgn",
]

ANALYTIC = [
    "stage1_fem",
    "stage1_to_voxel",
    "tikhonov_fem",
    "l1_fem",
    "elasticnet_fem",
    "fista_fem",
    "stomp_fem",
]

UNAVAILABLE = {
    "gisc_fmt": "Current method; excluded by user request.",
    "uhr_deepfmt": "Full-volume UHR OOMs on this 32GB GPU; ROI/lowres config not finalized.",
    "vox_dmrn": "Full-volume VoxDMRN is too large for formal rerun; vox_dmrn_lite not finalized.",
    "gaicn": "Requires trained FEM graph unrolling checkpoint; Stage1 assets are being generated first.",
}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {
        "created_at": now(),
        "updated_at": now(),
        "dataset": str(DATA_DIR),
        "exp": EXP,
        "current_method": CURRENT_METHOD,
        "train_epochs": TRAIN_EPOCHS,
        "unavailable": UNAVAILABLE,
        "stage1_generation": {},
        "methods": [],
    }


def save_manifest(manifest: dict) -> None:
    manifest["updated_at"] = now()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(MANIFEST)


def upsert_method(record: dict) -> None:
    manifest = load_manifest()
    methods = [m for m in manifest.get("methods", []) if m.get("method") != record["method"]]
    methods.append(record)
    manifest["methods"] = methods
    save_manifest(manifest)


def update_stage1(record: dict) -> None:
    manifest = load_manifest()
    manifest["stage1_generation"] = record
    save_manifest(manifest)


def run_logged(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        rc = proc.wait()
        log.write(f"\n[returncode] {rc}\n")
    return int(rc)


def stage1_counts() -> dict[str, int]:
    sample_root = DATA_DIR / "samples"
    sample_dirs = list(sample_root.glob("sample_*"))
    return {
        "samples": len(sample_dirs),
        "stage1_mesh": sum((p / "stage1_mesh.npy").exists() for p in sample_dirs),
        "stage1_voxel": sum((p / "stage1_voxel.npy").exists() for p in sample_dirs),
    }


def ensure_stage1_assets() -> bool:
    counts = stage1_counts()
    log_path = LOG_ROOT / "generate_stage1_fem_assets.log"
    checkpoint = "/home/foods/pro/DU2Vox/runs/stage1_uniform_1000_20k/checkpoints/best.pth"
    config = "/home/foods/pro/DU2Vox/configs/stage1/uniform_1000_20k.yaml"
    record = {
        "status": "existing" if counts["stage1_voxel"] == counts["samples"] else "started",
        "counts_before": counts,
        "checkpoint": checkpoint,
        "config": config,
        "log": str(log_path),
    }
    update_stage1(record)
    if counts["stage1_voxel"] == counts["samples"] and counts["samples"] > 0:
        return True
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/generate_stage1_fem_assets.py",
        "--data-dir",
        str(DATA_DIR),
        "--checkpoint",
        checkpoint,
        "--config",
        config,
        "--splits",
        "train",
        "val",
        "test",
        "--batch-size",
        "16",
        "--overwrite",
    ]
    rc = run_logged(cmd, log_path)
    counts_after = stage1_counts()
    record.update(
        {
            "status": "completed" if rc == 0 else "failed",
            "returncode": rc,
            "counts_after": counts_after,
        }
    )
    update_stage1(record)
    return rc == 0


def best_checkpoint(run_dir: Path) -> str | None:
    ckpts = glob.glob(str(run_dir / "checkpoints" / "*.ckpt"))
    if not ckpts:
        return None

    def key(path: str) -> tuple[float, float]:
        name = Path(path).name
        m = re.search(r"val_dice=([0-9.]+)", name)
        score = float(m.group(1).rstrip(".")) if m else -1.0
        return score, Path(path).stat().st_mtime

    return max(ckpts, key=key)


def metrics_summary(test_dir: Path) -> dict | None:
    path = test_dir / "metrics_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def eval_model(method: str, ckpt: str | None, test_dir: Path, log_path: Path) -> int:
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/eval_full_volume_fmt_simgen.py",
        f"model={method}",
        f"exp={EXP}",
        "data.dataset_type=fmt_simgen",
        "--ckpt_path",
        ckpt or "null",
        "--split",
        "test",
        "--threshold",
        "0.5",
        "--chunk_size",
        "65536",
        "--save_dir",
        str(test_dir),
        "--device",
        "cuda",
    ]
    return run_logged(cmd, log_path)


def train_model(method: str, run_dir: Path, log_path: Path) -> int:
    cmd = [
        "uv",
        "run",
        "python",
        "train.py",
        "fit",
        f"model={method}",
        f"exp={EXP}",
        "data.dataset_type=fmt_simgen",
        "logger=none",
        "data.batch_size=1",
        "data.eval_batch_size=1",
        "data.num_workers=2",
        "trainer.accumulate_grad_batches=4",
        f"trainer.max_epochs={TRAIN_EPOCHS}",
        "trainer.check_val_every_n_epoch=5",
        "callbacks.early_stopping=null",
        f"paths.output_dir={run_dir}",
    ]
    return run_logged(cmd, log_path)


def run_analytic(method: str) -> None:
    test_dir = TEST_ROOT / method
    log_path = LOG_ROOT / f"{method}_test.log"
    record = {
        "method": method,
        "status": "started",
        "checkpoint_kind": "analytic_no_checkpoint",
        "ckpt_path": None,
        "test_dir": str(test_dir),
        "test_log": str(log_path),
    }
    upsert_method(record)
    if not (test_dir / "metrics_summary.json").exists():
        rc = eval_model(method, None, test_dir, log_path)
        record["test_returncode"] = rc
        if rc != 0:
            record["status"] = "test_failed"
            upsert_method(record)
            return
    metrics = metrics_summary(test_dir)
    record["status"] = "completed" if metrics else "missing_metrics"
    record["metrics_summary"] = str(test_dir / "metrics_summary.json")
    if metrics:
        record.update({k: metrics.get(k) for k in metrics if k.endswith("_mean")})
    upsert_method(record)


def run_trainable(method: str) -> None:
    run_dir = RUN_ROOT / method
    test_dir = TEST_ROOT / method
    train_log = LOG_ROOT / f"{method}_train.log"
    test_log = LOG_ROOT / f"{method}_test.log"
    record = {
        "method": method,
        "status": "started",
        "run_dir": str(run_dir),
        "test_dir": str(test_dir),
        "train_log": str(train_log),
        "test_log": str(test_log),
        "train_epochs": TRAIN_EPOCHS,
    }
    upsert_method(record)
    ckpt = best_checkpoint(run_dir)
    if ckpt is None:
        rc = train_model(method, run_dir, train_log)
        record["train_returncode"] = rc
        if rc != 0:
            record["status"] = "train_failed"
            upsert_method(record)
            return
        ckpt = best_checkpoint(run_dir)
    if ckpt is None:
        record["status"] = "no_checkpoint"
        upsert_method(record)
        return
    record["ckpt_path"] = ckpt
    record["checkpoint_kind"] = f"best_val_dice_{TRAIN_EPOCHS}epoch_rerun"
    upsert_method(record)
    if not (test_dir / "metrics_summary.json").exists():
        rc = eval_model(method, ckpt, test_dir, test_log)
        record["test_returncode"] = rc
        if rc != 0:
            record["status"] = "test_failed"
            upsert_method(record)
            return
    metrics = metrics_summary(test_dir)
    record["status"] = "completed" if metrics else "missing_metrics"
    record["metrics_summary"] = str(test_dir / "metrics_summary.json")
    if metrics:
        record.update({k: metrics.get(k) for k in metrics if k.endswith("_mean")})
    upsert_method(record)


def write_summary_csv() -> None:
    manifest = load_manifest()
    fields = [
        "method",
        "status",
        "ckpt_path",
        "dice_mean",
        "iou_mean",
        "nrmse_mean",
        "psnr_mean",
        "ssim_mean",
        "cle_mean",
        "ple_mean",
        "cnr_mean",
        "flops_g_mean",
        "inference_time_ms_mean",
        "peak_gpu_memory_mb_mean",
        "metrics_summary",
    ]
    with SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in manifest.get("methods", []):
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    manifest["unavailable"] = UNAVAILABLE
    save_manifest(manifest)

    stage1_ok = ensure_stage1_assets()
    if stage1_ok:
        for method in ANALYTIC:
            run_analytic(method)
            write_summary_csv()
    else:
        manifest = load_manifest()
        manifest["unavailable"]["stage1_based_methods"] = "Stage1 asset generation failed."
        save_manifest(manifest)

    for method in TRAINABLE:
        run_trainable(method)
        write_summary_csv()

    write_summary_csv()
    print(json.dumps({"manifest": str(MANIFEST), "summary_csv": str(SUMMARY_CSV)}, indent=2))


if __name__ == "__main__":
    main()
