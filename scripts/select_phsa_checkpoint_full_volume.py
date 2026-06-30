#!/usr/bin/env python3
"""Select a PHSA checkpoint using validation-only full-volume Dice.

The training callback monitors proposal-biased sampled-query Dice. This selector
compares the saved top-k checkpoints and ``last.ckpt`` on a fixed validation subset
with the same sample-level proposal cache and full-volume decoder used at test time.
The test split is never touched during checkpoint selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--proposal-count", type=int, default=4096)
    parser.add_argument("--proposal-seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=32768)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoints(run_dir: Path) -> list[Path]:
    root = run_dir / "checkpoints"
    values = sorted(path.resolve() for path in root.glob("*.ckpt") if path.name != "last.ckpt")
    last = root / "last.ckpt"
    if last.exists():
        values.append(last.resolve())
    unique: list[Path] = []
    seen: set[str] = set()
    for path in values:
        fingerprint = checkpoint_sha256(path)
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(path)
    if not unique:
        raise FileNotFoundError(f"No checkpoints under {root}")
    return unique


def replace_symlink(link: Path, target: Path) -> None:
    if link.is_symlink() or link.exists():
        if link.is_dir() and not link.is_symlink():
            shutil.rmtree(link)
        else:
            link.unlink()
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target.resolve(), target_is_directory=target.is_dir())


def candidate_run(run_dir: Path, selection_root: Path, checkpoint: Path, index: int) -> Path:
    candidate = selection_root / "candidate_runs" / f"{index:02d}_{checkpoint.stem}"
    (candidate / "config").mkdir(parents=True, exist_ok=True)
    (candidate / "checkpoints").mkdir(parents=True, exist_ok=True)
    replace_symlink(candidate / "config" / "config.yaml", run_dir / "config" / "config.yaml")
    replace_symlink(candidate / "checkpoints" / "selected.ckpt", checkpoint)
    return candidate


def evaluate_candidate(
    args: argparse.Namespace,
    checkpoint: Path,
    index: int,
) -> dict[str, object]:
    candidate = candidate_run(args.run_dir, args.output_dir, checkpoint, index)
    evaluation = args.output_dir / "evaluations" / f"{index:02d}_{checkpoint.stem}"
    summary_path = evaluation / "summary.json"
    manifest_path = evaluation / "selection_manifest.json"
    fingerprint = checkpoint_sha256(checkpoint)
    manifest = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": fingerprint,
        "max_samples": args.max_samples,
        "proposal_count": args.proposal_count,
        "proposal_seed": args.proposal_seed,
        "chunk_size": args.chunk_size,
        "threshold": args.threshold,
    }
    cached_manifest = None
    if manifest_path.exists():
        try:
            cached_manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            cached_manifest = None
    if cached_manifest != manifest:
        shutil.rmtree(evaluation, ignore_errors=True)
    if not summary_path.exists():
        evaluation.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(ROOT / "scripts" / "eval_view_complementary_full_volume_paired_safe.py"),
            "--output_dir",
            str(evaluation),
            "--models",
            "phsa",
            "--phsa_run",
            str(candidate),
            "--split",
            "val",
            "--max_samples",
            str(args.max_samples),
            "--proposal_count",
            str(args.proposal_count),
            "--proposal_seed",
            str(args.proposal_seed),
            "--chunk_size",
            str(args.chunk_size),
            "--threshold",
            str(args.threshold),
            "--bootstrap_samples",
            "1000",
            "--device",
            args.device,
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
    summary = json.loads(summary_path.read_text())
    metrics = summary["models"]["phsa"]
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": fingerprint,
        "candidate_run": str(candidate.resolve()),
        "evaluation_dir": str(evaluation.resolve()),
        "num_samples": int(metrics["num_samples"]),
        "dice_mean": float(metrics["dice_mean"]),
        "nrmse_mean": metrics.get("nrmse_mean"),
        "component_recall_mean": metrics.get("component_recall_mean"),
    }
    # Predictions can be regenerated from the immutable checkpoint and consume most
    # of the selector disk space; retain metrics, manifest and sample IDs only.
    shutil.rmtree(evaluation / "predictions", ignore_errors=True)
    return result


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if not (args.run_dir / "config" / "config.yaml").exists():
        raise FileNotFoundError(args.run_dir / "config" / "config.yaml")

    candidates = checkpoints(args.run_dir)
    rows = [
        evaluate_candidate(args, checkpoint, index)
        for index, checkpoint in enumerate(candidates, start=1)
    ]
    selected = max(rows, key=lambda row: (float(row["dice_mean"]), str(row["checkpoint"])))
    selected_checkpoint = Path(str(selected["checkpoint"])).resolve()

    selected_run = args.output_dir / "selected_run"
    (selected_run / "config").mkdir(parents=True, exist_ok=True)
    (selected_run / "checkpoints").mkdir(parents=True, exist_ok=True)
    replace_symlink(selected_run / "config" / "config.yaml", args.run_dir / "config" / "config.yaml")
    replace_symlink(selected_run / "checkpoints" / "selected.ckpt", selected_checkpoint)

    report = {
        "selection_split": "val",
        "selection_samples": args.max_samples,
        "proposal_count": args.proposal_count,
        "proposal_seed": args.proposal_seed,
        "threshold": args.threshold,
        "candidates": rows,
        "selected": selected,
        "selected_run": str(selected_run.resolve()),
    }
    (args.output_dir / "selection.json").write_text(json.dumps(report, indent=2))
    (args.output_dir / "SELECTED_CHECKPOINT.txt").write_text(str(selected_checkpoint) + "\n")
    (args.output_dir / "SELECTED_RUN.txt").write_text(str(selected_run.resolve()) + "\n")

    lines = [
        "# PHSA Checkpoint Selection",
        "",
        f"Validation samples: {args.max_samples}",
        "",
    ]
    for row in sorted(rows, key=lambda value: float(value["dice_mean"]), reverse=True):
        marker = " **selected**" if row["checkpoint"] == selected["checkpoint"] else ""
        lines.append(
            f"- `{Path(str(row['checkpoint'])).name}`: Dice={float(row['dice_mean']):.6f}{marker}"
        )
    (args.output_dir / "selection.md").write_text("\n".join(lines) + "\n")
    print(selected_run.resolve())


if __name__ == "__main__":
    main()
