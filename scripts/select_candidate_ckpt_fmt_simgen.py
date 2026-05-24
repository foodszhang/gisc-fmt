#!/usr/bin/env python
"""Select FMT-SimGen checkpoints by candidate-domain validation.

This is a thin orchestrator around eval_candidate_dense_fmt_simgen.py. It evaluates a
small fixed checkpoint set on val50, chooses the best candidate_dice, and can optionally
run val200 for the selected checkpoint.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ROOT / "scripts" / "eval_candidate_dense_fmt_simgen.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--save_root", default="outputs/candidate_eval_compare")
    parser.add_argument("--val50_max_samples", type=int, default=50)
    parser.add_argument("--run_val200", action="store_true")
    parser.add_argument("--candidate_topk_ratio", type=float, default=0.10)
    parser.add_argument("--samples_per_candidate_cell", type=int, default=16)
    parser.add_argument("--outside_sample_num", type=int, default=131072)
    parser.add_argument("--chunk_size", type=int, default=32768)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. model=point_cqr")
    return parser.parse_args()


def epoch_from_name(path: Path) -> int | None:
    match = re.search(r"epoch=(\d+)", path.name)
    return int(match.group(1)) if match else None


def val_dice_from_name(path: Path) -> float | None:
    match = re.search(r"val_dice=([0-9]+(?:\.[0-9]+)?)", path.name)
    return float(match.group(1)) if match else None


def collect_candidates(run_dir: Path) -> list[Path]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint dir: {ckpt_dir}")
    ckpts = sorted(p for p in ckpt_dir.glob("*.ckpt") if p.name != "last.ckpt")
    selected: dict[str, Path] = {}

    with_val = [(val_dice_from_name(p), p) for p in ckpts]
    with_val = [(v, p) for v, p in with_val if v is not None]
    if with_val:
        selected["best_val_query"] = max(with_val, key=lambda item: item[0])[1]

    by_epoch = {epoch_from_name(p): p for p in ckpts if epoch_from_name(p) is not None}
    for target in (40, 50, 59):
        if target in by_epoch:
            selected[f"epoch{target}"] = by_epoch[target]

    last = ckpt_dir / "last.ckpt"
    if last.exists():
        selected["last"] = last

    # Preserve order while removing duplicate paths.
    out = []
    seen = set()
    for path in selected.values():
        resolved = path.resolve()
        if resolved not in seen:
            out.append(path)
            seen.add(resolved)
    if not out:
        raise FileNotFoundError(f"No candidate checkpoints found under {ckpt_dir}")
    return out


def run_eval(args: argparse.Namespace, ckpt: Path, save_dir: Path, max_samples: int | None) -> dict:
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        f"exp={args.exp}",
        f"ckpt_path={ckpt}",
        f"split={args.split}",
        f"eval.candidate_topk_ratio={args.candidate_topk_ratio}",
        "eval.candidate_mode=coarse_cell_sample",
        f"eval.samples_per_candidate_cell={args.samples_per_candidate_cell}",
        f"eval.outside_sample_num={args.outside_sample_num}",
        f"eval.chunk_size={args.chunk_size}",
        f"eval.threshold={args.threshold}",
        f"eval.seed={args.seed}",
        f"eval.save_dir={save_dir}",
        *args.overrides,
    ]
    if max_samples is not None:
        cmd.append(f"eval.max_samples={max_samples}")
    subprocess.run(cmd, cwd=ROOT, check=True)
    summary_path = save_dir / "metrics_summary.json"
    return json.loads(summary_path.read_text())


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    save_root = Path(args.save_root)
    if not save_root.is_absolute():
        save_root = ROOT / save_root

    candidates = collect_candidates(run_dir)
    run_name = run_dir.name
    rows = []
    for ckpt in candidates:
        tag = ckpt.stem.replace("=", "-")
        save_dir = save_root / f"{run_name}_val50_{tag}"
        summary = run_eval(args, ckpt, save_dir, args.val50_max_samples)
        rows.append(
            {
                "ckpt": str(ckpt),
                "save_dir": str(save_dir),
                "candidate_dice": float(summary["mean_candidate_dice"]),
                "precision": float(summary["mean_candidate_precision"]),
                "recall": float(summary["mean_candidate_recall"]),
                "outside_fp_rate": float(summary["mean_outside_fp_rate"]),
            }
        )

    best = max(rows, key=lambda row: row["candidate_dice"])
    result = {"run_dir": str(run_dir), "exp": args.exp, "val50": rows, "selected": best}

    if args.run_val200:
        ckpt = Path(best["ckpt"])
        tag = ckpt.stem.replace("=", "-")
        save_dir = save_root / f"{run_name}_val200_selected_{tag}"
        summary = run_eval(args, ckpt, save_dir, None)
        result["val200"] = {
            "ckpt": str(ckpt),
            "save_dir": str(save_dir),
            "candidate_dice": float(summary["mean_candidate_dice"]),
            "precision": float(summary["mean_candidate_precision"]),
            "recall": float(summary["mean_candidate_recall"]),
            "outside_fp_rate": float(summary["mean_outside_fp_rate"]),
        }

    out_path = save_root / f"{run_name}_selection_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"[INFO] Wrote {out_path}")


if __name__ == "__main__":
    main()
