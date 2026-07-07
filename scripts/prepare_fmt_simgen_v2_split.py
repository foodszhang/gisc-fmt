#!/usr/bin/env python
"""Create the fixed val/test split for fmt_simgen_v2_3k_20k."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k",
    )
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--val_size", type=int, default=300)
    parser.add_argument("--test_size", type=int, default=300)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def split_hash(names: list[str]) -> str:
    h = hashlib.sha256()
    for name in names:
        h.update(name.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def load_manifest_samples(path: Path) -> dict[str, dict]:
    obj = json.loads(path.read_text())
    samples = obj.get("samples", [])
    out = {}
    for sample in samples:
        sample_id = str(sample.get("id") or sample.get("sample_id"))
        if sample_id:
            out[sample_id] = sample
    return out


def read_split(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def stratified_partition(
    names: list[str],
    manifest: dict[str, dict],
    val_size: int,
    test_size: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    buckets: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for name in names:
        item = manifest.get(name, {})
        key = (str(item.get("num_foci", "unknown")), str(item.get("depth_tier", "unknown")))
        buckets[key].append(name)

    rng = np.random.default_rng(seed)
    val: list[str] = []
    test: list[str] = []
    for key in sorted(buckets):
        bucket = sorted(buckets[key])
        rng.shuffle(bucket)
        n = len(bucket)
        n_val = int(round(n * val_size / len(names)))
        n_val = max(0, min(n, n_val))
        val.extend(bucket[:n_val])
        test.extend(bucket[n_val:])

    def rebalance(left: list[str], right: list[str], target_left: int) -> None:
        if len(left) > target_left:
            move = sorted(left[target_left:])
            del left[target_left:]
            right.extend(move)
        elif len(left) < target_left:
            need = target_left - len(left)
            move = sorted(right[:need])
            del right[:need]
            left.extend(move)

    val = sorted(val)
    test = sorted(test)
    rebalance(val, test, val_size)
    test = sorted(test)
    if len(test) != test_size:
        raise ValueError(f"Expected test_size={test_size}, got {len(test)}")
    if set(val) & set(test):
        raise ValueError("val/test overlap")
    return sorted(val), sorted(test)


def distribution(names: list[str], manifest: dict[str, dict]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {"num_foci": {}, "depth_tier": {}}
    for name in names:
        item = manifest.get(name, {})
        for key in out:
            value = str(item.get(key, "unknown"))
            out[key][value] = out[key].get(value, 0) + 1
    return out


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser()
    split_dir = data_dir / "splits"
    train_path = split_dir / "train.txt"
    val_path = split_dir / "val.txt"
    test_path = split_dir / "test.txt"
    val_full_path = split_dir / "val_full_600.txt"
    manifest = load_manifest_samples(data_dir / "dataset_manifest.json")
    train_names = read_split(train_path)
    current_val = read_split(val_path)
    source_val = read_split(val_full_path) if val_full_path.exists() else current_val

    if len(train_names) != 2400:
        raise ValueError(f"Expected train split size 2400, got {len(train_names)}")
    if len(source_val) != args.val_size + args.test_size:
        raise ValueError(
            f"Expected source val size {args.val_size + args.test_size}, got {len(source_val)}"
        )
    val, test = stratified_partition(
        source_val,
        manifest,
        args.val_size,
        args.test_size,
        args.seed,
    )
    if set(train_names) & (set(val) | set(test)):
        raise ValueError("train overlaps val/test")

    report = {
        "data_dir": str(data_dir),
        "seed": args.seed,
        "train_count": len(train_names),
        "val_count": len(val),
        "test_count": len(test),
        "train_hash": split_hash(train_names),
        "val_hash": split_hash(val),
        "test_hash": split_hash(test),
        "val_full_hash": split_hash(source_val),
        "distribution": {
            "val_full": distribution(source_val, manifest),
            "val": distribution(val, manifest),
            "test": distribution(test, manifest),
        },
    }
    print(json.dumps(report, indent=2))
    if args.dry_run:
        return
    if not val_full_path.exists():
        shutil.copy2(val_path, val_full_path)
    val_path.write_text("".join(f"{name}\n" for name in val))
    test_path.write_text("".join(f"{name}\n" for name in test))
    (split_dir / "split_manifest_v2_3k_20k.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
