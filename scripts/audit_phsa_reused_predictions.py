#!/usr/bin/env python3
"""Validate reusable A2-U/A3-old full-volume prediction caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_view_complementary_full_volume_paired import (  # noqa: E402
    deterministic_proposal_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["a2u", "a3_old"])
    parser.add_argument("--proposal-count", type=int, required=True)
    parser.add_argument("--proposal-seed", type=int, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("--volume-shape", nargs=3, type=int, default=[190, 200, 104])
    return parser.parse_args()


def scalar(payload: np.lib.npyio.NpzFile, key: str) -> float:
    if key not in payload.files:
        raise RuntimeError(f"prediction cache is missing {key}")
    return float(np.asarray(payload[key]).reshape(()))


def main() -> None:
    args = parse_args()
    shape = tuple(int(value) for value in args.volume_shape)
    if any(value <= 0 for value in shape):
        raise ValueError(f"invalid volume shape: {shape}")
    root = args.output_dir.resolve()
    ids_path = root / "sample_ids.json"
    if not ids_path.exists():
        raise FileNotFoundError(ids_path)
    expected_ids = [str(value) for value in json.loads(ids_path.read_text())]
    if len(expected_ids) != args.expected_samples:
        raise RuntimeError(
            f"cache sample_ids has {len(expected_ids)} entries, expected {args.expected_samples}"
        )
    if len(set(expected_ids)) != len(expected_ids):
        raise RuntimeError("cache sample_ids.json contains duplicates")

    expected_set = set(expected_ids)
    for model in args.models:
        model_dir = root / "predictions" / model
        files = {path.stem: path for path in model_dir.glob("*.npz")}
        if set(files) != expected_set:
            missing = sorted(expected_set - set(files))[:10]
            extra = sorted(set(files) - expected_set)[:10]
            raise RuntimeError(f"{model} cache IDs mismatch: missing={missing}, extra={extra}")

        for index, sample_id in enumerate(expected_ids, start=1):
            path = files[sample_id]
            with np.load(path, allow_pickle=False) as payload:
                required = {"proposal_indices", "proposal_sha256", "threshold", "pred", "gt"}
                missing_keys = sorted(required - set(payload.files))
                if missing_keys:
                    raise RuntimeError(f"{path} is missing arrays: {missing_keys}")
                proposal_indices = payload["proposal_indices"].astype(np.int64, copy=False)
                if proposal_indices.ndim != 1 or len(proposal_indices) != args.proposal_count:
                    raise RuntimeError(
                        f"{path} proposal count={len(proposal_indices)}, expected={args.proposal_count}"
                    )
                expected = deterministic_proposal_indices(
                    shape,
                    args.proposal_count,
                    args.proposal_seed,
                    sample_id,
                )
                if not np.array_equal(proposal_indices, expected):
                    raise RuntimeError(f"{path} proposal indices do not match seed/count contract")
                digest = hashlib.sha256(proposal_indices.tobytes()).hexdigest()
                stored_digest = str(np.asarray(payload["proposal_sha256"]).reshape(()))
                if stored_digest != digest:
                    raise RuntimeError(f"{path} proposal SHA mismatch")
                if abs(scalar(payload, "threshold") - args.threshold) > 1.0e-8:
                    raise RuntimeError(f"{path} threshold does not match {args.threshold}")
                # Do not access pred/gt arrays here: checking their names in the NPZ
                # index avoids decompressing roughly eight million float16 values.
            if index % 50 == 0:
                print(f"[{model}] audited {index}/{len(expected_ids)}")
        print(f"[PASS] {model}: {len(expected_ids)} reusable predictions match proposal contract")

    print("PHSA BASELINE CACHE AUDIT: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
