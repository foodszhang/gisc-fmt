#!/usr/bin/env python
"""Forward smoke checks for E1b/E4/E6 FMT-SimGen experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import hydra
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.dataset.fmt_simgen_dataset import FmtSimGenProjDataset  # noqa: E402
from minr_fmt.model_factory import ModelFactory  # noqa: E402


def run_one(exp: str, data_dir: str, num_queries: int, device: str) -> None:
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                f"exp={exp}",
                f"data.data_dir={data_dir}",
                f"data.train_dir={data_dir}",
                f"data.val_dir={data_dir}",
                f"data.test_dir={data_dir}",
                f"data.sample_num={num_queries}",
                f"data.num_queries={num_queries}",
                "data.train_max_samples=2",
                "data.val_max_samples=2",
                "model.geometry.use_fmt_simgen_projection=true",
            ],
        )

    ds = FmtSimGenProjDataset(data_dir, config=cfg, split="train", is_training=True)
    batch = next(iter(DataLoader(ds, batch_size=1, num_workers=0)))
    p = batch["projections_packed"]
    proj_in = p.permute(1, 0, 2, 3, 4).reshape(
        p.shape[0] * p.shape[1], 1, p.shape[-2], p.shape[-1]
    )

    net = ModelFactory.create_model(cfg.model.name, config=cfg).to(device)
    net.eval()
    with torch.no_grad():
        pred, aux = net(
            proj_in.to(device),
            batch["points"].to(device),
            points_mm=batch["points_mm"].to(device),
            depth_maps=batch["depth_maps"].to(device),
        )

    expected = (1, num_queries, 1)
    if tuple(pred.shape) != expected:
        raise RuntimeError(f"{exp}: pred shape {tuple(pred.shape)} != {expected}")
    if not torch.isfinite(pred).all():
        raise RuntimeError(f"{exp}: prediction contains NaN/Inf")
    for key, value in aux.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"{exp}: aux[{key}] contains NaN/Inf")

    print(f"{exp}: pred_shape={tuple(pred.shape)} aux_keys={[str(k) for k in aux.keys()]}")
    if hasattr(net, "residual_scorer_input_dim"):
        print(f"{exp}: residual_scorer_input_dim={net.residual_scorer_input_dim}")
    if getattr(net, "last_s1_ptfa_evidence_stats", None):
        stats = net.last_s1_ptfa_evidence_stats
        print(
            f"{exp}: s1_ptfa_evidence mean/std/norm="
            f"{float(stats['mean']):.6f}/"
            f"{float(stats['std']):.6f}/"
            f"{float(stats['norm']):.6f}"
        )
    if getattr(net, "last_feature_refinement_stats", None):
        stats = net.last_feature_refinement_stats
        print(
            f"{exp}: feature_refinement "
            f"base_mean={float(stats['base_mean']):.6f} "
            f"ptfa_mean={float(stats['ptfa_mean']):.6f} "
            f"delta_norm={float(stats['delta_norm']):.6f} "
            f"refined_norm={float(stats['refined_norm']):.6f}"
        )
    if getattr(net, "last_reliability_gate_stats", None):
        stats = net.last_reliability_gate_stats
        print(
            f"{exp}: reliability_weights "
            f"mean/std/min/max="
            f"{float(stats['reliability_weights_mean']):.6f}/"
            f"{float(stats['reliability_weights_std']):.6f}/"
            f"{float(stats['reliability_weights_min']):.6f}/"
            f"{float(stats['reliability_weights_max']):.6f} "
            f"entropy={float(stats['reliability_weights_entropy']):.6f} "
            f"uniform_absdiff={float(stats['reliability_uniform_absdiff_mean']):.6f} "
            f"max_weight_mean={float(stats['reliability_max_weight_mean']):.6f} "
            f"valid_sum={float(stats['valid_weight_sum_mean']):.6f} "
            f"invalid_max={float(stats['invalid_weight_max']):.6f}"
        )
        print(
            f"{exp}: reliability_geom "
            f"valid_view_count_mean={float(stats['valid_view_count_mean']):.6f} "
            f"depth_eff_norm_mean/std="
            f"{float(stats['depth_eff_norm_mean']):.6f}/"
            f"{float(stats['depth_eff_norm_std']):.6f} "
            f"sigma_px_mean/std/min/max="
            f"{float(stats['sigma_px_mean']):.6f}/"
            f"{float(stats['sigma_px_std']):.6f}/"
            f"{float(stats['sigma_px_min']):.6f}/"
            f"{float(stats['sigma_px_max']):.6f}"
        )
    if getattr(net, "last_consensus_residual_gate_stats", None):
        stats = net.last_consensus_residual_gate_stats
        print(
            f"{exp}: consensus_confidence mean/std/min/max="
            f"{float(stats['confidence_mean']):.6f}/"
            f"{float(stats['confidence_std']):.6f}/"
            f"{float(stats['confidence_min']):.6f}/"
            f"{float(stats['confidence_max']):.6f}"
        )
        print(
            f"{exp}: consensus_delta "
            f"mean_norm={float(stats['ptfa_mean_norm']):.6f} "
            f"residual_update_norm={float(stats['ptfa_residual_update_norm']):.6f} "
            f"gamma_delta_norm={float(stats['ptfa_delta_norm']):.6f} "
            f"delta_abs_mean={float(stats['ptfa_delta_abs_mean']):.6f}"
        )
    if getattr(net, "last_ptfa_stats", None):
        for key in ("raw_depth_like_mm", "depth_eff_mm", "sigma_px"):
            value = net.last_ptfa_stats.get(key)
            if value is None:
                continue
            finite = value[torch.isfinite(value)]
            print(
                f"{exp}: {key} min/mean/max="
                f"{float(finite.min()):.6f}/"
                f"{float(finite.mean()):.6f}/"
                f"{float(finite.max()):.6f}"
            )
        weights = getattr(net, "last_query_aggregation_weights", None)
        if weights is not None:
            print(
                f"{exp}: gate_weight min/mean/max="
                f"{float(weights.min()):.6f}/"
                f"{float(weights.mean()):.6f}/"
                f"{float(weights.max()):.6f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/foods/pro/FMT-SimGen/data/uniform_1000_20k")
    parser.add_argument("--num_queries", type=int, default=512)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument(
        "--exp",
        action="append",
        default=[],
        help="Experiment override. Defaults to E1b, E4 lambda=0.2, and E6.",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    exps = args.exp or [
        "fmt_simgen_e1b_nongt_mixed",
        "fmt_simgen_e4_residual_scorer_lambda020",
        "fmt_simgen_e7_e4_plus_s1_ptfa_fixed",
    ]
    for exp in exps:
        run_one(exp, args.data_dir, args.num_queries, args.device)


if __name__ == "__main__":
    main()
