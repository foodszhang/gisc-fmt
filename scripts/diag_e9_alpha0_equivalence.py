#!/usr/bin/env python
"""Check E9 alpha=0 equivalence against E8-cED on the same sampled batch."""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.dataset.fmt_simgen_dataset import FmtSimGenProjDataset  # noqa: E402
from minr_fmt.loss import ScatterLightLoss  # noqa: E402
from minr_fmt.model_factory import ModelFactory  # noqa: E402


def _compose(exp: str, data_dir: str, num_queries: int):
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return hydra.compose(
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


def _projection_input(batch: dict[str, torch.Tensor]) -> tuple[dict, torch.Tensor]:
    p = batch["projections_packed"]
    B, V = p.shape[0], p.shape[1]
    proj_in = p.permute(1, 0, 2, 3, 4).reshape(B * V, 1, p.shape[-2], p.shape[-1])
    return batch["projections"], proj_in


def _loss_fn(cfg) -> ScatterLightLoss:
    loss_cfg = cfg.loss
    return ScatterLightLoss(
        init_scatter_weight=loss_cfg.scatter_weight,
        target_scatter_weight=loss_cfg.target_scatter_weight,
        start_decay_epoch=loss_cfg.start_decay_epoch,
        decay_epochs=loss_cfg.decay_epochs,
        pos_weight=loss_cfg.pos_weight,
        sparse_weight=loss_cfg.sparse_weight,
        lambda_dice=loss_cfg.dice_weight,
    )


def _max_mean_abs(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    d = (a - b).abs()
    return float(d.max().detach().cpu()), float(d.mean().detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/foods/pro/FMT-SimGen/data/uniform_1000_20k")
    parser.add_argument("--num_queries", type=int, default=512)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg_e8 = _compose("fmt_simgen_e8_ced_s1_ptfa_feature_refine", args.data_dir, args.num_queries)
    cfg_e9 = _compose("fmt_simgen_e9_stable_v2_alpha0_debug", args.data_dir, args.num_queries)

    ds = FmtSimGenProjDataset(args.data_dir, config=cfg_e8, split="train", is_training=True)
    batch = next(iter(DataLoader(ds, batch_size=1, num_workers=0)))
    projections, proj_in = _projection_input(batch)
    points = batch["points"].to(args.device)
    points_mm = batch["points_mm"].to(args.device)
    depth_maps = batch["depth_maps"].to(args.device)
    density = batch["point_densities"].unsqueeze(-1).to(args.device)
    proj_in = proj_in.to(args.device)
    projections = {str(k): v.to(args.device) for k, v in projections.items()}

    torch.manual_seed(args.seed)
    net_e8 = ModelFactory.create_model(cfg_e8.model.name, config=cfg_e8).to(args.device)
    net_e9 = ModelFactory.create_model(cfg_e9.model.name, config=cfg_e9).to(args.device)
    missing, unexpected = net_e9.load_state_dict(copy.deepcopy(net_e8.state_dict()), strict=False)
    print(f"load_state missing={missing}")
    print(f"load_state unexpected={unexpected}")
    net_e8.eval()
    net_e9.eval()

    loss_fn = _loss_fn(cfg_e8).to(args.device)
    with torch.no_grad():
        pred_e8, aux_e8 = net_e8(proj_in, points, points_mm=points_mm, depth_maps=depth_maps)
        pred_e9, aux_e9 = net_e9(proj_in, points, points_mm=points_mm, depth_maps=depth_maps)
        loss_e8 = loss_fn(aux_e8, projections, pred_e8, density)["total_loss"]
        loss_e9 = loss_fn(aux_e9, projections, pred_e9, density)["total_loss"]

    debug_e8 = net_e8.last_feature_refinement_debug
    debug_e9 = net_e9.last_feature_refinement_debug
    for key in ("f_ptfa", "f_refined"):
        if key not in debug_e8 or key not in debug_e9:
            raise RuntimeError(f"missing debug tensor {key}")

    pred_max, pred_mean = _max_mean_abs(pred_e8, pred_e9)
    ptfa_max, ptfa_mean = _max_mean_abs(debug_e8["f_ptfa"], debug_e9["f_ptfa"])
    refined_max, refined_mean = _max_mean_abs(debug_e8["f_refined"], debug_e9["f_refined"])
    loss_diff = float((loss_e8 - loss_e9).abs().detach().cpu())

    stats = getattr(net_e9, "last_mean_prior_residual_gate_stats", {})
    print(f"pred_abs_diff max={pred_max:.10g} mean={pred_mean:.10g}")
    print(f"f_ptfa_abs_diff max={ptfa_max:.10g} mean={ptfa_mean:.10g}")
    print(f"f_refined_abs_diff max={refined_max:.10g} mean={refined_mean:.10g}")
    print(f"loss_diff={loss_diff:.10g}")
    if stats:
        print(
            "e9_alpha0_stats "
            f"alpha={float(stats['alpha']):.10g} "
            f"residual_scale_mean={float(stats['residual_scale_mean']):.10g} "
            f"residual_scale_std={float(stats['residual_scale_std']):.10g} "
            f"anchor_loss={float(stats['anchor_loss']):.10g}"
        )
    print(
        "residual_scorer "
        f"e8={getattr(net_e8, 'residual_scorer_enabled', None)} "
        f"e9={getattr(net_e9, 'residual_scorer_enabled', None)}"
    )


if __name__ == "__main__":
    main()
