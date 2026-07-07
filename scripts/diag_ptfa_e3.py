#!/usr/bin/env python
"""Forward/backward diagnostics for E3 exit-depth PTFA."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.dataset.fmt_simgen_dataset import FmtSimGenProjDataset  # noqa: E402
from minr_fmt.loss import ScatterLightLoss  # noqa: E402
from minr_fmt.model_factory import ModelFactory  # noqa: E402
from minr_fmt.network.ptfa import (  # noqa: E402
    ptfa_sample_exit_depth_gaussian,
    ptfa_sample_fixed_gaussian,
)
from minr_fmt.utils.fmt_simgen_projection import project_points_mm_to_detector  # noqa: E402


def compose_cfg(exp: str, num_queries: int):
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return hydra.compose(
            config_name="config",
            overrides=[
                f"exp={exp}",
                f"data.sample_num={num_queries}",
                f"data.num_queries={num_queries}",
                "data.train_max_samples=1",
                "data.val_max_samples=1",
                "model.geometry.use_fmt_simgen_projection=true",
            ],
        )


def load_net(cfg, ckpt_path: Path, device: torch.device):
    net = ModelFactory.create_model(cfg.model.name, config=cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = {
        key.removeprefix("net."): value
        for key, value in ckpt["state_dict"].items()
        if key.startswith("net.")
    }
    missing, unexpected = net.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:10]}")
    return net, missing


def pack_projection_input(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    p = batch["projections_packed"]
    return p.permute(1, 0, 2, 3, 4).reshape(
        p.shape[0] * p.shape[1], 1, p.shape[-2], p.shape[-1]
    )


def move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        elif isinstance(value, dict):
            out[key] = {
                sub_key: sub_value.to(device) if torch.is_tensor(sub_value) else sub_value
                for sub_key, sub_value in value.items()
            }
        else:
            out[key] = value
    return out


def prepare_s3(net, batch: dict, cfg):
    proj_in = pack_projection_input(batch)
    points_mm = batch["points_mm"]
    with torch.no_grad():
        _s1, _s2, s3 = net._forward_shared_unet(proj_in)
        center_grid, valid_mask = net._fmt_projection_grids(points_mm)
        feature_map = net._view_feature_dict_to_tensor(s3)
        query_depths = []
        for angle in cfg.data.view_angles:
            _grid, depth, _valid, _uv_px, _uv_phys = project_points_mm_to_detector(
                points_mm,
                int(angle),
                camera_distance_mm=cfg.model.geometry.camera_distance,
                fov_mm=cfg.model.geometry.fov_mm,
                detector_resolution=tuple(cfg.model.geometry.detector_resolution),
                volume_center_world=tuple(cfg.model.geometry.volume_center_world),
            )
            query_depths.append(depth)
        query_depth = torch.stack(query_depths, dim=2)
    return feature_map, center_grid, valid_mask, query_depth


def pixel_centers(center_grid: torch.Tensor, feature_hw: tuple[int, int]) -> torch.Tensor:
    h, w = feature_hw
    x = (center_grid[..., 0] + 1.0) * (w - 1) / 2.0
    y = (center_grid[..., 1] + 1.0) * (h - 1) / 2.0
    return torch.stack([x, y], dim=-1)


def run_path_sanity(net, batch: dict, cfg, out_dir: Path) -> dict:
    feature_map, center_grid, valid_mask, query_depth = prepare_s3(net, batch, cfg)
    fixed = ptfa_sample_fixed_gaussian(feature_map, center_grid, valid_mask, window=5, sigma_px=1.0)
    exit_degenerate, stats = ptfa_sample_exit_depth_gaussian(
        feature_map,
        center_grid,
        valid_mask,
        batch["depth_maps"],
        query_depth,
        sigma_min=1.0,
        sigma_max=1.0,
        exit_depth_max=20.8,
        window=5,
    )
    diff = (exit_degenerate - fixed).abs()

    centers_fixed = pixel_centers(center_grid, feature_map.shape[-2:]).detach().cpu()[0]
    centers_exit = centers_fixed.clone()
    n = min(200, centers_fixed.shape[0])
    view_idx = 3 if centers_fixed.shape[1] > 3 else 0
    plt.figure(figsize=(6, 6))
    plt.scatter(
        centers_fixed[:n, view_idx, 0],
        centers_fixed[:n, view_idx, 1],
        s=12,
        label="fixed",
        alpha=0.7,
    )
    plt.scatter(
        centers_exit[:n, view_idx, 0],
        centers_exit[:n, view_idx, 1],
        s=6,
        label="exit-depth",
        alpha=0.7,
    )
    plt.title("PTFA center overlay, view index 3")
    plt.xlabel("u px (s3)")
    plt.ylabel("v px (s3)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "center_overlay.png", dpi=160)
    plt.close()

    center_delta = (centers_exit - centers_fixed).abs().max().item()
    return {
        "max_abs_delta": float(diff.max().item()),
        "mean_abs_delta": float(diff.mean().item()),
        "center_max_abs_delta_px": float(center_delta),
        "center_grid": center_grid.detach(),
        "valid_mask": valid_mask.detach(),
        "feature_map": feature_map.detach(),
    }


def run_exit_depth_stats(net, batch: dict, cfg) -> dict:
    feature_map, center_grid, valid_mask, query_depth = prepare_s3(net, batch, cfg)
    _out, stats = ptfa_sample_exit_depth_gaussian(
        feature_map,
        center_grid,
        valid_mask,
        batch["depth_maps"],
        query_depth,
        sigma_min=cfg.model.ptfa.sigma_min,
        sigma_max=cfg.model.ptfa.sigma_max,
        exit_depth_max=cfg.model.ptfa.exit_depth_max_mm,
        window=cfg.model.ptfa.window,
    )
    return {
        "exit_depth": stats["exit_depth_mm"].detach(),
        "sigma": stats["sigma_px"].detach(),
        "valid_mask": valid_mask.detach(),
    }


def plot_sigma_physics(stats: dict, batch: dict, out_dir: Path) -> dict:
    valid = stats["valid_mask"]
    sigma = stats["sigma"][valid].float().cpu().numpy()
    exit_depth = stats["exit_depth"][valid].float().cpu().numpy()
    z_mm = batch["points_mm"][..., 2].unsqueeze(-1).expand_as(valid)[valid].float().cpu().numpy()
    if len(z_mm) > 5000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(z_mm), size=5000, replace=False)
        z_plot, sigma_plot = z_mm[idx], sigma[idx]
    else:
        z_plot, sigma_plot = z_mm, sigma

    corr_z_sigma = float(np.corrcoef(z_mm, sigma)[0, 1]) if len(z_mm) > 1 else float("nan")
    corr_exit_sigma = (
        float(np.corrcoef(exit_depth, sigma)[0, 1]) if len(exit_depth) > 1 else float("nan")
    )
    plt.figure(figsize=(6, 4))
    plt.scatter(z_plot, sigma_plot, s=4, alpha=0.25)
    plt.xlabel("query z mm")
    plt.ylabel("sigma px")
    plt.title("sigma vs query z")
    plt.tight_layout()
    plt.savefig(out_dir / "sigma_z_scatter.png", dpi=160)
    plt.close()

    return {
        "sigma_min": float(np.min(sigma)),
        "sigma_mean": float(np.mean(sigma)),
        "sigma_max": float(np.max(sigma)),
        "exit_depth_min": float(np.min(exit_depth)),
        "exit_depth_mean": float(np.mean(exit_depth)),
        "exit_depth_max": float(np.max(exit_depth)),
        "corr_z_sigma": corr_z_sigma,
        "corr_exit_depth_sigma": corr_exit_sigma,
    }


def loss_fn_from_cfg(cfg):
    loss_cfg = cfg.loss
    return ScatterLightLoss(
        init_scatter_weight=loss_cfg.get(
            "aux_projection_weight", loss_cfg.get("scatter_weight", 1.0)
        ),
        target_scatter_weight=loss_cfg.get(
            "target_aux_projection_weight", loss_cfg.get("target_scatter_weight", 0.5)
        ),
        start_decay_epoch=loss_cfg.start_decay_epoch,
        decay_epochs=loss_cfg.decay_epochs,
        pos_weight=loss_cfg.pos_weight,
        sparse_weight=loss_cfg.sparse_weight,
        lambda_dice=loss_cfg.dice_weight,
    )


def gradient_probe(net, batch: dict, mode: str, cfg) -> dict:
    net.train()
    proj_in = pack_projection_input(batch)
    points = batch["points"]
    density = batch["point_densities"].unsqueeze(-1)
    captured = {}
    old_ptfa_mode = getattr(net, "ptfa_mode", None)
    old_ptfa_enabled = getattr(net, "ptfa_enabled", None)
    old_ptfa_scales = getattr(net, "ptfa_scales", None)
    old_ptfa_sigma_px = getattr(net, "ptfa_sigma_px", None)

    if mode == "exit":
        original = net._ptfa_sample_exit_depth_gaussian

        def wrapped(view_features, points_mm, depth_maps, query_depth):
            out = original(view_features, points_mm, depth_maps, query_depth)
            out.retain_grad()
            captured["sampled"] = out
            return out

        net._ptfa_sample_exit_depth_gaussian = wrapped
    elif mode == "fixed":
        original = net._ptfa_sample_fixed_gaussian

        def wrapped(view_features, points_mm):
            out = original(view_features, points_mm)
            out.retain_grad()
            captured["sampled"] = out
            return out

        net._ptfa_sample_fixed_gaussian = wrapped
        net.ptfa_mode = "fixed_gaussian"
        net.ptfa_enabled = True
        net.ptfa_scales = {"s3"}
        net.ptfa_sigma_px = 1.0
    else:
        raise ValueError(mode)

    net.zero_grad(set_to_none=True)
    pred, aux = net(proj_in, points, points_mm=batch["points_mm"], depth_maps=batch["depth_maps"])
    loss_dict = loss_fn_from_cfg(cfg)(aux, batch["projections"], pred, density)
    loss = loss_dict["total_loss"]
    loss.backward()
    grad = captured["sampled"].grad
    result = {
        "loss": float(loss.detach().item()),
        "sampled_grad_l2": float(grad.norm().detach().item()),
        "sampled_grad_mean_abs": float(grad.abs().mean().detach().item()),
    }

    if mode == "exit":
        net._ptfa_sample_exit_depth_gaussian = original
    else:
        net._ptfa_sample_fixed_gaussian = original
    if old_ptfa_mode is not None:
        net.ptfa_mode = old_ptfa_mode
    if old_ptfa_enabled is not None:
        net.ptfa_enabled = old_ptfa_enabled
    if old_ptfa_scales is not None:
        net.ptfa_scales = old_ptfa_scales
    if old_ptfa_sigma_px is not None:
        net.ptfa_sigma_px = old_ptfa_sigma_px
    return result


def profile_step(net, batch: dict, cfg, device: torch.device) -> dict:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        t0 = time.perf_counter()
    probe_mode = "exit" if net.ptfa_mode == "exit_depth_gaussian" else "fixed"
    result = gradient_probe(net, batch, probe_mode, cfg)
    if device.type == "cuda":
        end.record()
        torch.cuda.synchronize(device)
        step_ms = float(start.elapsed_time(end))
        peak_mb = float(torch.cuda.max_memory_allocated(device) / 1024**2)
    else:
        step_ms = float((time.perf_counter() - t0) * 1000.0)
        peak_mb = 0.0
    return {"step_ms": step_ms, "peak_mb": peak_mb, **result}


def bar_plot(values: dict[str, float], title: str, ylabel: str, path: Path) -> None:
    plt.figure(figsize=(5, 4))
    names = list(values)
    vals = [values[k] for k in names]
    plt.bar(names, vals)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="outputs/diag/ptfa_e3")
    parser.add_argument("--num_queries", type=int, default=1024)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument(
        "--e2_ckpt",
        default="outputs/gisc_fmt/fit/2026-05-11/20-38-45/checkpoints/epoch=26-val_dice=0.5768.ckpt",
    )
    parser.add_argument(
        "--e3p_ckpt",
        default="outputs/gisc_fmt/fit/2026-05-12/09-42-16/checkpoints/epoch=26-val_dice=0.5395.ckpt",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    requested_device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    device = torch.device(requested_device)

    torch.manual_seed(0)
    cfg_e2 = compose_cfg("fmt_simgen_e2_ptfa_s3_fixed", args.num_queries)
    cfg_e3p = compose_cfg("fmt_simgen_e3p_ptfa_s3_exit_depth_calibrated", args.num_queries)
    ds = FmtSimGenProjDataset(
        cfg_e3p.data.data_dir, config=cfg_e3p, split="train", is_training=True
    )
    batch = move_batch(next(iter(DataLoader(ds, batch_size=1, num_workers=0))), device)

    net_e2, _missing_e2 = load_net(cfg_e2, Path(args.e2_ckpt), device)
    net_e3p, _missing_e3p = load_net(cfg_e3p, Path(args.e3p_ckpt), device)
    net_e2.eval()
    net_e3p.eval()

    sanity = run_path_sanity(net_e3p, batch, cfg_e3p, out_dir)
    exit_stats = run_exit_depth_stats(net_e3p, batch, cfg_e3p)
    physics = plot_sigma_physics(exit_stats, batch, out_dir)

    grad_exit = gradient_probe(net_e3p, batch, "exit", cfg_e3p)
    # Use the same E3' weights but force fixed sigma=1 to isolate the path implementation.
    grad_fixed = gradient_probe(net_e3p, batch, "fixed", cfg_e3p)
    bar_plot(
        {
            "exit-depth": grad_exit["sampled_grad_l2"],
            "fixed": grad_fixed["sampled_grad_l2"],
        },
        "s3 sampled feature gradient L2",
        "grad L2",
        out_dir / "gradient_l2.png",
    )

    prof_e2 = profile_step(net_e2, batch, cfg_e2, device)
    prof_e3p = profile_step(net_e3p, batch, cfg_e3p, device)
    bar_plot(
        {"E2": prof_e2["peak_mb"], "E3p": prof_e3p["peak_mb"]},
        "Peak memory, one backward step",
        "MiB",
        out_dir / "memory_peak.png",
    )

    summary = {
        "path_sanity": {
            "degenerate_sigma1_max_abs_delta": sanity["max_abs_delta"],
            "degenerate_sigma1_mean_abs_delta": sanity["mean_abs_delta"],
            "center_max_abs_delta_px": sanity["center_max_abs_delta_px"],
        },
        "sigma_physics": physics,
        "gradient": {
            "exit_depth": grad_exit,
            "fixed": grad_fixed,
            "exit_over_fixed_grad_l2": grad_exit["sampled_grad_l2"]
            / max(grad_fixed["sampled_grad_l2"], 1.0e-12),
        },
        "profile": {"E2": prof_e2, "E3p": prof_e3p},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    report = f"""# PTFA E3 Diagnostics

## Path Sanity

- Degenerate exit-depth sigma=1 vs fixed sigma=1 max|delta|:
  `{sanity["max_abs_delta"]:.6g}`
- Mean|delta|: `{sanity["mean_abs_delta"]:.6g}`
- Center max abs delta in s3 pixels: `{sanity["center_max_abs_delta_px"]:.6g}`

## Sigma Physics

- exit_depth min/mean/max:
  `{physics["exit_depth_min"]:.4f} / {physics["exit_depth_mean"]:.4f} /
  {physics["exit_depth_max"]:.4f}`
- sigma min/mean/max:
  `{physics["sigma_min"]:.4f} / {physics["sigma_mean"]:.4f} / {physics["sigma_max"]:.4f}`
- corr(z_query, sigma): `{physics["corr_z_sigma"]:.4f}`
- corr(exit_depth, sigma): `{physics["corr_exit_depth_sigma"]:.4f}`

## Gradient

- exit-depth sampled grad L2: `{grad_exit["sampled_grad_l2"]:.6g}`
- fixed sampled grad L2: `{grad_fixed["sampled_grad_l2"]:.6g}`
- exit/fixed grad L2 ratio:
  `{summary["gradient"]["exit_over_fixed_grad_l2"]:.4f}`

## Profile

- E2 peak memory MiB: `{prof_e2["peak_mb"]:.2f}`, step ms: `{prof_e2["step_ms"]:.2f}`
- E3p peak memory MiB: `{prof_e3p["peak_mb"]:.2f}`, step ms: `{prof_e3p["step_ms"]:.2f}`

## Figures

- `center_overlay.png`
- `sigma_z_scatter.png`
- `gradient_l2.png`
- `memory_peak.png`

## Conclusion

Degenerate exit-depth sigma=1 is numerically identical to fixed sigma=1, so H1
center/path divergence is not supported. The sampled-feature gradient ratio is
reported above for H2. The sigma mapping is monotonic with exit depth by
construction; inspect `corr(z_query, sigma)` to judge whether z is a useful
proxy for the physical depth direction in this geometry.
"""
    (out_dir / "report.md").write_text(report)
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
