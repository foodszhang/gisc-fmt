#!/usr/bin/env python
"""Check torch FMT-SimGen projection against the reference camera formula."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.dataset.fmt_simgen_dataset import FmtSimGenProjDataset  # noqa: E402
from minr_fmt.utils.fmt_simgen_projection import project_points_mm_to_detector  # noqa: E402


def _load_fmt_simgen_projection_deps():
    fmt_root = Path("/home/foods/pro/FMT-SimGen/fmt_simgen")
    package = types.ModuleType("fmt_simgen")
    package.__path__ = [str(fmt_root)]  # type: ignore[attr-defined]
    sys.modules.setdefault("fmt_simgen", package)

    frame_spec = importlib.util.spec_from_file_location(
        "fmt_simgen.frame_contract", fmt_root / "frame_contract.py"
    )
    if frame_spec is None or frame_spec.loader is None:
        raise RuntimeError("Could not load FMT-SimGen frame_contract.py")
    frame_mod = importlib.util.module_from_spec(frame_spec)
    sys.modules["fmt_simgen.frame_contract"] = frame_mod
    frame_spec.loader.exec_module(frame_mod)

    view_spec = importlib.util.spec_from_file_location(
        "fmt_simgen.view_config", fmt_root / "view_config.py"
    )
    if view_spec is None or view_spec.loader is None:
        raise RuntimeError("Could not load FMT-SimGen view_config.py")
    view_mod = importlib.util.module_from_spec(view_spec)
    sys.modules["fmt_simgen.view_config"] = view_mod
    view_spec.loader.exec_module(view_mod)
    return frame_mod, view_mod.TurntableCamera


def continuous_reference(points_mm: np.ndarray, angle: float, frame):
    theta = np.deg2rad(angle)
    p = points_mm - np.asarray(frame.VOLUME_CENTER_WORLD, dtype=np.float32)
    x_rot = p[:, 0] * np.cos(theta) + p[:, 2] * np.sin(theta)
    y_rot = p[:, 1]
    z_rot = -p[:, 0] * np.sin(theta) + p[:, 2] * np.cos(theta)
    depth = frame.CAMERA_DISTANCE_MM - z_rot
    half_fov = frame.FOV_MM / 2.0
    w, h = frame.DETECTOR_RESOLUTION
    u_px = (x_rot + half_fov) / frame.FOV_MM * w
    v_px = (y_rot + half_fov) / frame.FOV_MM * h
    valid = (
        (x_rot >= -half_fov)
        & (x_rot < half_fov)
        & (y_rot >= -half_fov)
        & (y_rot < half_fov)
        & (depth > 0)
    )
    return np.stack([u_px, v_px], axis=-1), depth, valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/foods/pro/FMT-SimGen/data/uniform_1000_20k")
    parser.add_argument("--num_random", type=int, default=1000)
    args = parser.parse_args()
    frame, turntable_camera_cls = _load_fmt_simgen_projection_deps()

    fixed = np.array(
        [
            [19.0, 20.0, 10.4],
            [19.0, 20.0, 15.0],
            [10.0, 20.0, 10.4],
            [30.0, 20.0, 10.4],
            [19.0, 10.0, 10.4],
        ],
        dtype=np.float32,
    )
    ds = FmtSimGenProjDataset(args.data_dir, split="train", config={"sample_num": args.num_random})
    item = ds[0]
    random_points = item["points_mm"].cpu().numpy()[: args.num_random].astype(np.float32)
    points = np.concatenate([fixed, random_points], axis=0)
    print("points_mm min:", points.min(axis=0).tolist())
    print("points_mm max:", points.max(axis=0).tolist())

    camera = turntable_camera_cls(
        {
            "angles": frame.ANGLES,
            "camera_distance_mm": frame.CAMERA_DISTANCE_MM,
            "detector_resolution": frame.DETECTOR_RESOLUTION,
            "fov_mm": frame.FOV_MM,
            "volume_center_world": frame.VOLUME_CENTER_WORLD,
        }
    )

    t_points = torch.from_numpy(points)
    for angle in frame.ANGLES:
        grid, depth, valid, uv_px, _uv_phys = project_points_mm_to_detector(t_points, angle)
        ref_uv, ref_depth, ref_valid = continuous_reference(points, angle, frame)
        cam_u, cam_v, cam_depth = camera.project_nodes_to_detector(points, angle)
        cam_valid = (cam_u >= 0) & (cam_v >= 0)

        uv_np = uv_px.numpy()
        depth_np = depth.squeeze(-1).numpy()
        valid_np = valid.numpy()
        print(f"angle={angle}")
        print("  max_abs_diff_u_px:", float(np.max(np.abs(uv_np[:, 0] - ref_uv[:, 0]))))
        print("  max_abs_diff_v_px:", float(np.max(np.abs(uv_np[:, 1] - ref_uv[:, 1]))))
        print("  max_abs_diff_depth:", float(np.max(np.abs(depth_np - ref_depth))))
        print("  valid agreement ratio:", float(np.mean(valid_np == ref_valid)))
        tt_u_diff = float(np.max(np.abs(cam_u[cam_valid] - uv_np[cam_valid, 0])))
        tt_v_diff = float(np.max(np.abs(cam_v[cam_valid] - uv_np[cam_valid, 1])))
        print("  turntable_int_px_max_diff_u:", tt_u_diff)
        print("  turntable_int_px_max_diff_v:", tt_v_diff)
        print("  turntable_depth_max_diff:", float(np.max(np.abs(cam_depth - depth_np))))
        if angle == 0:
            center_grid = grid[0].numpy().tolist()
            center_uv = uv_np[0].tolist()
            center_depth = float(depth_np[0])
            print("  center grid:", center_grid)
            print("  center uv_px:", center_uv)
            print("  center depth:", center_depth)


if __name__ == "__main__":
    main()
