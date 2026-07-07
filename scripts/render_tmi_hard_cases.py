#!/usr/bin/env python3
"""Render TMI-style internal-mouse comparison panels for selected FMT-SimGen cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from skimage import measure

SPACING_MM = 0.2
METHOD_TITLES = {
    "gt": "Ground Truth",
    "fem_coarse": "FEM coarse prior (diagnostic)",
    "gisc_fmt": "GISC-FMT",
    "uhr_deepfmt": "UHR-DeepFMT",
    "pah2t_former": "PAH2T-Former",
    "gaicn": "GAICN",
    "two_stage_deepfmt_fixed": "Two-stage DeepFMT",
    "two_stage_deepfmt": "Two-stage DeepFMT legacy",
    "pgdpnn": "PGDPNN",
    "fem2vox_unet": "FEM2Vox U-Net",
    "fem2vox_unet_residual": "FEM2Vox residual",
    "vox_dmrn": "Vox-DMRN",
    "fmt_reconnet": "FMT-ReconNet",
    "map_pgan": "MAP-PGAN",
    "d2_recst": "D2-RecST",
    "dspgn": "DSPGN",
    "tikhonov_fem": "FEM Tikhonov",
    "l1_fem": "FEM L1",
    "elasticnet_fem": "FEM ElasticNet",
    "fista_fem": "FEM FISTA",
    "stomp_fem": "FEM StOMP",
}
PREDICTION_DIRS = {
    "gisc_fmt": "test/gisc_fmt/predictions",
    "uhr_deepfmt": "test/uhr_deepfmt/predictions",
    "pah2t_former": "current_code_runs/pah2t_former/selected_predictions/predictions",
    "gaicn": "current_code_runs/gaicn/test300/predictions",
    "two_stage_deepfmt_fixed": "current_code_runs/two_stage_deepfmt_fixed/test300/predictions",
    "two_stage_deepfmt": "test/two_stage_deepfmt/predictions",
    "pgdpnn": "test/pgdpnn/predictions",
    "fem2vox_unet": "test/fem2vox_unet/predictions",
    "fem2vox_unet_residual": (
        "current_code_runs/fem2vox_unet_residual/test300/predictions"
    ),
    "vox_dmrn": "test/vox_dmrn/predictions",
    "fmt_reconnet": "test/fmt_reconnet/predictions",
    "map_pgan": "test/map_pgan/predictions",
    "d2_recst": "test/d2_recst/predictions",
    "dspgn": "test/dspgn/predictions",
    "tikhonov_fem": "current_code_runs/tikhonov_fem/test300/predictions",
    "l1_fem": "current_code_runs/l1_fem/test300/predictions",
    "elasticnet_fem": "current_code_runs/elasticnet_fem/test300/predictions",
    "fista_fem": "current_code_runs/fista_fem/test300/predictions",
    "stomp_fem": "current_code_runs/stomp_fem/test300/predictions",
}
SUMMARY_DIRS = {
    method: str(Path(directory).parent) for method, directory in PREDICTION_DIRS.items()
}
SUMMARY_DIRS["pah2t_former"] = "current_code_runs/pah2t_former/test300"
for _method in ("tikhonov_fem", "l1_fem", "elasticnet_fem", "fista_fem", "stomp_fem"):
    SUMMARY_DIRS[_method] = f"current_code_runs/{_method}/test300"
ORGAN_STYLE = {
    4: ((0.72, 0.12, 0.12), 0.14),  # heart
    5: ((0.48, 0.70, 0.88), 0.10),  # lung
    6: ((0.48, 0.10, 0.10), 0.12),  # liver
    7: ((0.56, 0.25, 0.16), 0.14),  # kidney
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection_csv",
        type=Path,
        default=Path(
            "outputs/fmt_simgen_v2_3k_20k/paper_figures/tmi_hard_cases/selected_cases.csv"
        ),
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k/samples"),
    )
    parser.add_argument(
        "--shared_dir",
        type=Path,
        default=Path("/home/foods/pro/FMT-SimGen/output/shared_mesh_20k"),
    )
    parser.add_argument("--output_root", type=Path, default=Path("outputs/fmt_simgen_v2_3k_20k"))
    parser.add_argument("--save_dir", type=Path, default=None)
    parser.add_argument("--gisc_prediction_dir", type=Path, default=None)
    parser.add_argument("--gisc_summary_dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "uhr_deepfmt",
            "pah2t_former",
            "gaicn",
            "fem2vox_unet_residual",
            "gisc_fmt",
        ],
        help="Ordered comparison columns after Ground Truth. Missing prediction sets are skipped.",
    )
    return parser.parse_args()


def load_label_volume(shared_dir: Path) -> np.ndarray:
    path = shared_dir / "mcx_volume_trunk.bin"
    labels = np.fromfile(path, dtype=np.uint8)
    expected = 190 * 200 * 104
    if labels.size != expected:
        raise ValueError(f"Unexpected label volume size: {labels.size}, expected {expected}")
    return labels.reshape((104, 200, 190)).transpose(2, 1, 0)


def surface_from_mask(mask: np.ndarray) -> pv.PolyData | None:
    if not np.any(mask):
        return None
    vertices, faces, _, _ = measure.marching_cubes(mask.astype(np.float32), level=0.5)
    vertices = (vertices + 0.5) * SPACING_MM
    faces_pv = np.column_stack([np.full(len(faces), 3), faces]).astype(np.int64).ravel()
    return pv.PolyData(vertices, faces_pv)


def prediction_dir(args: argparse.Namespace, method: str) -> Path:
    if method == "gisc_fmt" and args.gisc_prediction_dir:
        return args.gisc_prediction_dir
    return args.output_root / PREDICTION_DIRS[method]


def load_prediction(
    args: argparse.Namespace, sample_dir: Path, sample_id: str, method: str
) -> tuple[np.ndarray, float]:
    if method == "gt":
        return np.load(sample_dir / "gt_voxels.npy").astype(np.float32), args.threshold
    if method == "fem_coarse":
        return np.load(sample_dir / "stage1_voxel.npy").astype(np.float32), args.threshold
    pred_path = prediction_dir(args, method) / f"{sample_id}.npz"
    with np.load(pred_path) as pred:
        threshold = float(pred["threshold"]) if "threshold" in pred else args.threshold
        return pred["pred"].astype(np.float32), threshold


def render_panel(
    body_surface: pv.PolyData,
    organ_surfaces: dict[int, pv.PolyData],
    volume: np.ndarray,
    output_path: Path,
    threshold: float,
) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=(760, 680))
    plotter.set_background("white")
    plotter.add_mesh(
        body_surface,
        color=(0.86, 0.80, 0.75),
        opacity=0.10,
        show_edges=False,
        smooth_shading=True,
    )
    for label, surface in organ_surfaces.items():
        color, opacity = ORGAN_STYLE[label]
        plotter.add_mesh(
            surface, color=color, opacity=opacity, show_edges=False, smooth_shading=True
        )
    source_surface = surface_from_mask(volume >= threshold)
    if source_surface is not None:
        plotter.add_mesh(
            source_surface,
            color=(0.90, 0.12, 0.06),
            opacity=0.96,
            show_edges=False,
            smooth_shading=True,
            specular=0.18,
        )
    plotter.camera_position = [
        (64.0, -45.0, 38.0),
        (19.0, 20.0, 10.0),
        (0.0, 0.0, 1.0),
    ]
    plotter.camera.zoom(1.18)
    plotter.screenshot(output_path)
    plotter.close()


def metric_label(row: dict[str, str], method: str) -> str:
    if method == "gt":
        return ""
    return f"Dice = {float(row[f'{method}_dice']):.3f}"


def meets_main_table_threshold(args: argparse.Namespace, method: str) -> bool:
    if method == "fem_coarse":
        summary_path = (
            args.output_root / "current_code_runs/fem_coarse/test300/metrics_summary.json"
        )
    elif method == "gisc_fmt" and args.gisc_summary_dir:
        summary_path = args.gisc_summary_dir / "metrics_summary.json"
    else:
        summary_path = args.output_root / SUMMARY_DIRS[method] / "metrics_summary.json"
    return float(json.loads(summary_path.read_text())["dice_mean"]) >= 0.4


def main() -> None:
    args = parse_args()
    save_dir = args.save_dir or args.selection_csv.parent / "renders"
    panel_dir = save_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    with args.selection_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    available_methods = []
    for method in args.methods:
        if method == "fem_coarse":
            available_methods.append(method)
            continue
        pred_dir = prediction_dir(args, method)
        predictions_complete = pred_dir.is_dir() and all(
            (pred_dir / f"{row['sample_id']}.npz").exists() for row in rows
        )
        if predictions_complete:
            available_methods.append(method)
        else:
            print(
                f"[WARN] Skipping {method}: "
                f"selected-case predictions are incomplete in {pred_dir}"
            )
    methods = [("Ground Truth", "gt")]
    for method in available_methods:
        suffix = "" if meets_main_table_threshold(args, method) else " [<0.4]"
        methods.append((METHOD_TITLES[method] + suffix, method))
    labels = load_label_volume(args.shared_dir)
    body_surface = surface_from_mask(labels > 0)
    if body_surface is None:
        raise SystemExit("Mouse body label volume is empty.")
    organ_surfaces = {
        label: surface
        for label in ORGAN_STYLE
        if (surface := surface_from_mask(labels == label)) is not None
    }

    figure, axes = plt.subplots(
        len(rows),
        len(methods),
        figsize=(3.2 * len(methods), 2.9 * len(rows)),
        squeeze=False,
    )
    for row_index, row in enumerate(rows):
        sample_id = row["sample_id"]
        sample_dir = args.data_root / sample_id
        for col_index, (title, method) in enumerate(methods):
            panel_path = panel_dir / f"{sample_id}_{method}.png"
            volume, threshold = load_prediction(args, sample_dir, sample_id, method)
            render_panel(body_surface, organ_surfaces, volume, panel_path, threshold)
            axis = axes[row_index, col_index]
            axis.imshow(plt.imread(panel_path))
            axis.axis("off")
            if row_index == 0:
                axis.set_title(title, fontsize=12, fontweight="bold")
            label = metric_label(row, method)
            if label:
                axis.text(
                    0.5,
                    0.03,
                    label,
                    ha="center",
                    va="bottom",
                    transform=axis.transAxes,
                    fontsize=10,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72},
                )
        axes[row_index, 0].text(
            0.01,
            0.98,
            f"{row['category'].replace('_', ' ')}\n{sample_id}\n"
            f"{row['num_foci']} foci | {row['shape_set']}",
            ha="left",
            va="top",
            transform=axes[row_index, 0].transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )
    figure.tight_layout(pad=0.3)
    figure.savefig(save_dir / "tmi_hard_cases.png", dpi=300, bbox_inches="tight")
    figure.savefig(save_dir / "tmi_hard_cases.pdf", bbox_inches="tight")
    plt.close(figure)
    print(save_dir / "tmi_hard_cases.png")


if __name__ == "__main__":
    main()
