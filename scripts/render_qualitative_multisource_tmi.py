#!/usr/bin/env python3
"""Render the TMI main-text qualitative comparison for multi-source cases."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from skimage import measure

SPACING_MM = 0.2
THRESHOLD = 0.5
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "fmt_simgen_v2_3k_20k"
DATA_ROOT = Path("/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k/samples")
SHARED_DIR = Path("/home/foods/pro/FMT-SimGen/output/shared_mesh_20k")
FIGURE_DIR = ROOT / "figures"
PANEL_DIR = FIGURE_DIR / "qualitative_multisource_tmi_panels"

METHODS = [
    ("UHR-DeepFMT", "uhr_deepfmt"),
    ("PAH²T-Former", "pah2t_former"),
    ("GAICN", "gaicn"),
    ("FEM2Vox-Res", "fem2vox_unet_residual"),
    ("GISC-FMT (Ours)", "gisc_fmt"),
    ("Ground Truth", "gt"),
]
PREDICTION_DIRS = {
    "uhr_deepfmt": [OUTPUT_ROOT / "test" / "uhr_deepfmt" / "predictions"],
    "pah2t_former": [
        OUTPUT_ROOT / "current_code_runs" / "pah2t_former" / "selected_predictions" / "predictions",
        OUTPUT_ROOT
        / "current_code_runs"
        / "pah2t_former"
        / "qualitative_multisource"
        / "predictions",
    ],
    "gaicn": [OUTPUT_ROOT / "current_code_runs" / "gaicn" / "test300" / "predictions"],
    "fem2vox_unet_residual": [
        OUTPUT_ROOT / "current_code_runs" / "fem2vox_unet_residual" / "test300" / "predictions"
    ],
    "gisc_fmt": [
        ROOT
        / "outputs"
        / "fmt_simgen_v2_e15_center_distance_precomputed"
        / "test_epoch52"
        / "predictions"
    ],
}
CASES = [
    {
        "panel": "(a)",
        "label": "Weak secondary focus",
        "sample_id": "sample_0409",
        "num_foci": 2,
        "shape_combo": "irregular + sphere",
        "depth_tier": "medium",
        "reason": (
            "The secondary sphere is substantially smaller than the irregular source; "
            "the case exposes incomplete weak-source recovery and boundary inflation."
        ),
    },
    {
        "panel": "(b)",
        "label": "Adjacent-source merging",
        "sample_id": "sample_1858",
        "num_foci": 3,
        "shape_combo": "ellipsoid + irregular",
        "depth_tier": "medium",
        "reason": (
            "The closest source centers are about 7.22 mm apart; baseline reconstructions "
            "miss or merge adjacent sources while the proposed method separates all three."
        ),
    },
    {
        "panel": "(c)",
        "label": "Mixed morphology",
        "sample_id": "sample_2483",
        "num_foci": 3,
        "shape_combo": "ellipsoid + irregular + sphere",
        "depth_tier": "medium",
        "reason": (
            "The three sources have distinct morphologies; baseline reconstructions retain "
            "only part of the combination while the proposed method preserves all sources."
        ),
    },
    {
        "panel": "(d)",
        "label": "Hard three-foci case",
        "sample_id": "sample_2945",
        "num_foci": 3,
        "shape_combo": "ellipsoid + sphere",
        "depth_tier": "deep",
        "reason": (
            "A deep three-source example with visible baseline reconstructions but residual "
            "missed-source, localization, and boundary errors."
        ),
    },
]
ORGAN_STYLE = {
    4: ((0.78, 0.48, 0.49), 0.14),
    5: ((0.60, 0.75, 0.86), 0.12),
    6: ((0.72, 0.50, 0.45), 0.13),
    7: ((0.70, 0.61, 0.53), 0.14),
}


def load_label_volume() -> np.ndarray:
    labels = np.fromfile(SHARED_DIR / "mcx_volume_trunk.bin", dtype=np.uint8)
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


def prediction_path(method: str, sample_id: str) -> Path:
    for directory in PREDICTION_DIRS[method]:
        path = directory / f"{sample_id}.npz"
        if path.exists():
            return path
    searched = ", ".join(str(path) for path in PREDICTION_DIRS[method])
    raise FileNotFoundError(f"Missing {method} prediction for {sample_id}; searched {searched}")


def load_volume(method: str, sample_id: str) -> np.ndarray:
    if method == "gt":
        return np.load(DATA_ROOT / sample_id / "gt_voxels.npy").astype(np.float32)
    with np.load(prediction_path(method, sample_id)) as prediction:
        return prediction["pred"].astype(np.float32)


def add_anatomical_context(
    plotter: pv.Plotter,
    body_surface: pv.PolyData,
    organ_surfaces: dict[int, pv.PolyData],
) -> None:
    plotter.add_mesh(
        body_surface,
        color=(0.80, 0.80, 0.80),
        opacity=0.10,
        show_edges=False,
        smooth_shading=True,
    )
    for label, surface in organ_surfaces.items():
        color, opacity = ORGAN_STYLE[label]
        plotter.add_mesh(
            surface,
            color=color,
            opacity=opacity,
            show_edges=False,
            smooth_shading=True,
        )


def render_panel(
    body_surface: pv.PolyData,
    organ_surfaces: dict[int, pv.PolyData],
    prediction: np.ndarray,
    gt: np.ndarray,
    output_path: Path,
    is_gt: bool,
) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=(820, 700))
    plotter.set_background("white")
    add_anatomical_context(plotter, body_surface, organ_surfaces)
    gt_surface = surface_from_mask(gt > 0.0)
    if is_gt:
        if gt_surface is not None:
            plotter.add_mesh(
                gt_surface,
                color=(0.00, 0.66, 0.78),
                opacity=0.86,
                show_edges=False,
                smooth_shading=True,
                specular=0.16,
            )
    else:
        prediction_surface = surface_from_mask(prediction >= THRESHOLD)
        if prediction_surface is not None:
            plotter.add_mesh(
                prediction_surface,
                color=(0.93, 0.19, 0.07),
                opacity=0.90,
                show_edges=False,
                smooth_shading=True,
                specular=0.16,
            )
        if gt_surface is not None:
            plotter.add_mesh(
                gt_surface,
                color=(0.00, 0.74, 0.86),
                opacity=0.42,
                style="wireframe",
                line_width=1.0,
            )
    plotter.camera_position = [
        (64.0, -45.0, 38.0),
        (19.0, 20.0, 10.0),
        (0.0, 0.0, 1.0),
    ]
    plotter.camera.zoom(1.18)
    plotter.screenshot(output_path)
    plotter.close()


def main() -> None:
    if any(case["num_foci"] < 2 for case in CASES):
        raise ValueError("The main-text qualitative figure must not contain single-focus cases.")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    (FIGURE_DIR / "qualitative_multisource_tmi_cases.json").write_text(
        json.dumps(
            {
                "threshold": THRESHOLD,
                "camera": "lateral view with slight dorsal oblique",
                "cases": CASES,
            },
            indent=2,
        )
        + "\n"
    )

    labels = load_label_volume()
    body_surface = surface_from_mask(labels > 0)
    if body_surface is None:
        raise ValueError("Mouse body surface is empty.")
    organ_surfaces = {
        label: surface
        for label in ORGAN_STYLE
        if (surface := surface_from_mask(labels == label)) is not None
    }
    figure, axes = plt.subplots(
        len(CASES),
        len(METHODS),
        figsize=(19.2, 11.1),
        squeeze=False,
    )
    for row_index, case in enumerate(CASES):
        sample_id = str(case["sample_id"])
        gt = load_volume("gt", sample_id)
        for col_index, (title, method) in enumerate(METHODS):
            panel_path = PANEL_DIR / f"{sample_id}_{method}.png"
            prediction = gt if method == "gt" else load_volume(method, sample_id)
            render_panel(body_surface, organ_surfaces, prediction, gt, panel_path, method == "gt")
            axis = axes[row_index, col_index]
            axis.imshow(plt.imread(panel_path))
            axis.axis("off")
            if row_index == 0:
                axis.set_title(title, fontsize=13, fontweight="bold", pad=8)
        label = (
            f"{case['panel']} {case['label']}\n"
            f"{case['num_foci']} foci | {case['shape_combo']}"
        )
        axes[row_index, 0].text(
            -0.09,
            0.50,
            label,
            ha="right",
            va="center",
            transform=axes[row_index, 0].transAxes,
            fontsize=10.5,
            linespacing=1.35,
        )
    figure.subplots_adjust(left=0.16, right=0.995, top=0.94, bottom=0.02, wspace=0.015, hspace=0.05)
    png_path = FIGURE_DIR / "qualitative_multisource_tmi.png"
    pdf_path = FIGURE_DIR / "qualitative_multisource_tmi.pdf"
    figure.savefig(png_path, dpi=360, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
