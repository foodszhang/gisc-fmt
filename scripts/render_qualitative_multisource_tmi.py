#!/usr/bin/env python3
"""Render TMI-style qualitative multi-source comparison figures.

The 3D block uses a fixed mouse-internal camera. The slice block uses the
Digimouse CT cropped/downsampled with the same trunk atlas crop as FMT-SimGen:
atlas voxel size 0.1 mm, trunk crop x=[0,38] mm, y=[34,74] mm, z=[0,20.8] mm,
then 2x downsampled to the common [190, 200, 104] voxel grid.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pyvista as pv
from matplotlib.lines import Line2D
from skimage import measure

SPACING_MM = 0.2
THRESHOLD = 0.5
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "fmt_simgen_v2_3k_20k"
DATA_ROOT = Path("/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k/samples")
SHARED_DIR = Path("/home/foods/pro/FMT-SimGen/output/shared_mesh_20k")
DIGMOUSE_DIR = Path("/home/foods/pro/FMT-SimGen/digmouse_data")

GT_COLOR = "#00A6B2"
PRED_COLOR = "#D95F02"
BODY_COLOR = "#9E9E9E"
BODY_EDGE_COLOR = "#8C8C8C"
ORGAN_COLOR = "#C9A79E"
BG_GRAY = "#F7F7F7"

METHODS_3D = [
    ("UHR-DeepFMT", "uhr_deepfmt"),
    ("PAH²T-Former", "pah2t_former"),
    ("GAICN", "gaicn"),
    ("FEM2Vox-Res", "fem2vox_unet_residual"),
    ("GISC-FMT (Ours)", "gisc_fmt"),
    ("Ground Truth", "gt"),
]
METHOD_TITLES = {method: title for title, method in METHODS_3D}
METHOD_TITLES["gt"] = "Ground Truth"

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
        "panel": "a",
        "label": "Weak secondary focus",
        "sample_id": "sample_0409",
        "num_foci": 2,
        "shape_combo": "irregular + sphere",
        "depth_tier": "medium",
        "reason": (
            "The secondary sphere is substantially smaller than the irregular source; "
            "the slice zoom highlights weak-source recovery and boundary inflation."
        ),
    },
    {
        "panel": "b",
        "label": "Adjacent-source merging",
        "sample_id": "sample_1858",
        "num_foci": 3,
        "shape_combo": "ellipsoid + irregular",
        "depth_tier": "medium",
        "reason": (
            "The closest source centers are about 7.22 mm apart; slice contours show "
            "whether adjacent lesions are merged or separated."
        ),
    },
    {
        "panel": "c",
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
        "panel": "d",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-slices", action="store_true", default=True)
    parser.add_argument("--slice-cases", default="a,b")
    parser.add_argument("--slice-methods", default="pah2t_former,gaicn,gisc_fmt,gt")
    parser.add_argument("--enhance-mouse-outline", action="store_true", default=True)
    parser.add_argument("--legend", action="store_true", default=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "paper_figures")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--format", default="png,pdf")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def requested_formats(value: str) -> list[str]:
    formats = [item.strip().lower() for item in value.split(",") if item.strip()]
    allowed = {"png", "pdf"}
    invalid = [item for item in formats if item not in allowed]
    if invalid:
        raise ValueError(f"Unsupported format(s): {invalid}. Allowed: {sorted(allowed)}")
    return formats or ["png", "pdf"]


def case_by_panel(panel: str) -> dict:
    for case in CASES:
        if case["panel"] == panel:
            return case
    raise KeyError(f"Unknown case panel: {panel}")


def load_label_volume() -> np.ndarray:
    labels = np.fromfile(SHARED_DIR / "mcx_volume_trunk.bin", dtype=np.uint8)
    expected = 190 * 200 * 104
    if labels.size != expected:
        raise ValueError(f"Unexpected label volume size: {labels.size}, expected {expected}")
    return labels.reshape((104, 200, 190)).transpose(2, 1, 0)


def extract_digmouse_ct_if_needed() -> Path:
    hdr = DIGMOUSE_DIR / "ct_data" / "ct_380x992x208.hdr"
    img = DIGMOUSE_DIR / "ct_data" / "ct_380x992x208.img"
    if hdr.exists() and img.exists():
        return hdr
    zip_path = DIGMOUSE_DIR / "ct_data.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing Digimouse CT archive: {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(DIGMOUSE_DIR)
    if not hdr.exists() or not img.exists():
        raise FileNotFoundError(f"CT archive did not contain expected files under {hdr.parent}")
    return hdr


def load_aligned_ct_volume() -> tuple[np.ndarray, str]:
    hdr = extract_digmouse_ct_if_needed()
    image = nib.load(str(hdr))
    ct = np.asarray(image.dataobj).astype(np.float32)
    if ct.ndim == 4 and ct.shape[-1] == 1:
        ct = ct[..., 0]
    if ct.shape != (380, 992, 208):
        raise ValueError(f"Unexpected Digimouse CT shape {ct.shape}; expected (380, 992, 208)")

    # FMT-SimGen trunk crop: x=[0,38] mm, y=[34,74] mm, z=[0,20.8] mm at 0.1 mm.
    crop = ct[0:380, 340:740, 0:208]
    if crop.shape != (380, 400, 208):
        raise ValueError(f"Unexpected CT crop shape {crop.shape}")
    downsampled = crop.reshape(190, 2, 200, 2, 104, 2).mean(axis=(1, 3, 5))
    return downsampled.astype(np.float32), "CT"


def fixed_ct_normalize(ct: np.ndarray, body_mask: np.ndarray) -> np.ndarray:
    values = ct[body_mask]
    lo, hi = np.percentile(values, [1.0, 99.0]) if values.size else np.percentile(ct, [1.0, 99.0])
    return np.clip((ct - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


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
    enhance_mouse_outline: bool,
) -> None:
    plotter.add_mesh(
        body_surface,
        color=BODY_COLOR,
        opacity=0.16 if enhance_mouse_outline else 0.10,
        show_edges=False,
        smooth_shading=True,
    )
    if enhance_mouse_outline:
        plotter.add_mesh(
            body_surface,
            color=BODY_EDGE_COLOR,
            opacity=0.18,
            style="wireframe",
            line_width=0.4,
        )
    for surface in organ_surfaces.values():
        plotter.add_mesh(
            surface,
            color=ORGAN_COLOR,
            opacity=0.075,
            show_edges=False,
            smooth_shading=True,
        )


def render_3d_panel(
    body_surface: pv.PolyData,
    organ_surfaces: dict[int, pv.PolyData],
    prediction: np.ndarray,
    gt: np.ndarray,
    output_path: Path,
    is_gt: bool,
    enhance_mouse_outline: bool,
) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=(820, 700))
    plotter.set_background("white")
    add_anatomical_context(plotter, body_surface, organ_surfaces, enhance_mouse_outline)
    gt_surface = surface_from_mask(gt > 0.0)
    if is_gt:
        if gt_surface is not None:
            plotter.add_mesh(
                gt_surface,
                color=GT_COLOR,
                opacity=0.62,
                show_edges=False,
                smooth_shading=True,
                specular=0.14,
            )
    else:
        prediction_surface = surface_from_mask(prediction >= THRESHOLD)
        if gt_surface is not None:
            plotter.add_mesh(
                gt_surface,
                color=GT_COLOR,
                opacity=0.50,
                show_edges=False,
                smooth_shading=True,
                specular=0.10,
            )
        if prediction_surface is not None:
            plotter.add_mesh(
                prediction_surface,
                color=PRED_COLOR,
                opacity=0.34,
                show_edges=False,
                smooth_shading=True,
                specular=0.16,
            )
            plotter.add_mesh(
                prediction_surface,
                color=PRED_COLOR,
                opacity=0.92,
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


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    from scipy import ndimage

    labels, num = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))
    comps = []
    for idx in range(1, num + 1):
        coords = np.argwhere(labels == idx)
        if coords.shape[0] >= 8:
            comps.append(coords)
    return comps


def choose_slice(gt_mask: np.ndarray) -> tuple[int, int]:
    components = connected_components(gt_mask)
    best = (-1, -1, -1, 0)
    for axis in range(3):
        for index in range(gt_mask.shape[axis]):
            covered = 0
            area = 0
            for coords in components:
                hits = coords[:, axis] == index
                if np.any(hits):
                    covered += 1
                    area += int(hits.sum())
            key = (covered, area, -abs(index - gt_mask.shape[axis] // 2), axis)
            if key > best:
                best = key
                best_axis = axis
                best_index = index
    return int(best_axis), int(best_index)


def slice2d(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return volume[index, :, :].T
    if axis == 1:
        return volume[:, index, :].T
    return volume[:, :, index].T


def projection2d(mask: np.ndarray, axis: int) -> np.ndarray:
    return np.any(mask, axis=axis).T


def crop_box_from_gt(gt_mask: np.ndarray, axis: int, padding: int = 12) -> tuple[slice, slice]:
    projection = projection2d(gt_mask, axis)
    coords = np.argwhere(projection)
    if coords.size == 0:
        return slice(None), slice(None)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    r0 = max(int(mins[0]) - padding, 0)
    c0 = max(int(mins[1]) - padding, 0)
    r1 = min(int(maxs[0]) + padding + 1, projection.shape[0])
    c1 = min(int(maxs[1]) + padding + 1, projection.shape[1])
    return slice(r0, r1), slice(c0, c1)


def draw_contours(ax, mask: np.ndarray, color: str, linewidth: float) -> None:
    for contour in measure.find_contours(mask.astype(np.float32), 0.5):
        ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth)


def render_slice_panel(
    ax,
    background: np.ndarray,
    background_label: str,
    gt: np.ndarray,
    prediction: np.ndarray | None,
    axis: int,
    index: int,
    crop: tuple[slice, slice],
    title: str,
    show_background_label: bool,
) -> None:
    bg_slice = slice2d(background, axis, index)[crop]
    gt_slice = slice2d(gt > 0.0, axis, index)[crop]
    ax.imshow(bg_slice, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    if prediction is not None:
        pred_slice = slice2d(prediction >= THRESHOLD, axis, index)[crop]
        draw_contours(ax, pred_slice, PRED_COLOR, 1.45)
    draw_contours(ax, gt_slice, GT_COLOR, 1.65)
    ax.set_title(title, fontsize=10.5, fontweight="bold" if "GISC" in title else "normal", pad=3)
    ax.axis("off")
    if show_background_label:
        ax.text(
            0.02,
            0.04,
            background_label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.5,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 1.5},
        )


def add_case_label(ax, case: dict, x: float = -0.08) -> None:
    label = f"({case['panel']}) {case['label']}\n{case['num_foci']} foci | {case['shape_combo']}"
    ax.text(
        x,
        0.50,
        label,
        ha="right",
        va="center",
        transform=ax.transAxes,
        fontsize=10.5,
        linespacing=1.35,
    )


def add_global_legend(fig, y: float = 0.025) -> None:
    handles = [
        Line2D([0], [0], color=GT_COLOR, lw=3, label="Cyan: Ground truth"),
        Line2D([0], [0], color=PRED_COLOR, lw=3, label="Orange-red: Prediction"),
        Line2D([0], [0], color=BODY_EDGE_COLOR, lw=3, label="Gray: anatomical context"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=3,
        frameon=False,
        fontsize=11,
    )


def ensure_can_write(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing figure(s):\n{formatted}\nUse --overwrite."
        )


def save_figure(fig, stem: str, output_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    paths = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi if fmt == "png" else None, bbox_inches="tight", facecolor="white")
        paths.append(path)
    return paths


def build_3d_panels(
    cases: list[dict],
    methods: list[tuple[str, str]],
    panel_dir: Path,
    labels: np.ndarray,
    enhance_mouse_outline: bool,
) -> dict[tuple[str, str], Path]:
    body_surface = surface_from_mask(labels > 0)
    if body_surface is None:
        raise ValueError("Mouse body surface is empty.")
    organ_surfaces = {
        label: surface
        for label in (4, 5, 6, 7, 8, 9)
        if (surface := surface_from_mask(labels == label)) is not None
    }
    panel_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for case in cases:
        sample_id = str(case["sample_id"])
        gt = load_volume("gt", sample_id)
        for _, method in methods:
            panel_path = panel_dir / f"{sample_id}_{method}.png"
            prediction = gt if method == "gt" else load_volume(method, sample_id)
            render_3d_panel(
                body_surface,
                organ_surfaces,
                prediction,
                gt,
                panel_path,
                method == "gt",
                enhance_mouse_outline,
            )
            out[(sample_id, method)] = panel_path
    return out


def make_3d_only(
    cases: list[dict],
    methods: list[tuple[str, str]],
    panel_paths: dict[tuple[str, str], Path],
    output_dir: Path,
    formats: list[str],
    dpi: int,
    legend: bool,
) -> None:
    fig, axes = plt.subplots(len(cases), len(methods), figsize=(18.2, 10.6), squeeze=False)
    for row, case in enumerate(cases):
        sample_id = str(case["sample_id"])
        for col, (title, method) in enumerate(methods):
            ax = axes[row, col]
            ax.imshow(plt.imread(panel_paths[(sample_id, method)]))
            ax.axis("off")
            if row == 0:
                ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        add_case_label(axes[row, 0], case)
    fig.subplots_adjust(
        left=0.16,
        right=0.995,
        top=0.94,
        bottom=0.075 if legend else 0.02,
        wspace=0.012,
        hspace=0.035,
    )
    if legend:
        add_global_legend(fig, y=0.015)
    save_figure(fig, "qualitative_comparison_3d_only", output_dir, formats, dpi)
    plt.close(fig)


def make_slices_only(
    slice_cases: list[dict],
    slice_methods: list[str],
    ct: np.ndarray,
    background_label: str,
    output_dir: Path,
    formats: list[str],
    dpi: int,
    legend: bool,
) -> None:
    fig, axes = plt.subplots(
        len(slice_cases),
        len(slice_methods),
        figsize=(4.2 * len(slice_methods), 3.2 * len(slice_cases)),
    )
    axes = np.atleast_2d(axes)
    for row, case in enumerate(slice_cases):
        sample_id = str(case["sample_id"])
        gt = load_volume("gt", sample_id)
        axis, index = choose_slice(gt > 0.0)
        crop = crop_box_from_gt(gt > 0.0, axis)
        for col, method in enumerate(slice_methods):
            prediction = None if method == "gt" else load_volume(method, sample_id)
            render_slice_panel(
                axes[row, col],
                ct,
                background_label,
                gt,
                prediction,
                axis,
                index,
                crop,
                METHOD_TITLES[method],
                show_background_label=(row == 0 and col == 0),
            )
        add_case_label(axes[row, 0], case, x=-0.13)
    fig.subplots_adjust(
        left=0.18,
        right=0.995,
        top=0.92,
        bottom=0.12 if legend else 0.04,
        wspace=0.045,
        hspace=0.26,
    )
    if legend:
        add_global_legend(fig, y=0.03)
    save_figure(fig, "qualitative_comparison_slices_only", output_dir, formats, dpi)
    plt.close(fig)


def make_3d_slice(
    cases: list[dict],
    methods_3d: list[tuple[str, str]],
    panel_paths: dict[tuple[str, str], Path],
    slice_cases: list[dict],
    slice_methods: list[str],
    ct: np.ndarray,
    background_label: str,
    output_dir: Path,
    formats: list[str],
    dpi: int,
    legend: bool,
) -> None:
    width = max(18.2, 4.1 * len(slice_methods))
    fig = plt.figure(figsize=(width, 14.2))
    outer = fig.add_gridspec(2, 1, height_ratios=[4.2, 1.35], hspace=0.08)
    top = outer[0].subgridspec(len(cases), len(methods_3d), wspace=0.012, hspace=0.035)
    for row, case in enumerate(cases):
        sample_id = str(case["sample_id"])
        first_ax = None
        for col, (title, method) in enumerate(methods_3d):
            ax = fig.add_subplot(top[row, col])
            if first_ax is None:
                first_ax = ax
            ax.imshow(plt.imread(panel_paths[(sample_id, method)]))
            ax.axis("off")
            if row == 0:
                ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        add_case_label(first_ax, case)

    bottom = outer[1].subgridspec(len(slice_cases), len(slice_methods), wspace=0.035, hspace=0.18)
    for row, case in enumerate(slice_cases):
        sample_id = str(case["sample_id"])
        gt = load_volume("gt", sample_id)
        axis, index = choose_slice(gt > 0.0)
        crop = crop_box_from_gt(gt > 0.0, axis)
        first_ax = None
        for col, method in enumerate(slice_methods):
            ax = fig.add_subplot(bottom[row, col])
            if first_ax is None:
                first_ax = ax
            prediction = None if method == "gt" else load_volume(method, sample_id)
            title = METHOD_TITLES[method] if row == 0 else ""
            render_slice_panel(
                ax,
                ct,
                background_label,
                gt,
                prediction,
                axis,
                index,
                crop,
                title,
                show_background_label=(row == 0 and col == 0),
            )
        add_case_label(first_ax, case, x=-0.10)
    fig.subplots_adjust(
        left=0.16,
        right=0.995,
        top=0.955,
        bottom=0.07 if legend else 0.02,
    )
    if legend:
        add_global_legend(fig, y=0.018)
    save_figure(fig, "qualitative_comparison_3d_slice", output_dir, formats, dpi)
    plt.close(fig)


def write_case_json(output_dir: Path, background_label: str) -> None:
    (output_dir / "qualitative_comparison_cases.json").write_text(
        json.dumps(
            {
                "threshold": THRESHOLD,
                "slice_background": background_label,
                "ct_alignment": (
                    "Digimouse CT ct_380x992x208 cropped to x=[0,38] mm, "
                    "y=[34,74] mm, z=[0,20.8] mm and mean-downsampled 2x to [190,200,104]."
                ),
                "cases": CASES,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    args = parse_args()
    if any(case["num_foci"] < 2 for case in CASES):
        raise ValueError("The main-text qualitative figure must not contain single-focus cases.")
    formats = requested_formats(args.format)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = [
        output_dir / f"qualitative_comparison_3d_slice.{fmt}" for fmt in formats
    ] + [
        output_dir / f"qualitative_comparison_3d_only.{fmt}" for fmt in formats
    ] + [
        output_dir / f"qualitative_comparison_slices_only.{fmt}" for fmt in formats
    ]
    ensure_can_write(expected, args.overwrite)

    labels = load_label_volume()
    body_mask = labels > 0
    try:
        ct_raw, background_label = load_aligned_ct_volume()
        background = fixed_ct_normalize(ct_raw, body_mask)
    except Exception as exc:
        print(f"[WARN] Falling back to anatomical mask background because CT loading failed: {exc}")
        background = body_mask.astype(np.float32) * 0.65 + (labels > 2).astype(np.float32) * 0.25
        background_label = "Anatomical mask"

    panel_dir = output_dir / "qualitative_comparison_panels"
    panel_paths = build_3d_panels(
        CASES,
        METHODS_3D,
        panel_dir,
        labels,
        enhance_mouse_outline=args.enhance_mouse_outline,
    )
    slice_cases = [
        case_by_panel(panel.strip()) for panel in args.slice_cases.split(",") if panel.strip()
    ]
    slice_methods = [method.strip() for method in args.slice_methods.split(",") if method.strip()]
    for method in slice_methods:
        if method != "gt" and method not in PREDICTION_DIRS:
            raise ValueError(f"Unknown slice method: {method}")

    make_3d_only(CASES, METHODS_3D, panel_paths, output_dir, formats, args.dpi, args.legend)
    if args.with_slices:
        make_slices_only(
            slice_cases,
            slice_methods,
            background,
            background_label,
            output_dir,
            formats,
            args.dpi,
            args.legend,
        )
        make_3d_slice(
            CASES,
            METHODS_3D,
            panel_paths,
            slice_cases,
            slice_methods,
            background,
            background_label,
            output_dir,
            formats,
            args.dpi,
            args.legend,
        )
    write_case_json(output_dir, background_label)
    for path in expected:
        if path.exists():
            print(path)


if __name__ == "__main__":
    main()
