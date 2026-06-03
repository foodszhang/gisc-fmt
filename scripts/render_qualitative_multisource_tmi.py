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
from matplotlib.patches import Rectangle
from skimage import measure

SPACING_MM = 0.2
THRESHOLD = 0.5
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "fmt_simgen_v2_3k_20k"
DATA_ROOT = Path("/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k/samples")
SHARED_DIR = Path("/home/foods/pro/FMT-SimGen/output/shared_mesh_20k")
DIGMOUSE_DIR = Path("/home/foods/pro/FMT-SimGen/digmouse_data")

GT_COLOR = "#11B7C8"
PRED_COLOR = "#D96B00"
BODY_COLOR = "#D8D8D8"
BODY_EDGE_COLOR = "#8A8A8A"
ORGAN_COLOR = "#CDB8A7"
BG_GRAY = "#F7F7F7"
SLICE_SLAB_RADIUS = 4

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

ORGAN_STYLE = {
    2: ("#B4B4B4", 0.12),
    4: ("#D96F77", 0.22),
    5: ("#78B6DD", 0.20),
    6: ("#D18A78", 0.21),
    7: ("#C3A16F", 0.22),
    8: ("#8FBA79", 0.18),
    9: ("#8FC7E8", 0.17),
}

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

CASE_POOLS = [
    {
        "panel": "a",
        "label": "Case I: Weak secondary-source recovery",
        "num_foci": 2,
        "shape_combo": "irregular + sphere",
        "reason": (
            "The secondary sphere is substantially smaller than the irregular source; "
            "the slice zoom highlights weak-source recovery and boundary inflation."
        ),
        "candidates": ["sample_0409", "sample_1837", "sample_2069"],
    },
    {
        "panel": "b",
        "label": "Case II: Adjacent-source separation",
        "num_foci": 3,
        "shape_combo": "ellipsoid + irregular",
        "reason": (
            "The closest source centers are close enough to stress separability; slice zooms show "
            "whether adjacent lesions are merged or separated."
        ),
        "candidates": ["sample_1858", "sample_1404", "sample_0473", "sample_2059"],
    },
    {
        "panel": "c",
        "label": "Case III: Mixed-morphology reconstruction",
        "num_foci": 3,
        "shape_combo": "ellipsoid + irregular + sphere",
        "reason": (
            "The three sources have distinct morphologies; baseline reconstructions retain "
            "only part of the combination while the proposed method preserves all sources."
        ),
        "candidates": ["sample_2483", "sample_2077", "sample_2357", "sample_1378"],
    },
    {
        "panel": "d",
        "label": "Case IV: Low-contrast three-focus localization",
        "num_foci": 3,
        "shape_combo": "ellipsoid + sphere",
        "reason": (
            "A deep three-source example with visible baseline reconstructions but residual "
            "missed-source, localization, and boundary errors."
        ),
        "candidates": ["sample_2945", "sample_0128", "sample_1741"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-slices", action="store_true", default=True)
    parser.add_argument("--slice-cases", default="a")
    parser.add_argument(
        "--slice-methods",
        default="uhr_deepfmt,pah2t_former,gaicn,fem2vox_unet_residual,gisc_fmt,gt",
    )
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


def case_by_panel(cases: list[dict], panel: str) -> dict:
    for case in cases:
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


def mask_outside_ratio(mask: np.ndarray, body_mask: np.ndarray) -> float:
    fg = mask.astype(bool)
    return float(np.logical_and(fg, ~body_mask).sum() / max(int(fg.sum()), 1))


def centroid_inside_body(mask: np.ndarray, body_mask: np.ndarray) -> bool:
    coords = np.argwhere(mask.astype(bool))
    if coords.size == 0:
        return False
    centroid = np.rint(coords.mean(axis=0)).astype(int)
    centroid = np.clip(centroid, 0, np.array(body_mask.shape) - 1)
    return bool(body_mask[tuple(centroid)])


def candidate_prediction_available(sample_id: str) -> list[str]:
    missing = []
    for _, method in METHODS_3D:
        if method == "gt":
            continue
        try:
            prediction_path(method, sample_id)
        except FileNotFoundError:
            missing.append(method)
    return missing


def select_filtered_cases(body_mask: np.ndarray, output_dir: Path) -> list[dict]:
    selected = []
    log_lines = [
        "panel\tsample_id\tstatus\tgt_outside_ratio\tpred_outside_ratio\treason",
    ]
    for pool in CASE_POOLS:
        selected_case = None
        for sample_id in pool["candidates"]:
            reasons = []
            gt = load_volume("gt", sample_id) > 0.0
            gt_outside = mask_outside_ratio(gt, body_mask)
            pred_outside = float("nan")
            missing = candidate_prediction_available(sample_id)
            if missing:
                reasons.append("missing_prediction:" + ",".join(missing))
            if gt_outside > 0.01:
                reasons.append(f"gt_outside_ratio>{0.01:g}")
            if not centroid_inside_body(gt, body_mask):
                reasons.append("gt_centroid_outside_body")
            if not missing:
                pred = load_volume("gisc_fmt", sample_id) >= THRESHOLD
                pred_outside = mask_outside_ratio(pred, body_mask)
                if pred_outside > 0.05:
                    reasons.append(f"pred_outside_ratio>{0.05:g}")
            status = "selected" if not reasons and selected_case is None else "rejected"
            if status == "selected":
                selected_case = {
                    **{key: value for key, value in pool.items() if key != "candidates"},
                    "sample_id": sample_id,
                    "gt_outside_ratio": gt_outside,
                    "pred_outside_ratio": pred_outside,
                }
                reason = "passed"
            else:
                reason = ";".join(reasons) if reasons else "lower_priority_candidate"
            log_lines.append(
                "\t".join(
                    [
                        str(pool["panel"]),
                        sample_id,
                        status,
                        f"{gt_outside:.6f}",
                        "nan" if np.isnan(pred_outside) else f"{pred_outside:.6f}",
                        reason,
                    ]
                )
            )
            if selected_case is not None:
                break
        if selected_case is None:
            raise RuntimeError(f"No valid qualitative case found for panel {pool['panel']}")
        selected.append(selected_case)
    (output_dir / "qualitative_case_filter_log.txt").write_text("\n".join(log_lines) + "\n")
    return selected


def add_anatomical_context(
    plotter: pv.Plotter,
    body_surface: pv.PolyData,
    organ_surfaces: dict[int, pv.PolyData],
    enhance_mouse_outline: bool,
) -> None:
    plotter.add_mesh(
        body_surface,
        color=BODY_COLOR,
        opacity=0.075 if enhance_mouse_outline else 0.055,
        show_edges=False,
        smooth_shading=True,
        specular=0.04,
        diffuse=0.86,
    )
    if enhance_mouse_outline:
        plotter.add_mesh(
            body_surface,
            color=BODY_EDGE_COLOR,
            opacity=0.075,
            style="wireframe",
            line_width=0.28,
        )
    for label, surface in organ_surfaces.items():
        color, opacity = ORGAN_STYLE.get(label, (ORGAN_COLOR, 0.14))
        plotter.add_mesh(
            surface,
            color=color,
            opacity=opacity,
            show_edges=False,
            smooth_shading=True,
            specular=0.06,
            diffuse=0.80,
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
                opacity=0.88,
                show_edges=False,
                smooth_shading=True,
                specular=0.22,
            )
            plotter.add_mesh(
                gt_surface,
                color=GT_COLOR,
                opacity=0.90,
                style="wireframe",
                line_width=0.85,
            )
    else:
        prediction_surface = surface_from_mask(prediction >= THRESHOLD)
        if prediction_surface is not None:
            plotter.add_mesh(
                prediction_surface,
                color=PRED_COLOR,
                opacity=0.86,
                show_edges=False,
                smooth_shading=True,
                specular=0.22,
            )
            plotter.add_mesh(
                prediction_surface,
                color=PRED_COLOR,
                opacity=0.90,
                style="wireframe",
                line_width=0.65,
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


def slice2d_slab(volume: np.ndarray, axis: int, index: int, radius: int, mask: bool) -> np.ndarray:
    start = max(index - radius, 0)
    stop = min(index + radius + 1, volume.shape[axis])
    slab = np.take(volume, indices=range(start, stop), axis=axis)
    reduced = np.any(slab > 0, axis=axis) if mask else np.mean(slab, axis=axis)
    return reduced.T


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


def target_coords_for_zoom(gt_mask: np.ndarray, panel: str) -> np.ndarray:
    components = connected_components(gt_mask)
    if not components:
        return np.argwhere(gt_mask)
    if panel == "a":
        return min(components, key=len)
    if panel == "b" and len(components) >= 2:
        centers = [coords.mean(axis=0) for coords in components]
        best_pair = (0, 1)
        best_dist = float("inf")
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = float(np.linalg.norm(centers[i] - centers[j]))
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (i, j)
        return np.concatenate([components[best_pair[0]], components[best_pair[1]]], axis=0)
    return np.argwhere(gt_mask)


def crop_box_from_coords(
    coords3d: np.ndarray,
    axis: int,
    shape2d: tuple[int, int],
    padding: int = 14,
) -> tuple[slice, slice]:
    if coords3d.size == 0:
        return slice(None), slice(None)
    if axis == 0:
        coords2d = np.column_stack([coords3d[:, 2], coords3d[:, 1]])
    elif axis == 1:
        coords2d = np.column_stack([coords3d[:, 2], coords3d[:, 0]])
    else:
        coords2d = np.column_stack([coords3d[:, 1], coords3d[:, 0]])
    mins = coords2d.min(axis=0)
    maxs = coords2d.max(axis=0)
    r0 = max(int(mins[0]) - padding, 0)
    c0 = max(int(mins[1]) - padding, 0)
    r1 = min(int(maxs[0]) + padding + 1, shape2d[0])
    c1 = min(int(maxs[1]) + padding + 1, shape2d[1])
    return slice(r0, r1), slice(c0, c1)


def zoom_crop_from_gt(gt_mask: np.ndarray, panel: str, axis: int, padding: int = 14) -> tuple[slice, slice]:
    shape2d = projection2d(gt_mask, axis).shape
    return crop_box_from_coords(target_coords_for_zoom(gt_mask, panel), axis, shape2d, padding)


def overlay_mask(ax, mask: np.ndarray, color: str, alpha: float) -> None:
    if not np.any(mask):
        return
    import matplotlib.colors as mcolors

    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[..., :3] = mcolors.to_rgb(color)
    rgba[..., 3] = mask.astype(np.float32) * alpha
    ax.imshow(rgba, interpolation="nearest")


def crop_bounds(crop: tuple[slice, slice]) -> tuple[int, int, int, int]:
    row_slice, col_slice = crop
    r0 = 0 if row_slice.start is None else int(row_slice.start)
    r1 = int(row_slice.stop)
    c0 = 0 if col_slice.start is None else int(col_slice.start)
    c1 = int(col_slice.stop)
    return r0, r1, c0, c1


def relative_crop(inner: tuple[slice, slice], outer: tuple[slice, slice]) -> tuple[slice, slice]:
    inner_r0, inner_r1, inner_c0, inner_c1 = crop_bounds(inner)
    outer_r0, _, outer_c0, _ = crop_bounds(outer)
    return (
        slice(max(inner_r0 - outer_r0, 0), max(inner_r1 - outer_r0, 0)),
        slice(max(inner_c0 - outer_c0, 0), max(inner_c1 - outer_c0, 0)),
    )


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
    roi_crop: tuple[slice, slice] | None = None,
) -> None:
    bg_slice = slice2d_slab(background, axis, index, SLICE_SLAB_RADIUS, mask=False)[crop]
    gt_slice = slice2d_slab(gt > 0.0, axis, index, SLICE_SLAB_RADIUS, mask=True)[crop]
    ax.imshow(bg_slice, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    overlay_mask(ax, gt_slice, GT_COLOR, 0.24)
    if prediction is not None:
        pred_slice = slice2d_slab(
            prediction >= THRESHOLD, axis, index, SLICE_SLAB_RADIUS, mask=True
        )[crop]
        overlay_mask(ax, pred_slice, PRED_COLOR, 0.22)
        draw_contours(ax, pred_slice, PRED_COLOR, 1.8)
    draw_contours(ax, gt_slice, GT_COLOR, 1.9)
    ax.set_title(title, fontsize=10.5, fontweight="bold" if "GISC" in title else "normal", pad=3)
    if roi_crop is not None:
        rr, cc = relative_crop(roi_crop, crop)
        r0, r1, c0, c1 = crop_bounds((rr, cc))
        ax.add_patch(
            Rectangle(
                (c0, r0),
                max(c1 - c0, 1),
                max(r1 - r0, 1),
                fill=False,
                edgecolor="#FFFFFF",
                linewidth=1.0,
                linestyle="--",
            )
        )
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
    case_label = str(case["label"])
    if ": " in case_label:
        case_id, description = case_label.split(": ", 1)
        label = (
            f"({case['panel']}) {case_id}\n"
            f"{description}\n"
            f"{case['num_foci']} foci | {case['shape_combo']}"
        )
    else:
        label = f"({case['panel']}) {case_label}\n{case['num_foci']} foci | {case['shape_combo']}"
    ax.text(
        x,
        0.50,
        label,
        ha="right",
        va="center",
        transform=ax.transAxes,
        fontsize=10.2,
        linespacing=1.28,
    )


def add_global_legend(fig, y: float = 0.025) -> None:
    handles = [
        Line2D([0], [0], color=GT_COLOR, lw=3, label="Cyan: Ground truth"),
        Line2D([0], [0], color=PRED_COLOR, lw=3, label="Orange-red: Prediction"),
        Line2D([0], [0], color=BODY_EDGE_COLOR, lw=3, label="Gray: anatomical context / CT"),
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
        for label in (2, 4, 5, 6, 7, 8, 9)
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
    save_figure(fig, "qualitative_comparison_3d_only_v2", output_dir, formats, dpi)
    plt.close(fig)


def make_slice_zoom(
    slice_cases: list[dict],
    slice_methods: list[str],
    ct: np.ndarray,
    background_label: str,
    output_dir: Path,
    formats: list[str],
    dpi: int,
    legend: bool,
) -> None:
    if len(slice_cases) != 1:
        slice_cases = slice_cases[:1]
    fig, axes = plt.subplots(
        2,
        len(slice_methods),
        figsize=(3.95 * len(slice_methods), 7.2),
    )
    axes = np.atleast_2d(axes)
    case = slice_cases[0]
    sample_id = str(case["sample_id"])
    gt = load_volume("gt", sample_id)
    axis, index = choose_slice(gt > 0.0)
    context_crop = crop_box_from_gt(gt > 0.0, axis, padding=42)
    zoom_crop = zoom_crop_from_gt(gt > 0.0, str(case["panel"]), axis, padding=18)
    for row, crop in enumerate([context_crop, zoom_crop]):
        for col, method in enumerate(slice_methods):
            prediction = None if method == "gt" else load_volume(method, sample_id)
            title = METHOD_TITLES[method] if row == 0 else ""
            render_slice_panel(
                axes[row, col],
                ct,
                background_label,
                gt,
                prediction,
                axis,
                index,
                crop,
                title,
                show_background_label=(row == 0 and col == 0),
                roi_crop=zoom_crop if row == 0 else None,
            )
        add_case_label(axes[row, 0], case, x=-0.13)
        axes[row, 0].text(
            -0.13,
            0.08,
            "Context slice" if row == 0 else "Local magnification",
            ha="right",
            va="center",
            transform=axes[row, 0].transAxes,
            fontsize=9.5,
            color="#444444",
        )
    fig.subplots_adjust(
        left=0.18,
        right=0.995,
        top=0.92,
        bottom=0.12 if legend else 0.04,
        wspace=0.045,
        hspace=0.18,
    )
    if legend:
        add_global_legend(fig, y=0.03)
    save_figure(fig, "qualitative_comparison_slice_zoom_v2", output_dir, ["png"], dpi)
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
    if len(slice_cases) != 1:
        slice_cases = slice_cases[:1]
    width = max(18.8, 3.25 * len(slice_methods))
    fig = plt.figure(figsize=(width, 14.1))
    outer = fig.add_gridspec(2, 1, height_ratios=[4.15, 1.05], hspace=0.075)
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

    bottom = outer[1].subgridspec(1, len(slice_methods), wspace=0.035, hspace=0.0)
    for row, case in enumerate(slice_cases):
        sample_id = str(case["sample_id"])
        gt = load_volume("gt", sample_id)
        axis, index = choose_slice(gt > 0.0)
        crop = crop_box_from_gt(gt > 0.0, axis, padding=42)
        zoom_crop = zoom_crop_from_gt(gt > 0.0, str(case["panel"]), axis, padding=18)
        first_ax = None
        for col, method in enumerate(slice_methods):
            ax = fig.add_subplot(bottom[0, col])
            if first_ax is None:
                first_ax = ax
            prediction = None if method == "gt" else load_volume(method, sample_id)
            title = METHOD_TITLES[method]
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
                show_background_label=(col == 0),
                roi_crop=zoom_crop,
            )
        add_case_label(first_ax, case, x=-0.10)
        first_ax.text(
            -0.10,
            0.08,
            "Context slice with zoom ROI",
            ha="right",
            va="center",
            transform=first_ax.transAxes,
            fontsize=9.5,
            color="#444444",
        )
    fig.subplots_adjust(
        left=0.16,
        right=0.995,
        top=0.955,
        bottom=0.07 if legend else 0.02,
    )
    if legend:
        add_global_legend(fig, y=0.018)
    save_figure(fig, "qualitative_comparison_3d_slice_v2", output_dir, formats, dpi)
    plt.close(fig)


def write_case_json(output_dir: Path, background_label: str, cases: list[dict]) -> None:
    (output_dir / "qualitative_comparison_cases.json").write_text(
        json.dumps(
            {
                "threshold": THRESHOLD,
                "slice_background": background_label,
                "ct_alignment": (
                    "Digimouse CT ct_380x992x208 cropped to x=[0,38] mm, "
                    "y=[34,74] mm, z=[0,20.8] mm and mean-downsampled 2x to [190,200,104]."
                ),
                "cases": cases,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    args = parse_args()
    if any(case["num_foci"] < 2 for case in CASE_POOLS):
        raise ValueError("The main-text qualitative figure must not contain single-focus cases.")
    formats = requested_formats(args.format)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = [
        output_dir / f"qualitative_comparison_3d_slice_v2.{fmt}" for fmt in formats
    ] + [
        output_dir / f"qualitative_comparison_3d_only_v2.{fmt}" for fmt in formats
    ] + [
        output_dir / "qualitative_comparison_slice_zoom_v2.png"
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

    cases = select_filtered_cases(body_mask, output_dir)
    panel_dir = output_dir / "qualitative_comparison_panels"
    panel_paths = build_3d_panels(
        cases,
        METHODS_3D,
        panel_dir,
        labels,
        enhance_mouse_outline=args.enhance_mouse_outline,
    )
    slice_cases = [
        case_by_panel(cases, panel.strip()) for panel in args.slice_cases.split(",") if panel.strip()
    ]
    slice_methods = [method.strip() for method in args.slice_methods.split(",") if method.strip()]
    for method in slice_methods:
        if method != "gt" and method not in PREDICTION_DIRS:
            raise ValueError(f"Unknown slice method: {method}")

    make_3d_only(cases, METHODS_3D, panel_paths, output_dir, formats, args.dpi, args.legend)
    if args.with_slices:
        make_slice_zoom(
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
            cases,
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
    write_case_json(output_dir, background_label, cases)
    for path in expected:
        if path.exists():
            print(path)


if __name__ == "__main__":
    main()
