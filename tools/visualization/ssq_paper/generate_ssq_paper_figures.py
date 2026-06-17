#!/usr/bin/env python3
"""Generate evidence-first SSQ-FMT simulation figures from fixed test300 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, stats

METHOD_RUNS = {
    "SSQ-FMT": "no_center_aux",
    "w/o transport-aware footprint": "no_ptfa",
    "w/o query-canonical reliability": "no_canonical_reliability",
    "w/o source-relative query cue": "no_source_cue",
}
DISPLAY_LABELS = {
    "SSQ-FMT": "SSQ-FMT",
    "w/o transport-aware footprint": "No footprint",
    "w/o query-canonical reliability": "No reliability",
    "w/o source-relative query cue": "No source cue",
}
METHOD_COLORS = {
    "SSQ-FMT": "#D55E00",
    "w/o transport-aware footprint": "#0072B2",
    "w/o query-canonical reliability": "#666666",
    "w/o source-relative query cue": "#9ECAE1",
}
ABLATION_METHODS = [m for m in METHOD_RUNS if m != "SSQ-FMT"]
GENERATION_COMMAND = "uv run python tools/visualization/ssq_paper/generate_ssq_paper_figures.py"
VOXEL_SIZE_MM = 0.2
BODY_LONG_AXIS_IJK = np.array([0.0, 1.0, 0.0], dtype=float)

EVAL_PROTOCOL = {
    "test_set": "fixed 300-sample test.txt",
    "voxel_threshold": 0.5,
    "component_iou_threshold": 0.01,
    "component_centroid_threshold_vox": 3.0,
    "component_connectivity": 26,
    "minimum_region_size": 10,
}


@dataclass
class Paths:
    repo: Path
    data_dir: Path
    ssq_eval_dir: Path
    out_dir: Path
    data_out: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir", default="/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k"
    )
    parser.add_argument("--ssq_eval_dir", default="outputs/fmt_simgen_v2_ssq_ablation/test300")
    parser.add_argument("--out_dir", default="outputs/paper_figures/ssq")
    parser.add_argument("--three_source_case", default=None)
    return parser.parse_args()


def set_tmi_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.8,
        }
    )


def git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo, text=True
        ).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dirs(paths: Paths) -> None:
    for sub in [
        paths.out_dir,
        paths.data_out,
        paths.out_dir / "qualitative_cases",
        paths.out_dir / "paired_improvement",
        paths.out_dir / "component_analysis",
        paths.out_dir / "separability_phase",
        paths.out_dir / "source_hypotheses",
        paths.out_dir / "mechanisms",
        paths.out_dir / "robustness",
    ]:
        sub.mkdir(parents=True, exist_ok=True)


def clean_outputs(paths: Paths) -> None:
    for pattern in [
        "**/fig_*.pdf",
        "**/fig_*.svg",
        "**/fig_*.png",
        "**/fig_qualitative_*_data.npz",
    ]:
        for p in paths.out_dir.glob(pattern):
            p.unlink()


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(base.with_suffix(f".{ext}"), **kwargs)
    plt.close(fig)


def read_metrics(paths: Paths, method: str) -> pd.DataFrame:
    run = METHOD_RUNS[method]
    df = pd.read_csv(paths.ssq_eval_dir / run / "metrics_per_sample.csv")
    df["method"] = method
    return df


def read_components(paths: Paths, method: str) -> pd.DataFrame:
    run = METHOD_RUNS[method]
    df = pd.read_csv(paths.ssq_eval_dir / run / "components" / "component_per_sample.csv")
    df["method"] = method
    return df


def load_all_tables(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.concat([read_metrics(paths, m) for m in METHOD_RUNS], ignore_index=True)
    comps = pd.concat([read_components(paths, m) for m in METHOD_RUNS], ignore_index=True)
    return metrics, comps


def load_sample_metadata(paths: Paths, sample_ids: list[str]) -> pd.DataFrame:
    rows = []
    for sample_id in sample_ids:
        sample_dir = paths.data_dir / "samples" / sample_id
        meta = json.loads((sample_dir / "tumor_params.json").read_text())
        centers = np.array([f["center"] for f in meta.get("foci", [])], dtype=float)
        intensities = np.array(
            [float(f.get("params", {}).get("intensity", np.nan)) for f in meta.get("foci", [])],
            dtype=float,
        )
        if len(centers) >= 2:
            dist = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
            dist[dist == 0] = np.inf
            min_dist = float(np.nanmin(dist))
        else:
            min_dist = np.nan
        weak_ratio = np.nan
        if len(intensities) and np.isfinite(intensities).all() and intensities.max() > 0:
            weak_ratio = float(intensities.min() / intensities.max())
        rows.append(
            {
                "sample_id": sample_id,
                "number_of_sources": int(meta.get("num_foci", len(centers))),
                "minimum_inter_source_distance_mm": min_dist,
                "weak_to_dominant_intensity_ratio": weak_ratio,
                "source_centers_mm": centers.tolist(),
                "source_intensities": intensities.tolist(),
                "shape_set": meta.get("shape_set"),
                "source_type": meta.get("source_type"),
            }
        )
    return pd.DataFrame(rows)


def success_mask(df: pd.DataFrame) -> pd.Series:
    return (
        (df["pred_component_count"] == df["gt_component_count"])
        & (df["matched_component_count"] == df["gt_component_count"])
        & (df["missed_component_count"] == 0)
        & (df["merge_count"] == 0)
        & (df["split_count"] == 0)
    )


def method_component_table(components: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    comp = components.merge(
        metrics[["sample_id", "method", "dice", "precision", "recall"]],
        on=["sample_id", "method"],
        how="left",
    )
    comp["success"] = success_mask(comp)
    return comp


def wide_component_outcomes(comp: pd.DataFrame) -> pd.DataFrame:
    base = comp[comp["method"] == "SSQ-FMT"].copy()
    base = base.rename(columns={c: f"ssq_{c}" for c in base.columns if c != "sample_id"})
    for method in ABLATION_METHODS:
        sub = comp[comp["method"] == method].copy()
        prefix = DISPLAY_LABELS[method].lower().replace(" ", "_")
        sub = sub.rename(columns={c: f"{prefix}_{c}" for c in sub.columns if c != "sample_id"})
        base = base.merge(sub, on="sample_id", how="left")
    return base


def centers_mm_to_ijk(centers_mm: np.ndarray) -> np.ndarray:
    return centers_mm / VOXEL_SIZE_MM - 0.5


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        raise ValueError("Cannot normalize near-zero vector")
    return v / n


def oblique_plane_from_centers(
    centers_ijk: np.ndarray, volume_shape: tuple[int, int, int], margin_mm: float = 5.0
) -> dict[str, Any]:
    if len(centers_ijk) < 2:
        raise ValueError("Oblique plane requires at least two source centers")
    origin = centers_ijk.mean(axis=0)
    u = normalize(centers_ijk[1] - centers_ijk[0])
    if len(centers_ijk) >= 3:
        normal = np.cross(u, centers_ijk[2] - centers_ijk[0])
        if np.linalg.norm(normal) < 1e-6:
            normal = np.cross(u, BODY_LONG_AXIS_IJK)
    else:
        normal = np.cross(u, BODY_LONG_AXIS_IJK)
        if np.linalg.norm(normal) < 1e-6:
            normal = np.cross(u, np.array([0.0, 0.0, 1.0]))
    normal = normalize(normal)
    v = normalize(np.cross(normal, u))
    rel = centers_ijk - origin[None, :]
    proj_u = rel @ u
    proj_v = rel @ v
    margin_vox = margin_mm / VOXEL_SIZE_MM
    u_min, u_max = float(proj_u.min() - margin_vox), float(proj_u.max() + margin_vox)
    v_min, v_max = float(proj_v.min() - margin_vox), float(proj_v.max() + margin_vox)
    u_axis = np.arange(np.floor(u_min), np.ceil(u_max) + 1)
    v_axis = np.arange(np.floor(v_min), np.ceil(v_max) + 1)
    max_size = 170
    if len(u_axis) > max_size:
        u_axis = np.linspace(u_axis.min(), u_axis.max(), max_size)
    if len(v_axis) > max_size:
        v_axis = np.linspace(v_axis.min(), v_axis.max(), max_size)
    uu, vv = np.meshgrid(u_axis, v_axis)
    coords = origin[:, None, None] + u[:, None, None] * uu[None] + v[:, None, None] * vv[None]
    coords[0] = np.clip(coords[0], 0, volume_shape[0] - 1)
    coords[1] = np.clip(coords[1], 0, volume_shape[1] - 1)
    coords[2] = np.clip(coords[2], 0, volume_shape[2] - 1)
    return {
        "origin_ijk": origin,
        "basis_u_ijk": u,
        "basis_v_ijk": v,
        "normal_ijk": normal,
        "u_axis": u_axis,
        "v_axis": v_axis,
        "coords": coords,
        "pixel_spacing_mm": VOXEL_SIZE_MM,
        "plane_shape": [int(len(v_axis)), int(len(u_axis))],
    }


def sample_plane(volume: np.ndarray, plane: dict[str, Any], order: int = 1) -> np.ndarray:
    return ndimage.map_coordinates(volume, plane["coords"], order=order, mode="constant", cval=0.0)


def load_prediction(paths: Paths, method: str, sample_id: str) -> tuple[np.ndarray, np.ndarray]:
    run = METHOD_RUNS[method]
    z = np.load(paths.ssq_eval_dir / run / "predictions" / f"{sample_id}.npz")
    return z["pred"].astype(float), z["gt"].astype(float)


def nearest_pair(centers_ijk: np.ndarray) -> tuple[int, int]:
    dist = np.linalg.norm(centers_ijk[:, None, :] - centers_ijk[None, :, :], axis=-1)
    dist[dist == 0] = np.inf
    return tuple(np.unravel_index(np.argmin(dist), dist.shape))  # type: ignore[return-value]


def profile_along_pair(
    volume: np.ndarray, centers_ijk: np.ndarray, pair: tuple[int, int], n: int = 240
) -> tuple[np.ndarray, np.ndarray]:
    a, b = centers_ijk[pair[0]], centers_ijk[pair[1]]
    t = np.linspace(-0.15, 1.15, n)
    pts = a[None, :] + (b - a)[None, :] * t[:, None]
    coords = [pts[:, 0], pts[:, 1], pts[:, 2]]
    values = ndimage.map_coordinates(volume, coords, order=1, mode="constant", cval=0.0)
    distance_mm = (t - t[0]) * np.linalg.norm(b - a) * VOXEL_SIZE_MM
    return distance_mm, values


def valley_ratio(values: np.ndarray) -> float:
    if len(values) < 5:
        return np.nan
    mid = values[len(values) // 5 : -len(values) // 5]
    peak = float(np.nanmax(values))
    if peak <= 0:
        return np.nan
    return float(np.nanmin(mid) / peak)


def weak_focus_retention(
    volume: np.ndarray, centers_ijk: np.ndarray, intensities: np.ndarray
) -> float:
    if len(centers_ijk) == 0 or len(intensities) == 0:
        return np.nan
    vals = ndimage.map_coordinates(
        volume,
        [centers_ijk[:, 0], centers_ijk[:, 1], centers_ijk[:, 2]],
        order=1,
        mode="constant",
        cval=0.0,
    )
    weak_idx = int(np.argmin(intensities))
    denom = float(np.nanmax(vals))
    if denom <= 0:
        return np.nan
    return float(vals[weak_idx] / denom)


def bridge_area_px(plane_pred: np.ndarray, plane_gt: np.ndarray) -> int:
    bridge = (plane_pred >= 0.5) & (plane_gt < 0.5)
    labels, nlab = ndimage.label(bridge, structure=np.ones((3, 3), dtype=bool))
    if nlab == 0:
        return 0
    return int(max(ndimage.sum(bridge, labels, index=np.arange(1, nlab + 1))))


def outcome_string(row: pd.Series, method: str) -> str:
    prefix = "ssq" if method == "SSQ-FMT" else DISPLAY_LABELS[method].lower().replace(" ", "_")
    pred = int(row[f"{prefix}_pred_component_count"])
    match = int(row[f"{prefix}_matched_component_count"])
    miss = int(row[f"{prefix}_missed_component_count"])
    merge = int(row[f"{prefix}_merge_count"])
    split = int(row[f"{prefix}_split_count"])
    return f"pred={pred}, match={match}, miss={miss}, merge={merge}, split={split}"


def any_ablation_failure(row: pd.Series) -> bool:
    for method in ABLATION_METHODS:
        prefix = DISPLAY_LABELS[method].lower().replace(" ", "_")
        if not bool(row[f"{prefix}_success"]):
            return True
    return False


def select_cases(
    paths: Paths,
    comp: pd.DataFrame,
    metrics: pd.DataFrame,
    sample_meta: pd.DataFrame,
    three_source_case: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = wide_component_outcomes(comp).merge(sample_meta, on="sample_id", how="left")
    multi = wide[wide["number_of_sources"].isin([2, 3])].copy()
    dist_q25 = multi["minimum_inter_source_distance_mm"].quantile(0.25)
    ratio_q25 = multi["weak_to_dominant_intensity_ratio"].quantile(0.25)

    adjacent = multi[
        (multi["minimum_inter_source_distance_mm"] <= dist_q25)
        & (multi["ssq_success"])
        & multi.apply(any_ablation_failure, axis=1)
    ].copy()
    if adjacent.empty:
        raise RuntimeError("No eligible adjacent-source case found")
    adjacent_idx = (adjacent["ssq_dice"] - adjacent["ssq_dice"].median()).abs().idxmin()

    three = multi[
        (multi["number_of_sources"] == 3)
        & (multi["ssq_gt_component_count"] == 3)
        & (multi["ssq_pred_component_count"] == 3)
        & (multi["ssq_matched_component_count"] == 3)
        & (multi["ssq_missed_component_count"] == 0)
        & (multi["ssq_merge_count"] == 0)
        & (multi["ssq_split_count"] == 0)
        & multi.apply(any_ablation_failure, axis=1)
    ].copy()
    if three.empty:
        raise RuntimeError("No eligible three-source success case found")
    if three_source_case:
        chosen = three[three["sample_id"] == three_source_case]
        if chosen.empty:
            raise ValueError(
                f"--three_source_case does not satisfy hard constraints: {three_source_case}"
            )
        three_idx = chosen.index[0]
    else:
        weak_three = three[three["weak_to_dominant_intensity_ratio"] <= ratio_q25]
        choose_from = weak_three if not weak_three.empty else three
        three_idx = (choose_from["ssq_dice"] - choose_from["ssq_dice"].median()).abs().idxmin()

    median_order = (three["ssq_dice"] - three["ssq_dice"].median()).abs().sort_values()
    contact = three.loc[median_order.index[:10]].copy()
    cases = pd.DataFrame(
        [
            {"case": "Adjacent sources", **wide.loc[adjacent_idx].to_dict()},
            {"case": "Three sources with a weak focus", **wide.loc[three_idx].to_dict()},
        ]
    )
    return cases, contact


def make_three_source_contact_sheet(
    paths: Paths, contact: pd.DataFrame, out_dir: Path, manifest: list[dict], commit: str
) -> None:
    rows = []
    fig, axes = plt.subplots(2, 5, figsize=(7.16, 3.2), constrained_layout=True)
    for ax, (_, row) in zip(axes.ravel(), contact.iterrows()):
        sample_id = row["sample_id"]
        pred, gt = load_prediction(paths, "SSQ-FMT", sample_id)
        centers = centers_mm_to_ijk(np.array(row["source_centers_mm"], dtype=float))
        plane = oblique_plane_from_centers(centers, gt.shape)
        gt_plane = sample_plane(gt, plane, order=0)
        pred_plane = sample_plane(pred, plane)
        ax.imshow(pred_plane, cmap="viridis", vmin=0, vmax=1, origin="lower")
        ax.contour(gt_plane, levels=[0.5], colors="white", linewidths=0.45)
        ax.set_xticks([])
        ax.set_yticks([])
        outcome = "; ".join(
            f"{DISPLAY_LABELS[m]}: {outcome_string(row, m)}" for m in ABLATION_METHODS
        )
        ax.set_title(
            f"{sample_id}\nDice={row['ssq_dice']:.2f}, CR={row['ssq_component_recall']:.2f}\n"
            f"d={row['minimum_inter_source_distance_mm']:.1f}, "
            f"w/d={row['weak_to_dominant_intensity_ratio']:.2f}",
            fontsize=5.5,
        )
        rows.append(
            {
                "sample_id": sample_id,
                "dice": row["ssq_dice"],
                "component_recall": row["ssq_component_recall"],
                "source_distance_mm": row["minimum_inter_source_distance_mm"],
                "weak_to_dominant_ratio": row["weak_to_dominant_intensity_ratio"],
                "ablation_outcomes": outcome,
            }
        )
    save_figure(fig, out_dir / "fig_three_source_candidate_contact_sheet")
    pd.DataFrame(rows).to_csv(out_dir / "three_source_candidate_contact_sheet.csv", index=False)
    manifest.append(
        {
            "figure_id": "fig_three_source_candidate_contact_sheet",
            "figure_path": str(out_dir / "fig_three_source_candidate_contact_sheet.pdf"),
            "source_data_path": str(out_dir / "three_source_candidate_contact_sheet.csv"),
            "methods": list(METHOD_RUNS),
            "samples": [r["sample_id"] for r in rows],
            "evaluation_protocol": EVAL_PROTOCOL,
            "generation_command": GENERATION_COMMAND,
            "git_commit": commit,
            "suitability": "supplementary",
        }
    )


def source_positions_on_plane(centers_ijk: np.ndarray, plane: dict[str, Any]) -> np.ndarray:
    rel = centers_ijk - plane["origin_ijk"][None, :]
    u = rel @ plane["basis_u_ijk"]
    v = rel @ plane["basis_v_ijk"]
    x = np.interp(u, plane["u_axis"], np.arange(len(plane["u_axis"])))
    y = np.interp(v, plane["v_axis"], np.arange(len(plane["v_axis"])))
    return np.stack([x, y], axis=1)


def generate_source_separation_profiles(
    paths: Paths,
    cases: pd.DataFrame,
    out_dir: Path,
    manifest: list[dict],
    audit_rows: list[dict],
    commit: str,
) -> None:
    methods = ["GT", *ABLATION_METHODS, "SSQ-FMT", "Density profile"]
    fig, axes = plt.subplots(2, 6, figsize=(7.16, 3.35), constrained_layout=True)
    profile_rows: list[dict[str, Any]] = []
    plane_records: dict[str, Any] = {}
    for row_idx, (_, case) in enumerate(cases.iterrows()):
        sample_id = case["sample_id"]
        centers_mm = np.array(case["source_centers_mm"], dtype=float)
        centers_ijk = centers_mm_to_ijk(centers_mm)
        intensities = np.array(case["source_intensities"], dtype=float)
        pred0, gt = load_prediction(paths, "SSQ-FMT", sample_id)
        plane = oblique_plane_from_centers(centers_ijk, gt.shape)
        gt_plane = sample_plane(gt, plane, order=0)
        center_xy = source_positions_on_plane(centers_ijk, plane)
        pair = nearest_pair(centers_ijk)
        plane_records[sample_id] = {
            "origin_ijk": plane["origin_ijk"].tolist(),
            "basis_u_ijk": plane["basis_u_ijk"].tolist(),
            "basis_v_ijk": plane["basis_v_ijk"].tolist(),
            "normal_ijk": plane["normal_ijk"].tolist(),
            "pixel_spacing_mm": plane["pixel_spacing_mm"],
            "plane_shape": plane["plane_shape"],
            "nearest_pair": [int(pair[0]), int(pair[1])],
        }
        for col, method in enumerate(methods):
            ax = axes[row_idx, col]
            if method == "GT":
                ax.imshow(gt_plane, cmap="Greys", vmin=0, vmax=1, origin="lower")
                ax.contour(gt_plane, levels=[0.5], colors="white", linewidths=0.55)
            elif method != "Density profile":
                pred, _ = load_prediction(paths, method, sample_id)
                pred_plane = sample_plane(pred, plane)
                ax.imshow(pred_plane, cmap="viridis", vmin=0, vmax=1, origin="lower")
                ax.contour(gt_plane, levels=[0.5], colors="white", linewidths=0.55)
                ax.contour(
                    pred_plane,
                    levels=[0.5],
                    colors=METHOD_COLORS[method],
                    linewidths=0.55,
                )
                ax.scatter(center_xy[:, 0], center_xy[:, 1], s=8, c="white", marker="+", lw=0.7)
                _, prof = profile_along_pair(pred, centers_ijk, pair)
                profile_rows.append(
                    {
                        "case": case["case"],
                        "sample_id": sample_id,
                        "method": method,
                        "inter_source_valley_ratio": valley_ratio(prof),
                        "bridge_area_px": bridge_area_px(pred_plane, gt_plane),
                        "weak_focus_retention": weak_focus_retention(
                            pred, centers_ijk, intensities
                        ),
                        "component_outcome": outcome_string(case, method),
                    }
                )
            else:
                for method2 in ABLATION_METHODS + ["SSQ-FMT"]:
                    pred, _ = load_prediction(paths, method2, sample_id)
                    dist_mm, prof = profile_along_pair(pred, centers_ijk, pair)
                    ax.plot(
                        dist_mm,
                        prof,
                        color=METHOD_COLORS[method2],
                        lw=1.6 if method2 == "SSQ-FMT" else 0.9,
                        label=DISPLAY_LABELS[method2] if row_idx == 0 else None,
                    )
                for source_idx in pair:
                    source_pos = (
                        np.linalg.norm(centers_ijk[source_idx] - centers_ijk[pair[0]])
                        * VOXEL_SIZE_MM
                    )
                    ax.axvline(source_pos, color="0.3", ls="--", lw=0.6)
                if len(centers_ijk) == 3:
                    for source_idx in range(3):
                        source_pos = (
                            np.linalg.norm(centers_ijk[source_idx] - centers_ijk[pair[0]])
                            * VOXEL_SIZE_MM
                        )
                        ax.axvline(source_pos, color="0.65", ls=":", lw=0.45)
                ax.set_ylim(0, 1.02)
                ax.set_xlabel("Distance (mm)")
                ax.set_ylabel("Density")
                ax.grid(axis="y")
            if row_idx == 0:
                ax.set_title(DISPLAY_LABELS.get(method, method))
            if col == 0:
                ax.set_ylabel(
                    f"{case['case']}\n"
                    f"d_min={case['minimum_inter_source_distance_mm']:.1f} mm\n"
                    f"weak/dom={case['weak_to_dominant_intensity_ratio']:.2f}",
                    fontsize=6,
                )
            if method != "Density profile":
                ax.set_xticks([])
                ax.set_yticks([])
        if row_idx == 0:
            axes[row_idx, 5].legend(frameon=False, fontsize=5.5, loc="upper right")
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap="viridis")
    fig.colorbar(sm, ax=axes[:, 1:5].ravel().tolist(), shrink=0.85, label="Predicted density")
    save_figure(fig, out_dir / "fig_source_separation_profiles")
    pd.DataFrame(profile_rows).to_csv(
        out_dir / "source_separation_profile_metrics.csv", index=False
    )
    save_json(out_dir / "source_separation_oblique_planes.json", plane_records)
    manifest.append(
        {
            "figure_id": "fig_source_separation_profiles",
            "figure_path": str(out_dir / "fig_source_separation_profiles.pdf"),
            "source_data_path": str(out_dir / "source_separation_profile_metrics.csv"),
            "methods": ["GT", *ABLATION_METHODS, "SSQ-FMT"],
            "samples": cases["sample_id"].tolist(),
            "evaluation_protocol": EVAL_PROTOCOL,
            "generation_command": GENERATION_COMMAND,
            "git_commit": commit,
            "suitability": "main text candidate",
        }
    )
    audit_rows.append(
        {
            "figure_id": "fig_source_separation_profiles",
            "intended_claim": (
                "SSQ-FMT separates adjacent and weak multi-source cases better than ablations."
            ),
            "exact_data_used": str(out_dir / "source_separation_profile_metrics.csv"),
            "sample_selection_rule": (
                "Hard component success constraints for selected adjacent and three-source cases."
            ),
            "population_subset": (
                "Fixed test300 multi-source samples satisfying selection constraints."
            ),
            "metric_definition": (
                "Oblique-plane profiles, valley ratio, bridge area, weak-focus retention, "
                "component outcome."
            ),
            "effect_size": "See source_separation_profile_metrics.csv.",
            "confidence_interval": "Not applicable to case study.",
            "statistical_test": "Not applicable to case study.",
            "supports_claim": "case-level visual evidence only",
            "placement": "main text candidate",
        }
    )


def bootstrap_net_ci(
    ssq_success: np.ndarray, ab_success: np.ndarray, n_boot: int = 5000
) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    n = len(ssq_success)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        rescued = np.sum(ssq_success[idx] & ~ab_success[idx])
        harmed = np.sum(~ssq_success[idx] & ab_success[idx])
        vals.append(rescued - harmed)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def generate_module_rescue_transitions(
    paths: Paths,
    comp: pd.DataFrame,
    sample_meta: pd.DataFrame,
    out_dir: Path,
    manifest: list[dict],
    audit_rows: list[dict],
    commit: str,
) -> None:
    merged = comp.merge(sample_meta, on="sample_id", how="left")
    subset = merged[
        (merged["method"] == "SSQ-FMT")
        & (merged["gt_component_count"].isin([2, 3]))
        & (merged["minimum_inter_source_distance_mm"] <= 8.0)
    ][["sample_id"]]
    sample_ids = set(subset["sample_id"])
    rows = []
    ssq = comp[(comp["method"] == "SSQ-FMT") & (comp["sample_id"].isin(sample_ids))].set_index(
        "sample_id"
    )
    for method in ABLATION_METHODS:
        ab = comp[(comp["method"] == method) & (comp["sample_id"].isin(sample_ids))].set_index(
            "sample_id"
        )
        common = sorted(set(ssq.index) & set(ab.index))
        ssq_s = success_mask(ssq.loc[common]).to_numpy(bool)
        ab_s = success_mask(ab.loc[common]).to_numpy(bool)
        both_success = int(np.sum(ssq_s & ab_s))
        rescued = int(np.sum(ssq_s & ~ab_s))
        harmed = int(np.sum(~ssq_s & ab_s))
        both_failure = int(np.sum(~ssq_s & ~ab_s))
        discordant = rescued + harmed
        pvalue = (
            float(stats.binomtest(min(rescued, harmed), discordant, 0.5).pvalue)
            if discordant
            else 1.0
        )
        ci = bootstrap_net_ci(ssq_s, ab_s)
        rows.append(
            {
                "ablation": method,
                "label": DISPLAY_LABELS[method],
                "num_samples": len(common),
                "both_success": both_success,
                "ssq_rescued": rescued,
                "ablation_only_success": harmed,
                "both_failure": both_failure,
                "net_rescued": rescued - harmed,
                "net_rescued_ci95_low": ci[0],
                "net_rescued_ci95_high": ci[1],
                "mcnemar_exact_p": pvalue,
                "supports_main_claim": bool(rescued > harmed and pvalue < 0.05),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "module_rescue_transitions.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.25), constrained_layout=True)
    for ax, (_, row) in zip(axes, result.iterrows()):
        mat = np.array(
            [
                [row["both_success"], row["ablation_only_success"]],
                [row["ssq_rescued"], row["both_failure"]],
            ],
            dtype=float,
        )
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(1, mat.max()))
        labels = [["Both\nsuccess", "Harmed"], ["Rescued", "Both\nfailure"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{labels[i][j]}\n{int(mat[i, j])}", ha="center", va="center")
        ax.set_title(DISPLAY_LABELS[row["ablation"]])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(
            f"net={int(row['net_rescued'])} "
            f"[{row['net_rescued_ci95_low']:.0f},{row['net_rescued_ci95_high']:.0f}]\n"
            f"McNemar p={row['mcnemar_exact_p']:.3f}"
        )
    save_figure(fig, out_dir / "fig_module_rescue_transitions")
    supports = bool((result["supports_main_claim"]).all())
    manifest.append(
        {
            "figure_id": "fig_module_rescue_transitions",
            "figure_path": str(out_dir / "fig_module_rescue_transitions.pdf"),
            "source_data_path": str(out_dir / "module_rescue_transitions.csv"),
            "methods": list(METHOD_RUNS),
            "samples": "test300 GT component count 2/3 and minimum source distance <= 8 mm",
            "evaluation_protocol": EVAL_PROTOCOL,
            "generation_command": GENERATION_COMMAND,
            "git_commit": commit,
            "suitability": "main text candidate" if supports else "supplementary or unsupported",
        }
    )
    audit_rows.append(
        {
            "figure_id": "fig_module_rescue_transitions",
            "intended_claim": (
                "SSQ-FMT rescues complete multi-source recovery failures made by ablations."
            ),
            "exact_data_used": str(out_dir / "module_rescue_transitions.csv"),
            "sample_selection_rule": "GT component count 2/3 and minimum source distance <= 8 mm.",
            "population_subset": (
                f"{int(result['num_samples'].max())} hard multi-source test samples."
            ),
            "metric_definition": "Complete recovery success event from component evaluator.",
            "effect_size": result[["label", "net_rescued"]].to_dict(orient="records"),
            "confidence_interval": result[
                ["label", "net_rescued_ci95_low", "net_rescued_ci95_high"]
            ].to_dict(orient="records"),
            "statistical_test": result[["label", "mcnemar_exact_p"]].to_dict(orient="records"),
            "supports_claim": "yes" if supports else "not consistently supported",
            "placement": "main text candidate" if supports else "supplementary or unsupported",
        }
    )


def generate_delta_dice_supplement(
    metrics: pd.DataFrame, out_dir: Path, manifest: list[dict], commit: str
) -> None:
    ssq = metrics[metrics["method"] == "SSQ-FMT"].set_index("sample_id")
    rows = []
    for method in ABLATION_METHODS:
        ab = metrics[metrics["method"] == method].set_index("sample_id")
        for sid in sorted(set(ssq.index) & set(ab.index)):
            rows.append(
                {
                    "sample_id": sid,
                    "ablation": method,
                    "label": DISPLAY_LABELS[method],
                    "delta_dice": float(ssq.loc[sid, "dice"] - ab.loc[sid, "dice"]),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "paired_delta_dice.csv", index=False)
    summary = df.groupby(["ablation", "label"], as_index=False)["delta_dice"].median()
    fig, ax = plt.subplots(figsize=(3.5, 1.65), constrained_layout=True)
    y = np.arange(len(summary))
    ax.barh(y, summary["delta_dice"], color=[METHOD_COLORS[m] for m in summary["ablation"]])
    ax.axvline(0, color="black", lw=0.7)
    ax.set_yticks(y, summary["label"])
    ax.set_xlabel("Median Delta Dice")
    ax.grid(axis="x")
    save_figure(fig, out_dir / "fig_delta_dice_supplementary")
    manifest.append(
        {
            "figure_id": "fig_delta_dice_supplementary",
            "figure_path": str(out_dir / "fig_delta_dice_supplementary.pdf"),
            "source_data_path": str(out_dir / "paired_delta_dice.csv"),
            "methods": list(METHOD_RUNS),
            "samples": "test300 paired by sample_id",
            "evaluation_protocol": EVAL_PROTOCOL,
            "generation_command": GENERATION_COMMAND,
            "git_commit": commit,
            "suitability": "supplementary",
        }
    )


def export_source_hypothesis_data(paths: Paths, sample_ids: list[str], out_dir: Path) -> None:
    rows = []
    for sid in sample_ids:
        heat_path = paths.data_dir / "samples" / sid / "proposal" / "meas_backproj_heatmap.npy"
        if not heat_path.exists():
            continue
        heat = np.load(heat_path)
        smooth = ndimage.gaussian_filter(heat.astype(float), sigma=1.0)
        peaks = (smooth == ndimage.maximum_filter(smooth, size=5)) & (smooth > smooth.max() * 0.1)
        coords = np.argwhere(peaks)
        vals = smooth[peaks]
        order = np.argsort(vals)[::-1][:5]
        np.savez_compressed(
            out_dir / f"source_hypotheses_{sid}.npz",
            heatmap=heat.astype(np.float32),
            peak_indices=coords[order].astype(np.int16),
            peak_scores=vals[order].astype(np.float32),
        )
        rows.append({"sample_id": sid, "num_saved_peaks": int(len(order))})
    pd.DataFrame(rows).to_csv(out_dir / "source_hypotheses_summary.csv", index=False)


def write_evidence_audit(
    paths: Paths, audit_rows: list[dict], script_sha: str, commit: str
) -> None:
    unsupported = [
        {
            "figure_id": "fig_ssq_module_mechanisms",
            "intended_claim": (
                "Mechanistic evidence for PTFA, query-canonical reliability, and source cue."
            ),
            "exact_data_used": "not generated",
            "sample_selection_rule": "requires analysis-mode internal exports",
            "population_subset": "not evaluated",
            "metric_definition": "actual footprint, reliability, and source-cue internals",
            "effect_size": "not available",
            "confidence_interval": "not available",
            "statistical_test": "not available",
            "supports_claim": "unsupported: actual internal exports are unavailable",
            "placement": "rejected",
        },
        {
            "figure_id": "fig_module_robustness",
            "intended_claim": "Inference-time robustness to detector and view perturbations.",
            "exact_data_used": "not generated",
            "sample_selection_rule": "requires controlled perturbation inference outputs",
            "population_subset": "not evaluated",
            "metric_definition": "Dice, complete recovery rate, merge/sample under perturbation",
            "effect_size": "not available",
            "confidence_interval": "not available",
            "statistical_test": "not available",
            "supports_claim": "unsupported: perturbation inference has not been run",
            "placement": "rejected",
        },
    ]
    all_rows = audit_rows + unsupported
    lines = [
        "# SSQ-FMT Evidence Audit",
        "",
        f"- generator_git_commit: `{commit}`",
        f"- generator_sha256: `{script_sha}`",
        f"- generation_command: `{GENERATION_COMMAND}`",
        "",
    ]
    for row in all_rows:
        lines.extend(
            [
                f"## {row['figure_id']}",
                "",
                f"- intended claim: {row['intended_claim']}",
                f"- exact data used: {row['exact_data_used']}",
                f"- sample-selection rule: {row['sample_selection_rule']}",
                f"- population/subset: {row['population_subset']}",
                f"- metric definition: {row['metric_definition']}",
                f"- effect size: {row['effect_size']}",
                f"- confidence interval: {row['confidence_interval']}",
                f"- statistical test: {row['statistical_test']}",
                f"- supports claim: {row['supports_claim']}",
                f"- placement: {row['placement']}",
                "",
            ]
        )
    (paths.out_dir / "evidence_audit.md").write_text("\n".join(lines), encoding="utf-8")
    save_json(paths.out_dir / "evidence_audit.json", all_rows)


def write_redesign_audit(paths: Paths, cases: pd.DataFrame, script_sha: str, commit: str) -> None:
    lines = [
        "# SSQ-FMT Figure Redesign Audit",
        "",
        f"- generator_git_commit: `{commit}`",
        f"- generator_sha256: `{script_sha}`",
        f"- generation_command: `{GENERATION_COMMAND}`",
        "",
        "## Removed Or Downgraded Figures",
        "",
        "- `fig_component_recovery_by_source_count`: not regenerated; summary trend was not "
        "sufficiently tied to a separability claim.",
        "- `fig_source_count_confusion`: not regenerated as a main figure.",
        "- `fig_paired_improvement`: replaced by supplementary Delta Dice only.",
        "- `fig_source_separability_phase`: CSV retained; heatmap not regenerated.",
        "- `fig_qualitative_comparison`: replaced by GT-source-center oblique planes.",
        "- `fig_qualitative_orthogonal_supplement`: not regenerated.",
        "- previous source-hypothesis result figures: not regenerated because actual "
        "source-relative cue exports are unavailable.",
        "",
        "## Selected Qualitative Cases",
        "",
    ]
    for _, row in cases.iterrows():
        lines.extend(
            [
                f"### {row['case']}",
                "",
                f"- sample_id: `{row['sample_id']}`",
                f"- source count: {int(row['number_of_sources'])}",
                f"- minimum source distance: {row['minimum_inter_source_distance_mm']:.3f} mm",
                f"- weak/dominant intensity ratio: {row['weak_to_dominant_intensity_ratio']:.3f}",
                f"- SSQ-FMT outcome: {outcome_string(row, 'SSQ-FMT')}",
                "",
            ]
        )
    lines.extend(
        [
            "## Scientific Expression Fixes",
            "",
            "- Removed the invalid distance-regularization paper-layer comparison.",
            "- Used `multi-view surface fluorescence measurements` terminology.",
            "- Stopped using synthetic source-cue visualizations.",
            "- Marked mechanism and robustness figures as rejected until actual internal exports "
            "or controlled perturbation inference outputs exist.",
            "",
        ]
    )
    (paths.out_dir / "redesign_audit.md").write_text("\n".join(lines), encoding="utf-8")


def write_readme(paths: Paths) -> None:
    text = f"""# SSQ-FMT Simulation Paper Evidence Figures

Method name: SSQ-FMT: Source-Separable Query-Canonical Fluorescence Molecular Tomography.

Rebuild all outputs from existing fixed test300 CSV/NPZ artifacts:

```bash
{GENERATION_COMMAND}
```

This generator does not use real-experiment outputs and does not retrain models. Main-text
candidate figures are restricted to evidence-bearing outputs:

- `fig_source_separation_profiles`
- `fig_module_rescue_transitions`, only if supported by paired rescue statistics

Mechanism and robustness figures are rejected until actual internal analysis exports or controlled
perturbation inference outputs exist. See `evidence_audit.md`.

The phase-diagram population CSV is retained at
`separability_phase/source_separability_per_sample.csv`, but the heatmap is not regenerated as a
main-text figure. Source-hypothesis NPZ files contain measurement-derived proposal peaks only;
actual source-relative cue export is unavailable, so no source-hypothesis result figure is
generated.
"""
    (paths.out_dir / "README.md").write_text(text, encoding="utf-8")


def write_latex(paths: Paths, manifest: list[dict]) -> None:
    main_text_ids = {"fig_source_separation_profiles", "fig_module_rescue_transitions"}
    captions = {
        "fig_source_separation_profiles": (
            "Source-separation profiles on GT-defined oblique planes. Each row uses the same "
            "source-center plane across all methods and reports the density profile along the "
            "nearest source pair."
        ),
        "fig_module_rescue_transitions": (
            "Paired complete-recovery transitions on hard multi-source samples. Rescued and "
            "harmed counts are computed from the shared component evaluator."
        ),
        "fig_delta_dice_supplementary": "Supplementary paired Delta Dice summary.",
    }
    chunks = []
    for item in manifest:
        if not item.get("figure_path", "").endswith(".pdf"):
            continue
        if item.get("suitability") == "rejected":
            continue
        fig_id = item["figure_id"]
        if fig_id not in main_text_ids:
            continue
        if fig_id == "fig_module_rescue_transitions" and not str(
            item.get("suitability", "")
        ).startswith("main text"):
            continue
        chunks.append(
            "\\begin{figure}[t]\n"
            "\\centering\n"
            f"\\includegraphics[width=\\linewidth]{{{item['figure_path']}}}\n"
            f"\\caption{{{captions.get(fig_id, fig_id)}}}\n"
            f"\\label{{fig:{fig_id}}}\n"
            "\\end{figure}\n"
        )
    (paths.out_dir / "latex_figures.tex").write_text("\n".join(chunks), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_tmi_style()
    repo = Path.cwd()
    paths = Paths(
        repo=repo,
        data_dir=Path(args.data_dir),
        ssq_eval_dir=Path(args.ssq_eval_dir),
        out_dir=Path(args.out_dir),
        data_out=Path(args.out_dir) / "data",
    )
    ensure_dirs(paths)
    clean_outputs(paths)
    commit = git_commit(repo)
    script_path = Path(__file__).resolve()
    script_sha = sha256_file(script_path)
    manifest: list[dict] = []
    audit_rows: list[dict] = []

    metrics, components = load_all_tables(paths)
    comp = method_component_table(components, metrics)
    sample_ids = sorted(metrics[metrics["method"] == "SSQ-FMT"]["sample_id"].unique())
    sample_meta = load_sample_metadata(paths, sample_ids)
    metrics.to_csv(paths.data_out / "all_method_metrics_per_sample.csv", index=False)
    components.to_csv(paths.data_out / "all_method_component_per_sample.csv", index=False)
    sample_meta.drop(columns=["source_centers_mm", "source_intensities"]).to_csv(
        paths.data_out / "sample_source_metadata.csv", index=False
    )

    cases, contact = select_cases(paths, comp, metrics, sample_meta, args.three_source_case)
    qual_dir = paths.out_dir / "qualitative_cases"
    make_three_source_contact_sheet(paths, contact, qual_dir, manifest, commit)
    cases.to_csv(qual_dir / "selected_source_separation_cases.csv", index=False)
    generate_source_separation_profiles(paths, cases, qual_dir, manifest, audit_rows, commit)

    rescue_dir = paths.out_dir / "paired_improvement"
    generate_module_rescue_transitions(
        paths, comp, sample_meta, rescue_dir, manifest, audit_rows, commit
    )
    generate_delta_dice_supplement(metrics, rescue_dir, manifest, commit)

    phase_dir = paths.out_dir / "separability_phase"
    comp.merge(sample_meta, on="sample_id", how="left").to_csv(
        phase_dir / "source_separability_per_sample.csv", index=False
    )

    export_source_hypothesis_data(
        paths, cases["sample_id"].tolist(), paths.out_dir / "source_hypotheses"
    )

    missing = [
        {
            "task": "analysis_mode_internal_exports",
            "reason": (
                "Actual PTFA, query-canonical reliability, and source-relative cue internal "
                "exports are unavailable; no mechanism figure is generated."
            ),
        },
        {
            "task": "actual_source_relative_cue_export",
            "reason": (
                "Only measurement-derived proposal peaks are exported. Hypothesis confidence, "
                "query-relative distances, soft ownership, ownership entropy/margin, and actual "
                "cue vectors before feature refinement are unavailable."
            ),
        },
        {
            "task": "controlled_perturbation_inference",
            "reason": (
                "No detector/view perturbation inference outputs exist; no robustness figure is "
                "generated."
            ),
        },
    ]
    save_json(paths.out_dir / "missing_data_manifest.json", missing)
    write_evidence_audit(paths, audit_rows, script_sha, commit)
    write_redesign_audit(paths, cases, script_sha, commit)
    write_readme(paths)
    write_latex(paths, manifest)
    save_json(
        paths.out_dir / "figure_manifest.json",
        {
            "generator_git_commit": commit,
            "generator_sha256": script_sha,
            "generation_command": GENERATION_COMMAND,
            "figures": manifest,
        },
    )
    save_json(
        paths.out_dir / "audit_report.json",
        {
            "generated_figures": len(
                [m for m in manifest if m.get("figure_path", "").endswith(".pdf")]
            ),
            "main_text_candidates": [
                m["figure_id"]
                for m in manifest
                if str(m.get("suitability", "")).startswith("main text")
            ],
            "missing_entries": len(missing),
            "methods": list(METHOD_RUNS),
            "generator_git_commit": commit,
            "generator_sha256": script_sha,
            "evaluation_protocol": EVAL_PROTOCOL,
        },
    )
    print(json.dumps(json.loads((paths.out_dir / "audit_report.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
