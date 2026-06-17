#!/usr/bin/env python3
"""Generate simulation-only SSQ-FMT paper figures from existing test300 artifacts."""

from __future__ import annotations

import argparse
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
from scipy import ndimage

METHOD_COLORS = {
    "SSQ-FMT": "#D55E00",
    "w/o transport-aware footprint": "#0072B2",
    "w/o query-canonical reliability": "#009E73",
    "w/o source-relative query cue": "#CC79A7",
    "w/o distance-field regularization": "#56B4E9",
}

METHOD_RUNS = {
    "SSQ-FMT": "no_center_aux",
    "w/o transport-aware footprint": "no_ptfa",
    "w/o query-canonical reliability": "no_canonical_reliability",
    "w/o source-relative query cue": "no_source_cue",
    "w/o distance-field regularization": "no_distance_aux",
}

GENERATION_COMMAND = "uv run python tools/visualization/ssq_paper/generate_ssq_paper_figures.py"

FOOTPRINT_METHODS = {
    "Point sampling": {
        "config": "configs/exp/fmt_simgen_e1b_nongt_mixed.yaml",
        "checkpoint": (
            "outputs/fmt_simgen_v2_3k_20k/ablation_runs/gisc_point/checkpoints/"
            "epoch=41-val_dice=0.5872.ckpt"
        ),
    },
    "Fixed footprint": {
        "config": "configs/exp/fmt_simgen_e2_ptfa_s3_fixed.yaml",
        "checkpoint": (
            "outputs/fmt_simgen_v2_3k_20k/ablation_runs/gisc_fixed_footprint/"
            "checkpoints/epoch=40-val_dice=0.6767.ckpt"
        ),
    },
    "Depth-conditioned footprint": {
        "config": "configs/exp/fmt_simgen_e3_ptfa_s3_exit_depth.yaml",
        "checkpoint": (
            "outputs/fmt_simgen_v2_3k_20k/ablation_runs/gisc_depth_footprint/"
            "checkpoints/epoch=47-val_dice=0.5874.ckpt"
        ),
    },
    "Adaptive constrained footprint": {
        "config": "configs/exp/fmt_simgen_v2_ssq_main.yaml",
        "checkpoint": (
            "outputs/fmt_simgen_v2_ssq_ablation/runs/no_center_aux/checkpoints/"
            "epoch=73-val_dice=0.7364.ckpt"
        ),
    },
}

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
    parser.add_argument("--max_qualitative_cases", type=int, default=4)
    return parser.parse_args()


def git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo, text=True
        ).strip()
    except Exception:
        return "unknown"


def ensure_dirs(paths: Paths) -> None:
    for sub in [
        paths.out_dir,
        paths.data_out,
        paths.out_dir / "footprint_design",
        paths.out_dir / "separability_phase",
        paths.out_dir / "component_analysis",
        paths.out_dir / "paired_improvement",
        paths.out_dir / "reliability_visualization",
        paths.out_dir / "source_hypotheses",
        paths.out_dir / "footprint_scale",
        paths.out_dir / "qualitative_cases",
    ]:
        sub.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def save_figure(fig: plt.Figure, base: Path) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ["pdf", "svg", "png"]:
        out = base.with_suffix(f".{ext}")
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(out, **kwargs)
        paths.append(str(out))
    plt.close(fig)
    return paths


def read_summary(eval_root: Path, run: str) -> dict[str, Any]:
    path = eval_root / run / "metrics_summary.json"
    return json.loads(path.read_text()) if path.exists() else {}


def read_metrics(eval_root: Path, run: str) -> pd.DataFrame:
    df = pd.read_csv(eval_root / run / "metrics_per_sample.csv")
    df["method"] = run
    return df


def read_components(eval_root: Path, run: str) -> pd.DataFrame:
    df = pd.read_csv(eval_root / run / "components" / "component_per_sample.csv")
    df["method"] = run
    return df


def run_to_method(run: str) -> str:
    for method, candidate in METHOD_RUNS.items():
        if candidate == run:
            return method
    return run


def load_all_method_tables(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    components = []
    for method, run in METHOD_RUNS.items():
        m = read_metrics(paths.ssq_eval_dir, run)
        c = read_components(paths.ssq_eval_dir, run)
        m["method"] = method
        c["method"] = method
        metrics.append(m)
        components.append(c)
    return pd.concat(metrics, ignore_index=True), pd.concat(components, ignore_index=True)


def component_count_label(n: int | float) -> str:
    try:
        i = int(n)
    except Exception:
        return ">3"
    return str(i) if i <= 3 else ">3"


def extract_sample_metadata(paths: Paths, sample_ids: list[str]) -> pd.DataFrame:
    rows = []
    samples_root = paths.data_dir / "samples"
    for sample_id in sample_ids:
        sample_dir = samples_root / sample_id
        meta_path = sample_dir / "tumor_params.json"
        gt_path = sample_dir / "gt_voxels.npy"
        if not meta_path.exists() or not gt_path.exists():
            rows.append({"sample_id": sample_id, "metadata_missing": True})
            continue
        meta = json.loads(meta_path.read_text())
        foci = meta.get("foci", [])
        centers = np.array([focus.get("center", [np.nan, np.nan, np.nan]) for focus in foci], float)
        intensities = np.array(
            [float(focus.get("params", {}).get("intensity", np.nan)) for focus in foci], float
        )
        num_sources = int(meta.get("num_foci", len(foci)))
        if len(centers) >= 2:
            dist_mat = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
            dist_mat[dist_mat == 0] = np.inf
            min_dist = float(np.nanmin(dist_mat))
        else:
            min_dist = np.nan
        if np.isfinite(intensities).all() and len(intensities) > 0 and np.nanmax(intensities) > 0:
            weak_ratio = float(np.nanmin(intensities) / np.nanmax(intensities))
        else:
            weak_ratio = np.nan
        gt = np.load(gt_path) > 0.5
        labels, nlab = ndimage.label(gt, structure=np.ones((3, 3, 3), dtype=bool))
        vols = ndimage.sum(gt, labels, index=np.arange(1, nlab + 1)) if nlab else []
        min_volume = float(np.min(vols)) if len(vols) else np.nan
        rows.append(
            {
                "sample_id": sample_id,
                "number_of_sources": num_sources,
                "minimum_inter_source_distance_mm": min_dist,
                "weak_to_dominant_intensity_ratio": weak_ratio,
                "minimum_source_volume": min_volume,
                "shape_set": meta.get("shape_set"),
                "source_type": meta.get("source_type"),
                "metadata_missing": False,
            }
        )
    return pd.DataFrame(rows)


def generate_footprint_design(
    paths: Paths, manifest: list[dict], missing: list[dict], commit: str
) -> None:
    out = paths.out_dir / "footprint_design"
    rows = []
    missing_ckpts = []
    for method, spec in FOOTPRINT_METHODS.items():
        ckpt = paths.repo / spec["checkpoint"]
        config = paths.repo / spec["config"]
        row = {
            "method": method,
            "config": spec["config"],
            "checkpoint": spec["checkpoint"],
            "prediction_dir": "",
            "status": "missing_unified_test300_prediction",
        }
        if not ckpt.exists():
            row["status"] = "missing_checkpoint"
            missing_ckpts.append({"method": method, "status": row["status"]})
        elif not config.exists():
            row["status"] = "missing_config"
            missing.append(
                {
                    "task": "footprint_design",
                    "method": method,
                    "reason": "Mapped checkpoint exists but mapped config file is missing.",
                    "config": "internal asset map",
                    "checkpoint": "internal asset map",
                }
            )
        else:
            missing.append(
                {
                    "task": "footprint_design",
                    "method": method,
                    "reason": (
                        "Checkpoint exists, but no existing unified 300-sample "
                        "prediction/evaluator output was found. Not using historical logs "
                        "to avoid incomparable numbers."
                    ),
                    "required_action": (
                        "Run eval_full_volume_fmt_simgen.py and eval_components_fmt_simgen.py "
                        "with the fixed protocol."
                    ),
                    "config": "internal asset map",
                    "checkpoint": (
                        "available checkpoint; internal path intentionally omitted from paper "
                        "manifest"
                    ),
                }
            )
        rows.append(row)
    df = pd.DataFrame(rows)
    metrics_cols = [
        "method",
        "dice",
        "iou",
        "precision",
        "recall",
        "assd",
        "hd95",
        "component_recall",
        "component_precision",
        "miss_per_sample",
        "merge_per_sample",
        "split_per_sample",
        "status",
    ]
    metrics_df = pd.DataFrame(columns=metrics_cols)
    metrics_df["method"] = df["method"]
    metrics_df["status"] = df["status"]
    metrics_df.to_csv(out / "footprint_design_metrics.csv", index=False)
    save_json(out / "footprint_design_metrics.json", metrics_df.to_dict(orient="records"))
    save_json(out / "missing_checkpoints.json", missing_ckpts)
    (out / "footprint_design_table.tex").write_text(
        metrics_df.to_latex(index=False, na_rep="--", escape=True), encoding="utf-8"
    )
    manifest.append(
        {
            "figure_id": "footprint_design_table",
            "figure_path": str(out / "footprint_design_table.tex"),
            "source_data_path": str(out / "footprint_design_metrics.csv"),
            "methods": list(FOOTPRINT_METHODS),
            "samples": "test300",
            "checkpoint": {
                m: "available or missing as recorded in missing data manifest"
                for m in FOOTPRINT_METHODS
            },
            "config": {m: "internal asset map" for m in FOOTPRINT_METHODS},
            "evaluation_protocol": EVAL_PROTOCOL,
            "generation_command": GENERATION_COMMAND,
            "git_commit": commit,
            "status": "metrics_missing_until_unified_inference_is_run",
        }
    )


def generate_component_analysis(
    paths: Paths, metrics: pd.DataFrame, components: pd.DataFrame, manifest: list[dict], commit: str
) -> None:
    out = paths.out_dir / "component_analysis"
    use_methods = [
        "SSQ-FMT",
        "w/o transport-aware footprint",
        "w/o query-canonical reliability",
        "w/o source-relative query cue",
        "w/o distance-field regularization",
    ]
    comp = components[components["method"].isin(use_methods)].copy()
    comp["source_group"] = comp["num_foci"].map(
        lambda x: f"{int(x)} source" + ("" if int(x) == 1 else "s")
    )
    comp["miss_rate"] = comp["missed_component_count"] / comp["gt_component_count"].clip(lower=1)
    comp["merge_rate"] = comp["merge_count"] / comp["gt_component_count"].clip(lower=1)
    comp["split_rate"] = comp["split_count"] / comp["gt_component_count"].clip(lower=1)
    grouped = comp.groupby(["method", "num_foci", "source_group"], as_index=False).agg(
        correctly_recovered_components=("matched_component_count", "mean"),
        missed_components=("missed_component_count", "mean"),
        merged_components=("merge_count", "mean"),
        split_components=("split_count", "mean"),
        component_recall=("component_recall", "mean"),
        component_precision=("component_precision", "mean"),
        source_count_accuracy=(
            "pred_component_count",
            lambda s: float(
                (s.to_numpy() == comp.loc[s.index, "gt_component_count"].to_numpy()).mean()
            ),
        ),
        miss_rate=("miss_rate", "mean"),
        merge_rate=("merge_rate", "mean"),
        split_rate=("split_rate", "mean"),
        num_samples=("sample_id", "count"),
    )
    grouped.to_csv(out / "component_error_by_foci.csv", index=False)
    grouped.to_csv(paths.data_out / "component_error_by_foci.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5), sharey=True)
    metrics_to_plot = [
        ("component_recall", "Component recall"),
        ("miss_rate", "Miss rate"),
        ("merge_rate", "Merge rate"),
        ("split_rate", "Split rate"),
    ]
    width = 0.17
    x = np.arange(len(metrics_to_plot))
    for ax, n_foci in zip(axes, [1, 2, 3]):
        sub = grouped[grouped["num_foci"] == n_foci]
        for i, method in enumerate(use_methods):
            vals = [float(sub[sub["method"] == method][key].iloc[0]) for key, _ in metrics_to_plot]
            ax.bar(
                x + (i - 2) * width,
                vals,
                width=width,
                label=method if n_foci == 1 else None,
                color=METHOD_COLORS[method],
                edgecolor="black",
                linewidth=0.3,
            )
        ax.set_title(f"{n_foci} source" + ("" if n_foci == 1 else "s"), fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [label for _, label in metrics_to_plot], rotation=25, ha="right", fontsize=7
        )
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25, linewidth=0.4)
    axes[0].set_ylabel("Rate", fontsize=8)
    axes[0].legend(frameon=False, fontsize=6, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    fig.text(0.01, 0.98, "(a)", fontsize=9, weight="bold")
    save_figure(fig, out / "fig_component_error_by_foci")

    confusion_rows = []
    for method in ["w/o transport-aware footprint", "SSQ-FMT"]:
        sub = comp[comp["method"] == method].copy()
        sub["gt_count_bin"] = sub["gt_component_count"].map(component_count_label)
        sub["pred_count_bin"] = sub["pred_component_count"].map(component_count_label)
        for gt_bin in ["1", "2", "3", ">3"]:
            denom = max(1, int((sub["gt_count_bin"] == gt_bin).sum()))
            for pred_bin in ["1", "2", "3", ">3"]:
                count = int(
                    ((sub["gt_count_bin"] == gt_bin) & (sub["pred_count_bin"] == pred_bin)).sum()
                )
                confusion_rows.append(
                    {
                        "method": method,
                        "gt_count": gt_bin,
                        "predicted_count": pred_bin,
                        "count": count,
                        "row_fraction": count / denom,
                    }
                )
    confusion = pd.DataFrame(confusion_rows)
    confusion.to_csv(out / "source_count_confusion.csv", index=False)
    confusion.to_csv(paths.data_out / "source_count_confusion.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))
    for ax, method in zip(axes, ["w/o transport-aware footprint", "SSQ-FMT"]):
        mat = (
            confusion[confusion["method"] == method]
            .pivot(index="gt_count", columns="predicted_count", values="row_fraction")
            .loc[["1", "2", "3", ">3"], ["1", "2", "3", ">3"]]
        )
        im = ax.imshow(mat.to_numpy(), vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(4), ["1", "2", "3", ">3"], fontsize=8)
        ax.set_yticks(range(4), ["1", "2", "3", ">3"], fontsize=8)
        ax.set_xlabel("Predicted source count", fontsize=8)
        ax.set_ylabel("GT source count", fontsize=8)
        ax.set_title(method, fontsize=8)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{mat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Row fraction")
    fig.text(0.01, 0.98, "(b)", fontsize=9, weight="bold")
    save_figure(fig, out / "fig_source_count_confusion")
    manifest.extend(
        [
            {
                "figure_id": "fig_component_error_by_foci",
                "figure_path": str(out / "fig_component_error_by_foci.pdf"),
                "source_data_path": str(out / "component_error_by_foci.csv"),
                "methods": use_methods,
                "samples": "test300",
                "checkpoint": "existing SSQ ablation checkpoints recorded in metrics_summary.json",
                "config": "existing SSQ ablation configs",
                "evaluation_protocol": EVAL_PROTOCOL,
                "generation_command": GENERATION_COMMAND,
                "git_commit": commit,
            },
            {
                "figure_id": "fig_source_count_confusion",
                "figure_path": str(out / "fig_source_count_confusion.pdf"),
                "source_data_path": str(out / "source_count_confusion.csv"),
                "methods": ["w/o transport-aware footprint", "SSQ-FMT"],
                "samples": "test300",
                "checkpoint": "existing SSQ ablation checkpoints recorded in metrics_summary.json",
                "config": "existing SSQ ablation configs",
                "evaluation_protocol": EVAL_PROTOCOL,
                "generation_command": GENERATION_COMMAND,
                "git_commit": commit,
            },
        ]
    )


def bootstrap_ci(
    values: np.ndarray, rng: np.random.Generator, n_boot: int = 2000
) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    boots = [
        float(np.median(rng.choice(values, size=len(values), replace=True))) for _ in range(n_boot)
    ]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def generate_paired_improvement(
    paths: Paths, metrics: pd.DataFrame, components: pd.DataFrame, manifest: list[dict], commit: str
) -> pd.DataFrame:
    out = paths.out_dir / "paired_improvement"
    ref_metrics = metrics[metrics["method"] == "SSQ-FMT"].set_index("sample_id")
    ref_comp = components[components["method"] == "SSQ-FMT"].set_index("sample_id")
    rows = []
    for method in [
        "w/o transport-aware footprint",
        "w/o query-canonical reliability",
        "w/o source-relative query cue",
        "w/o distance-field regularization",
    ]:
        m = metrics[metrics["method"] == method].set_index("sample_id")
        c = components[components["method"] == method].set_index("sample_id")
        common = sorted(set(ref_metrics.index) & set(m.index) & set(ref_comp.index) & set(c.index))
        for sid in common:
            rows.append(
                {
                    "sample_id": sid,
                    "comparison": method,
                    "num_foci": int(ref_metrics.loc[sid, "num_foci"]),
                    "delta_dice": float(ref_metrics.loc[sid, "dice"] - m.loc[sid, "dice"]),
                    "delta_component_recall": float(
                        ref_comp.loc[sid, "component_recall"] - c.loc[sid, "component_recall"]
                    ),
                    "delta_merge_count": float(
                        ref_comp.loc[sid, "merge_count"] - c.loc[sid, "merge_count"]
                    ),
                    "ssq_dice": float(ref_metrics.loc[sid, "dice"]),
                    "ablation_dice": float(m.loc[sid, "dice"]),
                    "ssq_merge_count": float(ref_comp.loc[sid, "merge_count"]),
                    "ablation_merge_count": float(c.loc[sid, "merge_count"]),
                }
            )
    paired = pd.DataFrame(rows)
    paired.to_csv(out / "paired_improvement.csv", index=False)
    paired.to_csv(paths.data_out / "paired_improvement.csv", index=False)

    rng = np.random.default_rng(42)
    summary = {}
    for method, sub in paired.groupby("comparison"):
        summary[method] = {}
        for key in ["delta_dice", "delta_component_recall", "delta_merge_count"]:
            vals = sub[key].to_numpy(float)
            lo, hi = bootstrap_ci(vals, rng)
            summary[method][key] = {
                "median": float(np.nanmedian(vals)),
                "mean": float(np.nanmean(vals)),
                "bootstrap_median_ci95": [lo, hi],
            }
        summary[method]["win_rate_delta_dice_positive"] = float((sub["delta_dice"] > 0).mean())
        summary[method]["merge_reduction_rate"] = float((sub["delta_merge_count"] < 0).mean())
    save_json(out / "paired_bootstrap_summary.json", summary)

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.6), sharex=True)
    plot_keys = [
        ("delta_dice", "Delta Dice"),
        ("delta_component_recall", "Delta component recall"),
        ("delta_merge_count", "Delta merge count"),
    ]
    comparisons = list(paired["comparison"].drop_duplicates())
    x = np.arange(len(comparisons))
    for ax, (key, ylabel) in zip(axes, plot_keys):
        for i, method in enumerate(comparisons):
            vals = paired[paired["comparison"] == method][key].to_numpy(float)
            parts = ax.violinplot(
                vals, positions=[i], widths=0.75, showmeans=False, showextrema=False
            )
            for body in parts["bodies"]:
                body.set_facecolor(METHOD_COLORS[method])
                body.set_alpha(0.35)
                body.set_edgecolor("black")
            jitter = (np.random.default_rng(i).random(len(vals)) - 0.5) * 0.18
            ax.scatter(
                np.full(len(vals), i) + jitter, vals, s=3, alpha=0.35, color=METHOD_COLORS[method]
            )
            med = np.nanmedian(vals)
            lo, hi = summary[method][key]["bootstrap_median_ci95"]
            ax.errorbar(
                i, med, yerr=[[med - lo], [hi - med]], color="black", marker="o", ms=3, lw=0.8
            )
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(axis="y", alpha=0.25, linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(
            ["Footprint", "Reliability", "Source cue", "Distance"],
            rotation=25,
            ha="right",
            fontsize=7,
        )
    fig.text(0.01, 0.98, "(a)", fontsize=9, weight="bold")
    save_figure(fig, out / "fig_paired_improvement")

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    markers = {2: "o", 3: "^"}
    sub = paired[paired["num_foci"].isin([2, 3])]
    for (method, nf), grp in sub.groupby(["comparison", "num_foci"]):
        ax.scatter(
            grp["delta_dice"],
            grp["delta_merge_count"],
            s=14,
            marker=markers[int(nf)],
            alpha=0.65,
            color=METHOD_COLORS[method],
            label=f"{method}, {nf} sources",
            linewidths=0,
        )
    ax.axvline(0, color="black", lw=0.6)
    ax.axhline(0, color="black", lw=0.6)
    ax.fill_between([0, ax.get_xlim()[1]], ax.get_ylim()[0], 0, color="#D55E00", alpha=0.08)
    ax.set_xlabel("Delta Dice", fontsize=8)
    ax.set_ylabel("Delta merge count", fontsize=8)
    ax.grid(alpha=0.25, linewidth=0.4)
    text = []
    for method, values in summary.items():
        text.append(
            f"{method.replace('w/o ', 'no ')}: win {values['win_rate_delta_dice_positive']:.2f}, "
            f"median {values['delta_dice']['median']:.3f}"
        )
    ax.text(0.02, 0.98, "\n".join(text), transform=ax.transAxes, va="top", fontsize=5.5)
    ax.legend(frameon=False, fontsize=5.5, loc="lower right", ncol=1)
    fig.text(0.01, 0.98, "(b)", fontsize=9, weight="bold")
    save_figure(fig, out / "fig_accuracy_separation_scatter")
    manifest.extend(
        [
            {
                "figure_id": "fig_paired_improvement",
                "figure_path": str(out / "fig_paired_improvement.pdf"),
                "source_data_path": str(out / "paired_improvement.csv"),
                "methods": ["SSQ-FMT"] + comparisons,
                "samples": "test300 paired by sample_id",
                "checkpoint": "existing SSQ ablation checkpoints recorded in metrics_summary.json",
                "config": "existing SSQ ablation configs",
                "evaluation_protocol": EVAL_PROTOCOL,
                "generation_command": GENERATION_COMMAND,
                "git_commit": commit,
            },
            {
                "figure_id": "fig_accuracy_separation_scatter",
                "figure_path": str(out / "fig_accuracy_separation_scatter.pdf"),
                "source_data_path": str(out / "paired_improvement.csv"),
                "methods": ["SSQ-FMT"] + comparisons,
                "samples": "test300 two- and three-source samples",
                "checkpoint": "existing SSQ ablation checkpoints recorded in metrics_summary.json",
                "config": "existing SSQ ablation configs",
                "evaluation_protocol": EVAL_PROTOCOL,
                "generation_command": GENERATION_COMMAND,
                "git_commit": commit,
            },
        ]
    )
    return paired


def generate_phase_diagram(
    paths: Paths,
    metrics: pd.DataFrame,
    components: pd.DataFrame,
    sample_meta: pd.DataFrame,
    manifest: list[dict],
    commit: str,
) -> None:
    out = paths.out_dir / "separability_phase"
    keep_methods = ["SSQ-FMT", "w/o transport-aware footprint", "w/o source-relative query cue"]
    comp = components[components["method"].isin(keep_methods)].merge(
        metrics[["sample_id", "method", "dice"]], on=["sample_id", "method"], how="left"
    )
    comp = comp.merge(sample_meta, on="sample_id", how="left")
    comp = comp[comp["number_of_sources"].isin([2, 3])].copy()
    comp["merge_probability"] = (comp["merge_count"] > 0).astype(float)
    comp["miss_count"] = comp["missed_component_count"]
    comp["predicted_source_count"] = comp["pred_component_count"]
    comp[
        [
            "sample_id",
            "method",
            "number_of_sources",
            "minimum_inter_source_distance_mm",
            "weak_to_dominant_intensity_ratio",
            "minimum_source_volume",
            "dice",
            "component_recall",
            "miss_count",
            "merge_count",
            "split_count",
            "predicted_source_count",
        ]
    ].to_csv(out / "source_separability_per_sample.csv", index=False)

    dist = comp["minimum_inter_source_distance_mm"].dropna()
    ratio = comp["weak_to_dominant_intensity_ratio"].dropna()
    if len(dist) < 4 or len(ratio) < 4:
        save_json(
            out / "source_separability_bins.json",
            {"status": "missing", "reason": "Insufficient source metadata for binning."},
        )
        return
    dist_edges = np.unique(np.quantile(dist, [0, 0.25, 0.5, 0.75, 1.0]))
    ratio_edges = np.unique(np.quantile(ratio, [0, 0.25, 0.5, 0.75, 1.0]))
    comp["dist_bin"] = pd.cut(
        comp["minimum_inter_source_distance_mm"], dist_edges, include_lowest=True
    )
    comp["ratio_bin"] = pd.cut(
        comp["weak_to_dominant_intensity_ratio"], ratio_edges, include_lowest=True
    )
    rows = []
    for method, sub in comp.groupby("method"):
        for d_bin in sub["dist_bin"].cat.categories:
            for r_bin in sub["ratio_bin"].cat.categories:
                cell = sub[(sub["dist_bin"] == d_bin) & (sub["ratio_bin"] == r_bin)]
                rows.append(
                    {
                        "method": method,
                        "distance_bin": str(d_bin),
                        "ratio_bin": str(r_bin),
                        "distance_mid_mm": (float(d_bin.left) + float(d_bin.right)) / 2,
                        "ratio_mid": (float(r_bin.left) + float(r_bin.right)) / 2,
                        "sample_count": int(len(cell)),
                        "component_recall": float(cell["component_recall"].mean())
                        if len(cell)
                        else np.nan,
                        "merge_probability": float(cell["merge_probability"].mean())
                        if len(cell)
                        else np.nan,
                    }
                )
    bins = pd.DataFrame(rows)
    ref = bins[bins["method"] == "SSQ-FMT"].copy()
    no_foot = bins[bins["method"] == "w/o transport-aware footprint"].copy()
    no_cue = bins[bins["method"] == "w/o source-relative query cue"].copy()
    key_cols = ["distance_bin", "ratio_bin"]
    ref = ref.merge(
        no_foot[key_cols + ["component_recall"]].rename(
            columns={"component_recall": "component_recall_no_footprint"}
        ),
        on=key_cols,
        how="left",
    ).merge(
        no_cue[key_cols + ["merge_probability"]].rename(
            columns={"merge_probability": "merge_probability_no_source_cue"}
        ),
        on=key_cols,
        how="left",
    )
    ref["component_recall_improvement_vs_no_footprint"] = (
        ref["component_recall"] - ref["component_recall_no_footprint"]
    )
    ref["merge_reduction_vs_no_source_cue"] = (
        ref["merge_probability_no_source_cue"] - ref["merge_probability"]
    )
    bins.to_csv(out / "source_separability_bins_all_methods.csv", index=False)
    ref.to_csv(out / "source_separability_bins.csv", index=False)
    ref.to_csv(paths.data_out / "source_separability_bins.csv", index=False)

    panels = [
        ("component_recall", "SSQ-FMT component recall", "viridis", 0, 1),
        ("merge_probability", "SSQ-FMT merge probability", "magma", 0, 1),
        (
            "component_recall_improvement_vs_no_footprint",
            "Recall improvement vs w/o transport-aware footprint",
            "BrBG",
            -0.5,
            0.5,
        ),
        (
            "merge_reduction_vs_no_source_cue",
            "Merge reduction vs w/o source-relative query cue",
            "BrBG",
            -0.5,
            0.5,
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.3), constrained_layout=True)
    d_bins = list(ref["distance_bin"].drop_duplicates())
    r_bins = list(ref["ratio_bin"].drop_duplicates())
    for ax, (key, title, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        mat = ref.pivot(index="ratio_bin", columns="distance_bin", values=key).reindex(
            index=r_bins, columns=d_bins
        )
        cnt = ref.pivot(index="ratio_bin", columns="distance_bin", values="sample_count").reindex(
            index=r_bins, columns=d_bins
        )
        im = ax.imshow(
            mat.to_numpy(), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto"
        )
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                n = cnt.iloc[i, j]
                label = "n=0" if pd.isna(n) else f"n={int(n)}"
                ax.text(j, i, label, ha="center", va="center", fontsize=6, color="white")
                if pd.isna(n) or n < 5:
                    rect = plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, facecolor="0.85", alpha=0.45, hatch="//"
                    )
                    ax.add_patch(rect)
        ax.set_title(title, fontsize=8)
        ax.set_xticks(
            range(len(d_bins)), [str(x) for x in d_bins], rotation=25, ha="right", fontsize=6
        )
        ax.set_yticks(range(len(r_bins)), [str(x) for x in r_bins], fontsize=6)
        ax.set_xlabel("Minimum inter-source distance (mm)", fontsize=8)
        ax.set_ylabel("Weak-to-dominant intensity ratio", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.text(0.01, 0.99, "(a)", fontsize=9, weight="bold")
    save_figure(fig, out / "fig_source_separability_phase")
    manifest.append(
        {
            "figure_id": "fig_source_separability_phase",
            "figure_path": str(out / "fig_source_separability_phase.pdf"),
            "source_data_path": str(out / "source_separability_bins.csv"),
            "methods": keep_methods,
            "samples": "test300 two- and three-source samples",
            "checkpoint": "existing SSQ ablation checkpoints recorded in metrics_summary.json",
            "config": "existing SSQ ablation configs",
            "evaluation_protocol": EVAL_PROTOCOL
            | {"binning": "quantile bins due empirical spacing distribution"},
            "generation_command": GENERATION_COMMAND,
            "git_commit": commit,
        }
    )


def central_slices(vol: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = vol > 0.5
    if mask.any():
        coords = np.argwhere(mask)
        center = np.round(coords.mean(axis=0)).astype(int)
    else:
        center = np.array(vol.shape) // 2
    x, y, z = [int(np.clip(center[i], 0, vol.shape[i] - 1)) for i in range(3)]
    return vol[:, :, z].T, vol[:, y, :].T, vol[x, :, :].T


def plot_case_grid(
    paths: Paths,
    sample_id: str,
    methods: list[str],
    out_base: Path,
    note: str,
) -> None:
    fig, axes = plt.subplots(len(methods), 4, figsize=(7.16, 1.45 * len(methods)))
    if len(methods) == 1:
        axes = axes[None, :]
    pred_paths = {
        method: paths.ssq_eval_dir / METHOD_RUNS[method] / "predictions" / f"{sample_id}.npz"
        for method in methods
        if method != "GT"
    }
    gt = np.load(next(iter(pred_paths.values())))["gt"] > 0.5
    vols = {"GT": gt.astype(float)}
    for method, p in pred_paths.items():
        vols[method] = (np.load(p)["pred"] >= 0.5).astype(float)
    for i, method in enumerate(methods):
        vol = vols[method]
        mip = vol.max(axis=2).T
        slices = central_slices(vol)
        images = [mip, *slices]
        labels = ["3D rendering", "Axial slice", "Coronal slice", "Sagittal slice"]
        for j, img in enumerate(images):
            ax = axes[i, j]
            ax.imshow(img, cmap="gray", origin="lower", vmin=0, vmax=1)
            if method != "GT":
                gt_img = [gt.max(axis=2).T, *central_slices(gt.astype(float))][j]
                ax.contour(gt_img, levels=[0.5], colors="white", linewidths=0.45)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(labels[j], fontsize=7)
            if j == 0:
                ax.set_ylabel(method, fontsize=7)
    fig.text(0.01, 0.99, "(a)", fontsize=9, weight="bold")
    fig.text(0.5, 0.01, note, fontsize=7, ha="center")
    save_figure(fig, out_base)


def generate_qualitative(
    paths: Paths,
    metrics: pd.DataFrame,
    components: pd.DataFrame,
    sample_meta: pd.DataFrame,
    paired: pd.DataFrame,
    manifest: list[dict],
    commit: str,
) -> None:
    out = paths.out_dir / "qualitative_cases"
    ssq = metrics[metrics["method"] == "SSQ-FMT"].merge(
        components[components["method"] == "SSQ-FMT"][
            ["sample_id", "merge_count", "missed_component_count"]
        ],
        on="sample_id",
    )
    ssq = ssq.merge(sample_meta, on="sample_id", how="left")
    source_cue_delta = paired[paired["comparison"] == "w/o source-relative query cue"].set_index(
        "sample_id"
    )
    candidates = []
    multi = ssq[ssq["num_foci"].isin([2, 3])].copy()
    if not multi.empty:
        median_sid = multi.iloc[(multi["dice"] - multi["dice"].median()).abs().argsort().iloc[0]][
            "sample_id"
        ]
        candidates.append(("fig_qualitative_adjacent", median_sid, "median multi-source case"))
    improved = source_cue_delta[source_cue_delta.index.isin(multi["sample_id"])]
    if not improved.empty:
        sid = improved.sort_values(
            ["delta_dice", "delta_merge_count"], ascending=[False, True]
        ).index[0]
        candidates.append(("fig_qualitative_weak_source", sid, "SSQ-FMT improved case"))
    three = ssq[ssq["num_foci"] == 3]
    if not three.empty:
        sid = three.iloc[(three["dice"] - three["dice"].median()).abs().argsort().iloc[0]][
            "sample_id"
        ]
        candidates.append(("fig_qualitative_three_source", sid, "representative three-source case"))
    failure = multi.sort_values(
        ["dice", "missed_component_count", "merge_count"], ascending=[True, False, False]
    )
    if not failure.empty:
        candidates.append(("fig_qualitative_failure", failure.iloc[0]["sample_id"], "failure case"))
    methods = [
        "GT",
        "w/o transport-aware footprint",
        "w/o query-canonical reliability",
        "w/o source-relative query cue",
        "SSQ-FMT",
    ]
    case_rows = []
    for fig_id, sample_id, note in candidates[:4]:
        plot_case_grid(paths, sample_id, methods, out / fig_id, note)
        neutral_data_path = out / f"{fig_id}_data.npz"
        arrays = {}
        for method in methods:
            key = method.lower().replace("SSQ-FMT".lower(), "ssq_fmt")
            key = (
                key.replace("w/o ", "without_")
                .replace(" ", "_")
                .replace("-", "_")
                .replace("/", "_")
            )
            if method == "GT":
                pred_path = (
                    paths.ssq_eval_dir / METHOD_RUNS["SSQ-FMT"] / "predictions" / f"{sample_id}.npz"
                )
                arrays[key] = np.load(pred_path)["gt"].astype(np.float16)
            else:
                pred_path = (
                    paths.ssq_eval_dir / METHOD_RUNS[method] / "predictions" / f"{sample_id}.npz"
                )
                arrays[key] = np.load(pred_path)["pred"].astype(np.float16)
        np.savez_compressed(neutral_data_path, **arrays)
        case_rows.append({"figure_id": fig_id, "sample_id": sample_id, "selection_note": note})
        manifest.append(
            {
                "figure_id": fig_id,
                "figure_path": str(out / f"{fig_id}.pdf"),
                "source_data_path": str(neutral_data_path),
                "methods": methods,
                "samples": [sample_id],
                "checkpoint": "existing SSQ ablation checkpoints recorded in metrics_summary.json",
                "config": "existing SSQ ablation configs",
                "evaluation_protocol": EVAL_PROTOCOL,
                "generation_command": GENERATION_COMMAND,
                "git_commit": commit,
            }
        )
    pd.DataFrame(case_rows).to_csv(out / "qualitative_case_selection.csv", index=False)
    pd.DataFrame(case_rows).to_csv(paths.data_out / "qualitative_case_selection.csv", index=False)


def local_maxima3d(volume: np.ndarray, max_peaks: int = 5) -> tuple[np.ndarray, np.ndarray]:
    smooth = ndimage.gaussian_filter(volume.astype(float), sigma=1.0)
    mx = ndimage.maximum_filter(smooth, size=5)
    mask = (smooth == mx) & (smooth > smooth.max() * 0.1)
    coords = np.argwhere(mask)
    vals = smooth[mask]
    if len(vals) == 0:
        return np.empty((0, 3), int), np.empty((0,), float)
    order = np.argsort(vals)[::-1][:max_peaks]
    return coords[order], vals[order]


def generate_source_hypotheses(
    paths: Paths, sample_meta: pd.DataFrame, manifest: list[dict], missing: list[dict], commit: str
) -> None:
    out = paths.out_dir / "source_hypotheses"
    candidates = (
        sample_meta[sample_meta["number_of_sources"].isin([2, 3])].head(3)["sample_id"].tolist()
    )
    rows = []
    for sample_id in candidates:
        sample_dir = paths.data_dir / "samples" / sample_id
        heat_path = sample_dir / "proposal" / "meas_backproj_heatmap.npy"
        proj_path = sample_dir / "proj.npz"
        pred_path = paths.ssq_eval_dir / METHOD_RUNS["SSQ-FMT"] / "predictions" / f"{sample_id}.npz"
        if not heat_path.exists() or not proj_path.exists() or not pred_path.exists():
            missing.append(
                {
                    "task": "source_hypotheses",
                    "sample_id": sample_id,
                    "reason": "Required proposal, measurement, or prediction file is missing.",
                }
            )
            continue
        heat = np.load(heat_path)
        peaks, scores = local_maxima3d(heat, max_peaks=5)
        proj = np.load(proj_path)
        views = [k for k in proj.files if not k.startswith("depth_")]
        pred_npz = np.load(pred_path)
        pred = pred_npz["pred"]
        gt = pred_npz["gt"]
        np.savez_compressed(
            out / f"source_hypotheses_{sample_id}.npz",
            heatmap=heat.astype(np.float32),
            peak_indices=peaks.astype(np.int16),
            peak_scores=scores.astype(np.float32),
            prediction=pred.astype(np.float16),
            gt=gt.astype(np.float16),
        )
        fig, axes = plt.subplots(1, 5, figsize=(7.16, 1.8))
        meas = np.stack([proj[v] for v in views])
        axes[0].imshow(meas.max(axis=0), cmap="magma")
        axes[0].set_title("Surface measurements", fontsize=7)
        axes[1].imshow(heat.max(axis=2).T, cmap="viridis", origin="lower")
        axes[1].set_title("Coarse proposal field", fontsize=7)
        axes[2].imshow(heat.max(axis=2).T, cmap="viridis", origin="lower")
        if len(peaks):
            axes[2].scatter(
                peaks[:, 0], peaks[:, 1], c="#D55E00", s=14, edgecolor="white", linewidth=0.3
            )
        axes[2].set_title("Source hypotheses", fontsize=7)
        axes[3].imshow(pred.max(axis=2).T, cmap="gray", origin="lower")
        if len(peaks):
            center = peaks[0]
            yy, xx = np.indices(pred.shape[:2])
            dist = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2)
            ownership = np.exp(-(dist**2) / (2 * 12.0**2))
            axes[3].contour(ownership.T, levels=[0.5], colors="#D55E00", linewidths=0.7)
        axes[3].set_title("Source-relative query cues", fontsize=7)
        axes[4].imshow(pred.max(axis=2).T, cmap="gray", origin="lower")
        axes[4].contour((gt > 0.5).max(axis=2).T, levels=[0.5], colors="white", linewidths=0.5)
        axes[4].set_title("Final reconstruction", fontsize=7)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.text(0.01, 0.98, "(a)", fontsize=9, weight="bold")
        save_figure(fig, out / f"fig_source_hypotheses_{sample_id}")
        rows.append(
            {
                "sample_id": sample_id,
                "num_hypotheses": int(len(peaks)),
                "max_score": float(scores[0]) if len(scores) else np.nan,
            }
        )
        manifest.append(
            {
                "figure_id": f"fig_source_hypotheses_{sample_id}",
                "figure_path": str(out / f"fig_source_hypotheses_{sample_id}.pdf"),
                "source_data_path": str(out / f"source_hypotheses_{sample_id}.npz"),
                "methods": ["SSQ-FMT"],
                "samples": [sample_id],
                "checkpoint": "existing SSQ-FMT prediction",
                "config": "SSQ-FMT source hypothesis config",
                "evaluation_protocol": EVAL_PROTOCOL,
                "generation_command": GENERATION_COMMAND,
                "git_commit": commit,
                "caption_note": (
                    "Hypotheses are over-complete candidates, not final detections; "
                    "GT source centers are not used during inference."
                ),
            }
        )
    pd.DataFrame(rows).to_csv(out / "source_hypotheses_summary.csv", index=False)


def generate_missing_analysis_files(
    paths: Paths, missing: list[dict], manifest: list[dict], commit: str
) -> None:
    for task, subdir, reason in [
        (
            "query_canonical_reliability",
            "reliability_visualization",
            (
                "No analysis-mode export with projected detector coordinate, local footprint "
                "weights, view reliability, and aggregated feature norm was found in existing "
                "artifacts."
            ),
        ),
        (
            "footprint_scale",
            "footprint_scale",
            (
                "No per-query-view export of exit depth, effective footprint radius, and "
                "reliability was found in existing artifacts."
            ),
        ),
    ]:
        out = paths.out_dir / subdir
        save_json(out / "missing_data.json", {"task": task, "reason": reason})
        missing.append(
            {
                "task": task,
                "reason": reason,
                "required_action": (
                    "Run an analysis-mode export pass; do not infer these quantities from "
                    "final predictions."
                ),
            }
        )
        manifest.append(
            {
                "figure_id": task,
                "figure_path": "",
                "source_data_path": str(out / "missing_data.json"),
                "methods": ["SSQ-FMT"],
                "samples": [],
                "checkpoint": "not evaluated",
                "config": "analysis export required",
                "evaluation_protocol": EVAL_PROTOCOL,
                "generation_command": GENERATION_COMMAND,
                "git_commit": commit,
                "status": "missing_analysis_export",
            }
        )


def generate_latex(paths: Paths, manifest: list[dict]) -> None:
    snippets = []
    captions = {
        "fig_source_separability_phase": (
            "Source-separability phase diagram for SSQ-FMT on the fixed 300-sample "
            "simulation test set. Bins summarize two- and three-source samples by minimum "
            "inter-source distance and weak-to-dominant intensity ratio. Hatched cells have "
            "fewer than five samples."
        ),
        "fig_component_error_by_foci": (
            "Component-level error decomposition grouped by source count. SSQ-FMT improves "
            "multi-source recovery while preserving the shared voxel threshold and component "
            "matching protocol."
        ),
        "fig_source_count_confusion": (
            "Source-count confusion matrices for SSQ-FMT and the ablation without the "
            "transport-aware detector footprint."
        ),
        "fig_paired_improvement": (
            "Per-sample paired improvement of SSQ-FMT over ablations using identical sample IDs."
        ),
        "fig_accuracy_separation_scatter": (
            "Accuracy-separation trade-off for paired changes. The favorable region has "
            "positive Delta Dice and negative Delta merge count."
        ),
    }
    for item in manifest:
        fig_id = item["figure_id"]
        fig_path = item.get("figure_path", "")
        if not fig_path.endswith(".pdf"):
            continue
        caption = captions.get(
            fig_id, "SSQ-FMT simulation result generated from fixed test-set predictions."
        )
        snippets.append(
            "\\begin{figure}[t]\n"
            "\\centering\n"
            f"\\includegraphics[width=\\linewidth]{{{fig_path}}}\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{fig:{fig_id}}}\n"
            "\\end{figure}\n"
        )
    (paths.out_dir / "latex_figures.tex").write_text("\n".join(snippets), encoding="utf-8")


def generate_readme(paths: Paths, missing: list[dict]) -> None:
    text = f"""# SSQ-FMT Simulation Paper Figures

Method name: SSQ-FMT: Source-Separable Query-Canonical Fluorescence Molecular Tomography.

This directory contains simulation-only figures generated from the fixed 300-sample test split
with voxel threshold 0.5 and the shared component evaluator.

## Reproduction

```bash
uv run python tools/visualization/ssq_paper/generate_ssq_paper_figures.py
```

## Existing Prediction Inputs

The generated component, paired-improvement, source-separability, qualitative, and source-hypothesis
figures use existing 300-sample prediction/evaluator outputs under:

```text
{paths.ssq_eval_dir}
```

No real-experiment outputs are used.

## Missing Data

See `missing_data_manifest.json` for assets that are not available as comparable fixed-protocol
simulation outputs. Missing entries are not replaced with hand-filled values.

## Output Data

All figures have corresponding CSV/JSON/NPZ data files under `data/` or the figure-specific
subdirectory. Figure-level provenance is recorded in `figure_manifest.json`.
"""
    if missing:
        text += "\nCurrent missing-data entries: " + str(len(missing)) + "\n"
    (paths.out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = Path.cwd()
    paths = Paths(
        repo=repo,
        data_dir=Path(args.data_dir),
        ssq_eval_dir=Path(args.ssq_eval_dir),
        out_dir=Path(args.out_dir),
        data_out=Path(args.out_dir) / "data",
    )
    ensure_dirs(paths)
    commit = git_commit(repo)
    manifest: list[dict] = []
    missing: list[dict] = []

    metrics, components = load_all_method_tables(paths)
    metrics.to_csv(paths.data_out / "all_method_metrics_per_sample.csv", index=False)
    components.to_csv(paths.data_out / "all_method_component_per_sample.csv", index=False)
    sample_ids = sorted(metrics[metrics["method"] == "SSQ-FMT"]["sample_id"].unique().tolist())
    sample_meta = extract_sample_metadata(paths, sample_ids)
    sample_meta.to_csv(paths.data_out / "sample_source_metadata.csv", index=False)

    generate_footprint_design(paths, manifest, missing, commit)
    generate_component_analysis(paths, metrics, components, manifest, commit)
    paired = generate_paired_improvement(paths, metrics, components, manifest, commit)
    generate_phase_diagram(paths, metrics, components, sample_meta, manifest, commit)
    generate_qualitative(paths, metrics, components, sample_meta, paired, manifest, commit)
    generate_source_hypotheses(paths, sample_meta, manifest, missing, commit)
    generate_missing_analysis_files(paths, missing, manifest, commit)

    save_json(paths.out_dir / "figure_manifest.json", manifest)
    save_json(paths.out_dir / "missing_data_manifest.json", missing)
    generate_latex(paths, manifest)
    generate_readme(paths, missing)
    audit = {
        "generated_figures": len(
            [m for m in manifest if str(m.get("figure_path", "")).endswith(".pdf")]
        ),
        "missing_entries": len(missing),
        "methods": list(METHOD_RUNS.keys()),
        "evaluation_protocol": EVAL_PROTOCOL,
        "git_commit": commit,
    }
    save_json(paths.out_dir / "audit_report.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
