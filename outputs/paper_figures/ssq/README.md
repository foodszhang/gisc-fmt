# SSQ-FMT Simulation Paper Figures

Method name: SSQ-FMT: Source-Separable Query-Canonical Fluorescence Molecular Tomography.

This directory contains simulation-only figures generated from the fixed 300-sample test split
with voxel threshold 0.5 and the shared component evaluator.

## Reproduction

```bash
uv run python tools/visualization/ssq_paper/generate_ssq_paper_figures.py
```

## Existing Prediction Inputs

The generated component, paired-improvement, source-separability, and qualitative figures use
existing 300-sample prediction/evaluator outputs under:

```text
outputs/fmt_simgen_v2_ssq_ablation/test300
```

No real-experiment outputs are used.

## Missing Data

See `missing_data_manifest.json` for assets that are not available as comparable fixed-protocol
simulation outputs. Missing entries are not replaced with hand-filled values.

## Output Data

All figures have corresponding CSV/JSON/NPZ data files under `data/` or the figure-specific
subdirectory. Figure-level provenance is recorded in `figure_manifest.json`.

## Source Hypotheses

Proposal and peak data are exported, but actual source-relative cue export unavailable. No
source-hypotheses result figure is generated until analysis mode exports hypothesis centers,
hypothesis confidence, query-relative distances, soft ownership, entropy/margin, and cue
embedding or an interpretable scalar projection.

Current missing-data entries: 7
