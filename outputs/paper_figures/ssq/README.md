# SSQ-FMT Simulation Paper Evidence Figures

Method name: SSQ-FMT: Source-Separable Query-Canonical Fluorescence Molecular Tomography.

Rebuild all outputs from existing fixed test300 CSV/NPZ artifacts:

```bash
uv run python tools/visualization/ssq_paper/generate_ssq_paper_figures.py
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
