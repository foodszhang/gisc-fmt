# SSQ-FMT Figure Redesign Audit

## Removed or Demoted Outputs

- Removed `w/o distance-field regularization` from paper-layer comparisons because the current mainline has the distance head and distance loss disabled.
- Removed the old grouped-bar component figure and replaced it with a two-panel source-count trend figure.
- Removed the old paired violin/jitter plot and stopped emitting the accuracy-separation scatter as a main-text figure.
- Stopped generating source-hypotheses result figures because actual source-relative cue export is unavailable.

## Scientific Corrections

- Qualitative slices are selected from GT component centers once per sample and reused for all methods.
- Prediction panels show continuous density rather than binarized masks.
- MIP panels are no longer labeled as 3D rendering.
- Source-hypotheses data are exported without drawing artificial Gaussian ownership cues.

## Qualitative Cases and Shared Slice Indices

| case | sample_id | main_plane | sagittal_index | coronal_index | axial_index |
| --- | --- | --- | --- | --- | --- |
| Adjacent sources | sample_0712 | sagittal | 107 | 171 | 47 |
| Weak secondary source | sample_2505 | coronal | 116 | 132 | 26 |
| Three-source case | sample_1516 | axial | 56 | 130 | 32 |

## Figure Suitability

- `footprint_design_table`: `outputs/paper_figures/ssq/footprint_design/footprint_design_table.tex`; suitability: main text; data: `outputs/paper_figures/ssq/footprint_design/footprint_design_metrics.csv`.
- `fig_component_recovery_by_source_count`: `outputs/paper_figures/ssq/component_analysis/fig_component_recovery_by_source_count.pdf`; suitability: main text; data: `outputs/paper_figures/ssq/component_analysis/component_error_by_foci.csv`.
- `fig_source_count_confusion`: `outputs/paper_figures/ssq/component_analysis/fig_source_count_confusion.pdf`; suitability: main text; data: `outputs/paper_figures/ssq/component_analysis/source_count_confusion.csv`.
- `fig_paired_improvement`: `outputs/paper_figures/ssq/paired_improvement/fig_paired_improvement.pdf`; suitability: main text; data: `outputs/paper_figures/ssq/paired_improvement/paired_improvement.csv`.
- `fig_source_separability_phase`: `outputs/paper_figures/ssq/separability_phase/fig_source_separability_phase.pdf`; suitability: main text; data: `outputs/paper_figures/ssq/separability_phase/source_separability_bins.csv`.
- `fig_qualitative_comparison`: `outputs/paper_figures/ssq/qualitative_cases/fig_qualitative_comparison.pdf`; suitability: main text; data: `outputs/paper_figures/ssq/qualitative_cases/fig_qualitative_comparison_data.npz`.
- `fig_qualitative_orthogonal_supplement`: `outputs/paper_figures/ssq/qualitative_cases/fig_qualitative_orthogonal_supplement.pdf`; suitability: supplementary; data: `outputs/paper_figures/ssq/qualitative_cases/qualitative_case_selection.csv`.

## Reproduction

```bash
uv run python tools/visualization/ssq_paper/generate_ssq_paper_figures.py
```
