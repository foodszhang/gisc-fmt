# SSQ-FMT Figure Redesign Audit

- generator_repo_head_at_generation: `92496da`
- generator_script_git_blob_sha: `666e59823bdb47aa70f0c886145e83dc2bb641fe`
- generator_script_sha256: `ff22009048099b6f805fa3328200a35ffd1173713b5f88bf89678cc5676e6657`
- generation_command: `uv run python tools/visualization/ssq_paper/generate_ssq_paper_figures.py`

## Removed Or Downgraded Figures

- `fig_component_recovery_by_source_count`: not regenerated; summary trend was not sufficiently tied to a separability claim.
- `fig_source_count_confusion`: not regenerated as a main figure.
- `fig_paired_improvement`: replaced by supplementary Delta Dice only.
- `fig_source_separability_phase`: CSV retained; heatmap not regenerated.
- `fig_qualitative_comparison`: replaced by GT-source-center oblique planes.
- `fig_qualitative_orthogonal_supplement`: not regenerated.
- previous source-hypothesis result figures: not regenerated because actual source-relative cue exports are unavailable.

## Selected Qualitative Cases

### Adjacent sources

- sample_id: `sample_0712`
- source count: 2
- minimum source distance: 5.231 mm
- weak/dominant intensity ratio: 0.544
- SSQ-FMT outcome: pred=2, match=2, miss=0, merge=0, split=0

### Three sources with a weak focus

- sample_id: `sample_0300`
- source count: 3
- minimum source distance: 7.091 mm
- weak/dominant intensity ratio: 0.419
- SSQ-FMT outcome: pred=3, match=3, miss=0, merge=0, split=0

## Scientific Expression Fixes

- Removed the invalid distance-regularization paper-layer comparison.
- Used `multi-view surface fluorescence measurements` terminology.
- Stopped using synthetic source-cue visualizations.
- Marked mechanism and robustness figures as rejected until actual internal exports or controlled perturbation inference outputs exist.
