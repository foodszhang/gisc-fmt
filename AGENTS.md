# Repository Guidelines

## Project Structure & Module Organization

This is a Python research codebase for GISC-FMT training and evaluation. The main entrypoint is `train.py`, which dispatches Hydra tasks such as `fit`, `validate`, and `test`. Core package code lives in `minr_fmt/`: models in `minr_fmt/models/`, network blocks in `minr_fmt/network/`, datasets in `minr_fmt/dataset/`, and helpers in `minr_fmt/utils/`. Hydra configuration is under `configs/`. Tests live in `tests/`, utility scripts in `scripts/`, and split checkpoint artifacts in `pretrained/`.

## Build, Test, and Development Commands

- `uv sync`: create/update the project environment from `pyproject.toml` and `uv.lock`.
- `uv run pytest`: run the pytest suite configured for `tests/test_*.py`.
- `uv run ruff check .`: run lint checks for pycodestyle, pyflakes, warnings, and import ordering.
- `uv run black .`: format Python files using Black.
- `uv run python scripts/quick_run_bg.py`: run a lightweight background-branch sanity check.
- `uv run python train.py fit model=gisc_fmt`: start a training run with the GISC-FMT model.
- `uv run python train.py fit model=<baseline> data.dataset_type=fmt_simgen`: train an adapted voxel-domain baseline through the shared Hydra/Lightning entrypoint.
- `uv run python train.py test model=<baseline> ckpt_path=<path> data.dataset_type=fmt_simgen`: evaluate a baseline checkpoint with the shared voxel metrics and output layout.
- `bash scripts/reassemble_pretrained.sh`: rebuild the provided split checkpoint before evaluation.
- `uv run python train.py test model=gisc_fmt ckpt_path=pretrained/gisc_fmt_brain1000_best.ckpt`: evaluate a checkpoint.

## Coding Style & Naming Conventions

Use Python 3.12-compatible code. Keep formatting Black-compatible and respect the Ruff line length of 100 characters. Prefer explicit configuration keys over hidden code defaults, matching the Hydra/OmegaConf style. Use `snake_case` for functions, variables, config keys, and modules; use `PascalCase` for model or module classes. Keep Bash scripts guarded with `set -euo pipefail`.

## Baseline Integration Guidelines

Do not add independent baseline training entrypoints. New comparison models must plug into `ModelFactory`, use `configs/model/<name>.yaml`, and run through `train.py fit/validate/test`. Query-wise GISC variants should keep the existing `(density_pred, aux_outputs)` protocol. Voxel-domain baselines should return `{"pred_voxel": logits, "aux_outputs": ...}` and use the shared `TrainingLightningModule` voxel loss/metrics path. Keep adapted baselines from importing GISC-only query projection, footprint scheduling, or confidence completion mechanisms unless the model is explicitly an internal CQR ablation.

## Testing Guidelines

Tests use `pytest` and should be placed in `tests/` with names matching `test_*.py`. Add focused tests for model branches, tensor shapes, config behavior, and backward compatibility. Keep tensors small enough for CPU execution unless GPU behavior is required. Run `uv run pytest` before submitting changes.

## Commit & Pull Request Guidelines

The current history uses concise imperative commit messages, for example `Add split pretrained checkpoint` and `Initial public release (split pretrained)`. Follow that style: describe the concrete change, not the process. Pull requests should include a summary, commands run, dataset/checkpoint assumptions, and metric or behavior changes. Link issues when applicable.

## Security & Configuration Tips

Do not commit private datasets, generated `outputs/`, API keys, or unsplit large checkpoints unless intentionally part of a release. Prefer Hydra command-line overrides for local paths, such as `data.train_dir=/abs/path/to/train`, instead of hard-coding machine-specific paths in tracked configs.

## Current FMT-SimGen v2 Findings

Use `/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k` for the v2 full comparison unless the user requests a different dataset. The fixed split keeps `train.txt` at 2400 samples and splits the original 600 validation samples into `val.txt` 300 and `test.txt` 300 with stratification by `(num_foci, depth_tier)`. The dataset includes descattered projections, and GISC-FMT should use `proj_noscatter.npz` where configured.

The active comparison set excludes the CQR ablation series by user request and focuses on GISC-FMT plus paper baselines: `fem2vox_unet`, `uhr_deepfmt`, `vox_dmrn`, `two_stage_deepfmt`, `fmt_reconnet`, `pgdpnn`, `map_pgan`, `d2_recst`, and `dspgn`. Keep all comparison models on the shared Hydra/Lightning entrypoint and common v2 exp config.

The historical `gisc_fmt` baseline selected `epoch=44-val_dice=0.6930.ckpt` by
candidate Dice. Its test300 Dice is about 0.662, IoU about 0.512, ASSD about 0.579,
and HD95 about 2.123 at threshold 0.5. Keep this checkpoint only when a table
explicitly needs the historical baseline. Do not report it as the current best method.

The strongest archived GISC-FMT test300 result is E13-MSQ-fixed:
`outputs/fmt_simgen_v2_multisource/runs/e13_msq_fixed_from_mpb/checkpoints/epoch=05-val_dice=0.7432.ckpt`.
On the same 300-sample test split, Dice is about 0.741, IoU about 0.603, ASSD about
0.350, and HD95 about 1.120. By `num_foci`, Dice is about 0.799 for one focus, 0.747
for two foci, and 0.692 for three foci. Recall is about 0.892, 0.810, and 0.696,
respectively. Multi-source recovery remains the main improvement target.

For the active paper summary and TMI-style figures, use the E15 center-distance
checkpoint:
`outputs/fmt_simgen_v2_e15_center_distance_precomputed/checkpoints/epoch=52-val_dice=0.7414.ckpt`.
Its test300 Dice is about 0.725. Keep E13-MSQ-fixed recorded as the stronger archived
test result, but do not mix E13 predictions into E15 paper figures.

FEM-domain methods must be mapped to the common `[190, 200, 104]` voxel grid before
metrics or figures. Do not report mesh-only Dice as a paper comparison. The verified
`fem_coarse` and `fem_to_voxel` internal diagnostic voxel Dice is about 0.574. Do not
present either diagnostic as a paper comparison method. Traditional FEM outputs and
GAICN must use the shared barycentric mesh-to-voxel mapper before evaluation. The
available per-sample FEM assets already store a barycentric `[190, 200, 104]`
interpolation. Reuse their fixed graph, mapping, measurement, and initialization assets
to accelerate GAICN training, but do not expose their upstream method identity in paper
tables or figures.

Keep deep-baseline training serial: run only one trainer at a time. Use a 200-train /
50-val short gate before a formal run when adapting a baseline. A deep method below
0.4 Dice remains available for supplementary metrics and figures, but mark it as below
the main-table threshold. The recovered paper-like UHR checkpoint remains the valid UHR
comparison at about 0.525 test300 Dice. PAH2T-Former is about 0.518. Two-stage DeepFMT
axis and loss fixes improve test300 Dice only to about 0.277, so retain it as a marked
supplementary comparison unless a later adaptation crosses 0.4.
The full-data lightweight FEM2Vox residual-prior adaptation reaches about 0.592 test300
Dice.
For GAICN, keep training loss in mesh-node space when `gt_nodes.npy` is available, then
interpolate to the common voxel grid for validation, test metrics, and figures. Cache
the measurement backprojection once per batch and use two correction phases with the
cached FEM initialization. The full-data run reaches about 0.559 val Dice and about
0.559 test300 voxel Dice.

For paper figures, do not display internal FEM-prior diagnostics. Use the fixed-view
mouse-internal renderer in `scripts/render_tmi_hard_cases.py`. Keep a main figure for
valid deep comparisons, a marked supplementary deep-baseline figure for methods below
0.4 Dice, and a separate traditional-FEM supplementary figure. Hard-case selection must
require complete GISC-FMT component recovery and should include two-focus, three-focus,
and irregular-shape examples.

Template-prior residual adaptation does not rescue the PGDPNN/FMT-ReconNet family on
the current v2 data. The PGDPNN 200-train / 50-val gate remains below 0.06 val Dice,
while the FMT-ReconNet smoke remains near the existing low baseline. Retain their
test300 data and marked supplementary figures; do not promote them to the main table.

## E15 Center-Distance Separation

Use E15 for multi-source separation work. The goal is to improve three-focus and mixed-shape cases without changing the E13 main path.

- Keep the main query-density head unchanged.
- Add only lightweight auxiliary query heads for center heatmap and distance-to-boundary supervision.
- Generate auxiliary targets from `gt_voxels` only during training; do not feed them into inference or sampling.
- Keep the non-GT sampler unchanged and avoid GT ROI leakage.
- Preferred starting config should inherit from the E13 MPB chain and use `model.aux_heads.*` plus `loss.center_*` settings.
- Relevant eval slices are full-volume test300, grouped by `num_foci` and shape class, plus component-level recall / missed / merge reporting.
- Focus metrics: all Dice / Precision / Recall, foci=3 Recall, component recall@3, missed/sample@3, merge/sample@3, mixed-two-shape Dice, and mixed-three-shape Dice.
- Do not reintroduce experimental-number naming into code-level symbols; name helpers by task semantics only.

## Current Comparison And Training Entry Points

- E15 training configs: `configs/exp/fmt_simgen_v2_e15_center.yaml` and `configs/exp/fmt_simgen_v2_e15_center_distance.yaml`.
- E15 uses the E13 MPB chain as its base and keeps the main query-density head intact.
- The auxiliary heads are only supervision helpers; they do not alter non-GT sampling or inference inputs.
- The completed E15 center-distance checkpoint `epoch=52-val_dice=0.7414.ckpt`
  reaches about 0.725 test300 Dice. Use it for the active paper summary and figures.
- Use the shared entrypoint for formal runs: `uv run python train.py fit model=gisc_fmt exp=fmt_simgen_v2_e15_center_distance data.dataset_type=fmt_simgen`.
- For comparison work, keep the paper baselines on the shared v2 protocol and record the selected checkpoint plus grouped metrics.
- The most relevant comparison slices remain full-volume test300, grouped by `num_foci`, shape class, and depth tier, with component recall / missed / merge reporting.
