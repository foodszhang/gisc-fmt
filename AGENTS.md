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

Latest test300 result for `gisc_fmt` selected `epoch=44-val_dice=0.6930.ckpt` by candidate Dice. Test Dice is about 0.662, IoU about 0.512, ASSD about 0.579, and HD95 about 2.123 at threshold 0.5. The main GISC-FMT weakness is multi-source recovery, not depth. By `num_foci`, Dice is about 0.761 for one focus, 0.673 for two foci, and 0.579 for three foci. Recall drops from about 0.866 to 0.558 from one to three foci, while precision drops less, indicating missed or incomplete secondary foci rather than only false positives. Depth is secondary: deep, medium, and shallow Dice are about 0.674, 0.665, and 0.647.

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
