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
