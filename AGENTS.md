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
- `bash scripts/reassemble_pretrained.sh`: rebuild the provided split checkpoint before evaluation.
- `uv run python train.py test model=gisc_fmt ckpt_path=pretrained/gisc_fmt_brain1000_best.ckpt`: evaluate a checkpoint.

## Coding Style & Naming Conventions

Use Python 3.12-compatible code. Keep formatting Black-compatible and respect the Ruff line length of 100 characters. Prefer explicit configuration keys over hidden code defaults, matching the Hydra/OmegaConf style. Use `snake_case` for functions, variables, config keys, and modules; use `PascalCase` for model or module classes. Keep Bash scripts guarded with `set -euo pipefail`.

## Testing Guidelines

Tests use `pytest` and should be placed in `tests/` with names matching `test_*.py`. Add focused tests for model branches, tensor shapes, config behavior, and backward compatibility. Keep tensors small enough for CPU execution unless GPU behavior is required. Run `uv run pytest` before submitting changes.

## Commit & Pull Request Guidelines

The current history uses concise imperative commit messages, for example `Add split pretrained checkpoint` and `Initial public release (split pretrained)`. Follow that style: describe the concrete change, not the process. Pull requests should include a summary, commands run, dataset/checkpoint assumptions, and metric or behavior changes. Link issues when applicable.

## Security & Configuration Tips

Do not commit private datasets, generated `outputs/`, API keys, or unsplit large checkpoints unless intentionally part of a release. Prefer Hydra command-line overrides for local paths, such as `data.train_dir=/abs/path/to/train`, instead of hard-coding machine-specific paths in tracked configs.
