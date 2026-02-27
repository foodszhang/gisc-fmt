# GISC-FMT

**Geometry-Aware Implicit Reconstruction with Multi-View Scattering-Compensated Learning for NIR-II Fluorescence Molecular Tomography**

This repository provides training and evaluation code for **GISC-FMT**, a geometry-calibrated multi-view implicit framework for NIR-II FMT reconstruction.

## Abstract

Near-infrared II (NIR-II) fluorescence molecular tomography (FMT) enables deep-tissue tumor localization, yet multi-view 3D reconstruction from boundary measurements remains highly ill-posed under diffusion-dominated scattering, frequently causing boundary shifts, lesion merging, and leakage. We propose **GISC-FMT**, a geometry-calibrated multi-view implicit framework that predicts a continuous 3D tumor occupancy field via scattering-compensated learning. GISC-FMT establishes query-wise cross-view correspondence via calibrated projections, and extracts view-aligned multi-scale descriptors for each 3D query, enabling location-consistent evidence comparison in continuous space. To aggregate uncertain multi-view cues, we introduce a confidence-aware base--residual complementary completion that preserves view-specific information while suppressing unreliable observations. We further regularize training with a Monte Carlo low-scattering pseudo-label, serving as a scattering-compensation auxiliary learning signal that is removed at inference. Experiments on synthetic data and in-vivo measurements show improved overlap, boundary accuracy, and instance fidelity over two representative voxel-based learning baselines, suggesting the potential of the proposed query-wise multi-view implicit formulation for morphology-faithful NIR-II FMT under realistic scattering conditions.

## What is included

- **Dependency specification**: `pyproject.toml` (authoritative) and `requirements.txt` (minimal list)
- **Training code**: `train.py` (`fit/validate/test` via Hydra + PyTorch Lightning)
- **Evaluation code**: `train.py test` and `scripts/eval_gisc_fmt.sh`
- **(Pre-)trained model(s)**: split parts under `pretrained/` (reassemble into `pretrained/gisc_fmt_brain1000_best.ckpt`)
- **Reproducibility**: precise commands in this README + result table with commands

## Repository layout

- `train.py`: training/evaluation entrypoint
- `minr_fmt/`: core package (models / networks / datasets / utils)
- `configs/`: Hydra configuration tree
- `scripts/`: runnable training/evaluation helpers
- `pretrained/`: provided checkpoint(s)

## Installation

### Option A: uv

```bash
uv sync
```

### Option B: pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
# minimal
uv run python train.py fit model=gisc_fmt

# override paths if needed
uv run python train.py fit \
  model=gisc_fmt \
  data.train_dir=/abs/path/to/train \
  data.val_dir=/abs/path/to/val \
  data.test_dir=/abs/path/to/test \
  data.block_dir=/abs/path/to/blocks
```

Or:

```bash
bash scripts/train_gisc_fmt.sh
```

## Evaluation

### Reassemble the checkpoint

```bash
bash scripts/reassemble_pretrained.sh
```

### Evaluate with the provided checkpoint

```bash
uv run python train.py test \
  model=gisc_fmt \
  ckpt_path=pretrained/gisc_fmt_brain1000_best.ckpt
```

Or:

```bash
bash scripts/eval_gisc_fmt.sh pretrained/gisc_fmt_brain1000_best.ckpt
```

## Results

| Method | Checkpoint | Metric(s) | Command |
|---|---|---|---|
| GISC-FMT | `pretrained/gisc_fmt_brain1000_best.ckpt` | reported by `train.py test` (e.g., `val_dice`, `test_*`) | `uv run python train.py test model=gisc_fmt ckpt_path=pretrained/gisc_fmt_brain1000_best.ckpt` |

## Citation

If you find this code useful, please cite our paper:

```bibtex
@inproceedings{gisc_fmt_2026,
  title   = {Geometry-Aware Implicit Reconstruction with Multi-View Scattering-Compensated Learning for NIR-II Fluorescence Molecular Tomography},
  author  = {Anonymized Authors},
  year    = {2026}
}
```

## Notes

- Output directory: `outputs/${model.name}/${task}/...` (see `configs/paths/default.yaml`).
- Sanity check script: `uv run python scripts/quick_run_bg.py`.
