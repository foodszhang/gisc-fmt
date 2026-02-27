# Pretrained checkpoints

This repo includes a GISC-FMT checkpoint for convenience:

- `pretrained/gisc_fmt_brain1000_best.ckpt`

## Use for evaluation

```bash
uv run python train.py test \
  model=gisc_fmt \
  ckpt_path=pretrained/gisc_fmt_brain1000_best.ckpt
```

Notes:
- This checkpoint is provided as-is.
- You still need to place the dataset locally (see `DATASET.md`).
