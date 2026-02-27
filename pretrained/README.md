# Pretrained checkpoints

GitHub blocks files larger than 100MB in regular Git history.
To keep the checkpoint in this repository, we store it as split parts:

- `pretrained/gisc_fmt_brain1000_best.ckpt.part000`
- `pretrained/gisc_fmt_brain1000_best.ckpt.part001`
- `pretrained/gisc_fmt_brain1000_best.ckpt.part002`

## Reassemble

```bash
bash scripts/reassemble_pretrained.sh
```

This will create:

- `pretrained/gisc_fmt_brain1000_best.ckpt`

and verify the SHA256 checksum (see `pretrained/SHA256SUMS`).

## Use for evaluation

```bash
uv run python train.py test \
  model=gisc_fmt \
  ckpt_path=pretrained/gisc_fmt_brain1000_best.ckpt
```
