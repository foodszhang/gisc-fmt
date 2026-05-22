# Baseline Adaptations

All adapted baselines use the existing Hydra + PyTorch Lightning entrypoint:

```bash
uv run python train.py fit model=<name> data.dataset_type=fmt_simgen
uv run python train.py test model=<name> ckpt_path=<path_to_ckpt> data.dataset_type=fmt_simgen
```

The FMT-SimGen dataset, train/val/test split handling, projection inputs, and voxel
metrics are shared through `TrainingDataModule` and `TrainingLightningModule`.
Voxel-domain models return `{"pred_voxel": logits, "aux_outputs": ...}` and are
supervised against `gt_voxels.npy` from the same sample folder.

## Existing Baselines

- `uhr_deepfmt`: UHR-DeepFMT-adapted projection-to-volume baseline. It keeps the
  multi-view projection stacking, 3D U-Net encoder/decoder, SE skip fusion, and
  full voxel output. It does not use GISC query projection or adaptive footprints.
- `vox_dmrn`: generic voxel baseline using the existing residual 2D encoder and
  MLP voxel head. It now exposes full ROI voxel logits when `output_type=voxel`.

## Internal Mechanism Baselines

- `fem2vox_unet`: direct coarse projection-prior to 3D U-Net refinement. It uses
  a mean projection backprojection-style learned seed volume and no query features.
- `point_cqr`: GISC query path with calibrated projection sampling but no footprint
  aggregation (`ptfa.enabled=false`).
- `fixed_footprint_cqr`: GISC query path with fixed Gaussian footprint sampling.
- `depth_footprint_cqr`: GISC query path with hand-written exit-depth footprint
  scheduling and no learned correction.
- `unconstrained_adaptive_cqr`: configured as an unconstrained adaptive CQR ablation
  placeholder for the current implementation. It disables physical-prior flags in
  config and uses the fixed footprint path until an unconstrained sigma predictor is
  added without changing GISC's constrained PCFS path.

## Adapted Paper Baselines

- `two_stage_deepfmt`: projection restoration/enhancement network followed by voxel
  reconstruction. Projection restoration loss is disabled by default because current
  FMT-SimGen training samples do not provide a separate high-quality projection label.
- `fmt_reconnet`: coarse prior generation, identity-initialized 3D spatial transformer,
  and V-Net-style reconstruction.
- `pgdpnn`: prior information generation with a distribution prediction head. The
  distribution target is adapted to the current dataset as a soft target derived from
  the voxel GT when loss support is extended.
- `map_pgan`: multi-branch per-view encoders, attention fusion, and voxel generator.
  GAN training is disabled by default (`use_gan=false`, `lambda_adv=0.0`) to avoid
  adding a second optimizer path to the shared Lightning loop.
- `d2_recst`: dual-domain adapted baseline exposing multi-scale decoder features for
  feature-consistency/perceptual losses without external natural-image VGG features.
- `dspgn`: lightweight graph morphology prior baseline with fixed grid graph nodes,
  kNN message passing, and scatter-style global update before voxel decoding. It does
  not depend on PyTorch Geometric.

## Outputs

Checkpoints, resolved config, TensorBoard logs, and test outputs follow the existing
`paths.output_dir` and `paths.checkpoint_dir` configuration. Test runs write summary
metrics JSON/CSV and prediction volumes under `${paths.output_dir}/test` unless
`test.save_dir` is overridden.

Current full-grid test metrics include Dice, IoU, NRMSE, PSNR, SSIM, centroid
localization error (CLE), peak localization error (PLE), volume error, ASSD, and HD95.

## Known Limits

These are adapted implementations for sparse-view voxel reconstruction in this
repository, not full reproductions of each original paper. GAN adversarial training,
PGDPNN distribution supervision, and unconstrained learned footprint prediction are
configured/documented but intentionally conservative to keep the shared training
entrypoint stable.
