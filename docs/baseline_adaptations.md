# Baseline Adaptations

All adapted baselines use the existing Hydra + PyTorch Lightning entrypoint:

```bash
uv run python train.py fit model=<name> data.dataset_type=fmt_simgen
uv run python train.py test model=<name> ckpt_path=<path_to_ckpt> data.dataset_type=fmt_simgen
```

Voxel-domain models return `{"pred_voxel": logits, "aux_outputs": ...}` and are
supervised against `gt_voxels.npy`. Model internals should not apply sigmoid to final
logits; metrics apply `torch.sigmoid(pred_voxel)`.

## Stage 1 FEM Assets

DU2Vox Stage 1 uses shared FEM assets such as `mesh.npz`, `system_matrix.A.npz`,
and graph Laplacian files. The default inspected path is:

```text
/home/foods/pro/FMT-SimGen/output/shared_mesh_20k
```

This path may come from the older uniform1000/20k DU2Vox setup, while the current
main comparison dataset is FMT-SimGen v2 3k/20k. Therefore FEM baselines are gated by
diagnostics and shape checks instead of assuming asset compatibility. Run:

```bash
uv run python scripts/check_fem_stage1_assets.py exp=fmt_simgen_v2_3k_20k_common data.dataset_type=fmt_simgen
```

Traditional FEM baselines are implemented only when FMT-SimGen provides the required
system matrix and measurement vector. GAICN-adapted reuses DU2Vox Stage 1 FEM system
matrix and mesh graph where available. If those assets are missing or node counts do
not match, the corresponding model should fail instead of fabricating FEM inputs.

The current mesh-to-voxel path defaults to a cached barycentric FEM interpolation map.
The deterministic nearest-node mapper remains only as an explicit diagnostic fallback.

## Method Summary

| Method | Original input/output | FMT-SimGen adaptation | Stage 1 FEM | A matrix | Mesh graph | Domain | Core mechanisms kept | Interface changes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `vox_dmrn` | Projection to voxel/source volume | Existing residual 2D encoder and voxel head | No | No | No | Voxel | Residual projection encoder, voxel output | Runs through shared voxel loss |
| `uhr_deepfmt` | Multi-view FMT projection to 3D volume | Existing multi-view 3D U-Net-style baseline | No | No | No | Voxel | Multi-view stacking, 3D decoder, SE skip fusion | Uses FMT-SimGen projections and full voxel GT |
| `two_stage_deepfmt` | Blurred projections to restored projections to slices | `RestorationNet` plus slice-wise `IRadonNet` | No | No | No | Voxel | Projection restoration, per-z profile reconstruction | Projection loss disabled when restored projection targets are absent |
| `pgdpnn` | Target surface, template surface/source, STN, V-Net | Projection boundary shell, nearest template, affine STN, V-Net | No | No | No | Voxel/template | Radiomics-like template selection, STN deformation, V-Net refinement | Templates built with `scripts/build_pgdpnn_templates.py` |
| `pah2t_former` | 24x64x64 surface photon density to FMT reconstruction | V sparse-view projections resized to Vx64x64, interpreted as projection-depth tensor | No | No | No | Voxel transformer | IMSM intra/inter attention, SC-PAM, 4-stage U-shaped hierarchy | Default full output; ROI mode can paste ROI logits back to full volume |
| `cnn3d_baseline` | Plain 3D CNN comparator | Boundary-shell projection embedding to 3D U-Net/V-Net | No | No | No | Voxel | Local 3D convolution only | Included as PAH2T paper-style 3D-CNN comparator |
| `transunet3d_baseline` | 3D TransUNet comparator | Boundary-shell projection embedding, CNN encoder/decoder, transformer bottleneck | No | No | No | Voxel transformer | Generic bottleneck self-attention | Does not include IMSM or SC-PAM; this is the contrast with PAH2T |
| `stage1_fem` | FEM Stage 1 mesh/source estimate | Reads `stage1_mesh`/`coarse_d` or Stage 1 voxel if present | Yes | Optional | Optional | Mesh or voxel | Uses real Stage 1 output only | Fails if no real Stage 1 output is available |
| `stage1_to_voxel` | FEM node field to voxel grid | Mesh node field mapped to voxel grid | Yes | Optional | Optional | Mesh-to-voxel | Direct Stage 1 transfer | Uses cached barycentric FEM interpolation by default |
| `stage1_unet` | Coarse FEM/voxel prior refined by 3D U-Net | Stage 1 volume input to V-Net-style 3D U-Net | Yes | No | No | Voxel | Ordinary voxel refinement only | No projections, no GISC query sampling, no footprint |
| `tikhonov_fem` | `min ||Ax-y||^2 + lambda ||x||^2` | Iterative nonnegative gradient solve | Yes | Yes | No | Mesh-to-voxel | L2-regularized FEM inverse problem | Uses `measurement_b.npy`; lambda configured in YAML |
| `l1_fem` | `0.5||Ax-y||^2 + lambda ||x||_1` | ISTA with soft thresholding | Yes | Yes | No | Mesh-to-voxel | Sparse FEM inverse problem | Uses `measurement_b.npy` |
| `elasticnet_fem` | L1 + L2 FEM inverse problem | ISTA-style elastic net updates | Yes | Yes | No | Mesh-to-voxel | Sparse plus shrinkage regularization | Uses configured lambda pair |
| `fista_fem` | Accelerated proximal sparse inverse problem | FISTA updates with nonnegative clamp | Yes | Yes | No | Mesh-to-voxel | Forward-backward splitting predecessor | Uses `measurement_b.npy` |
| `stomp_fem` | Stagewise orthogonal matching pursuit sparse inverse problem | Greedy support selection on FEM `A` and `measurement_b` | Yes | Yes | No | Mesh-to-voxel | StOMP sparse reconstruction | Only valid when A/y are present |
| `gaicn` | FEM graph unrolling with system matrix and graph attention/contraction | Lightweight GAICN-like FBS unrolling over FEM mesh graph | Yes | Yes | Yes | Mesh graph | System matrix gradient step, mesh message passing, learnable contraction | No PyG dependency; errors if FEM graph assets are missing |
| `gisc_fmt` | DU2Vox/GISC query-wise reconstruction | Existing GISC-FMT model | Uses configured priors where implemented | No for baseline path | No for baseline path | Query/voxel | Projection sampling, adaptive footprint/CQR path | CQR variants kept as internal ablations, not main external baselines |

## Alignment Rules

The shared voxel loss and evaluation no longer resize prediction tensors to target
shape implicitly. Valid output modes are:

```text
full_voxel: pred_voxel.shape == gt_voxels.shape
roi_voxel: crop target explicitly or paste prediction back to full volume before loss
mesh: convert x_mesh to pred_voxel through a declared mesh_to_voxel mapper
```

Each FEM model reports `output_space` and `alignment_mode` in `aux_outputs` where
applicable. Shape mismatch now raises an error instead of silently interpolating ROI
predictions to the full target.

## PAH2T-Former Adaptation

PAH2T-Former is added as `model=pah2t_former` based on the IEEE TCI 2025 paper
"PAH2T-Former: Paired-Attention Hybrid Hierarchical Transformer for Synergistically
Enhanced FMT Reconstruction Quality and Efficiency" (DOI `10.1109/TCI.2025.3559431`).
The original input is a `24x64x64` surface photon density tensor. FMT-SimGen provides
`V` sparse-view projection images instead, so the adaptation resizes projections to
`Vx64x64` and treats `V` as the projection/depth dimension for 3D attention.

The implementation keeps the paper-specific modules:

```text
IMSM = IMSMintra(channel/spatial pooled attention) + IMSMinter(projection-axis attention)
SC-PAM = shared common Q/K, low-rank spatial attention, channel attention, paired fusion
```

This is intentionally different from `transunet3d_baseline`, which uses a conventional
CNN encoder-decoder with a generic transformer bottleneck and does not include IMSM or
SC-PAM. The default PAH2T config outputs full `[190,200,104]` logits with `base_channels=8`,
AMP-compatible operations, and gradient checkpointing. For ROI experiments, set
`data.voxel_ranges` and `model.pah2t_former.output_shape` to the ROI shape and keep
`model.pah2t_former.paste_roi_to_full=true`; ROI logits are pasted into the full volume
before loss/evaluation rather than interpolated to the full GT.

## Required Commands

Voxel/deep baselines:

```bash
uv run python train.py fit model=uhr_deepfmt exp=fmt_simgen_v2_3k_20k_common data.dataset_type=fmt_simgen
uv run python train.py fit model=vox_dmrn exp=fmt_simgen_v2_3k_20k_common data.dataset_type=fmt_simgen
uv run python train.py fit model=two_stage_deepfmt exp=fmt_simgen_v2_3k_20k_common data.dataset_type=fmt_simgen
uv run python train.py fit model=pgdpnn exp=fmt_simgen_v2_3k_20k_common data.dataset_type=fmt_simgen
uv run python train.py fit model=pah2t_former exp=fmt_simgen_v2_3k_20k_common data.dataset_type=fmt_simgen data.batch_size=1
uv run python train.py fit model=cnn3d_baseline exp=fmt_simgen_v2_3k_20k_common data.dataset_type=fmt_simgen
uv run python train.py fit model=transunet3d_baseline exp=fmt_simgen_v2_3k_20k_common data.dataset_type=fmt_simgen
uv run python train.py fit model=stage1_unet exp=fmt_simgen_v2_3k_20k_common data.dataset_type=fmt_simgen
uv run python train.py fit model=gisc_fmt exp=fmt_simgen_v2_3k_20k_common data.dataset_type=fmt_simgen
```

FEM baselines when diagnostics pass:

```bash
uv run python train.py test model=stage1_fem exp=fmt_simgen_v2_3k_20k_common data.dataset_type=fmt_simgen
uv run python train.py test model=stage1_to_voxel exp=fmt_simgen_v2_3k_20k_common data.dataset_type=fmt_simgen
uv run python train.py test model=tikhonov_fem exp=fmt_simgen_v2_3k_20k_common data.dataset_type=fmt_simgen
uv run python train.py test model=l1_fem exp=fmt_simgen_v2_3k_20k_common data.dataset_type=fmt_simgen
uv run python train.py test model=elasticnet_fem exp=fmt_simgen_v2_3k_20k_common data.dataset_type=fmt_simgen
uv run python train.py test model=fista_fem exp=fmt_simgen_v2_3k_20k_common data.dataset_type=fmt_simgen
uv run python train.py test model=stomp_fem exp=fmt_simgen_v2_3k_20k_common data.dataset_type=fmt_simgen
uv run python train.py fit model=gaicn exp=fmt_simgen_v2_3k_20k_common data.dataset_type=fmt_simgen
```

## Known Limits

`pgdpnn` requires templates built before training:

```bash
uv run python scripts/build_pgdpnn_templates.py \
  --data_dir /home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k \
  --split train \
  --out /home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k/templates/pgdpnn_templates_k20.npz \
  --k 20 \
  --roi_shape 190 200 104
```

GAN adversarial training and the large CQR variant set remain outside the external
main comparison table.
