# SSQ-FMT Baseline Implementation and Reporting Audit

This document defines the comparison protocol for the current SSQ-FMT paper.
It is an experiment specification and does not imply that every named method has
been reproduced exactly.

## 1. Common physical-space protocol

All methods use the same samples, seven-view surface fluorescence measurements,
physical reconstruction region, and final evaluation grid.

- Common physical region: `38.0 x 40.0 x 20.8 mm^3`.
- Common reference grid: `190 x 200 x 104`.
- Voxel networks may reconstruct on memory-safe native grids.
- Native predictions are mapped to the reference grid by fixed trilinear
  interpolation only.
- Continuous fluorescence density targets are trilinearly resampled to the
  native grid for training. The loss is not computed by forcing the network to
  emit the full reference grid.
- Interpolation is applied before thresholding.
- No learned super-resolution or method-specific morphology post-processing is
  permitted in the primary comparison.
- Distance-based metrics are reported in millimetres using `0.2 mm` spacing.

The native grid is an internal numerical representation rather than a different
field of view. A `64 x 64 x 32` network and SSQ-FMT can therefore be evaluated
in the same physical coordinates without allocating full-resolution 3-D feature
maps inside every baseline.

## 2. Fidelity levels

| Level | Meaning | Reporting rule |
|---|---|---|
| `official` | Official authors' implementation and configuration | Exact method name may be used. |
| `paper_faithful` | Defining modules, input formation, objective, and training schedule can be reproduced from the paper | State `our reimplementation`. |
| `mechanism_preserving_adaptation` | Important method-specific mechanisms are retained, but some inputs or details are adapted | Append `(adapted)`. |
| `architecture_proxy` | Only a generic architectural idea or a subset of defining mechanisms is present | Use a mechanism-based control name; do not claim an exact reproduction. |
| `controlled` | Deliberately designed internal control with matched data and capacity | Report as a controlled baseline. |

The executable registry is implemented in
`minr_fmt/benchmark/baseline_protocol.py`. Every run writes a
`baseline_manifest.json` recording fidelity, reporting tier, physical extent,
and native/reference grids.

## 3. Method-by-method audit

### UHR-DeepFMT

**Published defining mechanisms.** The paper describes a 3-D fusion dual-sampling
convolutional architecture and squeeze-and-excitation skip fusion.

**Current implementation.** The repository contains a 3-D encoder-decoder,
dilated upsampling, and SE-style skip weighting. It reconstructs on a `64^3`
internal grid and maps the result to the common reference grid.

**Unresolved discrepancy.** The current input constructor repeats each 2-D view
along a volume depth axis. The paper's defining dual-sampling operation cannot be
verified from this implementation or from an official public implementation.

**Status.** `UHR-DeepFMT-inspired 3D SE-UNet`, `architecture_proxy`, supplementary
or development use only. It must not be reported as an exact UHR-DeepFMT
reimplementation. DOI: `10.1109/TMI.2021.3071556`.

### PAH2T-Former

**Published defining mechanisms.** The method uses a paired-attention hybrid
hierarchical transformer with intra/inter modulation and paired spatial-channel
attention.

**Current implementation.** The repository implements IMSM-like and SC-PAM-like
blocks with gradient checkpointing.

**Unresolved discrepancy.** Seven acquisition views are treated as a shallow
3-D depth dimension and subsequently resized to reconstruction depth. This data
formation has not been independently verified against the original method.

**Status.** `PAH2T-Former (adapted)`, supplementary only until input formation and
training objective are verified. DOI: `10.1109/TCI.2025.3559431`.

### MAP-PGAN

**Published defining mechanisms.** The method uses a parameterized multi-branch
generator, attention prior, parameterized skip connections, adversarial training,
and gradient-penalty regularization.

**Current implementation.** The repository contains independent view branches,
attention fusion, a discriminator module, and a projection-to-volume decoder.
Surface lifting now occurs on the declared internal grid to avoid a full-grid
latent tensor.

**Missing mechanisms.** Alternating adversarial optimization and gradient penalty
are not implemented; `use_gan` and `lambda_adv` remain disabled. The published
parameterized skip and attention-prior objectives are not reproduced.

**Status.** `MAP-PGAN-inspired adaptation`, supplementary/development only. DOI:
`10.1364/BOE.469505`.

### D2-RecST

**Published defining mechanisms.** D2-RecST introduces adversarial transfer in the
image domain and perceptual transfer in a feature domain.

**Current implementation.** The repository has a projection-to-volume backbone
and exposes decoder features. Surface lifting now occurs on its internal grid.

**Missing mechanisms.** Perceptual-domain supervision is inactive and image-domain
adversarial training is absent.

**Status.** `D2-RecST-inspired adaptation`, supplementary/development only. DOI:
`10.1016/j.cmpb.2022.107293`.

### DSPGN

**Published defining mechanisms.** DSPGN embeds an FMT imaging-system prior and
uses graph convolution over system/FEM topology.

**Current implementation.** The repository samples a small regular set of nodes,
uses one k-nearest-neighbour message-passing step, and broadcasts a pooled graph
feature to a voxel decoder. Surface lifting occurs on the internal grid.

**Missing mechanisms.** The FEM mesh, system matrix, and paper-specific graph
construction are not used.

**Status.** `DSPGN-inspired adaptation`, supplementary/development only. DOI:
`10.1016/j.cmpb.2025.108948`.

### FMT-ReconNet and PGDPNN

**Current implementation.** Both names instantiate the same generic pipeline:
nearest-template selection, affine 3-D spatial transformation, and V-Net residual
refinement.

**Unresolved discrepancy.** Available descriptions are insufficient to verify that
this shared pipeline reproduces either named method. The entries are not
independent implementations. Full-volume warping and refinement are also unsafe
at the current reference grid.

**Status.** Merge conceptually as `Template-STN reconstruction control` and keep
in supplementary experiments. Preflight blocks large-grid execution until a
native-grid template library and explicit physical-space mapping are available.
FMT-ReconNet DOI: `10.1109/EMBC53108.2024.10781645`.

### Two-stage projection-to-volume control

The current implementation uses a projection restoration network followed by a
learned fully connected profile-to-slice mapping and 2-D refinement. The mapping
is not a verified inverse-Radon operator.

**Status.** Controlled supplementary experiment only.

### Vox-DMRN

The current implementation consumes one view and uses a fully connected head to
emit all ROI voxels. Expanding this head to `190 x 200 x 104` is impractical and
does not match the seven-view protocol.

**Status.** Single-view supplementary experiment only. Full-reference execution
is blocked by preflight.

### Controlled 3-D CNN and 3-D TransUNet

These internal capacity controls reconstruct on `64 x 64 x 32`. Their continuous
fluorescence density targets are resampled to that grid for training, and their
predictions are mapped to the reference grid only for common evaluation. They are
not literature-method reproductions.

**Status.** Main-table eligible as controlled baselines.

## 4. Recommended reporting structure

### Main comparison

- Tikhonov-FEM.
- FISTA-L1-FEM.
- StOMP-FEM when space permits.
- Coarse FEM and deterministic FEM-to-voxel physical controls.
- FEM-prior residual 3-D CNN control.
- Native-grid 3-D CNN capacity control.
- Native-grid 3-D TransUNet capacity control.
- SSQ-FMT.

### Supplementary or development-only results

- UHR-DeepFMT-inspired 3D SE-UNet.
- PAH2T-Former `(adapted)`.
- MAP-PGAN-inspired adaptation.
- D2-RecST-inspired adaptation.
- DSPGN-inspired adaptation.
- Template-STN reconstruction control.
- Two-stage projection-to-volume control.
- Vox-DMRN single-view adaptation.
- Graph-unrolled FEM/GAICN-like control.

## 5. Conditions for promoting an adapted method

A method may be promoted to `paper_faithful` only after all of the following are
satisfied:

1. The paper's defining input formation is implemented.
2. Defining losses and optimization stages are active rather than instantiated
   but zero-weighted.
3. Native output coordinates and physical extent are explicit.
4. Forward/backward smoke tests and a small-dataset overfit test pass.
5. At least one reported numerical result, parameter count, or original ablation
   is independently reproduced within an explainable tolerance.
6. The manuscript discloses the exact implementation source and adaptations.

Until these conditions are met, a lower result from a proxy cannot be used as
evidence that the original published method is inferior to SSQ-FMT.
