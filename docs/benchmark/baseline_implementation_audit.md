# SSQ-FMT Baseline Implementation and Reporting Audit

This document defines the comparison protocol for the current SSQ-FMT paper.
It is part of the experiment specification, not a claim that every named method
has been reproduced exactly.

## 1. Common evaluation protocol

All methods use the same samples, seven-view surface fluorescence measurements,
physical reconstruction region, and final evaluation grid.

- Common physical region: `38.0 x 40.0 x 20.8 mm^3`.
- Common reference grid: `190 x 200 x 104`.
- A voxel network may reconstruct on its own memory-safe native grid.
- Native predictions are mapped to the reference grid by fixed trilinear
  interpolation only.
- The interpolation is applied to continuous fluorescence-density logits or
  probabilities before thresholding.
- No learned super-resolution or method-specific morphology post-processing is
  permitted in the main comparison.
- Distance-based metrics must be reported in physical units.

The native grid is an internal numerical representation. It does not change the
physical field of view. Therefore, a `64 x 64 x 32` network and SSQ-FMT may be
compared on the common reference grid without forcing the voxel network to
allocate full-resolution 3-D feature maps.

## 2. Fidelity levels

| Level | Meaning | Reporting rule |
|---|---|---|
| `official` | Official authors' implementation and configuration | Exact method name may be used. |
| `paper_faithful` | Defining modules, input formation, loss, and training schedule are reproducible from the paper | Use `our reimplementation` in Implementation Details. |
| `mechanism_preserving_adaptation` | Important method-specific modules are retained, but the input formation or some implementation details are adapted | Append `(adapted)` to the method name. |
| `architecture_proxy` | Only a generic architectural idea or a subset of mechanisms is present | Use a mechanism-based control name; do not present as an exact literature reproduction. |
| `controlled` | Deliberately designed internal control with matched data and capacity | Report as a controlled baseline, not as prior work. |

The executable registry is implemented in
`minr_fmt/benchmark/baseline_protocol.py`. A run writes
`baseline_manifest.json` into its output directory.

## 3. Method-by-method audit

### UHR-DeepFMT

**Published defining mechanisms.** The paper describes a 3-D fusion dual-sampling
convolutional architecture and squeeze-and-excitation based skip fusion for
ultra-high-resolution FMT reconstruction.

**Current implementation.** The repository retains a 3-D encoder-decoder,
dilated upsampling, and SE-based skip fusion. It now reconstructs on a `64^3`
native grid and uses a fixed mapping to the common reference grid.

**Unresolved discrepancy.** The current input constructor repeats each 2-D view
along a volume depth axis. This is not verified as the paper's exact dual-sampling
input formation.

**Status.** `UHR-DeepFMT (adapted)`. It may be included in the main table only
with this explicit label and a limitation statement. DOI:
`10.1109/TMI.2021.3071556`.

### PAH2T-Former

**Published defining mechanisms.** The method is a paired-attention hybrid
hierarchical transformer with intra/inter modulation and paired spatial-channel
attention.

**Current implementation.** The repository implements IMSM-like and SC-PAM-like
blocks and gradient checkpointing.

**Unresolved discrepancy.** Seven acquisition views are treated as a shallow
3-D depth dimension and are subsequently resized to the reconstruction volume.
This data formation has not been verified against the original implementation.

**Status.** `PAH2T-Former (adapted)`, appendix only until the input formation and
training objective are independently verified. DOI: `10.1109/TCI.2025.3559431`.

### MAP-PGAN

**Published defining mechanisms.** The published method uses a parameterized
multi-branch generator, attention prior, parameterized skip connections,
adversarial training, and gradient-penalty regularization.

**Current implementation.** The repository contains independent view branches,
attention fusion, a discriminator module, and a projection-to-volume decoder.
The expensive surface lifting has been moved to the declared internal grid.

**Missing mechanisms.** The current trainer does not implement alternating
adversarial optimization or gradient penalty; `use_gan` and `lambda_adv` remain
disabled. The paper-specific parameterized skip connections and attention-prior
objective are not reproduced.

**Status.** `MAP-PGAN-inspired adaptation`, appendix/development only. It must
not be displayed as `MAP-PGAN` in the main table. DOI: `10.1364/BOE.469505`.

### D2-RecST

**Published defining mechanisms.** The method introduces adversarial transfer in
the image domain and perceptual transfer in a feature domain.

**Current implementation.** The repository has a projection-to-volume backbone
and exposes decoder features.

**Missing mechanisms.** The perceptual loss is currently inactive and the
adversarial image-domain training is absent.

**Status.** `D2-RecST-inspired adaptation`, appendix/development only. DOI:
`10.1016/j.cmpb.2022.107293`.

### DSPGN

**Published defining mechanisms.** DSPGN embeds the FMT imaging-system prior and
uses graph convolution over system/FEM topology to preserve spatial relations.

**Current implementation.** The repository samples a small regular set of nodes,
performs one k-nearest-neighbour message-passing step, and broadcasts the mean
node feature to a voxel decoder. Surface lifting now occurs on the internal grid.

**Missing mechanisms.** The FEM mesh, system matrix, and paper-specific graph
construction are not used.

**Status.** `DSPGN-inspired adaptation`, appendix/development only. DOI:
`10.1016/j.cmpb.2025.108948`.

### FMT-ReconNet and PGDPNN

**Current implementation.** Both names instantiate the same generic pipeline:
nearest-template selection, affine 3-D spatial transformation, and V-Net
residual refinement.

**Unresolved discrepancy.** Available descriptions are insufficient to verify
that this shared pipeline reproduces either named method. The two entries are
not independent implementations. At the current reference grid, full-volume
warping and V-Net refinement are also memory intensive.

**Status.** Merge conceptually as `Template-STN reconstruction control`; appendix
only. The preflight blocks large-grid execution until a native-grid template
library and physical-space template mapping are implemented. FMT-ReconNet DOI:
`10.1109/EMBC53108.2024.10781645`.

### Two-stage DeepFMT control

**Current implementation.** A projection restoration network is followed by a
learned fully connected profile-to-slice mapping and 2-D refinement.

**Unresolved discrepancy.** The mapping is not a verified inverse-Radon operator.

**Status.** `Two-stage projection-to-volume control`, appendix only.

### Vox-DMRN

**Current implementation.** The model consumes one view and uses a fully
connected head to emit all ROI voxels.

**Limitation.** Expanding this head to `190 x 200 x 104` creates an impractical
output layer and does not match the seven-view comparison protocol.

**Status.** Single-view appendix experiment only. Full-reference execution is
blocked by preflight.

### Controlled 3-D CNN and 3-D TransUNet

These are internal capacity controls. They reconstruct on `64 x 64 x 32`, share
the same physical region as SSQ-FMT, and are mapped to the reference grid by
fixed interpolation. They are not literature-method reproductions.

**Status.** Main-table eligible as controlled baselines.

## 4. Recommended main and supplementary tables

### Main comparison

- Tikhonov-FEM.
- FISTA-L1-FEM.
- StOMP-FEM when space permits.
- Coarse FEM / deterministic FEM-to-voxel physical control.
- 3-D CNN capacity control.
- 3-D TransUNet capacity control.
- UHR-DeepFMT `(adapted)`, with explicit disclosure.
- SSQ-FMT.

### Supplementary or development-only results

- PAH2T-Former `(adapted)`.
- MAP-PGAN-inspired adaptation.
- D2-RecST-inspired adaptation.
- DSPGN-inspired adaptation.
- Template-STN reconstruction control.
- Two-stage projection-to-volume control.
- Vox-DMRN single-view adaptation.
- Graph-unrolled FEM/GAICN-like control.

## 5. Conditions for promoting an adapted method

An adapted method may be promoted only after all of the following are satisfied:

1. The paper's defining input formation is implemented.
2. Defining losses and optimization stages are active rather than instantiated
   but zero-weighted.
3. Native output coordinates and physical extent are explicit.
4. A forward/backward smoke test and a small-dataset overfit test pass.
5. The implementation is independently compared against at least one numerical
   result, parameter count, or ablation reported by the original paper.
6. The exact implementation status is disclosed in the manuscript.

Until these conditions are met, a lower result from a proxy cannot be used as
evidence that the original published method is inferior to SSQ-FMT.
