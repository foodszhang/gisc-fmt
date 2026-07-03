# Morphology–Amplitude Factorized Continuous Reconstruction

## Method question

The candidate third contribution addresses a reconstruction ambiguity rather than a
new PHSA training trick. In diffuse optical imaging, diffusion-induced blurring can
couple the spatial support of a fluorescent region with its relative density
amplitude. A scalar decoder may reduce its loss through different support–amplitude
trade-offs, including an enlarged low-density region or a contracted high-density
region.

The proposed operational factorization is

\[
\widehat{\rho}(\mathbf x)
=
\widehat p(\mathbf x)\widehat a(\mathbf x),
\]

where \(\widehat p(\mathbf x)\) is an operational source-support probability and
\(\widehat a(\mathbf x)\) is the support-conditioned relative density amplitude. The
final output remains one fluorescence density distribution. The support output is an
internal reconstruction variable, not a second physical image and not an absolute
concentration estimate.

## Checkpoint-compatible implementation

The reproduced Phase-A checkpoint reaches sampled-query validation Dice 0.7507228.
Its final density path is

`strong shared fusion -> UnifiedDensityDecoder shared path -> scalar density head`.

The original scalar head is retained as the amplitude predictor. After the historical
checkpoint is loaded, a support head is attached to the same decoder pre-activation.
Its final layer is initialized to output a probability near one, so enabling product
composition does not randomly destroy the reproduced Phase-A function.

The implementation rejects any candidate-conditioned call. It cannot run with
`ablation=full` inside the decoder or with a nonzero candidate context scale. This is
intentional: previous Phase-B/Phase-C candidate-context training was unsuccessful and
is not reused to validate this contribution.

## Trainable scope

During screening, the following modules are frozen:

- surface encoder and query-dependent surface sampler;
- proposal evidence and source-hypothesis construction;
- PHSA/candidate aggregation and candidate-conditioned adapters;
- candidate residual and candidate branch heads.

Only the reproduced shared fusion, the shared part of `UnifiedDensityDecoder`, its
original amplitude head, and the optional support head are trainable. Candidate
auxiliary losses are disabled during this continuation.

## Required screening comparisons

All continuations start from the same 0.7507 checkpoint and use the same seed, split,
query count, trainable shared modules, optimizer budget, and checkpoint-selection
metric.

1. `phase_a_reference`: untouched checkpoint;
2. `scalar_control`: equal-budget continuation without a support head;
3. `support_aux`: support supervision is present, but final density remains the
   original scalar amplitude;
4. `factorized_core`: support supervision and product composition are both active.

The third contribution is supported only if `factorized_core` exceeds both
`scalar_control` and `support_aux`. Improvement over the untouched checkpoint alone is
insufficient because it may result from continued optimization. Improvement over
`support_aux` is required to attribute the gain to the factorized reconstruction
rather than auxiliary morphology supervision.

## Evaluation hierarchy

The first screening uses validation data only. Sampled-query Dice is used for training
monitoring, but the decision is based on full-volume reconstruction and component-level
metrics:

- Dice, precision, recall, IoU, NRMSE, volume error, ASSD, and HD95;
- component recall and precision;
- small-component recall;
- matched-component IoU;
- merge and split counts;
- results stratified by one-, two-, and three-source samples.

If the single-seed screening is positive, rerun the three continuation variants with
three seeds and only then evaluate the selected configuration on the held-out test
set. Threshold sensitivity and component-balanced losses are follow-up experiments,
not part of the first screening.

## Decision rule

- Retain as a candidate third contribution only if the product factorization improves
  full-volume or component recovery beyond both controls without materially degrading
  NRMSE or false-positive volume.
- Treat it as auxiliary supervision if `support_aux` and `factorized_core` are
  statistically indistinguishable.
- Reject it if gains appear only in sampled-query Dice or only relative to the
  untouched checkpoint.
