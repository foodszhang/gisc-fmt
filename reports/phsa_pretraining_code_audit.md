# PHSA Pretraining Code Audit

Date: 2026-06-30  
Branch: `ssq-fmt-final-method-refactor`

## Audited method contract

The configured view-complementary model is instantiated as `SSQFMTPatch2`, not the
base `SSQFMT` aggregation implementation. For `a3_geometry_only`, the direct
candidate-specific cross-view weights are

\[
w_{v,m}=\frac{I_{v,m}(\epsilon+g_{v,m})}
{\sum_{v'}I_{v',m}(\epsilon+g_{v',m})}.
\]

Candidate support is therefore absent from the direct reliability multiplier.
However, Patch-2 explicitly encodes `[candidate support, separability]` as a
candidate descriptor before aggregation, and support also participates in
hypothesis construction/existence estimation. The valid claim is therefore
"support-free direct view reliability," not "support-free aggregation."

The geometry-only separability path has no learned measurement correction. The
rejected fixed hypothesis grid, bounded routing and continuous-applicability
variants remain disabled.

## Correctness issues fixed before the long run

1. **Training/inference hypothesis mismatch**
   - Previous training constructed source hypotheses from the current resampled
     density-query set.
   - Full-volume inference constructed hypotheses once from a fixed sample-level
     proposal set and reused them across decoder chunks.
   - The audited training entrypoint now uses a deterministic sample-level proposal
     set with the same sample-ID hash, count, seed, coordinate conversion and
     proposal-cache definition as full-volume inference.

2. **Duplicate/unused normalization work**
   - The dataset previously applied per-view max normalization before the network
     repeated the same normalization.
   - The audited run supplies raw measurements; the network owns per-view
     normalization.
   - The unused candidate percentile normalizer is replaced by a parameter-free
     pass-through in the view-complementary path.

3. **Incorrect from-scratch optimization protocol**
   - The failed direct-full run used fine-tuning learning rates for randomly
     initialized shared and decoder modules.
   - The audited 50-epoch run is split into 20-epoch Phase A, 14-epoch Phase B and
     16-epoch joint Phase C.

4. **Phase-B shared-path drift**
   - The shared common projection and shared density head are frozen in Phase B.
   - Only the candidate support/separability projection, candidate context,
     candidate normalization and candidate input projection are updated.

5. **Patch-2 optimizer-group coupling**
   - Patch-2 places the shared common projection and candidate descriptor projection
     inside the same aggregation module.
   - Joint Phase C therefore uses the same conservative learning rate for that
     whole aggregation module instead of applying the larger candidate-context rate
     to the shared projection.

6. **Ineffective epoch-wise query resampling**
   - `current_epoch` is process-local dataset state. Persistent DataLoader workers
     would retain epoch-0 dataset copies.
   - Persistent workers are disabled so workers are recreated after `set_epoch`,
     making `resample_queries_each_epoch=true` effective.

7. **Checkpoint cadence and selection**
   - All stage lengths are divisible by the validation interval, so the last epoch
     of every stage is validated and checkpointed.
   - The final checkpoint is not selected solely by proposal-biased sampled-query
     Dice. Saved top-k checkpoints and `last.ckpt` are compared on validation-only
     full volumes before the selected checkpoint is evaluated on test data.

8. **Reused baseline cache safety**
   - Reused A2-U/A3-old predictions are audited for sample IDs, proposal count,
     deterministic proposal sequence, SHA-256 and threshold before paired testing.

9. **Unused full GT transfer**
   - Full GT volumes are still loaded on CPU because query labels and component
     supervision are generated from them.
   - They are removed during collation after those targets are constructed, avoiding
     unnecessary pinning and host-to-device transfer.

## Fail-fast checks executed before training

`scripts/preflight_phsa_curriculum.py` checks:

- requested train/validation/test counts and sample-ID disjointness;
- actual `SSQFMTPatch2` instantiation;
- sample-level proposal count and seed;
- raw dataset input plus network-owned normalization;
- exact geometry-only weight invariance to support changes;
- explicit support-descriptor retention;
- Phase-B freezing and per-module learning rates;
- Phase-C optimizer coverage and per-module learning rates;
- finite Phase-A and Phase-B forward/backward passes on a small real batch;
- gradient norms for the active encoder, evidence, constructor, candidate and
  decoder modules.

The long run must not begin unless the preflight exits successfully.

## Intentional method/experiment choices, not hidden code bugs

1. **Component-level training supervision**
   - Candidate evidence and source-hypothesis construction use GT component centers
     and covariances during simulation training.
   - Query allocation remains non-GT and inference uses only measurement/geometry
     inputs.
   - The manuscript must disclose this source-level training supervision.

2. **Candidate confidence threshold**
   - Training keeps proposal anchors valid to preserve gradients.
   - Evaluation applies the configured candidate confidence threshold (`0.4`).
   - This is an inference-selection policy and should be reported in implementation
     details.

3. **Detector-footprint approximation**
   - PHSA currently uses an isotropic detector scale derived from the mean 3-D
     covariance scale plus the query footprint scale; it does not project the full
     covariance ellipse.

4. **Proposal budget**
   - The formal run uses 4096 deterministic sample-level proposal points and 4096
     independently resampled density queries.
   - This is a fixed compute/coverage choice, not a mathematical guarantee.

5. **Validation checkpoint subset**
   - Full-volume checkpoint selection uses a fixed validation subset, never the test
     split. The subset size is configurable through `SELECTION_SAMPLES`.

## Items that static auditing cannot guarantee

- that PHSA improves Dice or component recovery over A2-U;
- that the current candidate auxiliary-loss weights are optimal;
- that 4096 proposal points are the best accuracy/runtime trade-off;
- that storage throughput is sufficient to keep a particular GPU fully occupied;
- that a single seed establishes statistical robustness.

These are experiment-dependent questions. They should not later be presented as
newly discovered implementation defects unless the fail-fast contract itself is
violated.
