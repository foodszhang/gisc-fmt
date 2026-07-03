# Phase-A Morphology–Amplitude Factorization Experiment

This experiment evaluates whether explicitly separating operational source support
from within-support relative density improves continuous FMT reconstruction without
introducing a second density field or any FEM/forward-model input.

The approximately 0.75 Phase-A checkpoint was produced before the later
`unified_density_decoder` refactor. Its active reconstruction path is retained
exactly as:

`complementary_aggregation -> shared_density_logit_decoder`.

The historical scalar density head is used as the amplitude head. A separate support
decoder with the same input contract is attached only after the checkpoint has been
loaded, and its final layer is initialized to output a value near one. No parameter is
mapped into the newer unified decoder, and the unified decoder is inactive throughout
this experiment. The factorized model therefore begins from an almost identical
reconstruction function:

\[
\widehat{\rho}(\mathbf x)=\widehat p(\mathbf x)\widehat a(\mathbf x).
\]

Four matched continuations are run from the same checkpoint:

1. `scalar_control`: ordinary scalar continuation on the historical decoder;
2. `support_aux`: support supervision, but density remains the scalar amplitude;
3. `factorized_core`: product composition plus support and amplitude losses;
4. `factorized_component`: factorized core plus component-balanced density error.

The untouched checkpoint is also evaluated as `phase_a_reference`. The comparison
separates gains caused by continued optimization, extra supervision, and the actual
product factorization. All variants use the same split, seed, query count, optimizer
budget, checkpoint selection metric, and full-volume evaluator.

The support target is an operational target defined by a fixed normalized-density
threshold. It is not claimed to be a unique biological boundary. The final output
remains one fluorescence density distribution.
