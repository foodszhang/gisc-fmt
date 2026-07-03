# Phase-A Morphology–Amplitude Factorization Experiment

This experiment evaluates whether explicitly separating operational source support
from within-support relative density improves continuous FMT reconstruction without
introducing a second density field or any FEM/forward-model input.

The scalar Phase-A density head is retained as the amplitude head. A support head is
attached after loading the Phase-A checkpoint and initialized to output a value near
one. The factorized model therefore begins from an almost identical reconstruction
function:

\[
\widehat{\rho}(\mathbf x)=\widehat p(\mathbf x)\widehat a(\mathbf x).
\]

Four matched continuations are run from the same checkpoint:

1. `scalar_control`: ordinary scalar continuation;
2. `support_aux`: support supervision, but density remains the scalar amplitude;
3. `factorized_core`: product composition plus support and amplitude losses;
4. `factorized_component`: factorized core plus component-balanced density error.

The comparison separates gains caused by extra supervision from gains caused by the
actual product factorization. All variants use the same split, seed, query count,
optimizer budget, checkpoint selection metric, and full-volume evaluator.

The support target is an operational target defined by a fixed normalized-density
threshold. It is not claimed to be a unique biological boundary. The final output
remains one fluorescence density distribution.
