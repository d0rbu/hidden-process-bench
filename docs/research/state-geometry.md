# Learning state coordinates and update rules

[Primary protocol](independent-discovery.md) · [Earlier operator method](observable-operators.md)

This restricted method first learns a linear predictive-state process, then checks
whether its operators admit a stochastic latent Markov realization. It receives no
process family or order, and uses no HMM optimization adapter. The coordinate and
recognition priors are built into the method and must be disclosed in comparisons.
The [development results](../results/discovery-development.md) include ten paired
output/activation runs. None demonstrated structure recovery.

## Fit an autonomous process

The inputs are cached tokens, neural probabilities and a chosen public feature array.
Output-only features are the probability vectors. Internal-access features can be
cached activations. Neither includes latent-state labels. A fixed split holds out
entire sequences. PCA and whitening fit only training sequences during selection.

For candidate dimension d, use coordinates s(h) = [1; whitened features(h)]. Learn
one linear operator B_x for each observed token using the relation

    B_x s(h) ≈ q(x | h) s(hx).

Token-conditioned ridge regression fits this relation on adjacent observations.
A separate linear readout fits the neural next-token probabilities. Its coefficients
sum to the constant coordinate, and become the first rows of the operators. An unseen
token has no fitted nonconstant update. Inference starts from the average empty-history
state and runs autonomously from tokens, with no features or neural calls at test time.

Validation scores the entire autonomous rollout against neural probabilities.
Target features at validation positions are not used to predict those positions.
Search dimensions from one through the numerically supported feature rank plus one,
capped by vocabulary size, and ridge strengths 1e-6, 1e-3 and 0.1. Choose the smallest
dimension within one sequence-level standard error of the best validation error,
then the best ridge at that dimension; refit all permitted observations.

The standalone predictor uses the earlier operator executable's explicit probability
floor and state-reset rules. Those repairs can break the exact linear realization;
the count of repairs belongs in the result. No original transformer is bundled.

## Recognize a compatible latent process

An invertible node-emitting latent Markov process has column-state operators

    B_x = F Tᵀ diag(E[:, x]) F⁻¹.

Let B = sum_x B_x. If B is invertible, then D_x = B⁻¹ B_x share a real eigenbasis.
A fixed generic linear combination of the D_x supplies a candidate joint basis.
Normalize each basis column using the linear state's normalizer, then transform
the initial state, total transition and emission factors into that basis.

Accept numerical compatibility only if all transformed parameters are nonnegative
and stochastic and reconstruct every original operator to the predeclared 1e-7
tolerance. Reject total-operator or basis condition numbers above 1e8. A one-dimensional
positive realization is memoryless. Failed diagonalization, positivity or normalization
leaves the family unresolved; it never triggers a forced HMM fit.

The test recognizes a particular algebraic family. It cannot establish arbitrary
program discovery. Singular transitions, repeated emission spectra, noisy coordinates,
nonlinear predictive features, vocabulary-limited dimension and finite observation
coverage can prevent recognition of a true HMM. Compatibility is a numerical diagnostic,
not exact equivalence, minimality, identifiable latent semantics or causal recovery.
Actual prediction and structure errors still require the private evaluator.

Tests check latent realization recovery under arbitrary invertible coordinate changes,
a noncommuting edge-emitting counterexample, singular/complex/invalid realizations,
autonomous predictions from output features, missing tokens and isolated teacher-free
execution. No trained-teacher result follows from these fixtures.

## Package a method

```python
from pathlib import Path
from hidden_process_bench.state_geometry import prepare_bundle

prepare_bundle(Path("outputs/geometry-method"), features="outputs")
```

Use `features="activations"` for the internal-access variant. Execute the bundle using
the neutral runtime's `extract` command. Freeze the whole cohort before private scoring;
retain failed regressions and all searched dimensions. Paired variants must use the
same histories and probabilities and disclose their different information access.
