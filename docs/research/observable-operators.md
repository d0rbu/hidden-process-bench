# Observable-operator identification

[Documentation](../README.md) · [Discovery protocol](independent-discovery.md) · [Measured attempt](../results/operator-development.md)

This restricted classical baseline learns a linear predictive-state program directly
from anonymous neural observations. It receives neither an HMM constructor nor a
component/state count, and uses no feature-to-HMM fitting adapter. It still assumes
a finite-dimensional homogeneous linear representation: **it does not choose among
generating families or demonstrate discovering HMM semantics**. It is a baseline for
the primary task, not a substitute for the requested family-discovery methods.

The update operators are inspired by the observable representation in
[Hsu, Kakade and Zhang](https://arxiv.org/abs/0811.4413). Their theoretical assumptions
do not automatically apply to this adaptation's overlapping windows, approximate
neural probabilities, dimension selection or numerical repairs.

## Learning and execution

The only inputs used are cached sampled tokens and their aligned next-token
probabilities. A fixed method RNG splits whole sequences into 75% training and 25%
validation, with at least two sequences on either side. No private oracle is queried.

Training estimates one-, two- and three-token moments across all available windows.
The last symbol in the pair/triple is integrated using neural next-token probabilities;
the other symbols are sampled. The SVD of the pair moment supplies observable
coordinates. Every dimension from one through the numerically supported rank is
tried; the vocabulary limits this search. The rank cutoff and pseudoinverse cutoff
are both relative 1e-7.

Select the smallest dimension within one sequence-level standard error of the best
validation teacher KL. Retain every dimension's loss, repairs and failures, then refit
the chosen dimension on the complete permitted cache. Neither the true order nor
private scoring results enter this selection.

The method submits an initial coordinate vector, a normalization row and one square
operator per token. For state `s`, the raw probability of token `a` is `r @ B[a] @ s`.
After observing `a`, update to `B[a] @ s`, normalized by its mass under `r`. The
executable floors predictions at 1e-9 and renormalizes; nonfinite/nonpositive state
mass resets to the learned initial state. Invalid raw readouts use a learned marginal
fallback. These repairs belong to the proposed executable and can break exact linear
word-probability semantics. Signed coordinates are not latent probability states.

Parameter files, code, selection logs and the mechanism description are frozen before
private evaluation. Prediction uses only the submitted files and history, with no
teacher, original observation cache or refitting. There are zero additional neural
executions. A one-dimensional positive operator system is history-independent; a
larger selected dimension alone does not identify a generator's latent order.

## Run

```bash
uv run --locked python -m hidden_process_bench.discovery_cli prepare-operators \
  --out outputs/operator-method
uv run --locked python -m hidden_process_bench.discovery_cli extract \
  --program outputs/operator-method --entry main.py \
  --challenge outputs/anonymous-challenge --out outputs/operator-attempt
```

Then use the neutral [private scoring command](../reference/discovery-runtime.md).
The published development receipt includes the exact method as `method.zip`; unpack
it to reproduce that version rather than assuming the current source has identical
bytes. The extracted file hashes must match `method_sha256` in its summary.

Tests compare exact moment predictions against independently enumerated hidden paths,
check equivalence under an invertible coordinate change, exercise numerical failures,
and run real extraction/standalone prediction in the namespace/cgroup sandbox.
These checks validate the implementation; the measured development failures determine
what it achieved under the actual observation budget.
