# Research design

[Documentation](../README.md) · [Related work](related-work.md) · [Artifacts](../reference/artifacts.md)

## Question and scope

Can an interpretability method recover a known data-generating process from a trained
transformer, and does internal access help relative to output-only system identification?
The first executable task is a mixture of four HMMs. The target is a minimal equivalent
HMM, with parameters compared up to component/state permutation in the identifiable
family below. The number of components and minimal states is disclosed. Unknown model
order, arbitrary program synthesis, static weights-only
extraction, and causal circuit identification are later research tasks.

There are two useful claims to separate: recovering an equivalent predictive process,
and recovering the particular process parameters modulo relabeling. The latter needs
identifiability assumptions. A method can predict well while parameter recovery fails.
A successful probe is also not evidence that its decoded variable is causally used.

## Generating process and exact oracle

Choose one component C ~ w for an entire sequence. Draw z_0 ~ pi_C, emit
x_t ~ E_C[z_t, :], then transition z_(t+1) ~ A_C[z_t, :]. Components share a token
vocabulary; the transformer receives neither C nor z_t.

Let b_t(c,s) = P(C=c, z_t=s | x_<t). Then:

```text
b_0(c,s)      = w_c pi_c(s)
p_t(v)        = sum_(c,s) b_t(c,s) E_c(s,v)
posterior     = b_t(c,s) E_c(s,x_t) / p_t(x_t)
b_(t+1)(c,j)  = sum_s posterior(c,s) A_c(s,j)
```

The oracle normalizes every step in float64. Its output has L+1 positions for a history
of length L; training/scoring uses the first L. Tests compare this recursion with
independent enumeration over all hidden paths for small examples.

Each random component has a self-loop and a directed cycle, plus optional edges with
configurable retention probability (default 0.5). Mixing its transition matrix with 0.55 times the identity makes it strictly
diagonally dominant and full rank. Its initial state is the stationary distribution.
Emissions are Dirichlet(0.4) draws with a 2% uniform floor. Mixture weights are uniform.

The default uses 12 joint states and 16 symbols. Runs require at least as many symbols
as joint states for the sufficient rank screen below. Full row rank of the stacked
emission matrix, positive initial joint mass, stationarity error, and transition singular
values are reported. Treating component/state pairs as one block-diagonal HMM connects
this family to [population identifiability up to state relabeling](https://arxiv.org/abs/1306.4657).
This is a mathematical design argument, not a finite-data success guarantee.

The sampler also requires the numerical rank of H[u,v] = P(uv), using empty/one-symbol
prefixes and suffixes, to equal N = components × states. Every equivalent weighted
automaton, hence every HMM, needs at least rank(H) states. Our N-state construction
provides the matching upper bound when the rank saturates; see the
[Hankel rank argument](https://arxiv.org/html/2009.01217v1). This is a sufficient
minimality witness, computed in float64 with tolerance 1e-10, not an exact certificate.
An unsaturated short rank is inconclusive, so the sampler rejects it as
`minimality_unresolved`. The unrestricted HMM API still supports redundant controls.

Minimality alone does not imply uniqueness up to permutation for general probabilistic
or weighted automata: equivalent minimal weighted representations can differ by a
change of basis. Permutation scoring here relies on the HMM identifiability assumptions.

## Sampling informative tasks

[PAutomaC](https://proceedings.mlr.press/v21/verwer12a.html) varied graph and symbol
sparsity, drew weights from Dirichlet distributions, and used a 3-gram baseline when
selecting tasks with varied difficulty. This motivates explicit structural controls
and a pre-training screen here. Our stationary, fixed-length mixture family and exact
entropy thresholds are new prototype choices, not a reproduction of its generator.

For each component c, enumerate all token histories of length d = 0, 1, 2, 3, marginalize
hidden paths, and compute H_d(c) = H(X_t | X_(t-d:t), C=c), in nats/token. Stationarity
makes H_0 - H_3 the best unigram's excess cross entropy over the three-token-history
oracle. H_2 - H_3 is the corresponding excess for the best trigram predictor, whose
history has two tokens. These are lower bounds on the benefit available to a longer
history. They do not depend on a fitted n-gram, Monte Carlo noise, or test sequences.

The default `[sampling]` table requires **every component** to have:

- Minimum stationary state mass at least 0.05, avoiding rarely visited nominal states.
- H_0 - H_3 at least 0.02 nats/token, excluding nearly IID components.
- H_2 - H_3 at least 0.001 nats/token, requiring some benefit beyond two-token memory.

Conditioning on component identity prevents a mixture of IID distributions from passing
merely because the history reveals its component. A fully observed first-order Markov
chain also fails the third check. Tests cover both controls and compare the entropy
calculation against independent hidden-path enumeration.

The first acceptable candidate is selected, with at most 128 attempts. The first uses
`seed`; subsequent seeds come from `SeedSequence(seed, spawn_key=(attempt_index,))`.
Every candidate's seed, metrics, and rejection reasons are flushed to evaluator-only
`private/sampling.jsonl`, including on exhaustion. No training occurs after exhaustion.
The audit includes the observable rank, joint state count, and numerical rank tolerance.
The exact enumeration is bounded to four million component/token strings.

These are declared development filters, not a guarantee of difficult extraction. In
particular, a positive trigram gap does not prove a trigram fails the separate 0.03
checkpoint criterion. Broader studies should freeze difficulty strata and report
acceptance rates, entropy profiles, actual edge density, and all checkpoint failures.
The thresholds, sparsity, and Dirichlet concentration must not be tuned to obtain a
desired extraction result. Setting gain thresholds to zero permits explicit IID controls.

## Training and checkpoint qualification

A two-layer, width-48 causal transformer trains on fresh samples with standard next-token
cross entropy. No oracle probabilities or latent labels are used as training targets.
There is no dropout. AdamW uses learning rate 0.003, weight decay 0.01, and gradient-norm
clipping at 1.0. The [demo config](../../configs/demo.toml) bounds training at 2,000 steps.

For each independent evaluation sequence, average KL(p_oracle || p_model) across
positions, in nats/token. Report the mean, sequence-level standard error, and mean +
1.96 SE. The stopping validation set is fixed; a distinct test set is evaluated once
after training. The default tolerance is 0.03 nats/token. Both the aggregate and **every
component group** must have an upper bound at or below tolerance on validation and test.
A group with fewer than two sequences cannot qualify.

This normal approximation is a diagnostic uncertainty estimate, not a simultaneous
finite-sample guarantee. Repeated validation is selection, not independent testing.
Positions within a sequence are correlated and are not counted as independent samples.
Passing establishes only the declared average-context criterion at the tested lengths;
rare histories and longer contexts can still fail. Undertrained runs retain all artifacts
and recovery scores but are marked `diagnostic_only_unqualified_checkpoint`.

Generator randomness is private. Model initialization, model sampling, and extraction
initialization have independent config seeds: the values exposed to an extractor must
not be reversible offsets of the generating seed. The public demo specifies all four;
for a hidden task change the generating seed independently. Evaluator-only data streams
use the offsets below:

| Use | Seed |
|---|---|
| Generator candidate stream | `seed`, then spawned candidate seeds; all evaluator-only |
| Transformer initialization | `model_seed` |
| Training samples | `seed + 10` |
| Stopping validation | `seed + 11` |
| Final test | `seed + 12` |
| Model-generated extraction histories | `observation_seed` |
| Extraction initializations | `recovery_seed + restart index` |
| Supervised probe fit / test | `seed + 40` / `seed + 41` |

## Recovery methods and information budgets

All three methods optimize the same known-shape, differentiable HMM family. Softmax
parameters keep probability rows valid. Adam uses learning rate 0.05. Default runs use
two random restarts and 250 steps per restart. Each method selects its best final objective,
without looking at true parameters or evaluator scores.

- **Samples:** minimize negative log likelihood of transformer-generated sequences.
- **Probabilities:** minimize cross entropy against the transformer's next-token
  probabilities on those exact prefixes. This is KL distillation up to constant entropy.
- **Activations:** add 0.1 times MSE between standardized residual activations and a
  learned linear map of the candidate HMM's joint beliefs. Standardization is per feature
  over the extraction set. The map includes an intercept. No true beliefs or labels are supplied.

The third objective is an experimental activation-assisted baseline, motivated by
belief-geometry work. Its linear-decoding assumption can be wrong, and fitting it can
hurt recovery. It does not trace a circuit or read HMM matrices directly out of weights.

Default observations comprise 128 model-generated sequences of length 24: 3,072 token
queries and 24 batched forward calls per method, with 38,400 prefix tokens processed.
The runner collects this set once and reuses it. These are equivalent per-method budgets,
not three independently incurred query streams. Batched calls and token queries are
reported separately; no KV cache is used. Methods differ in information bandwidth and
the activation method has extra fitted parameters and compute. Equal queries are not
equal total cost; fitting time and parameter counts are recorded.

Unlimited output access can identify suitable HMM families. Internal access is not
assumed to contain an impossible information advantage. Useful follow-up comparisons
include fewer queries, shorter histories, optimizer/sample efficiency, and richer
processes, always with a strong output-only baseline.

## Scoring

Parameter scoring enumerates state permutations within every candidate/reference
component pair, then component permutations. It minimizes the mean of four equally
weighted MAEs: mixture weights, initial distributions, transitions, and emissions.
Each matrix entry has equal weight within its field. The report retains all four errors
and the chosen permutations. Parameter errors require matching counts; the baseline
extractors do not search over model order.

Predictive comparison remains meaningful across different state/component counts.
The standalone scorer accepts such candidates and returns `parameters: null`; it still
reports predictive KL, numerical equivalence, and both complexity profiles.

`equivalence.py` views an HMM as a weighted automaton with symbol operators
M_v = blockdiag(diag(E_c[:, v]) A_c). A reachable-space search compares all finite-word
probabilities in exact arithmetic, pruning linearly dependent rows. The implementation
uses float64, two orthogonalization passes, and tolerance 1e-10. A detected difference
includes a token-string witness and both probabilities. `numerically_equivalent` is
a numerical diagnostic, not an exact certificate or a uniform approximate-distance bound.
Ill-conditioned or rare differences can be missed; predictive KL remains a separate score.

`split_states(hmm, copies=2)` creates equivalent controls by copying emission rows and
dividing incoming transition and initial mass among copies. This changes nominal state
count without changing the output process. `short_hankel_rank` measures the numerical
rank of P(prefix + suffix) for empty/one-token prefixes and suffixes; it is an observable
linear-complexity lower bound, not a general solver for minimum HMM order. Reports also
include raw state counts, edge counts, dense-family parameter degrees of freedom, and
effective initial joint-state count. See the [scaling protocol](scaling.md).

Transition-support F1 treats a true positive edge as probability > 0 and a predicted
edge as probability >= 0.05 after alignment. This arbitrary, declared threshold is an
auxiliary diagnostic, not an exact sparsity claim; small true edges can be missed.
Predictive KL is also evaluated on separate ground-truth histories, using the same
mixture oracle and sequence-level uncertainty summaries as the transformer check.

A regression test constructs two HMMs with identical emission rows and different
transitions: their outputs coincide while parameter errors remain positive. This
negative control prevents equating observational equivalence with exact recovery.

Supervised ridge probes predict true joint beliefs from either model probabilities
or residual activations. Fit and test sequences are independent. Ridge strength is
fixed at 0.001; R-squared is aggregate variance explained across all belief coordinates.
These probes see evaluator-only labels and are never submitted as extraction results.
When emissions have full row rank, exact next-token probabilities already determine
beliefs linearly given the emissions. This makes the probability probe an essential control.

## Before a research claim

The prototype is a single-task development scaffold, not a validated benchmark suite.
Freeze hyperparameters on development generators, use unseen evaluator-only generator
seeds and multiple model seeds, and compare paired budgets across a query sweep. Report
checkpoint failure rates, poor recovery, and runtime as well as successes. Add random
and shuffled-activation controls and targeted causal interventions before claiming the
activation method found the mechanism. Do not tune against final-test or hidden recovery scores.

For finite automata or program generators, first specify what counts as an equivalent
solution, an exact oracle where possible, an observable access protocol, and appropriate
held-out/counterfactual tests. Reuse the experimental discipline, not unsupported HMM claims.
