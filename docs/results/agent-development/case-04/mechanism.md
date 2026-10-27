# Persistent predictive regimes with token evidence

## Claim and prior knowledge
The mechanism is a six-regime predictive-state filter with a sparse belief-transfer graph and a small local readout correction. Its state is a probability vector over six regimes, with five continuous degrees of freedom. It is an independently executable autoregressive stochastic process, not a copy of the neural network.

My prior knowledge was generic: categorical Markov models, Bayesian filtering, nonnegative mixtures, and discounted token counts. No generating family, state count, token meanings, or hidden generator parameters were assumed. Only the supplied instructions, metadata, and cached token/probability arrays were used. No neural weights were available or used.

## Exact executable rules
All coefficients are in parameters.json. Let b be a row-vector belief, A a row-stochastic transition matrix, L a positive token-evidence table, and E six row-stochastic predictive profiles.

1. Retain the last min(history length, 23) tokens. Initialize b to initial_belief independently for each history.
2. For each retained token x, transfer belief r = b A, then incorporate evidence:

       b[j] = r[j] L[j,x] / sum_i r[i] L[i,x].

3. For a nonempty retained history of length n and last token x, compute q[v] = sum_j b[j] E[j,v]. With C = output_calibration, return

       p = softmax(log(q) + C[x,:] + C[16,:]
                   + (log(1+n)/3) C[17,:]).

4. For empty history, return empty_history_probabilities: the common observed initial distribution, normalized to remove floating-point summation error. This initial readout is separate from the initial belief used for evidence updates.

The columns of L are normalized across regimes for convenience. Multiplying a whole column by any positive constant leaves all updates unchanged. L is not assumed to be a token-emission distribution. E describes predictions after token evidence, not necessarily the likelihood of the token just observed.

These rules define an observation-driven generative process: sample a first token from the initial readout, update the belief using that token, sample the next token from the readout, and repeat. Applying the history window defines a bounded-memory extension to arbitrarily long histories. The account does not assert that b is an exact posterior of a conventional hidden Markov chain. A literal HMM was tested and performed less well; separating recognition evidence from predictive profiles is intentional.

## Structure
The final graph has 21 nonzero entries out of 36, including six self-links. Its approximate matrix is:

       0       1       2       3       4       5
  0  .89604  .08354  .02042  0       0       0
  1  .27097  .32618  .39340  .00468  0       .00476
  2  .15085  .15071  .69343  .00500  0       0
  3  0       0       0       .68803  .29637  .01560
  4  0       0       0       .33563  .64838  .01599
  5  0       0       .00310  .04312  0       .95378

The graph is almost separated into groups {0,1,2} and {3,4,5}. Regimes 0 and 5 are particularly persistent. Salient predictive-profile masses are:

* Regime 0: token 7 (.419), 14 (.204), 13 (.105).
* Regime 1: broad profile, notably 15 (.159), 7 (.136), 8 (.133), 10 (.103).
* Regime 2: tokens 6 (.194), 12 (.179), 9 (.084).
* Regime 3: tokens 4 (.278), 12 (.154), 1 (.116), 9 (.108).
* Regime 4: tokens 1 (.342), 5 (.123), 12 (.092), 4 (.088).
* Regime 5: tokens 14 (.298), 13 (.279), 9 (.166).

These are overlapping predictive profiles, not hard token classes. Evidence shifts relative regime masses while the graph carries information across positions. Histories ending in the same token can therefore yield different predictions. Rare links permit transfers between broad groups, unlike a permanently fixed sequence-level topic. Regime names and permutations are arbitrary. A describes belief transfer; interpreting its diagonal as a literal hidden-state dwell-time law would exceed the evidence.

The readout correction preserves this main structure. Its largest absolute coefficient is about .206 and mean absolute coefficient about .054. Last-token and smooth-length terms account for small reproducible neural discrepancies. The clock is an empirical adjustment, not evidence that the underlying generator fundamentally depends on absolute time.

## Evidence and complexity selection
The observations contain 128 sequences of 24 tokens and 3,072 complete 16-way prediction vectors. Position j predicts from the preceding j tokens. Entire sequences, not adjacent positions, were held out. Split A fitted rows 0..95 and validated 96..127. Split B fitted 32..127 and validated 0..31. These are model-selection validations, not a pristine final test. Detailed results are in evidence.json.

Centered output singular values begin 10.240, 6.401, 5.116, 3.513, 1.642, then .527, .395, .320. Five dominant variation directions suggest approximately six convex predictive regimes, but the residual is nonzero. Nonnegative mixtures with weights inferred separately from held-out target distributions give KL .00750 with five profiles and .00199 with six; seven/eight give .00159/.00109. This is geometry evidence only: those weights use target probabilities and are not executable predictions from histories.

Runnable validation measures mean KL(neural distribution || independent mechanism), in nats per position, over all 24 positions. On split A:

* Marginal baseline: .33451; last-token table: .15719; two-token table: .11883.
* Conventional HMM, 4/5/6 states: .07247/.01967/.01428. Seven/eight did not consistently improve.
* Six-regime filter with evidence before transition: .01231.
* Transition-then-evidence filters with 5/6/7/8 regimes: .01738/.01187/.01186/.01426.
* Pruning the six-regime graph to 21 edges: .01187; pruning to 17: .01275.
* Regularized local readout calibration on the dense six-regime filter improves .01187 to .01097.

On split B, transition-then-evidence filters with 5/6/7 regimes score .01831/.01160/.01129. Calibration improves the six-regime result to .01076. Six regimes were selected for parsimony: removing one costs substantially, whereas adding a seventh offers only small gains. This does not establish an exact six-state generator. Graph pruning and readout calibration were tested as separate ablations; the final combined refit has no new untouched validation split.

Distinct alternatives were tested. A six-state token-labelled, edge-emitting model validated at .01355. Static mixtures of token Markov chains fit training data but generalized poorly (six components: .06032). Multiscale count-evidence filters validated around .0183-.0222. Tempered Bayesian evidence and prior-reset variants did not materially improve conventional filtering. These comparisons favor persistent token-informed regime dynamics over a short n-gram or permanently fixed topic, without uniquely identifying the source mechanism.

After selection, the sparse six-regime core and regularized correction were refitted to every supplied row. Training KL is approximately .01103 for the core and .01006 for the submitted readout. These figures are in-sample descriptions, not held-out scores. No additional teacher queries, neural executions, hidden scores, or external data were used.

## Assumptions, uncertainties, and alternatives
* Tokens are categorical; their numeric ordering has no mechanistic meaning.
* Profiles and evidence likelihoods are effective predictive coordinates, not uniquely recovered physical states or emission laws. Alternative state bases, semi-Markov processes, richer continuous filters, or a larger latent process could explain these observations.
* Small graph entries and exact zeros are weakly identified. Sparsity is a modeling choice supported by an ablation, not proof of forbidden source transitions.
* Prediction error is real, with unusually large errors on some histories. Target-dependent mixture reconstruction is better than the executable filter, suggesting imperfect dynamics, missing mechanisms, or neural approximation artifacts.
* The local correction and clock describe the supplied neural predictions; they do not establish causal properties of the original training generator.
* Only prefixes of lengths 0..23 are observed. The predictor retains 23 tokens because the reported 24-position context may include BOS. The 23-versus-24 boundary and behavior beyond observed lengths are not identified. This window keeps inference within the observed prefix-length range but has unvalidated long-history agreement.
* Seven regimes have a small validation advantage in one split. Six is the compact approximation supported by the major spectral and predictive improvements, not a known true latent count.
* Validation informed selection and is optimistic as a final generalization estimate. No private evaluation results or confidence intervals are asserted.

## Execution and resources
Run python predictor.py with one JSON object on stdin containing histories and vocab_size=16. The output contains probabilities in token order 0..15. Every row starts with a fresh belief. There is no fitting, communication between rows, persistence across requests, neural dependency, or access to challenge files. The frozen predictor reads only its bundled parameter JSON and uses the Python standard library.

There are 538 stored numerical coefficients, counting stored zeros, 288 readout-correction coefficients, and the initial readout; no observation rows are bundled. Inference costs O(min(n,23)*6^2 + 6*16) per history. Search used approximately 264,500 analytic-gradient optimization steps and 17,000 nonnegative-factorization/update iterations across candidates and splits, using NumPy and standard Python. submission.json records the supplied cached budget and zero additional teacher queries.
