# Frozen mixed-process campaign

[Documentation](../README.md) · [Discovery objective](independent-discovery.md) · [Private suite](mixed-suite.md)

This campaign tests whether methods can infer a mechanism without being told its
family or order. The [protocol](final-campaign/protocol.json) and six executable
method bundles were frozen before new tasks were generated. This public copy has
SHA-256 `688671ed0546c468d6c064aa6c90b15b33e5477129b8030d46a57291c6420b4d`.
It was published before private final scoring. The original protocol bytes and method
bundles remain unchanged. The completed [48-trial results](../results/final-campaign-v2.md)
include the registered programs, all six teachers and separate structural assessments.

## Tasks and teacher qualification

The evaluator generates six fresh tasks: two screened HMM mixtures, two delayed-token
rules and two stack processes. Their identities, parameters, random streams and
qualification records stay outside the methods' contexts and filesystems. Every
transformer has vocabulary 16, context 24, width 64, four attention heads and two
layers. Output/internal pairs receive identical cached tokens and distributions.

All six teachers in the current suite passed separate stopping-validation and test
gates, including every mixture component or stack-depth group. Validation requires
mean KL + 1.96 sequence standard errors at most 0.024 nats/token; the independent
final-test cutoff is 0.03. Qualification is approximate and limited to those sampled
histories. It is not exact Bayesian optimality or evidence of mechanism recovery.

The [earlier whole-suite attempt](../results/qualification-attempts.json) retained one
failed deep-stack test group and started no extraction. The new whole suite uses
fresh tasks and random streams and a stricter stopping-validation margin declared
before generation. No failed teacher was resumed using test feedback. Method bundles
and the entire agent configuration are unchanged between those attempts.

## Methods and information

| Variant | Inputs and approach | Fixed bundle |
|---|---|---|
| Observable operators | Outputs; moment-based linear-state identification | [Source](final-campaign/operators.zip) |
| Output geometry | Outputs; unsupervised coordinates and autonomous update regression | [Source](final-campaign/geometry.zip) |
| PCA | Cached neural residuals plus outputs; same autonomous state learner | [Source](final-campaign/pca.zip) |
| Sparse autoencoder | Learned residual features; same state learner | [Source](final-campaign/sae.zip) |
| Transcoder | Learned MLP features, with an additional neural replay | [Source](final-campaign/transcoder.zip) |
| Head ablation | Greedy neural pruning followed by the state learner | [Source](final-campaign/causal.zip) |
| Research agent, outputs | Fresh neutral context; chooses and writes its own process | Prompt and configuration in the protocol |
| Research agent, internals | Same prompt, with weights and counted neural execution | Prompt and configuration in the protocol |

There are 48 declared trials, including failures. The six programmed variants have a
restricted linear-state prior; five also test a particular latent Markov realization
analytically after learning the operators. These are disclosed method assumptions,
not unrestricted family discovery. The agent can choose an arbitrary program. No
variant receives an HMM fitter, candidate template, hidden labels or true state count.
The neutral predictor interface accepts token histories and returns probabilities.

Each programmed extraction has 30 seconds wall time, 20 seconds CPU time and 1 GB RAM.
Each agent trial has 20 actions and 30 minutes total, with at most 10 minutes per model
call; numerical actions have their own bounds. Two agent trials run concurrently.
The model is `gpt-6.1-sol` with `xhigh` reasoning. Agent prompts, binary/catalog hashes,
method files, query accounting and limits are recorded in the protocol.

The original cache has 128 sequences of length 24. Internal variants can execute the
weights; transcoder, pruning and agent execution require additional query counts.
Paired observations do not make this an equal-query or equal-compute comparison.
This experiment cannot establish that weights contain more recoverable information
than outputs in general.

## Registration, scoring and structural review

All 48 trials must terminate before any private final score is computed. Programs
are fixed by external receipt hashes. Failed, timed-out and unsubmitted trials remain
in the roster; intermediate artifacts are not rescued. The final agent calls wait
for the ten paired development trials to finish, without receiving their hypotheses
or scores. No final method adaptation is allowed.

Held-out scoring uses 128 independent sequences of length 24. Longer scoring uses
128 sequences of length 72. Report true-process-to-program, teacher-to-program and
true-process-to-teacher KL separately. The teacher is undefined beyond context 24,
so longer scoring measures extrapolation to the generating process alone. Preserve
sequence uncertainty estimates, execution errors, batch-invariance checks and
concrete diagnostic histories.

Review the actual submitted executable and mechanism after registration. Assess the
proposed family, dependencies, update rules and observable complexity. Apply HMM
parameter alignment only if the method independently chose a compatible form. Accept
an equivalent representation without requiring the generator's original notation.
A family assertion, matching dimension or low KL alone does not establish recovery.
Conversely, approximation error alone does not rule out partial structural recovery.

Report predictive agreement, structural resemblance, equivalence, minimality and
causal evidence separately. Numerical equivalence is not an all-history certificate.
Head pruning measures neural output effects; a claim of causal process recovery needs
an explicit correspondence between source mechanisms and neural interventions.

The frozen record's hashes and bundles are bound by a [manifest](final-campaign/manifest.json).
Final programs, errors, inputs, teachers and assessments are in the
[completed receipt dataset](../results/final-campaign-v2/summary.json).
