# Paired agent development results

[Documentation](../README.md) · [Discovery protocol](../research/independent-discovery.md) · [Earlier agent cohort](agent-development.md)

Both agent conditions independently recovered the delayed rule's lag-three permutation.
On one HMM task, the internal-access agent inferred the correct four-regime, four-local-state
organization. Its calibrated filter differs from the source's Bayesian update. These are
core and partial structural findings; no exact equivalent source or neural circuit was
established. This is a development cohort, not the fresh final campaign.

Six of ten trials submitted programs. Four failed during local client preflight because
the login file was temporarily absent, before any model inference. Those failures are
retained and were not retried. They do not measure the extraction method's ability on
those two processes. All ten outcomes were registered before any private scoring.

## Prediction scores

KL against the true process, nats/token; lower is better. Every scored entry uses the
same 128 private histories of length 24 as its paired condition. Transformer scores
come from those same histories, separately from teacher qualification.

| Development case | Transformer | Agent: outputs | Agent: internals | Structural finding |
|---|---:|---:|---:|---|
| Delayed rule A | 0.02390 | 0.01959 | 0.02072 | Lag-three permutation recovered in both conditions |
| Delayed rule B | 0.02078 | — | — | Both preflights failed; no submission |
| HMM mixture A | 0.02302 | 0.01352 | 0.01604 | Persistent-mixture family; internal condition inferred correct 4 × 4 dimensions |
| HMM mixture B | 0.02280 | 0.01308 | 0.01445 | Approximate latent filters; equivalent HMM not established |
| Stack rule | 0.00846 | — | — | Both preflights failed; no submission |

The submitted programs can be closer to the source than the transformer. They choose
simpler mechanisms and smooth some neural prediction errors; this does not establish
correct structure by itself. [Raw scores, sequence uncertainty and receipts](agent-development-v2/summary.json)
separate source-to-program, teacher-to-program and source-to-teacher KL. Longer-history
scores use length 72 and no teacher comparison, since the teacher's context is 24.

## What was recovered?

**Delayed rule A.** Both methods selected a three-token FIFO and the exact 16-entry
vocabulary permutation in the source. Three interleaved observed-symbol Markov chains
are an equivalent description of that core dependency. The designated, frozen
predictors preserve the source's modal permutation on **all 4,096 possible length-three
histories**, checked after registration without refitting. Their bundled simple cores
use a modal probability of 0.70; the source's is 0.69349. The designated predictors add
neural confidence, position and other-token corrections, so their full probability laws
differ from the source. This is recovery of the core rule, not exact recovery of the
noise law, minimality, or all-history equivalence. See the [modal diagnostic](agent-development-v2/lag-modal-diagnostic.json).

**HMM mixture A.** The output-only agent independently selected four persistent source
blocks with 4/5/4/5 local states, 18 total. Its executable uses ordinary HMM filtering
after a separately calibrated initial boundary. The source has four blocks of four,
16 total. A larger representation could in principle be equivalent; a state-count
mismatch alone is not a rejection. No such equivalence or reduction was established.

The internal-access agent selected four persistent regimes with four local states each,
matching the source organization. Its transition matrix has four exact 4 × 4 blocks
and zero cross-block entries. The actual predictor uses separate recognition and
emission tables and a last-token logit correction. It therefore does not execute the
literal Bayesian filter of its proposed generative core. Alignment of that **unaltered,
method-proposed core** gives mean parameter MAE 0.03896 and transition-support F1 0.9091
at edge threshold 0.05. This diagnostic neither refits nor replaces the designated
predictor. It supports partial structural resemblance; it does not establish equivalent
execution or a recovered neural circuit. See the [core diagnostic](agent-development-v2/generative-core-diagnostic.json).

**HMM mixture B.** The output-only program uses six regimes, nearly arranged into two
triples, with separate recognition/readout tables and nonlinear output calibration.
The internal program uses eight coordinates and two-sided token-conditioned evidence.
The source is two permanently disconnected three-state HMMs. These programs found
compact filtering explanations and some persistent organization. A six-coordinate
count alone does not identify the source, and a different coordinate count alone does
not rule out equivalence. Their actual update rules, uncertainties and prediction
witnesses leave equivalent source recovery unestablished.

The [manual review](agent-development-v2/structural-review.json) is bound to the exact
registered programs. Its two positive recovery flags refer specifically to the delayed
core rule. They do not assert full probabilistic equivalence, minimality or causal circuit
recovery. Submitted source programs and the source processes remain available for inspection.

## Access, work and limitations

The model was `gpt-6.1-sol`, reasoning `xhigh`, in fresh neutral contexts. It received no
HMM template, family/order hint, hidden labels, generator code, earlier hypothesis or
private feedback. Each trial had 20 actions, 30 minutes total and at most 10 minutes
per model call. This protocol also encourages an early working candidate; it is a
separate version from the earlier five-trial cohort, whose timeouts remain unchanged.

Output-only trials used 128 cached histories and 3,072 distributions. Internal trials
received the same cache plus weights, residuals and a generic neural execution helper.
Their reported successful additional executions were:

| Internal trial | Batched forwards | Returned prediction vectors |
|---|---:|---:|
| Delayed rule A | 24 | 47,232 |
| HMM mixture A | 5 | 35,336 |
| HMM mixture B | 65 | 78,336 |

The last trial also reported work lost to failed actions, with incomplete exact counts
and declared upper bounds. Source/action logs and resource declarations are preserved.
These are declared execution counts, not an enforced equal-query budget. Internal
agents mostly used weights to obtain extra distributions and controlled history tests;
they did not recover a neural circuit. Cached-activation decoding in HMM mixture B is
correlational evidence. Query access, compute and candidate searches differ, and there
is only one agent trial per case/access. No general weights-versus-outputs ranking follows.

All programs run independently of the transformer at scoring time. They do not refit
on evaluation histories. The original unsubmitted artifacts from preflight failures
were not scored or rescued. Repeated development validation and method exploration
mean these cases are not an unbiased final test. The [fresh frozen campaign](../research/final-campaign.md)
uses unchanged methods and new private tasks; it remains in progress.

## Evidence and verification

The [manifest](agent-development-v2/manifest.json) binds source, original registration,
protocol, all ten outcomes, six complete submissions, action trajectories, private
prediction scores, diagnostic histories, source processes and teacher receipts.
Runtime source is archived separately from method-written predictors. Authentication
contents are never included; the failed preflight stderr records the missing path.

A separate post-registration audit recomputed all five teachers' saved samples,
groups, exact oracle probabilities, checkpoint predictions, validation/test gates,
paired packet hashes and separate scoring streams. The [audit](agent-development-v2/qualification-audit.json)
records their hashes. Published per-sequence KL/group arrays reproduce every qualification
bound without exposing training seeds. Local tests rerun all six registered predictors
on private diagnostic histories, repeat the exhaustive modal checks, and recompute the
HMM-core alignment. These checks support the stated results, not a broader discovery claim.
