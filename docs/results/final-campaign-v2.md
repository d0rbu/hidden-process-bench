# Fresh discovery campaign: 48 frozen trials

[Documentation](../README.md) · [Protocol frozen before task generation](../research/final-campaign.md)

Agents independently recovered both delayed rules' lags and exact token permutations.
They also discovered typed-stack behavior and persistent HMM regimes, with important
parts of those mechanisms unresolved. **Full HMM structure recovery was not established.**
None demonstrated exact source equivalence, minimality or a recovered neural circuit.
The 36 restricted linear-state trials did not demonstrate source structure recovery.

All 48 trials submitted programs and completed both scoring sets without scoring
errors. No intermediate program was rescued and no trial was selectively retried.
Failed numerical actions inside a trial are preserved in its trajectory. The
[registration](final-campaign-v2/registration.json) fixes every program before any
private final scoring; its SHA-256 is
`d2775305545856e1dd463817de717aa78d634a4366bc1894ffa69bcb4824ea22`.

## What was concealed

Methods received opaque case IDs, token IDs, the same neural architecture and declared
access budgets. No process family, latent labels, state count, generator seed, HMM
constructor, fitting adapter, private oracle or scoring feedback was supplied. Each
agent began in a fresh neutral context; it could choose a program, automaton, HMM or
another representation. The same fixed prompt and method sources preceded all six
new tasks. The complete twelve-agent context/packet audit passed.

There were two fresh tasks each from HMM mixtures, delayed-token rules and unbounded
typed-stack processes. HMM candidates passed dependency, occupancy and numerical
minimal-order screens; all rejected candidates are retained. The two accepted HMMs
had three-by-three and two-by-three component/state shapes, with numerical joint
observable ranks nine and six. Those shapes were revealed only after registration.
An earlier whole suite failed one teacher's deep-stack test group and began no
extraction; its [negative receipt](qualification-attempts.json) is retained.

## How close were the transformers?

Every transformer had two layers, width 64, four heads, context 24, vocabulary 16 and
103,744 parameters. Training used two CPU threads. Stopping validation required
mean KL plus 1.96 sequence standard errors at most .024; a separate 2,048-sequence
test required at most .03 overall **and in every component/depth group**. All six
qualified. These are approximate sampled-context guarantees, not Bayes optimality
on every possible history.

| Process | Training steps | Test mean KL | Largest test upper bound |
|---|---:|---:|---:|
| Stack A | 1800 | 0.01069 | 0.02372 |
| HMM A | 800 | 0.01749 | 0.02179 |
| Lag three | 100 | 0.02004 | 0.02014 |
| HMM B | 400 | 0.01815 | 0.01925 |
| Lag four | 100 | 0.01918 | 0.01929 |
| Stack B | 1900 | 0.00899 | 0.01918 |

The frozen-runtime audit regenerated every candidate process, screening decision,
qualification/held-out sample, oracle and teacher prediction from its original seeds.
Checkpoint predictions and saved arrays matched exactly. Published per-sequence
qualification statistics let the local tests recompute all group bounds.

## Prediction results

KL is true process to submitted program, in nats/token; lower is better. Each row
uses the same 128 independent histories for all eight methods. Short histories have
24 prediction positions, longer histories 72. Transformer scores below use the same
short scoring histories, rather than the separate qualification set above. There
is no transformer prediction beyond its context; longer results assess source
extrapolation only.

| Process | Transformer | Outputs agent | Internals agent | Outputs, long | Internals, long |
|---|---:|---:|---:|---:|---:|
| Stack A | 0.00907 | 0.00769 | 0.00805 | 0.01522 | 0.01571 |
| HMM A | 0.01849 | 0.01094 | 0.01354 | 0.01041 | 0.01079 |
| Lag three | 0.02037 | 0.01735 | 0.01744 | 0.01709 | 0.01700 |
| HMM B | 0.01679 | 0.01176 | 0.01388 | 0.01202 | 0.01673 |
| Lag four | 0.01921 | 0.01613 | 0.01595 | 0.01231 | 0.01264 |
| Stack B | 0.01015 | 0.00599 | 0.00530 | 0.01361 | 0.03526 |

The [machine-readable summary](final-campaign-v2/summary.json) also gives sequence
standard errors, teacher-to-program KL and maximum total-variation errors. Every
trial has concrete short/long witness histories. A candidate can beat the transformer
against the source by imposing a helpful structural prior; this is neither evidence
of leakage by itself nor proof that its entire mechanism was recovered.

## What the submitted programs recovered

**Delayed rules: four scoped successes.** Both agents on each task independently
selected lag three/four and the exact 16-symbol bijection. The actual frozen predictors
use the symbol at that lag, including noisy observations, rather than following an
unobserved clean orbit. They match the source's preferred token on every one of the
4,096 length-three and 65,536 length-four histories, respectively, and on 1,024 uniform
histories each at lengths 23 and 72. This establishes the tested dependency core.
Their calibrated probabilities still contain position/other-token effects, and do
not recover the exact constant source noise law. Separate simple core programs were
not substituted for the designated predictors. This is not all-history equivalence
or a minimal-automaton certificate.

**HMMs: family and grouping, incomplete state dynamics.** Both access conditions
independently chose mixtures of hidden-state chains. On HMM A they inferred three
persistent groups with 3/3/2 or 2/3/3 modes; the source has 3/3/3. On HMM B both chose
3/2 modes across two regimes; the source has 3/3. These are meaningful partial
organizational findings. They are not dismissed merely for using another state
count or coordinate system. The actual updates also use confidence discounting,
separate recognition factors, token traces or readout corrections; no equivalent
observable law was established. The shape-incompatible cores were not refitted or
forced into the HMM parameter-alignment scorer.

**Stacks: roles and nested restoration, incomplete mismatch/memory rules.** All four
agents found the two opener/closer pairs and restoration of an older opening after
a nested pair closes. The source pops only a matching top, ignores mismatches and
has unbounded depth. Actual submissions instead use bounded suffixes, a finite noisy
stack, any-close popping, or selective deletion of a buried matching opener. For
example, Stack B history `[7,15,5,4]` leaves opening 7 in the source; the internal
agent's selective-deletion parser becomes empty. Its submitted probability vector
has TV about .267 on that probe. These are partial structural discoveries, not
recovery of the complete pushdown process.

The [manual review](final-campaign-v2/structural-review.json) assesses every executable
and binds it to its original submission receipt. The criteria were written before
opening final truth. Four flags indicate **dependency-core recovery only**, eight
agent trials have partial structure, and 36 programmed trials remain unresolved or
incompatible. Exact source recovery, equivalence, minimality and neural-circuit flags
are separate and all remain false. The [diagnostics](final-campaign-v2/structural-diagnostics.json)
contain the exhaustive modal checks and full stack-probe vectors; local tests replay
the original submitted programs without changing them.

## Restricted programmed methods

All six share signed autonomous linear-state prediction, with explicit probability
floors and invalid-state resets. Their representation/recognition assumptions are
declared in the protocol; they are not unrestricted program search. Lag-rule fits
collapsed to memoryless prediction. Stack fits did not identify typed updates.
None recognized a compatible HMM realization on the HMM cases. This weak shared
learner does not settle the capabilities of PCA, autoencoders, transcoders or circuit
analysis in general.

| Process | Operators | Geometry | PCA | Autoencoder | Transcoder | Head ablation |
|---|---:|---:|---:|---:|---:|---:|
| Stack A | 0.07808 | 0.05429 | 0.05908 | 0.06095 | 0.05842 | 0.05918 |
| HMM A | 0.23216 | 0.07209 | 0.08650 | 0.29401 | 0.09878 | 0.15624 |
| Lag three | 1.45611 | 1.45697 | 1.45697 | 1.45697 | 1.45697 | 1.45697 |
| HMM B | 0.17379 | 0.05711 | 0.14115 | 0.07971 | 0.15665 | 0.15628 |
| Lag four | 1.16946 | 1.16241 | 1.16241 | 1.16241 | 1.16241 | 1.16241 |
| Stack B | 0.05810 | 0.05230 | 0.05146 | 0.05200 | 0.05108 | 0.05156 |

## Access, compute and limits

Each case's output/internal packets have identical cached histories and probabilities:
128 length-24 sequences, 3,072 probability vectors, 24 collection calls and 38,400
processed prefix tokens. Outputs agents made no additional neural queries. The
internal agents declared these **additional** helper executions:

| Process | Forward calls | Returned vectors |
|---|---:|---:|
| Stack A | 34 | 92,468 |
| HMM A | 61 | 41,664 |
| Lag three | 19 | 91,056 |
| HMM B | 25 | 34,968 |
| Lag four | 15 | 24,704 |
| Stack B | 16 | 21,456 |

Accounting is method-reported, with per-instance records and failed-action caveats
preserved; Stack A includes inferred completed forwards from two rolled-back actions.
Some agents also recomputed attention diagnostics. Each transcoder adds one full
cache replay (3,072 vectors); head-ablation adds nine (27,648). Operators, output
geometry, PCA and the autoencoder add no full neural execution. Fitting/search
work and every agent action are retained. Agent trials used `gpt-6.1-sol`, `xhigh`,
at most 20 actions/1,800 seconds, with at most 600 seconds per model call.

Weights were principally used to execute/query and inspect the neural model, not to
recover a source graph directly from static tensors. Ablations and top-state probes
are useful diagnostics; no trial established semantic source-state/neural-intervention
alignment or a complete circuit. Extra queries, compute and search differ, so this
is not an equal-budget ranking or evidence that internals contain more information
than outputs. Six cases at one transformer size do not establish scaling, minimality
bias or a general recovery rate. An independently chosen HMM still has to be fitted
from evidence; supplying that family to the method would instead be an informed control.

## Complete evidence

The [manifest](final-campaign-v2/manifest.json) binds all published files. The dataset
includes the original neutral packets, six teacher checkpoints, generating processes,
all sampling decisions, per-sequence qualification statistics, original method/runtime
archives, twelve agent trajectories/preflight captures, 48 frozen submissions, both
scoring receipts per trial, all original scoring arrays and structural assessments.
NPZ files are preserved byte-for-byte in parts under 1 MiB; each adjacent JSON
descriptor gives their order, length and original SHA-256. Concatenate the parts
before loading the NPZ. Local tests reconstruct them and reproduce every KL, sequence
standard error and maximum TV. Operational audit scripts are
archived separately; their local absolute paths require adaptation to rerun elsewhere.
The teacher/source audit used the exact frozen runtime, not newer evaluator math.

These now-public cases are development data for future work. A new confirmatory run
needs new hidden tasks, frozen methods and a fresh registration barrier. Historical
informed HMM fits remain [controls](five-method-pilot.md), never primary discovery results.
