# State coordinates, learned features and interventions

[Primary protocol](../research/independent-discovery.md) · [Frozen receipts](discovery-development/summary.json) · [Agent results](agent-development.md)

All 25 programs ran and submitted executable processes. **None demonstrated recovery
of the generating structure.** These are restricted process-identification methods:
they learn their own state dimension and update operators without an HMM fitting
adapter, but retain a linear-state prior and a fixed analytical family recognizer.
They do not have the research agent's arbitrary program search space.

Two development cohorts used the same five qualified teachers: ten paired
output/activation runs, then fifteen representation/intervention runs. Each whole
cohort was registered before its private scoring. Later development could see earlier
results, so this is not one jointly blind 25-trial campaign or an unbiased ranking.

| Source process | Output geometry | PCA | Sparse autoencoder | Transcoder | Head ablation |
|---|---:|---:|---:|---:|---:|
| Delayed rule A | 1.1649 | 1.1649 | 1.1649 | 1.1649 | 1.1649 |
| Delayed rule B | 1.4269 | 1.4269 | 1.4269 | 1.4269 | 1.4269 |
| HMM mixture A | 0.1143 | 0.1889 | 0.1594 | 0.1889 | 0.1889 |
| HMM mixture B | 0.1175 | 0.2303 | 0.2162 | 0.2958 | 0.2494 |
| Stack rule | 0.0481 | 0.0558 | 0.0523 | 0.0528 | 0.0546 |

Values are mean KL from the true process to each frozen program, in nats/token,
on 128 private sequences of length 24. Lower is better. The corresponding transformer
scores are 0.0239, 0.0208, 0.0230, 0.0228 and 0.0085. Exact receipts also retain
sequence-level standard errors, teacher-to-program KL, length-72 scores and diagnostic
total-variation witnesses. The teacher is undefined beyond its length-24 context.

## What these methods actually did

[Output geometry and PCA](../research/state-geometry.md) regress autonomous token-update
operators in whitened probability or activation coordinates. Validation selects state
dimension and regularization using complete rollouts from tokens. The frozen predictor
does not need target features or the teacher at inference time.

The [sparse autoencoder and transcoder](../research/discovery-features.md) train
representations on the fixed public training sequences, then use those codes for the
same operator learner. The transcoder predicts the last MLP contribution. The
head-ablation variant prunes attention heads while preserving neural predictions on
the public training split, then uses the retained model's residuals as features.

None receives a process family, state count, latent labels, generator or HMM adapter.
After learning operators, a built-in algebraic test checks for a compatible latent
Markov realization. Failed compatibility leaves the latent family unresolved.
The same pipeline recognizes a simple two-state process from exact synthetic
probabilities in its correctness test; that control is not a trained-teacher result.

## Why this is not structural recovery

Every delayed-rule run selected a single coordinate: its predictions are independent
of history and miss the delay. On HMM mixture B, output geometry selected six
coordinates, matching the source's joint state count, but the learned operators failed
the latent-family checks. No transition/emission recovery or equivalence follows from
the matching dimension. On mixture A, dimensions ranged from one to fourteen.

The stack runs selected two or three coordinates but did not recover push/pop rules.
Their nonzero private prediction discrepancies also contradict exact equality on
those histories. All programs use explicit clipping/reset rules that can break a
pure linear-process interpretation. No minimality certificate is established.

Head pruning measures effects on neural outputs. It does not align a neural
intervention with an intervention on the recovered generating process, and is not
scored as causal process recovery. The summary records every removal and its scope.

This is a weak set of adaptations sharing one operator-learning stage. Poor results
can reflect that stage, unsuitable coordinates, finite data or neural approximation
error. They do not show that sparse autoencoders, transcoders or causal methods in
general cannot recover these processes, or that weights contain less useful information.

## Budgets, failures and evidence

All variants started from identical histories and 3,072 original probability vectors.
Output geometry, PCA and the autoencoder added no neural execution. Each transcoder
replayed 3,072 vectors; each head-pruning run returned 27,648 additional vectors across
nine executions, including the original replay and eight head interventions. These
are different access/query/compute conditions, not an equal-budget information test.

Extraction used networkless namespaces with 30 seconds wall time, 20 seconds process
CPU and 1 GB RAM per program. The encoder/transcoder used 800 fixed-seed optimizer
steps. Full source bundles, parameters, selection grids, representation objectives,
execution counts and programs are archived by case/method. The
[manifest](discovery-development/manifest.json) binds them to the registrations.

The geometry scorer initially failed before predictions because its immutable source
capsule lacked project metadata for provenance. Exact metadata/source mirrors were
added and scoring resumed in new directories from all ten original submissions.
The failed attempt remains local; no extraction or fitting was repeated.

The separately running paired research-agent cohort and a fresh final campaign remain
unfinished. The first fresh-suite attempt qualified only five of six teachers, so
no final extraction began. A new whole-suite attempt must retain that failure and
qualify every teacher before evaluating the frozen methods.
The [qualification-attempt receipt](qualification-attempts.json) records all six
teachers: the failed stack group's test upper bound was 0.03069 against a 0.03 limit.
