# Five recovery method families: measured pilot

[Documentation](../README.md) · [Protocol](../research/method-study.md) · [Raw summary](five-method-pilot/summary.json)

All five families ran. Belief features helped on one new independent task, while the
ordinary probability fit remained competitive elsewhere. **No candidate recovered a
numerically equivalent process.** These are bounded adaptations, not reproductions
of full published systems or a general method ranking.

## Prediction recovery

Predictive KL in nats/token; lower is better. Each column uses the same 512 held-out
oracle histories for every method. All four checkpoints passed the unchanged 0.03
transformer qualification gate. The first two are older development examples sharing
a generator seed; A and B use independently drawn private generator and model seeds.

| Method | Demo | Earlier pilot | Independent A | Independent B |
|---|---:|---:|---:|---:|
| Adam: samples | 0.0860 | 0.1147 | 0.0987 | 0.0725 |
| Adam: probabilities (control) | 0.0137 | 0.0226 | 0.0310 | 0.0215 |
| Baum–Welch / EM | 0.1167 | 0.1865 | 0.1162 | 0.0972 |
| Spectral → HMM adapter | 0.2503 | 0.2437 | 0.2131 | 0.2038 |
| Belief geometry: block 1 | 0.0179 | 0.0454 | 0.0167 | 0.0253 |
| Belief geometry: block 2 | 0.0180 | 0.0371 | 0.0182 | 0.0201 |
| Belief geometry: both blocks | 0.0210 | 0.0291 | 0.0180 | 0.0256 |
| Final-residual PCA | 0.0146 | 0.0350 | 0.0315 | 0.0199 |
| Sparse autoencoder | 0.0154 | 0.0424 | 0.0169 | 0.0269 |
| Last-MLP transcoder | 0.0163 | 0.0432 | 0.0310 | 0.0263 |
| Pruned-model HMM | 0.0137 | 0.0223 | 0.0320 | 0.0215 |
| Agent: static weights | 0.4016 | 0.3837 | — | — |
| Agent: cached observations | 0.0089 | 0.0110 | — | — |

The shared extraction budget is 128 generated histories of length 24. Most numerical
methods use two restarts of 250 HMM fitting steps. SAE/transcoder learning adds 400
steps. The observation agent instead chose 12 restarts of 1,800 steps plus a 400-step
refit, with a preliminary optimization attempt also retained in its report. It is a
stronger probability-fitting procedure with **more compute**, rather than evidence of
an advantage from weights. It inspected weights/activations but did not use them in
its fitting objective. The static agent made zero transformer calls; the other agent
used the same cached observations and made zero additional transformer calls.

The circuit method uses 15 additional batched forwards: 13 for discovery/controls and
two for interchange. Its HMM distills the pruned model; its access budget is larger.

## What changed our view

- On independent A, block-1 belief features reduced KL from 0.0310 to 0.0167, and SAE
  features reached 0.0169. Parameter MAE also improved: 0.0628 → 0.0408 and 0.0426.
  This is one task and one recovery initialization schedule, not a general access advantage.
- On independent B, final-residual PCA reached 0.0199 versus the probability control’s
  0.0215. Other feature choices were mixed. On the older examples, feature assistance
  generally hurt predictive recovery. Selecting a different best layer after seeing each
  test score would overstate the result; every frozen layer trial is shown.
- EM stayed in poorer optima under these two-restart budgets. The spectral adapter
  needed clipping/normalization on 90.9%–92.7% of prediction vectors. Its poor scores
  are a finite-sample failure of this adaptation, not a verdict on spectral learning.
- PCA/SAE/transcoder held-out reconstruction errors were about 10.2%–12.5%,
  3.9%–4.8%, and 2.7%–3.9% of standardized-target energy. Better reconstruction did
  not reliably imply better HMM recovery. The transcoder was a single-layer sparse
  replacement fit, not an attribution-graph or full circuit-tracing evaluation.
- Observable belief-update regression had held-out normalized errors of 2.6%–14.1%,
  depending on task/layer. Those are one-step coordinate predictions. The HMM adapter
  uses coordinate reconstruction, rather than enforcing the fitted operator matrices.
- Circuit pruning retained 10/10, 9/10, 9/10 and 10/10 tested head/MLP nodes. Held-out
  KL from the original model was 0, 0.00187, 0.00134 and 0, respectively. This did not
  isolate a small circuit. The interchange mapping yielded change-error ratios of
  0.86–0.90 relative to a zero-change baseline of 1; counterfactual distribution KL
  remained much larger than ordinary candidate/model KL. It gives limited support
  for that particular localized belief-state mapping, not recovered causal dynamics.

[Paired sequence-level differences](five-method-pilot/paired-deltas.json) include
diagnostic 95% intervals against the ordinary probability control. They quantify
context sampling variation within each fixed task, not uncertainty across generators,
and are not corrected for comparing multiple methods.

## Structural recovery

Permutation-aligned parameter MAE; lower is better. A predictive improvement need not
recover the true parameters. State/component counts are disclosed to all fitters.

| Method | Demo | Earlier pilot | Independent A | Independent B |
|---|---:|---:|---:|---:|
| Adam: samples | 0.0965 | 0.1074 | 0.1174 | 0.1033 |
| Adam: probabilities (control) | 0.0315 | 0.0505 | 0.0628 | 0.0699 |
| Baum–Welch / EM | 0.1300 | 0.1126 | 0.1304 | 0.1154 |
| Spectral → HMM adapter | 0.1222 | 0.1163 | 0.0886 | 0.1096 |
| Belief geometry: block 1 | 0.0507 | 0.0741 | 0.0408 | 0.0657 |
| Belief geometry: block 2 | 0.0521 | 0.0481 | 0.0572 | 0.0672 |
| Belief geometry: both blocks | 0.0638 | 0.0704 | 0.0552 | 0.0672 |
| Final-residual PCA | 0.0395 | 0.0451 | 0.0692 | 0.0636 |
| Sparse autoencoder | 0.0444 | 0.0542 | 0.0426 | 0.0642 |
| Last-MLP transcoder | 0.0503 | 0.0544 | 0.0650 | 0.0640 |
| Pruned-model HMM | 0.0315 | 0.0465 | 0.0613 | 0.0699 |
| Agent: static weights | 0.1448 | 0.1396 | — | — |
| Agent: cached observations | 0.0253 | 0.0353 | — | — |

All 48 submissions were numerically distinguished from the reference at tolerance
1e-10. The static candidates use uniform transitions and do not claim to recover
temporal dynamics. Agent results are limited to the two older checkpoints; they
are not independent hidden-task replications.

## Receipts and reproduction

The numerical source was committed cleanly at
`0481e282c083e137c4afae97345ac5297c41df27` (use the full revision and source hashes in each plan for exact provenance).
All candidates were fixed and hashed before evaluator feedback. The agent conditions
had fresh contexts and public-only staged inputs, with instruction/API boundaries
rather than an operating-system sandbox. The inherited agent model identifier was not
exposed in tool metadata; these agent trials are less reproducible than the numerical fits.

| Evidence | Files |
|---|---|
| Frozen configs and software/source hashes | [Existing plan](five-method-pilot/existing/plan.json), [independent plan](five-method-pilot/independent/plan.json) |
| Full numerical scores, budgets, timings and diagnostics | [Existing report](five-method-pilot/existing/report.json), [independent report](five-method-pilot/independent/report.json) |
| Independent training configs, all rejected candidates and qualification | [Training plan](five-method-pilot/training/plan.json), [A](five-method-pilot/training/case-0/report.json), [B](five-method-pilot/training/case-1/report.json) |
| Fixed agent methods and accounting | [Static report](five-method-pilot/agent-static/report.json), [observation report](five-method-pilot/agent-observations/report.json), [protocols](five-method-pilot/agent-protocols.json) |
| Agent scoring and submission hashes | [Evaluation](five-method-pilot/agent-evaluation/evaluation.json) |
| Exact agent source as experimental receipts | [Static](five-method-pilot/agent-static/extract_static.py.txt), [observation](five-method-pilot/agent-observations/extract.py.txt) |
| All published receipt hashes | [Manifest](five-method-pilot/manifest.json) |

Candidate HMM JSON and post-submission evaluator truth copies accompany each numerical
case. Exact agent sources retain their original bytes as `.py.txt` experiment receipts;
Python can execute these files directly in a directory with the staged public inputs.
Checkpoints, generated histories, fitted observable-operator arrays and complete local
run directories remain under ignored `outputs/`. Checkpoint hashes bind the published
results to those saved bytes. Published synthetic truths/seeds are now development data,
not suitable for a future blind challenge.

Run the numerical protocol on any newly trained checkpoint using the command in the
[protocol](../research/method-study.md), appending `--run` for each task. It refuses
existing output directories. The new training plan froze a 4,000-step cap and up to
512 sampling attempts per task before training; both tasks qualified before the cap.
No GPUs were rented. Local checks: 206 Python tests and 99.29% branch-inclusive coverage
before running the study; the final publication gate is recorded in the PR.

Historical receipts are preserved. Scores here use a new common evaluation sample,
so the older demo’s historical 0.0129 probability KL becomes 0.0137 here without
changing its candidate parameters. Longer histories, unknown order, additional query
budgets, repeated recovery seeds and more independent generators remain untested.
