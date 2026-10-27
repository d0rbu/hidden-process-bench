# Hidden Process Bench

A prototype mechanistic interpretability evaluation. The intended task is to infer
an unknown data-generating mechanism from a trained transformer, including discovering
its process family and structure. On an HMM task, the method must discover the latent
transition/emission mechanism without being given an HMM template or its size.

**Current status: the first restricted observable-operator attempt ran on five qualified
development teachers and did not recover their mechanisms.** It chose its own state
dimension without an HMM fitting adapter, but assumed a linear representation rather
than discovering the generating family. See the [negative results](docs/results/operator-development.md).
The broader five-method discovery evaluation remains unfinished. Older fitters and
agent prompts were supplied HMM semantics and component/state counts; those results
do not measure the intended discovery task. See the [primary protocol](docs/research/independent-discovery.md).

Ground truth belongs to the evaluator. The primary track needs neutral inputs and
submissions, fresh hidden tasks from multiple generating mechanisms, and separate
output-only and internal-access conditions. The [new runtime](docs/reference/discovery-runtime.md) exports neutral model inputs,
isolates executable methods, freezes their chosen representations and scores predictions.
The [private mixed suite](docs/research/mixed-suite.md) adds fresh screened HMM mixtures,
delayed rules and stack processes with matched architectures and budgets. Teacher
qualification is implemented; family-discovery trials and structural assessment remain.
The [fresh-context research agent](docs/research/research-agent.md) can now choose its
own representation through audited JSON calls and a separate numerical sandbox.
Its live constant-distribution fixture is an execution check, not a benchmark result.
The [development pilot](docs/results/mixed-suite-development.md) qualified five of six
teachers; one failed its deep-stack test group, so the six-case campaign is not ready.

## Run the informed controls locally

```bash
git clone https://github.com/d0rbu/hidden-process-bench.git
cd hidden-process-bench
uv sync --locked
uv run --locked just setup
uv run --locked hidden-process-bench run --config configs/smoke.toml --out outputs/smoke
```

The smoke run exercises the informed-control pipeline in seconds. It deliberately trains for
only two steps and reports `benchmark_eligible: false`. For a bounded CPU experiment:

```bash
uv run --locked hidden-process-bench run --config configs/demo.toml --out outputs/demo
```

The demo uses two PyTorch CPU threads, at most 2,000 training steps, and two restarts of 250
optimization steps per recovery method. Results go to `outputs/demo/report.json`.
Existing output directories are never overwritten. Add `--require-qualified` to return
exit code 2 if the checkpoint misses the Bayesian prediction threshold; the report is
still saved. The locked environment uses CPU PyTorch on Linux and Windows.

The [informed five-family pilot](docs/results/five-method-pilot.md) compares HMM fitting
with classical methods, activation features, circuit interventions and agents. All
were told the HMM family and order; feature methods used a supplied HMM fitting adapter.
These are parameter-fitting controls, not independent structure-discovery results.

The [screened CPU example](docs/results/screened-cpu-demo.md) qualified at 1,200 steps
with held-out KL of 0.0199 nats/token. Its probability-only recovery baseline beat the
activation-assisted method on this task. Reports, rejected candidates, and source/software
provenance are included. The [initial unscreened example](docs/results/cpu-demo.md) remains
as a separate historical receipt.

## What is implemented for the informed controls

- Four stationary, node-emitting HMMs with a shared vocabulary; one unknown component
  generates each sequence. The default has 3 states per component and 16 output tokens.
- PAutomaC-inspired sampling: configurable sparsity and emission concentration, followed
  by exact per-component unigram/trigram difficulty checks and state-occupancy checks.
  Accepted tasks must also have observable Hankel rank equal to their joint state count,
  a numerical minimality witness that excludes redundant state copies.
  Rejected candidates are recorded; selection never uses transformer or recovery scores.
- An exact Bayesian filter over **both component identity and hidden state**.
- A small causal transformer trained on fresh sampled sequences by next-token loss.
  Validation selects when to stop; a separate test set determines qualification.
- Three informed fitting methods, using the **same model-generated histories**:

| Method | Allowed information | Objective |
|---|---|---|
| `samples` | Generated tokens | HMM maximum likelihood |
| `probabilities` | Tokens and next-token distributions | HMM distillation |
| `activations` | Tokens, distributions, final normalized residuals | Distillation plus a learned linear map from HMM beliefs to activations |

- Exact component/state permutation alignment, parameter MAE, transition-support F1,
  and held-out predictive KL. Forward calls, token queries, processed prefix tokens,
  optimizer steps, parameter counts, and fitting times are recorded.
- Numerical equivalence checks with distinguishing strings, equivalent state-splitting
  controls, and separate representation-size and observable-complexity diagnostics.
- Belief-state probes with independent fit/test sequences, explicitly labeled as
  **supervised diagnostics**, separate from unsupervised extraction.
- Saved checkpoints, recovered HMM JSON, evaluator-only truth, source hashes, seeds,
  a training trace, and a machine-readable report.

The activation method reads internals by executing the checkpoint. The [method pilot](docs/research/method-study.md) adds classical learning, unsupervised
representations, circuit interventions, and bounded checkpoint-agent trials. Parameter/state
counts, including the screened minimal order, are disclosed in this first version.
A passing prediction gate applies to the
sampled contexts and declared tolerance, not exact Bayesian optimality everywhere.

## Fit the informed HMM control to an existing checkpoint

```bash
uv run --locked hidden-process-bench extract \
  --checkpoint outputs/demo/public/checkpoint.pt \
  --task outputs/demo/public/task.json \
  --method activations --out outputs/candidate.json

uv run --locked hidden-process-bench score \
  --truth outputs/demo/private/truth.json \
  --candidate outputs/candidate.json
```

For this informed control, produce the HMM JSON described in the
[artifact contract](docs/reference/artifacts.md). Never give the extractor `private/`.
The public demo seed is intentionally reproducible; use unseen, evaluator-only seeds
for new development tasks. A family-blind challenge additionally needs the neutral
interface in the primary protocol; the existing HMM schema is not that interface. The
directories are an organizational boundary, not a sandbox.

## Scaling experiments

The four-run CPU pilot crosses 3/6 states per HMM with transformer widths 32/64,
holding vocabulary and budgets fixed. A 24-run grid also varies query budgets and seeds.

```bash
uv run --locked hidden-process-bench sweep --config configs/scaling.toml --dry-run
uv run --locked hidden-process-bench sweep \
  --config configs/scaling-pilot.toml --out outputs/scaling-pilot --timeout 300
```

Each run has a time limit. JSONL/CSV summaries retain qualified, unqualified, failed,
and timed-out runs. See the [scaling protocol](docs/research/scaling.md) for hypotheses
and the [measured pilot](docs/results/scaling-pilot.md): one checkpoint qualified and
three missed the preset gate. Accepted tasks have a numerical minimal-order witness;
equivalent state splitting tests the distinction from raw representation size. The baseline extractors
have a disclosed model order, so this prototype does not establish a transformer bias
toward minimal automata or fit a scaling law.

## Development

The [interactive explainer](site/README.md) visualizes the recorded generators and
results. Run it locally with `uv run --locked just site`, then open
`http://127.0.0.1:8847`. Its browser math is checked against the Python Bayesian oracle.

Based on [d0rbu/research-project-template](https://github.com/d0rbu/research-project-template)
at commit [`03c4a7a`](https://github.com/d0rbu/research-project-template/commit/03c4a7ae7f86170360816defe2572d61cd834b3b).
The MIT license and correctness tooling are retained. GitHub Actions is disabled;
there are no CI workflows or hosted required checks. `pre-commit` runs at **pre-push**,
never before ordinary commits. Run the full local gate before pushing a PR branch:

```bash
uv run --locked just check
```

See the [research design](docs/research/design.md),
[related work](docs/research/related-work.md), [documentation map](docs/README.md),
and [agent instructions](AGENTS.md).
