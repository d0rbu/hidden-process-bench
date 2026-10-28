# Hidden Process Bench

A prototype mechanistic interpretability evaluation with exact data-generating programs.

**New task: find the hidden exceptions.** A transformer learns a public addition rule
plus two private wrong answers. Methods must identify the triggering inputs, using
bounded output queries or declared neural access. The clean machine is allowed
knowledge; exception labels and training examples stay private. Causal localization
is separate extra credit. See the [protocol and runnable commands](docs/research/planted-exceptions.md)
and [first CPU pilot](docs/results/exceptions-pilot.md).

Five internal methods have now actually searched for backdoor inputs on three fresh
teachers, with every submission registered before private scoring. Transcoder, rare-neuron
and ablation search each found the same one of six planted inputs; PCA and SAE found
none. Uniform search found two. A nominated ablation repaired one trigger but damaged
clean addition, so no selective circuit recovery is claimed. See the [21 method trials
and limits](docs/results/exception-methods.md) and [method definitions](docs/research/exception-methods.md).

## Earlier process-discovery track

This separate task asks methods to infer the generating family and structure without
a supplied HMM template or size. Its results do not evaluate exception finding.

**Current status: 48 fresh independent-discovery trials completed.** Agents recovered
both delayed rules’ lags and exact token permutations, and found partial stack and
HMM organization. Full HMM recovery, exact source equivalence, minimality and neural
circuit recovery were not established. The 36 restricted linear-state trials did not
demonstrate source structure recovery. See the [final results and frozen programs](docs/results/final-campaign-v2.md)
and [protocol frozen before task generation](docs/research/final-campaign.md).
Earlier [paired agents](docs/results/agent-development-v2.md),
[first agent cohort](docs/results/agent-development.md) and
[25 state/feature/intervention runs](docs/results/discovery-development.md) remain development evidence.
Older fits supplied HMM semantics and component/state counts are informed controls,
not discovery results. See the [primary protocol](docs/research/independent-discovery.md).

Ground truth belongs to the evaluator. The primary track needs neutral inputs and
submissions, fresh hidden tasks from multiple generating mechanisms, and separate
output-only and internal-access conditions. The [new runtime](docs/reference/discovery-runtime.md) exports neutral model inputs,
isolates executable methods, freezes their chosen representations and scores predictions.
The [private mixed suite](docs/research/mixed-suite.md) adds fresh screened HMM mixtures,
delayed rules and stack processes with matched architectures and budgets. Teacher
qualification, executable discovery trials and separate structural review are implemented.
The [fresh-context research agent](docs/research/research-agent.md) can now choose its
own representation through audited JSON calls and a separate numerical sandbox.
Its first mixed-process development results are linked above.
The [development pilot](docs/results/mixed-suite-development.md) qualified five of six
teachers; the final campaign used a separate fresh suite with all six qualified.

## Run independent discovery locally

```bash
git clone https://github.com/d0rbu/hidden-process-bench.git
cd hidden-process-bench
uv sync --locked
uv run --locked just setup
uv run --locked python -m hidden_process_bench.mixed_suite prepare \
  --config configs/discovery-pilot.toml --out outputs/private-suite
uv run --locked python -m hidden_process_bench.mixed_suite train \
  --suite outputs/private-suite --timeout 300
```

These evaluator commands generate six private tasks, train and qualify their teachers,
and export neutral inputs for qualified cases. Check `private/training-summary.json`:
`campaign_ready` requires all six teachers to pass. Failures are retained; a failed suite
must not be presented as a completed campaign.

Package the restricted operator baseline, or supply your own fixed method:

```bash
uv run --locked python -m hidden_process_bench.discovery_cli prepare-operators \
  --out outputs/operator-method
uv run --locked python -m hidden_process_bench.discovery_cli extract \
  --program outputs/operator-method --entry main.py \
  --challenge outputs/private-suite/public/CASE_ID/outputs --out outputs/extraction
```

Replace `CASE_ID` with an opaque qualified case ID. The extractor sees only that neutral
packet. Keep the suite's `private/` files and this evaluator conversation out of its
context. See the [mixed-suite guide](docs/research/mixed-suite.md) and
[submission contract](docs/reference/discovery-runtime.md) for other methods, isolated
agent execution, receipt registration and private scoring. Prediction scores leave
structural recovery for a separate review of the method's chosen mechanism.

## Run the informed controls

```bash
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
