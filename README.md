# Hidden Process Bench

A prototype mechanistic interpretability evaluation: train a transformer on four
hidden Markov models, then recover the generating processes from the trained model.
The ground truth makes both predictive fidelity and structural recovery measurable.

The central question is whether access to model internals helps at a fixed observation
budget. An output-only method may already recover the process. That is a baseline to
measure, not an assumption to hide.

## Run locally

```bash
git clone https://github.com/d0rbu/hidden-process-bench.git
cd hidden-process-bench
uv sync --locked
uv run --locked just setup
uv run --locked hidden-process-bench run --config configs/smoke.toml --out outputs/smoke
```

The smoke run exercises the entire pipeline in seconds. It deliberately trains for
only two steps and reports `benchmark_eligible: false`. For a bounded CPU experiment:

```bash
uv run --locked hidden-process-bench run --config configs/demo.toml --out outputs/demo
```

The demo uses two CPU threads, at most 1,200 training steps, and two restarts of 250
optimization steps per recovery method. Results go to `outputs/demo/report.json`.
Existing output directories are never overwritten. Add `--require-qualified` to return
exit code 2 if the checkpoint misses the Bayesian prediction threshold; the report is
still saved. The locked environment uses CPU PyTorch on Linux and Windows.

## What is implemented

- Four stationary, node-emitting HMMs with a shared vocabulary; one unknown component
  generates each sequence. The default has 3 states per component and 16 output tokens.
- An exact Bayesian filter over **both component identity and hidden state**.
- A small causal transformer trained on fresh sampled sequences by next-token loss.
  Validation selects when to stop; a separate test set determines qualification.
- Three recovery methods, using the **same model-generated histories**:

| Method | Allowed information | Objective |
|---|---|---|
| `samples` | Generated tokens | HMM maximum likelihood |
| `probabilities` | Tokens and next-token distributions | HMM distillation |
| `activations` | Tokens, distributions, final normalized residuals | Distillation plus a learned linear map from HMM beliefs to activations |

- Exact component/state permutation alignment, parameter MAE, transition-support F1,
  and held-out predictive KL. Forward calls, token queries, processed prefix tokens,
  optimizer steps, parameter counts, and fitting times are recorded.
- Belief-state probes with independent fit/test sequences, explicitly labeled as
  **supervised diagnostics**, separate from unsupervised extraction.
- Saved checkpoints, recovered HMM JSON, evaluator-only truth, source hashes, seeds,
  a training trace, and a machine-readable report.

The activation method reads internals by executing the checkpoint. Static weights-only
reconstruction and causal circuit validation are not implemented. Parameter/state
counts are disclosed in this first version. A passing prediction gate applies to the
sampled contexts and declared tolerance, not exact Bayesian optimality everywhere.

## Extract from an existing checkpoint

```bash
uv run --locked hidden-process-bench extract \
  --checkpoint outputs/demo/public/checkpoint.pt \
  --task outputs/demo/public/task.json \
  --method activations --out outputs/candidate.json

uv run --locked hidden-process-bench score \
  --truth outputs/demo/private/truth.json \
  --candidate outputs/candidate.json
```

For an independent extractor, produce the HMM JSON described in the
[artifact contract](docs/reference/artifacts.md). Never give the extractor `private/`.
The public demo seed is intentionally reproducible; use unseen, evaluator-only seeds
for a real challenge. The directories are an organizational boundary, not a sandbox.

## Development

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
