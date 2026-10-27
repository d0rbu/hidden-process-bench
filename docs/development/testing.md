# Testing

[Documentation](../README.md) · [Correctness](correctness.md)

```bash
uv run --locked python -m pytest
```

The suite checks the Bayesian filter against independent enumeration of hidden paths,
causal masking and BOS alignment, differentiable filtering against NumPy and finite
differences, permutation-invariant scoring, a nonidentifiable negative control,
recovery of known simple distributions, checkpoint/public artifact and seed boundaries, and
real end-to-end CLI runs. The inherited template contract and Markdown tests remain.
Sampling entropies are checked against hidden-path enumeration. IID mixtures and fully
observed Markov chains test the difficulty screen; state splitting and temporal witnesses
test equivalence. Redundant state copies fail the minimal-order screen while retaining
identical temporal information. Sweep tests exercise a real child process and timeout, crash, malformed
receipt, missing-report, and qualification-status paths.

The website snapshot is reproduced from public receipts in `test_site_data.py`.
Delayed-token and stack walkthroughs check every frame against the process oracle;
replaying only the displayed memory must retain the full-history prediction.
Final campaign tests bind all 48 original programs to the preregistered sources,
neutral agent contexts and teacher receipts. They recompute qualification bounds,
recompute every final KL/standard error and maximum TV from original scoring arrays,
replay held-out prediction witnesses, exhaustively check the recovered short lag
rules and reproduce stack mismatch probes without refitting.
`uv run --locked just site-check` uses Node.js 22+ to compare every browser belief and
prediction with NumPy fixtures, and verifies equivalent state copies and sampling order.
It is part of the local pre-push gate. See the [website guide](../../site/README.md) for
browser and public-byte verification.

Discovery-runtime tests launch real Linux namespace/cgroup sandboxes: they check private
file/environment/network isolation, NumPy availability, read-only sources, bounded output,
timeouts, failed searches, frozen-manifest tampering, and row-invariance violations.
Generic constant-process fixtures validate execution and scoring; they are not discovery
results. These tests require Bubblewrap, prlimit and a working user systemd manager.

Fresh-agent tests use a synthetic executable client in a real sparse namespace,
without contacting a model provider. They reject injected context/tool descriptions,
exercise local HTTP request capture, check numerical workspace rollback, enforce action
and wall budgets, and preserve failed-trial receipts. The captured context fixture
contains neutral text and the inert question-tool definition, never account headers.

Mixed-suite tests compare stack predictions against an independent parser on all
short histories, check delayed-rule causality with bounded Hypothesis cases, and
verify seeded sampling/serialization. Real child processes exercise all three
generators. Separate tests recompute qualification, verify paired public observations,
reject source/task/checkpoint mismatches, and retain failed preparation and timeouts.
The positive smoke fixtures use a deliberately loose tolerance to test exports;
they are not evidence of research qualification.

Tests use two CPU threads, no external network, no paid providers, and no GPUs. The measured
coverage source is `hidden_process_bench` plus repository `scripts`, not test code.
The template's 95% branch coverage threshold is retained. Warnings, unknown markers,
unknown pytest configuration, and unexpected xfail passes are errors.

Focused iteration avoids applying the whole-suite coverage threshold to a subset:

```bash
uv run --locked python -m pytest --no-cov -k filter
uv run --locked python -m pytest --no-cov -m property
```

Full local gate, **before pushing a PR branch**, not before every commit:

```bash
uv run --locked pre-commit run --hook-stage pre-push --all-files
```

This checks the lockfile, Ruff lint/format, ty, Vulture, pytest/coverage, local Markdown
links, justfile syntax, and file hygiene. GitHub Actions is disabled. New files must be
staged for file-based hooks to include them. Expensive tests can use the inherited
`slow` marker; run those explicitly with `-m slow --no-cov` if any are added.

Test coverage is evidence about contracts and regressions. It is not evidence that an
extraction method finds a transformer circuit or succeeds on a distribution of tasks.

Exception-track tests check addition against an independent integer oracle, carry and
token alignment, uniform private deviations, split separation, per-trigger gates and
local controls. Real isolated searches exercise batched/duplicate query accounting,
over-budget rejection, failures, candidate limits, external registration and tampering.
Oracle-backed export fixtures are synthetic contract tests, not teacher performance.
Published development receipts are hash-checked and their census/search scores recomputed.
