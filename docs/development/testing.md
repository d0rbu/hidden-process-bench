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
test equivalence. Sweep tests exercise a real child process and timeout, crash, malformed
receipt, missing-report, and qualification-status paths.

Tests use two CPU threads, no network, no paid providers, and no GPUs. The measured
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
