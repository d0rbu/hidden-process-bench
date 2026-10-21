# AGENTS.md — Hidden Process Bench

This is a correctness-first Python research prototype for recovering known hidden
processes from trained transformers. Keep the implementation small and explicit.

## Read first

- [README](README.md) and [documentation map](docs/README.md)
- [Research design](docs/research/design.md): hypotheses, information access, metrics
- [Artifact contract](docs/reference/artifacts.md): what extractors may read
- [Workflows](docs/onboarding/workflows.md) and [testing](docs/development/testing.md)
- [Architecture](docs/reference/architecture.md) and [file map](docs/reference/file-reference.md)

## Local workflow

- Use `uv sync --locked` and `uv run --locked ...`. Python is 3.13.
- Install **only the pre-push hook** with `uv run --locked just setup`.
- Run the full local gate immediately before pushing a PR branch:
  `uv run --locked pre-commit run --hook-stage pre-push --all-files`.
- Do not run pre-commit before each commit. During development, run focused checks
  appropriate to the change. Stage new files before the full gate so file hooks see them.
- There is no GitHub Actions CI, no actionlint, and no required hosted status check.
  Do not reintroduce those. The default-branch policy requires a PR, with zero required
  approvals and resolved review threads; repository settings are applied separately.
- Preserve unrelated changes and existing experiment outputs. Use a fresh run directory.
- Keep `pyproject.toml` and `uv.lock` together. Use `uv add`/`uv lock`, never edit the lock by hand.

## Research correctness

- Extractors must not receive true HMMs, latent labels, generating seeds, or scorer feedback.
  Supervised probes are diagnostics, never extraction results.
- Keep training, stopping validation, final test, model generation, and probe RNG streams separate.
  Model/observation/recovery seeds must not reveal the private generator seed through an offset.
- Match query budgets and report access levels. Query equality is not equal compute or bandwidth.
- Qualify against the joint Bayesian oracle and check every generating component.
  Do not relabel an undertrained checkpoint as a successful benchmark.
- Score state/component permutations and predictive equivalence separately. Do not claim
  identifiability for arbitrary generators or a causal mechanism from a linear probe.
- The activation baseline is not a static weights-only extractor.
- Record config, source/checkpoint hashes, actual steps, and incomplete/failed runs honestly.
- Screen candidate tasks before training, retain all rejected candidates, and never select
  generators using extraction scores. Numerical equivalence is not an exact certificate.
- Separate raw state count from observable complexity. Scaling summaries must retain
  unqualified checkpoints, errors, and timeouts; fixed-order fitting does not test minimality bias.
- Defaults use two PyTorch CPU threads and bounded steps. Broader training and provider rentals
  need a concrete research reason and must follow the user's shared spending policy.

## Code and validation

Use typed boundaries, phantom types for scalar invariants, jaxtyping/beartype for array
contracts, explicit semantic checks, and bounded Hypothesis tests. Keep imports at the
top and use `@override` on overridden methods. Do not lower the 95% branch coverage gate,
exclude production code, or add broad suppressions to make checks pass. Inspect Vulture
findings; known dynamic schema/fixture names are individually documented in configuration.
Update the owning docs and file map when behavior changes. Report what was actually run.
