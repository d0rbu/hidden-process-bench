# Configuration

[Documentation](../README.md) · [Setup](../onboarding/getting-started.md)

[`pyproject.toml`](../../pyproject.toml) owns Python/tool configuration and
[`uv.lock`](../../uv.lock) pins dependencies. Python, Ruff, and ty target 3.13.
Hatchling installs `hidden_process_bench` from `src/`; the executable is
`hidden-process-bench`. Torch comes from an explicit CPU wheel index on Linux/Windows
and PyPI on macOS. This prototype executes on CPU.

## Local gate and hooks

[`.pre-commit-config.yaml`](../../.pre-commit-config.yaml) sets both
`default_install_hook_types` and `default_stages` to `pre-push`.

```bash
uv run --locked pre-commit install --hook-type pre-push
uv run --locked pre-commit run --hook-stage pre-push --all-files
```

The second command is also `uv run --locked just check`. No pre-commit hook is installed
at commit time. The gate includes lock validation, Ruff lint and formatting, ty, Vulture,
pytest with coverage, local Markdown links, justfile syntax, JSON/YAML/TOML checks,
merge markers, large files, and whitespace. Actionlint was removed with the CI workflows.

The Markdown checker is offline and checks local files/images, not remote URLs or
heading anchors. Vulture's known false positives are individually named in `ignore_names`:
TypedDict keys are accessed as strings, and the `cpu_threads` fixture is registered via
pytest's `autouse`. No production module is excluded from lint, types, or coverage.

## Recipes

| Recipe | Purpose |
|---|---|
| `setup` | Locked install and pre-push hook |
| `check` | Full local gate before PR pushes |
| `lint`, `format-check`, `fmt` | Lint, check format, or safely fix/format |
| `typecheck`, `deadcode` | ty and Vulture |
| `test`, `coverage` | pytest, optionally HTML coverage |
| `lock-check`, `docs-check` | Dependency and local link checks |
| `site`, `site-data`, `site-check` | Static website preview, reproducible snapshot, Node.js 22+ syntax/math checks |

## Experiments

[`configs/demo.toml`](../../configs/demo.toml) declares all model dimensions, seeds,
training/query/optimization budgets, and the KL tolerance. [`configs/smoke.toml`](../../configs/smoke.toml)
exercises all stages with intentionally inadequate training. Unknown config keys and
invalid counts, probabilities, dimensions, or combinations fail at the boundary.

The nested `[sampling]` table owns optional-edge probability, emission concentration,
minimum stationary occupancy, minimum unigram/trigram gains in nats/token, and the
attempt cap. [Design](../research/design.md) defines the exact screen. A run requires at
least four tokens so its history includes the screen's three-token context.
The minimal-order screen also requires `vocab >= components * states` and rejects any
candidate whose short Hankel rank does not reach that joint state count. These constraints
apply to generated benchmark tasks; the HMM library and standalone scorer still accept
redundant representations for equivalence controls.

[`configs/scaling-pilot.toml`](../../configs/scaling-pilot.toml) and
[`configs/scaling.toml`](../../configs/scaling.toml) select a base TOML relative to their
own directory and Cartesian grid axes. Supported axes are generator/model seed, states,
components, vocabulary, width, layers, extraction sequences, and training steps. Counts
and each resolved `RunConfig` are validated before execution. The run-count and per-child
time limits are CLI options; use `sweep --dry-run` to inspect their total budgets.

## GitHub policy

GitHub Actions is disabled in repository settings. There are no `.github/workflows`
or Dependabot files. The [portable ruleset](../../.github/rulesets/default-branch.json)
requires a pull request and resolved conversations, with zero required review approvals.
It has **no required status checks**. Applying its JSON is a separate repository setting:

```bash
gh api --method POST repos/d0rbu/hidden-process-bench/rulesets \
  --input .github/rulesets/default-branch.json
gh api repos/d0rbu/hidden-process-bench/rules/branches/main
```

Inspect existing rulesets before applying; update an existing ID instead of duplicating
it. A copied template file does not itself enforce GitHub settings.
