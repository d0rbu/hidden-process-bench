# Workflows

[Documentation](../README.md) · [Testing](../development/testing.md)

## Implement a change

Inspect Git status and the relevant contract. Keep reusable code in `src/hidden_process_bench/`
and orchestration in `experiment.py`/`cli.py`. Write tests for scientific invariants and
external behavior, then update the owning docs and [file map](../reference/file-reference.md).

During iteration, run targeted checks such as:

```bash
uv run --locked just fmt
uv run --locked just test --no-cov -k permutation
uv run --locked just typecheck
```

Ordinary `git commit` does not run pre-commit. The full gate belongs immediately before
pushing a PR branch:

```bash
git add <changed-paths>
uv run --locked just check
git push -u origin <branch>
gh pr create
```

An installed pre-push hook runs the gate on push automatically. File-based hooks select
the pushed diff; `just check` explicitly checks all tracked files. A manual check is
useful before the first push or when hooks have not yet been installed. Do not run the
full gate repeatedly between commits. Review any whitespace fixes before pushing.

The [PR-only ruleset](../../.github/rulesets/default-branch.json) has no CI requirement.
Merge after local checks pass and review threads are resolved. No human approval is
required by the default policy. Local test results are not a hosted GitHub status.

## Dependencies

Use `uv add <package>`, `uv add --dev <tool>`, or an intentional `uv lock` after editing
metadata. Review both `pyproject.toml` and `uv.lock`, then `uv sync --locked`. Dependency
maintenance is local; there is no hosted dependency-update workflow.

## New recovery method

Start from `fit_hmm`'s information boundary. Read only the permitted observation tier,
return a validated `MixtureHMM`, and let the evaluator score it. Compare against the
output-only baselines on identical cached histories and declared budgets. Select
restarts using the method's objective, never hidden ground-truth scores. See the
[design](../research/design.md) before making an interpretability claim.
