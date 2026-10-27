# Getting started

[Documentation](../README.md) · [Workflows](workflows.md)

Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.8.22 or
newer. `uv` provisions Python 3.13 if needed.

```bash
uv sync --locked
uv run --locked just setup
uv run --locked hidden-process-bench run --config configs/smoke.toml --out outputs/smoke
```

`just setup` installs a **pre-push** hook. It does not install a commit hook. If reusing
a clone that already has the template's managed commit hook, remove that old hook with
`uv run --locked pre-commit uninstall --hook-type pre-commit`.

The command above exercises the historical informed HMM-fitting control. For the primary
independent-discovery task, follow the [mixed-suite quickstart](../../README.md#run-independent-discovery-locally),
then the [neutral submission interface](../reference/discovery-runtime.md). Its methods
choose their own mechanisms without a supplied family, template or state count.

The smoke run should finish with `benchmark_eligible: false`: it exercises plumbing
without pretending two training steps establish optimality. The [demo config](../../configs/demo.toml)
is a larger, still bounded CPU experiment. Every run needs a new output directory.
See the [artifact contract](../reference/artifacts.md) before sharing run outputs.

Common commands:

```bash
uv run --locked just
uv run --locked just fmt
uv run --locked just test --no-cov -k filter
uv run --locked just typecheck
uv run --locked just docs-check
```

Before pushing a PR branch, run `uv run --locked just check`. No GitHub Actions run is
expected. CPU-only PyTorch is selected for Linux/Windows; macOS uses its PyPI wheel.
The checked reference platform is Linux CPU. No cloud credentials or GPU are needed.
