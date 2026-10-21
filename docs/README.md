# Documentation

Start with the [project README](../README.md) or [agent guide](../AGENTS.md).

| Area | Documents |
|---|---|
| Research | [Design](research/design.md), [related work](research/related-work.md) |
| Setup | [Getting started](onboarding/getting-started.md), [workflows](onboarding/workflows.md), [glossary](onboarding/glossary.md) |
| Correctness | [Contracts](development/correctness.md), [testing](development/testing.md) |
| Experiments | [Lifecycle](pipelines/experiment-lifecycle.md) |
| Results | [Single-seed CPU example and receipts](results/cpu-demo.md) |
| Reference | [Architecture](reference/architecture.md), [artifacts](reference/artifacts.md), [configuration](reference/configuration.md), [file map](reference/file-reference.md) |

These documents own durable project knowledge. Keep commands runnable from the repo
root. `uv run --locked just docs-check` validates local Markdown file and image links.
