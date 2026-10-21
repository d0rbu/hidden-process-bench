# File reference

[Documentation](../README.md) · [Architecture](architecture.md)

| Path | Owner / purpose |
|---|---|
| `README.md`, `AGENTS.md`, `CLAUDE.md` | Project entry points; Claude pointer inherited from the template |
| `pyproject.toml`, `uv.lock`, `.python-version` | Installable package, locked environment, Python 3.13 |
| `.pre-commit-config.yaml`, `justfile` | Local checks and pre-push setup, with no commit-time gate |
| `.github/rulesets/default-branch.json` | PR-only policy with no hosted status requirement |
| `.gitignore`, `.editorconfig`, `LICENSE` | Artifact exclusions, editor defaults, inherited MIT license |
| `src/hidden_process_bench/__init__.py` | Research package |
| `src/hidden_process_bench/config.py` | Validated shapes, budgets, and TOML settings |
| `src/hidden_process_bench/process.py` | HMM sampling, exact oracle, JSON serialization |
| `src/hidden_process_bench/model.py` | Causal transformer, checkpoint and observation API |
| `src/hidden_process_bench/recovery.py` | Three observation-tier recovery objectives |
| `src/hidden_process_bench/metrics.py` | Permutations, KL, qualification, rank checks, probes |
| `src/hidden_process_bench/experiment.py` | Training, split seeds, orchestration, provenance |
| `src/hidden_process_bench/cli.py` | `run`, `extract`, and `score` commands |
| `configs/demo.toml`, `configs/smoke.toml` | Explicit bounded experiment configurations |
| `tests/conftest.py` | CPU thread bound for tests |
| `tests/test_process.py`, `tests/test_metrics.py` | Oracle enumeration, invariants, scoring and negative controls |
| `tests/test_model.py`, `tests/test_recovery.py` | Causality, checkpoints, gradients, information-tier recovery |
| `tests/test_config.py`, `tests/test_experiment.py` | Boundary validation and real end-to-end workflows |
| `tests/test_correctness_tools.py` | Inherited runtime/phantom/array examples |
| `scripts/check_markdown_links.py`, `scripts/__init__.py` | Offline documentation validation |
| `tests/test_markdown_links.py` | Markdown parser and Git file-discovery regressions |
| `docs/research/` | Design, assumptions, limitations, literature |
| `docs/results/cpu-demo.md` | Recorded CPU example, comparison and limits |
| `docs/results/cpu-demo.json`, `docs/results/cpu-demo-provenance.json` | Unmodified report and source/software receipts from the documented run |
| `docs/onboarding/`, `docs/development/` | Setup, workflows, vocabulary, correctness and testing |
| `docs/pipelines/experiment-lifecycle.md` | How to run and interpret an experiment |
| `docs/reference/` | Architecture, artifact schema, configuration and this file map |

Generated runs live under ignored `outputs/`. See the [artifact contract](artifacts.md)
for their layout. No generated training corpus or checkpoint is committed to Git.
