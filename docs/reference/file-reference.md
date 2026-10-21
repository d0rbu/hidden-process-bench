# File reference

[Documentation](../README.md) · [Architecture](architecture.md)

| Path | Owner / purpose |
|---|---|
| `README.md`, `AGENTS.md`, `CLAUDE.md` | Project entry points; Claude pointer inherited from the template |
| `pyproject.toml`, `uv.lock`, `.python-version` | Installable package, locked environment, Python 3.13 |
| `.pre-commit-config.yaml`, `justfile` | Local checks and pre-push setup, with no commit-time gate |
| `.github/rulesets/default-branch.json` | PR-only policy with no hosted status requirement |
| `.gitignore`, `.editorconfig`, `LICENSE` | Artifact exclusions, editor defaults, inherited MIT license |
| `src/hidden_process_bench/__init__.py`, `src/hidden_process_bench/__main__.py` | Research package and subprocess entry point |
| `src/hidden_process_bench/config.py` | Validated shapes, budgets, and TOML settings |
| `src/hidden_process_bench/process.py` | HMM sampling, exact oracle, JSON serialization |
| `src/hidden_process_bench/sampling.py` | Entropy and minimal-order screens, bounded rejection sampling, private audit |
| `src/hidden_process_bench/equivalence.py` | Numerical word-equivalence check, redundant-state controls, complexity |
| `src/hidden_process_bench/model.py` | Causal transformer, checkpoint and observation API |
| `src/hidden_process_bench/recovery.py` | Three observation-tier recovery objectives |
| `src/hidden_process_bench/metrics.py` | Permutations, KL, qualification, rank checks, probes |
| `src/hidden_process_bench/experiment.py` | Training, split seeds, orchestration, provenance |
| `src/hidden_process_bench/scaling.py` | Frozen Cartesian plans, subprocess limits, complete status summaries |
| `src/hidden_process_bench/cli.py` | `run`, `extract`, `score`, and `sweep` commands |
| `configs/demo.toml`, `configs/smoke.toml` | Explicit bounded experiment configurations |
| `configs/scaling-pilot.toml`, `configs/scaling.toml` | Four-run pilot and 24-run process/model/query sweep |
| `tests/conftest.py` | CPU thread bound for tests |
| `tests/test_process.py`, `tests/test_metrics.py` | Oracle enumeration, invariants, scoring and negative controls |
| `tests/test_model.py`, `tests/test_recovery.py` | Causality, checkpoints, gradients, information-tier recovery |
| `tests/test_config.py`, `tests/test_experiment.py` | Boundary validation and real end-to-end workflows |
| `tests/test_sampling.py`, `tests/test_equivalence.py`, `tests/test_scaling.py` | Enumeration, degeneracy/equivalence controls, bounded sweep and failure retention |
| `tests/test_correctness_tools.py` | Inherited runtime/phantom/array examples |
| `scripts/check_markdown_links.py`, `scripts/__init__.py` | Offline documentation validation |
| `tests/test_markdown_links.py` | Markdown parser and Git file-discovery regressions |
| `scripts/build_site_data.py`, `tests/test_site_data.py` | Reproduce public website data, receipt/source validation |
| `site/index.html`, `site/style.css`, `site/app.js`, `site/icon.svg` | Static explainer, interactive HMM diagrams and recorded results |
| `site/process.mjs`, `site/test_process.mjs` | Browser sampling/filtering/equivalence and NumPy-parity tests |
| `site/data.json`, `site/README.md` | Public snapshot with provenance and website operations |
| `docs/research/` | Design, assumptions, limitations, literature |
| `docs/results/cpu-demo.md` | Historical unscreened CPU example, comparison and limits |
| `docs/results/cpu-demo.json`, `docs/results/cpu-demo-provenance.json` | Unmodified report and source/software receipts from the documented run |
| `docs/results/screened-cpu-demo.md`, `docs/results/screened-cpu-demo/` | Current screened demo and unmodified report/config/provenance/sampling receipts |
| `docs/results/scaling-pilot.md`, `docs/results/scaling-pilot/` | Four-case process/model-size results, frozen plan, statuses, and full receipts |
| `docs/onboarding/`, `docs/development/` | Setup, workflows, vocabulary, correctness and testing |
| `docs/pipelines/experiment-lifecycle.md` | How to run and interpret an experiment |
| `docs/reference/` | Architecture, artifact schema, configuration and this file map |

Generated runs live under ignored `outputs/`. See the [artifact contract](artifacts.md)
for their layout. No generated training corpus or checkpoint is committed to Git.
