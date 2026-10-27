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
| `src/hidden_process_bench/neural_numpy.py`, `tests/test_neural_numpy.py`, `docs/reference/neural-execution.md` | Standalone public-neural execution, counted layer traces, head/MLP interventions and PyTorch parity |
| `src/hidden_process_bench/generators.py` | Evaluator-only shared process interface, HMM adapter, delayed and stack oracles |
| `src/hidden_process_bench/mixed_suite.py` | Private balanced plans, archived source, pre-training screens, bounded matched training and neutral exports |
| `tests/test_generators.py`, `tests/test_mixed_suite.py` | Independent oracle checks, source/seed boundaries, real child workflows and failure retention |
| `configs/discovery-smoke.toml`, `configs/discovery-pilot.toml`, `docs/research/mixed-suite.md` | Matched suite settings and private campaign protocol |
| `src/hidden_process_bench/challenge.py` | Neutral neural snapshots and method instructions, with no generator metadata |
| `src/hidden_process_bench/submissions.py` | Family-neutral executable freezing, hashes and external registration |
| `src/hidden_process_bench/isolation.py`, `src/hidden_process_bench/isolated_worker.py` | Linux namespace/cgroup execution, bounded temporary files and retained failures |
| `src/hidden_process_bench/discovery.py`, `src/hidden_process_bench/discovery_cli.py` | Isolated extraction, generic prediction and private scoring commands |
| `tests/test_challenge.py`, `tests/test_submissions.py`, `tests/test_isolation.py`, `tests/test_discovery.py`, `tests/test_discovery_cli.py` | Declared access, tamper rejection, real sandbox isolation and neutral end-to-end scoring |
| `docs/reference/discovery-runtime.md` | Executable discovery interface, assumptions and limits |
| `src/hidden_process_bench/agent_client.py`, `src/hidden_process_bench/agent_preflight.py`, `src/hidden_process_bench/research_agent.py` | Fresh neutral client, captured-context audit, iterative numerical actions and frozen agent submission |
| `tests/test_agent_client.py`, `tests/test_agent_preflight.py`, `tests/test_research_agent.py`, `tests/fixtures/agent-request.json` | Real client namespace checks, context injection rejection, workspace rollback, budgets and retained failures |
| `docs/results/agent-development.md`, `docs/results/agent-development/`, `tests/test_agent_receipts.py` | Five-case family-neutral agent results, preserved timeouts, structural review and bound source/submission/trajectory receipts |
| `docs/research/research-agent.md` | Family-choosing agent protocol, access boundary and local command |
| `src/hidden_process_bench/operator_method.py`, `src/hidden_process_bench/operator_predictor.py`, `tests/test_operator_method.py` | Unknown-dimension operator baseline, standalone execution, exact moment and coordinate-equivalence checks |
| `src/hidden_process_bench/state_geometry.py`, `src/hidden_process_bench/operator_structure.py`, `tests/test_state_geometry.py`, `docs/research/state-geometry.md` | Unsupervised state coordinates, autonomous token-update regression, restricted algebraic family recognition and isolated execution |
| `src/hidden_process_bench/discovery_features.py`, `tests/test_discovery_features.py`, `docs/research/discovery-features.md` | Sparse autoencoder/transcoder and circuit-pruned feature variants feeding autonomous dynamics, with counted neural execution |
| `docs/research/observable-operators.md`, `docs/results/operator-development.md`, `docs/results/operator-development/` | Restricted identification prior, five-case negative development results and frozen source/submission receipts |
| `src/hidden_process_bench/disclosure.py` | Prior disclosures and explicit informed-control scope for new artifacts |
| `docs/research/independent-discovery.md` | Primary discovery objective, neutral inputs/submissions and remaining experiment work |
| `docs/results/mixed-suite-development.md`, `docs/results/mixed-suite-development.json` | Aggregate teacher-only pilot; failed preparation and unqualified stack case retained, task mappings private |
| `docs/results/disclosure-correction.json` | Scope correction for unchanged historical informed-control receipts |
| `src/hidden_process_bench/recovery.py` | Three observation-tier recovery objectives |
| `src/hidden_process_bench/classical.py` | Scaled mixture Baum–Welch and spectral observable operators |
| `src/hidden_process_bench/representations.py` | Counted layer traces, PCA/SAE/transcoder fits, observable belief updates |
| `src/hidden_process_bench/circuits.py` | Greedy head/MLP ablation and candidate-belief interchange |
| `src/hidden_process_bench/method_study.py` | Public-only extraction submissions followed by evaluator-only scoring |
| `tests/test_classical.py`, `tests/test_representations.py`, `tests/test_circuits.py`, `tests/test_method_study.py` | Exact EM path checks, synthetic dynamics, interventions and study boundaries |
| `docs/research/method-study.md` | Five-family pilot protocol, access and adaptation limits |
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
| `docs/results/five-method-pilot.md`, `docs/results/five-method-pilot/` | Four-checkpoint, five-family comparison, fixed candidates, agent source, hash manifests and raw receipts |
| `tests/test_method_receipts.py` | Receipt hashes, candidate validity, summary and scoring consistency |
| `docs/results/cpu-demo.md` | Historical unscreened CPU example, comparison and limits |
| `docs/results/cpu-demo.json`, `docs/results/cpu-demo-provenance.json` | Unmodified report and source/software receipts from the documented run |
| `docs/results/screened-cpu-demo.md`, `docs/results/screened-cpu-demo/` | Current screened demo and unmodified report/config/provenance/sampling receipts |
| `docs/results/scaling-pilot.md`, `docs/results/scaling-pilot/` | Four-case process/model-size results, frozen plan, statuses, and full receipts |
| `docs/onboarding/`, `docs/development/` | Setup, workflows, vocabulary, correctness and testing |
| `docs/pipelines/experiment-lifecycle.md` | How to run and interpret an experiment |
| `docs/reference/` | Architecture, artifact schema, configuration and this file map |

Generated runs live under ignored `outputs/`. See the [artifact contract](artifacts.md)
for their layout. No generated training corpus or checkpoint is committed to Git.
