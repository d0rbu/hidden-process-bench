# Scaling and equivalent processes

[Documentation](../README.md) · [Design](design.md) · [Artifacts](../reference/artifacts.md)

There are two separate hypotheses in the motivating discussion:

1. More model capacity permits learning and recovering processes with higher observable
   complexity, given adequate training and extraction budgets.
2. Among equivalent process representations, a transformer favors a minimal representation.

Neither follows from larger nominal state count. Splitting a hidden state into identical
copies preserves every finite-string probability. No observation-only method can identify
which equivalent description was used as ground truth. Model internals may use an efficient
representation, a redundant one, or neither; output agreement alone does not establish that.

## Executable sweeps

```bash
uv run --locked hidden-process-bench sweep --config configs/scaling.toml --dry-run
uv run --locked hidden-process-bench sweep \
  --config configs/scaling-pilot.toml --out outputs/scaling-pilot --timeout 300
```

The pilot crosses 3/6 states per component with transformer widths 32/64: four runs.
The larger preset crosses 3/4/6 states, widths 32/64, 64/256 extraction sequences, and
generator seeds 7/19: 24 runs. Both keep four components, vocabulary 24, sequence length
24, two layers, and the 0.03 nats/token qualification threshold fixed. Full configurations
are resolved and saved before execution. The same shape/seed selects the same process
across width/query variants. Model initialization uses its independent configured seed.

`--max-runs` defaults to 32 and is checked before outputs are created. `--timeout` is a
per-run subprocess limit (default 300 seconds), not a throughput estimate. The plan reports
the maximum total training-token budget and subprocess-time budget. Child logs, sampling
audits, failed runs, and partial timed-out artifacts are retained. `cases.jsonl` is flushed
after each case; `cases.csv` and `summary.json` are written on completion. A sweep exits 2
if any case is unqualified, failed, or timed out. Inspect statuses before pooling metrics.

The implementation runs each Cartesian point independently, including repeated training
across query-budget variants. This is simple and reproducible but does not reuse checkpoints.
Dimensions currently stay within the six-state/six-component exact parameter-scoring limit.
This is a CPU prototype, not a demonstrated large-model training stack.

## What to compare

Report checkpoint qualification rate before extraction accuracy. Compare recovery on
qualified checkpoints at matched process, model seed, and observation budget; preserve
unqualified rows as failures of the complete pipeline. Conditioning on qualification can
bias a cross-size comparison, so publish both denominators. Vary training tokens separately
to distinguish an optimization shortfall from an insufficient-capacity hypothesis.

Process complexity has several recorded measures:

| Measure | Interpretation and limitation |
|---|---|
| Joint hidden states | Size of the supplied representation; changes under redundant splitting |
| Dense-family degrees of freedom | Parameter count of the unconstrained mixture family; not the stationary generator's degrees of freedom |
| Transition edges | Structural sparsity of the supplied representation |
| Effective joint states | exp(entropy of initial component/state mass); also representation-dependent |
| Short Hankel rank | Observable linear-complexity lower bound from empty/one-token prefix-suffix probabilities, tolerance 1e-10 |
| Unigram/trigram gains | Componentwise temporal information at the declared context lengths |

A truncated Hankel rank can underestimate complexity. It is not a general proof of
minimal HMM order. Plot recovery against several measures rather than naming one scalar
"simplicity". Use multiple independent generating processes per stratum; do not fit a
correlation or scaling law to the four-point development pilot.

## Equivalent-automaton control

The library exposes an equivalent state-splitting construction:

```python
from pathlib import Path

from hidden_process_bench.equivalence import compare_processes, split_states
from hidden_process_bench.process import MixtureHMM

original = MixtureHMM.load(Path("outputs/demo/private/truth.json"))
redundant = split_states(original, copies=2)
diagnostic = compare_processes(original, redundant)
```

The construction must stay within the supported state count. Tests verify equivalent predictions across
different model orders and distinguish non-equivalent processes with identical unigrams.

The current extraction methods receive a fixed model order. They cannot establish
automatic minimal-order discovery or a transformer preference for minimal automata.
Testing that hypothesis requires order search and held-out predictive validation, or a
separate representation/intervention analysis. Keep any explicit simplicity penalty in
the extractor distinct from a bias attributed to the transformer itself.
