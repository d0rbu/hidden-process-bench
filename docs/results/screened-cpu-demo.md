# Screened CPU prototype example

[Documentation](../README.md) · [Design](../research/design.md) · [Scaling pilot](scaling-pilot.md)

Measured on 2026-10-21 UTC from clean source
[`10a68ed`](https://github.com/d0rbu/hidden-process-bench/commit/10a68ed0a5d354a1cfa0a13cb29000454c31f5c2).
The [report](screened-cpu-demo/report.json), [provenance](screened-cpu-demo/provenance.json),
[resolved config](screened-cpu-demo/config.json), and [complete sampling audit](screened-cpu-demo/sampling.jsonl)
are unmodified receipts. All recorded source hashes were checked against the source files.
This is one public development task, not a blind benchmark study.

```bash
uv sync --locked
uv run --locked hidden-process-bench run \
  --config configs/demo.toml --out outputs/reproduce-screened-demo --require-qualified
```

The sampler rejected four candidates and accepted the fifth before training. Each of
its four components passed the preset occupancy and temporal-information checks:

| Component | Minimum stationary mass | Unigram gain (nats/token) | Trigram gain (nats/token) |
|---|---:|---:|---:|
| 0 | 0.129660 | 0.112820 | 0.010999 |
| 1 | 0.254433 | 0.079123 | 0.009156 |
| 2 | 0.085030 | 0.117700 | 0.014676 |
| 3 | 0.082999 | 0.086938 | 0.002508 |

The [historical unscreened example](cpu-demo.md) used the rejected first candidate.
Its results are preserved separately and are not pooled with this run.

## Prediction and recovery

The 59,376-parameter transformer stopped after 1,200 steps / 1,843,200 tokens.
Validation and a distinct 256-sequence test set both passed the preset 0.03 nats/token
gate, including every component group. Test mean KL was **0.019926**, aggregate upper
bound 0.021539, and largest component upper bound 0.023859. These normal-approximation
bounds are diagnostic, not simultaneous finite-sample guarantees.

All recovery methods used the same 128 model-generated sequences of length 24:
3,072 token queries per method, two restarts, and 250 optimizer steps per restart.

| Method | Aligned parameter MAE ↓ | Held-out predictive KL ↓ |
|---|---:|---:|
| samples | 0.096522 | 0.089592 |
| probabilities | 0.031457 | 0.012912 |
| activations | 0.043963 | 0.016446 |

The probability-only baseline performed better than the activation-assisted baseline
on both measures in this example. None exactly recovered the process; the equivalence
diagnostic found distinguishing strings for all three. This is evidence to retain,
not a reason to select a different generator or tune against the held-out test.
It does not establish a general ranking of access levels.

The complete run took 14.7 seconds locally with two PyTorch CPU threads, Python 3.13.2,
NumPy 2.5.0, and PyTorch 2.14.1+cpu. This single timing is not a throughput claim.
No GPU or paid provider was used. Generated checkpoints and observations remain outside
Git; the raw report records the checkpoint hash. Supervised belief probes are labeled
separately and must not be interpreted as extraction or causal evidence.
