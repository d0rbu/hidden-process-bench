# Process-size and model-size pilot

[Documentation](../README.md) · [Scaling protocol](../research/scaling.md) · [Screened demo](screened-cpu-demo.md)

Four CPU runs completed on 2026-10-21 UTC from clean source
[`d9a0d4c`](https://github.com/d0rbu/hidden-process-bench/commit/d9a0d4c64a76c918bf90ff1375d0059a8ba0be4b).
The [frozen plan](scaling-pilot/plan.json), [summary](scaling-pilot/summary.json),
[CSV](scaling-pilot/cases.csv), and [JSONL](scaling-pilot/cases.jsonl) retain every case.
Individual receipts are listed below. Source hashes and identical ground-truth bytes
within each width comparison were verified.

These are reruns of the same seeds used by
[`10a68ed`](https://github.com/d0rbu/hidden-process-bench/commit/10a68ed0a5d354a1cfa0a13cb29000454c31f5c2),
now with the minimal-order filter enabled. All four selected ground truths and
checkpoints were byte-identical to their earlier counterparts, and prediction/recovery
metrics matched exactly. These reruns are not independent replications.

```bash
uv run --locked hidden-process-bench sweep \
  --config configs/scaling-pilot.toml --out outputs/reproduce-scaling --timeout 300
```

The command exits **2** because three checkpoints are unqualified. This is expected
from this recorded run and does not mean its artifacts are missing. There were no
execution failures or timeouts. The four child runs took about 93 seconds in total;
this is a local timing, not a throughput guarantee.

## Prediction qualification

Every case uses four HMMs, 24 output symbols, context length 24, two transformer layers,
generator seed 7, independent model seed 8, 128 extraction sequences, and the same
2,000-step training cap. The sampling screen accepted candidate 5 for three-state HMMs
and candidate 57 for six-state HMMs. Model size does not affect which process is selected.
No caps, thresholds, or generators were changed after seeing these outcomes.

| States per HMM | Observable rank | Width | Parameters | Steps | Test mean KL | Largest validation upper bound | Largest test upper bound | Qualified |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | 12 | 32 | 27,808 | 2,000 | 0.024371 | 0.034394 | 0.029531 | no |
| 3 | 12 | 64 | 104,768 | 1,700 | 0.019728 | 0.027480 | 0.027550 | yes |
| 6 | 24 | 32 | 27,808 | 2,000 | 0.039026 | 0.063517 | 0.047102 | no |
| 6 | 24 | 64 | 104,768 | 2,000 | 0.028719 | 0.044683 | 0.041762 | no |

Upper bounds are componentwise mean KL + 1.96 sequence-level standard errors;
qualification also requires the aggregate bound and checks both validation and test.
In particular, the first case is unqualified even though its final test passes:
its stopping validation did not pass before the cap. We do not select using that final test.
The observable rank is the short-Hankel numerical diagnostic, with tolerance 1e-10.
It matches the joint state count in both tasks, passing the sufficient minimal-order
screen; redundant state copies cannot create a larger accepted task.

The wider model had lower test KL on both matched processes. Only one checkpoint
qualified, and there are only two process shapes, each with one generator
seed and one model seed. This pilot does not identify a scaling law, a capacity boundary,
a simplicity–recoverability correlation, or a bias toward minimal equivalent automata.
Insufficient training is a competing explanation for the failures.

## Recovery and receipts

For the sole qualified checkpoint (three states per HMM, width 64):

| Method | Aligned parameter MAE ↓ | Held-out predictive KL ↓ |
|---|---:|---:|
| samples | 0.107447 | 0.109650 |
| probabilities | 0.050510 | 0.021020 |
| activations | 0.051498 | 0.035352 |

Unqualified checkpoints still have diagnostic recovery scores in the CSV and raw
reports; they are not pooled into a qualified recovery comparison. Parameter MAE
depends on matrix dimensions, so compare it within a fixed process shape. Predictive
KL is the primary cross-shape measure here. Query counts do not equalize information
bandwidth or compute, and fixed-order extractors do not test minimal-order discovery.

| Case | Report | Provenance | All sampling attempts |
|---|---|---|---|
| 000: 3 states, width 32 | [Report](scaling-pilot/case-000-report.json) | [Source/software](scaling-pilot/case-000-provenance.json) | [Audit](scaling-pilot/case-000-sampling.jsonl) |
| 001: 3 states, width 64 | [Report](scaling-pilot/case-001-report.json) | [Source/software](scaling-pilot/case-001-provenance.json) | [Audit](scaling-pilot/case-001-sampling.jsonl) |
| 002: 6 states, width 32 | [Report](scaling-pilot/case-002-report.json) | [Source/software](scaling-pilot/case-002-provenance.json) | [Audit](scaling-pilot/case-002-sampling.jsonl) |
| 003: 6 states, width 64 | [Report](scaling-pilot/case-003-report.json) | [Source/software](scaling-pilot/case-003-provenance.json) | [Audit](scaling-pilot/case-003-sampling.jsonl) |

All linked JSON, JSONL, and CSV receipts are copied unchanged from the run. The larger
24-run preset is supplied for follow-up; it was **not run** as part of this pilot.
