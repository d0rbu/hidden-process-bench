# CPU prototype example

**Scope correction:** all fits here were given the HMM family and order. These are
informed fitting controls, not independent process-discovery results. Checkpoint
qualification concerns transformer predictions only. See the
[primary protocol](../research/independent-discovery.md). Raw receipts are unchanged.

[Documentation](../README.md) · [Research design](../research/design.md)

**Historical, unscreened task.** This receipt predates the PAutomaC-inspired sampling
screen. Its first component fails the new minimum occupancy and unigram-gain checks.
The current default therefore selects another process. These numbers are retained as
development history and must not be attributed to the current screened generator.

A single development task run on 2026-10-21 UTC, from a clean source tree at
[`f8c2bfd`](https://github.com/d0rbu/hidden-process-bench/commit/f8c2bfd9aaf11cba89e588fbaec7fd1a6b1b2836).
The [raw report](cpu-demo.json) and [source/software provenance](cpu-demo-provenance.json)
are copied unchanged from that run. Source hashes were checked against the committed files.
This is a public, known-seed example, not a held-out benchmark study.

## Reproduce

Use a separate checkout of the source revision linked above; the current `demo.toml`
and sampler have changed. The command below applies to that historical checkout.

```bash
uv sync --locked
uv run --locked hidden-process-bench run \
  --config configs/demo.toml --out outputs/reproduce-cpu-demo --require-qualified
```

The [config](../../configs/demo.toml) uses generator seed 7, independent model seed 8,
observation seed 27, and recovery seed 37. Python was 3.13.2, NumPy
2.5.0, and PyTorch 2.14.1+cpu, with two PyTorch CPU threads.
The measured end-to-end run took 18.9 seconds after dependency setup;
this is one local timing, not a portability or throughput guarantee. No GPU was rented.

## Prediction qualification

Training stopped at 1,100 steps / 1,689,600 tokens,
before the 1,200-step cap. Validation and final test both passed the preset 0.03 nats/token
gate, including all four component groups. The final test had 256 independent sequences.

- Mean held-out transformer KL: **0.019597 nats/token**.
- Aggregate normal-approximation upper bound: 0.021285.
- Largest component upper bound: 0.028046.
- Joint emission rank: 12 of 12; transitions are full rank and starts are stationary.

The same seed was used for preliminary implementation checks; those runs are not
independent replications and are not pooled into this result. No thresholds or optimizer
budgets were selected using the final test to obtain a pass.

## Recovery comparison

All methods used the same 128 generated sequences of length 24, equivalent to 3,072
token queries per method, and two restarts of 250 fitting steps. Counts and timings are
in the raw report. Parameter error averages the four aligned parameter-field MAEs.

| Method | Parameter MAE ↓ | Held-out predictive KL ↓ | Transition-support F1 ↑ |
|---|---:|---:|---:|
| samples | 0.085632 | 0.058166 | 0.821 |
| probabilities | 0.054641 | 0.011232 | 0.900 |
| activations | 0.040181 | 0.011717 | 0.933 |

The activation-assisted fit has lower aggregate parameter error in this example; the
probability-only fit has slightly lower predictive KL. Neither is exact parameter
recovery. This single paired example cannot establish an advantage for internal access,
and query equality does not match information bandwidth, fitting parameters, or compute.
A recovered HMM can predict better than the imperfect teacher because its constrained
process family can smooth teacher errors; the scores above compare each directly to truth.

Supervised belief-probe test R-squared was
0.9311 from probabilities and
0.9271 from residual activations.
These diagnostics use true belief labels and are not extraction methods or causal evidence.

## Artifacts and limits

The checkpoint is generated locally under `outputs/reproduce-cpu-demo/public/` when
reproducing. The recorded checkpoint SHA-256 is
`f964cd956b876da1900ab8f80dbed12bcfab47322a3abd5b822bc7641b53d52d`.
Generated checkpoints, observations, and true parameters stay outside Git. These small
JSON files are selected experiment receipts. Cross-platform byte equality is not promised.

The tiny smoke config remains intentionally unqualified. The implemented internal-access
method executes a checkpoint to read activations; static weights-only reconstruction,
unknown model order, multi-seed performance, and causal intervention tests remain future work.
