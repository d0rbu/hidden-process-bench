# First observable-operator attempt

[Documentation](../README.md) · [Method](../research/observable-operators.md) · [Full scores and assumptions](operator-development/summary.json) · [File hashes](operator-development/manifest.json)

**This attempt did not recover the generating mechanisms.** It supplied its own
executable representation and selected its own dimension, without an HMM constructor,
family label, hidden-state count or fitting adapter. However, its built-in linear
operator representation is a restricted prior; it did not independently choose a
generating family. The broader five-method discovery evaluation remains unfinished.

The same fixed method ran on all five qualified development teachers: two HMM mixtures,
two delayed rules and one stack rule. The sixth teacher remains unqualified and was
not exported. This is development evidence, not a ready final campaign or a method
ranking. All five submissions were registered before any private scoring.

## Results

KL in nats/token; lower is better. Each row uses 128 private sequences at length 24
and a separate 128 at length 72. Teacher KL here is measured on the extraction-scoring
split, not the separate teacher-qualification test.

| Process, disclosed only after registration | Selected dimension | Oracle → teacher | Oracle → submission | Teacher → submission | Oracle → submission, length 72 |
|---|---:|---:|---:|---:|---:|
| Delayed rule A | 1 | 0.02390 | 1.16650 | 1.17997 | 1.27853 |
| Delayed rule B | 1 | 0.02078 | 1.42893 | 1.41117 | 1.67614 |
| HMM mixture A | 1 | 0.02302 | 0.20239 | 0.17109 | 0.23581 |
| HMM mixture B | 1 | 0.02280 | 0.49362 | 0.46821 | 0.46911 |
| Stack rule | 2 | 0.00846 | 0.06259 | 0.05323 | 0.09009 |

The four dimension-one submissions are IID predictors: their positive scalar operators
normalize back to the same state after every token. They miss the history dependence
of the HMMs and delayed rules. The stack submission proposes a two-dimensional repaired
linear update, without recovering stack contents or push/pop rules. On held-out histories,
the largest oracle/submission total-variation discrepancy is 0.384–0.764 across the
five cases. These provide numerical counterexamples to predictive equivalence; they
are not formal minimality certificates or causal evidence.

The stack candidate repaired 80 of 3,072 short-history predictions and reset state twice;
at length 72 it repaired 240 of 9,216 predictions and reset once. The selected IID
candidates needed no repairs. Larger dimensions often produced unstable signed
predictions during public validation; all 16 dimension trials per task are preserved.
Do not interpret the chosen dimensions as estimates of the true minimal generators.

## Information and receipts

Each extraction saw exactly one anonymous public packet with 128 generated histories
of length 24 and neural next-token probabilities. It had NumPy and the standard library,
120 seconds wall time, 100 seconds CPU and 1 GB memory. It received no weights,
activations, private process source, oracle probabilities, task seeds or scorer feedback.
The 96/32 public sequence split selected dimension; all public data then fit the frozen
submission. No neural executions were added. No GPU was rented.

The [summary](operator-development/summary.json) binds the private plan, complete
registration roster and exact method source hashes. Each case directory contains its
frozen executable, registration receipt, full dimension-selection log, short and long
score receipts, and separate structural assessment in the summary. Private task IDs,
truth parameters, seeds and checkpoint mappings are not published. Full inputs,
predictions, per-request runtime logs and registration timestamps remain in the local
`outputs/operator-development-v1/` evidence directory. Signed coordinates and equivalent
representations are allowed; none of these submissions established an equivalent process.

The orchestration script initially used the wrong teacher qualification field after
scoring the first case. Its traceback is retained. Resumption reused all five frozen
submissions and both completed score receipts; extraction and fitting were not repeated.

A separate research-agent preflight captured outgoing requests to a local fake endpoint.
Both tested clients exposed additional tools despite disabling flags. No discovery
model call or agent trial was run, and this is not counted as an agent failure on the
benchmark. Fresh-context agent evaluation and the other interpretation methods remain
required work, alongside a fully qualified fresh final suite and structural review.
