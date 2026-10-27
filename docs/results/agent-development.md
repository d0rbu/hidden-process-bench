# Family-neutral agent: development results

[Primary protocol](../research/independent-discovery.md) · [Agent method](../research/research-agent.md) · [Receipts](agent-development/summary.json)

One of five fresh-context agent runs submitted an executable process. Four exceeded
the 180-second model-call limit. **No independent structure recovery was demonstrated.**
These are development results; a fresh final campaign and the other method families
remain unfinished.

Each agent received 128 model-generated sequences of length 24 and their next-token
probabilities. It received no weights, process-family hint, state count, generator code,
HMM constructor, fitting adapter or scorer feedback. It chose its own mechanism and
code. All five runs terminated and the roster was registered before private scoring.
The cases are the same five qualified teachers used in the earlier
[operator attempt](operator-development.md); the sixth planned teacher was unqualified.

| Process, revealed after registration | Transformer KL | Program KL | Program KL, length 72 | Outcome |
|---|---:|---:|---:|---|
| Delayed rule A | 0.02390 | — | — | Call timeout |
| Delayed rule B | 0.02078 | — | — | Call timeout |
| HMM mixture A | 0.02302 | — | — | Call timeout |
| HMM mixture B | 0.02280 | 0.01354 | 0.01332 | Six-regime filter submitted |
| Stack rule | 0.00846 | — | — | Call timeout |

KL is measured against the exact generating process, in nats/token; lower is better.
Each split has 128 independent sequences. The teacher's context is 24, so no teacher
score is reported for length 72. Missing programs have no prediction score. A timeout
is an operational failure, not evidence that an intermediate hypothesis was wrong.

## What the submitted program recovered

The agent independently selected six persistent predictive regimes, with an approximately
two-block transition graph. The private generator has two permanently separate
three-state HMM components. That resemblance is useful partial evidence.

The submitted program deliberately separates evidence likelihoods from predictive
profiles, then adds last-token and length corrections and retains only 23 tokens.
It does not claim to be a literal HMM. Its transition graph has small links between
the two groups, whereas the generating mixture never switches components. Correct
regime count alone does not identify the latent states or transition/emission rules.

An equivalent representation would be accepted, but this program has no equivalence
certificate. Its largest observed total-variation discrepancies are 0.2415 at length
24 and 0.4376 at length 72. Frozen examples appear in
[the short-history witness](agent-development/case-04/heldout-witness.json) and
[the long-history witness](agent-development/case-04/long-witness.json).
These are post-registration diagnostic examples, not all-history error bounds.
There is no minimality certificate or causal intervention evidence.

The program matches the transformer at KL 0.01038 on the private short split.
Its lower average error against the source process than the transformer's error
can occur when a fitted process smooths neural approximation errors. This does not
establish structural or circuit recovery. The executable, coefficients and the
agent's own qualifications are preserved in
[its frozen submission](agent-development/case-04/mechanism.md).

## Execution and reproducibility

Protocol version 1 used `gpt-6.1-sol` with `xhigh` reasoning, at most 20 actions,
1,800 seconds per trial, and 180 seconds per model call. Numerical actions had
120 seconds wall time, 100 seconds process CPU, 1 GB RAM and two CPU cores. Two trials
ran concurrently. The declared observation budget was identical across cases;
compute was not matched to the classical baseline.

There were 83 attempted model calls. The summary records completed-call token usage
separately: timed-out calls do not emit terminal usage, so the totals are lower bounds.
No intermediate files from failed runs were rescued or scored, including a JSON
`run` action emitted by one call that never completed. The submitted program uses
only the standard library and its frozen coefficients; it does not call the teacher.

[Method source](agent-development/method.zip), per-case action files, trajectory
archives, exact score receipts, input/source hashes and structural reviews are bound
by the [manifest](agent-development/manifest.json). The source archive includes the
controller's package snapshot; it was not exposed to the evaluated agent. Trajectory
archives contain model prompts, accepted actions and numerical stdout/stderr, not
authentication or the evaluator's process files. Full raw client logs remain local.

Protocol version 2 now shows the remaining budget, encourages an early executable,
and allows 600 seconds per model call within the unchanged total budget. It also
provides a generic neural execution helper for internal-access trials. Those changes
do not alter or repair these version-1 results. Paired access trials, broader method
coverage, and a new fully qualified final suite remain required.
