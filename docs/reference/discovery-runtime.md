# Executable discovery interface

[Documentation](../README.md) · [Primary protocol](../research/independent-discovery.md)

This runtime lets a method choose and submit its own process representation. It
supplies no generating family, state count, constructor or fitting adapter. The
[mixed private generator suite](../research/mixed-suite.md) now has a training path;
the [first restricted operator attempt](../results/operator-development.md) ran through
this interface and failed to recover the mechanisms. Broader family-discovery method
evaluations remain pending; passing runtime tests is not a recovery result.

## Public inputs

Export a neural checkpoint into a fresh directory:

```bash
uv run --locked python -m hidden_process_bench.discovery_cli export \
  --checkpoint outputs/my-run/public/checkpoint.pt --access outputs \
  --sequences 128 --length 24 --seed 101 --out outputs/challenge
```

This is a development example. Previously published checkpoints are not hidden test
cases. The observation seed controls model sampling only; it must be independent of
the private generating seed.

The snapshot contains `instructions.txt`, `challenge.json`, and only the permitted
arrays. `outputs` exposes cached generated tokens and next-token distributions;
`weights` exposes neural weights only; `internals` exposes those weights and the
same cached tokens, distributions and final normalized residuals. Neural architecture,
BOS alignment and tensor conventions are specified. Generator seeds, type, size,
source, latent labels, qualification diagnostics and evaluator paths are absent.
For paired conditions, reuse the same cached `Observations` through the Python API;
the CLI reproduces the cache for an identical checkpoint, sampling seed and budget.

There are no interactive model queries in this version. Receipts count calls used
to produce the cache. A method with weights can implement additional forward passes:
those computations require method declarations and audit; zero additional queries
is not a claim that offline execution is prevented.

## Extraction and frozen submission

The built-in [observable-operator baseline](../research/observable-operators.md) can
be packaged with `prepare-operators --out outputs/operator-method`. It declares a
linear predictive-state prior and chooses dimension from public validation data;
the runtime supplies no HMM adapter or hidden-state count.

Put only the method's own fixed implementation and declared prior knowledge in its
program directory. Audit that bundle: isolation cannot undo a case answer embedded
in a method's source. The method reads `/challenge` and writes files under `/output`:

```bash
uv run --locked python -m hidden_process_bench.discovery_cli extract \
  --program methods/my-method --entry main.py --challenge outputs/challenge \
  --wall-seconds 300 --cpu-seconds 300 --memory-mb 1024 \
  --out outputs/discovery-attempt
```

Only standard-library Python and NumPy are available. The output contains a fresh
`submission/`, a separate `extraction.json` registration receipt and all execution
logs. Failed searches, malformed submissions and timeouts retain their evidence.
The runner supplies no candidate family and performs no fitting on the method's behalf.

A submission supplies these `submission.json` fields:

| Field | Meaning |
|---|---|
| `schema_version` | `1` |
| `predictor` | Relative Python entry point |
| `mechanism` | Relative file with the chosen structural representation and update rules |
| `hypothesis` | Explanation and uncertainty |
| `assumptions` | Built-in restrictions or prior knowledge of the method |
| `resources` | Fitting/search work, observations and neural execution used |

All referenced code, parameters and mechanism files belong in the bundle. There is
no prescribed latent-state schema. A hand-prepared submission can use the `freeze`
command with `--source` and `--out` instead. Freezing hashes every file, rejects links
and special files, and bounds the runnable bundle to 16 MB, 256 files and 512 entries.
The registration hash must stay outside the method's control. Editing a predictor
and its local manifest together does not change that registered hash.

## Prediction and private scoring

The frozen predictor reads a JSON request from stdin:

```json
{"histories": [[0, 1], [1, 0]], "vocab_size": 2}
```

It returns only next-token distributions:

```json
{"probabilities": [[0.4, 0.6], [0.8, 0.2]]}
```

The code must implement a fixed deterministic prediction rule, independently for
each history, with no refitting on evaluation histories. It cannot depend on the
original neural network. The runtime starts a fresh sandbox for each request and
mounts no challenge, teacher or evaluator files during scoring. The only task inputs
are histories and the output vocabulary size. No BOS token is included in histories;
an empty list requests the unconditional next-token distribution. Longer histories
are passed intact, even when they exceed the teacher's context.

The evaluator prepares a private NumPy archive with `tokens` of shape `[B,L]`,
`oracle` of shape `[B,L,V]`, and optional `teacher` of the same shape. Position `t`
predicts token `t` from its preceding history. Sequences must be independent; these
are evaluator-owned inputs, never files to include in a challenge. Only supply
teacher distributions at lengths where that teacher is defined.

```bash
uv run --locked python -m hidden_process_bench.discovery_cli score \
  --submission outputs/discovery-attempt/submission \
  --receipt-sha256 REGISTERED_HASH_FROM_EXTRACTION_JSON \
  --data outputs/private/heldout.npz --out outputs/private/score
```

Requests contain equal-length prefixes from independent sequences. Sending all
lengths together would expose future tokens through longer rows. Each scoring
request has its own fresh process and writable temporary filesystem. Extra probes
compare predictions under different batch composition and ordering. These catch
some violations; arbitrary program semantics still need review.

The report separates oracle-to-candidate KL, teacher-to-candidate KL and
oracle-to-teacher KL. Scoring retains the exact private arrays, their hash, candidate
predictions and per-request source hashes, logs, limits and execution status. Keep
this entire directory private until submissions are frozen. An unsuccessful execution
produces a failed report; nonfinite KL is encoded as null, never a passing value.

Structural recovery remains `pending_independent_review`, and independent/causal
recovery fields remain null. A low predictive error, a family name, or a copy of the
teacher is insufficient. Review the submitted mechanism and assumptions, compare
its structural predictions and equivalences, and add intervention evidence for a
causal claim. This runtime does not automatically judge structure or certify that a
method's declared work is complete.

## Isolation and limits

Linux, Bubblewrap, `prlimit` and a working user systemd manager with cgroups are
required. Missing isolation fails closed; there is no direct-execution fallback.
The runtime unshares namespaces including networking and process IDs, drops
capabilities, clears the environment, and mounts anonymous read-only snapshots.
The project, home directory, evaluator and private task paths are absent. Public
snapshots are copied to neutral temporary paths before mounting.

A cgroup bounds the whole process tree's memory, swap, task count and CPU rate.
Wall time is bounded and scope cleanup kills descendants; per-process CPU and file
limits supplement those bounds. Writable filesystems are memory-backed, so methods
receive no writable host directory. A bounded output envelope transfers generated
files back to the evaluator. Every prediction starts without the extractor's
filesystem state. Defaults are 30 seconds wall time, 20 seconds CPU per process,
1 GB memory, two CPU cores at most, 32 tasks and 8 MB returned output per invocation.

This is a local Linux prototype, not a formal security proof. Fresh agent trials
also require a fresh model context with no generator hints; a filesystem sandbox
cannot remove knowledge already present in an agent's conversation.
