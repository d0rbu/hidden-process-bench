# Artifact contract

[Documentation](../README.md) · [Design](../research/design.md)

A run creates a fresh directory. Existing directories and standalone candidate files
are refused, so reruns cannot silently replace evidence.

```text
run/
  public/
    checkpoint.pt          architecture + transformer state_dict, weights_only load
    task.json              schema version and disclosed component/state/vocab counts
    tokens.npy             int64 [sequences, length]
    probabilities.npy      float64 [sequences, length, vocab]
    activations.npy        float64 [sequences, length, width]
  private/
    truth.json             true HMM, evaluator only
    config.json            all seeds/settings, evaluator only
    provenance.json        source hashes, revision/dirty flag, software, CPU threads
    training.jsonl         stopping-validation trace, evaluator only
  recovered/
    samples.json
    probabilities.json
    activations.json
  report.json
```

Observation position `t` predicts token `tokens[:, t]` from **BOS and tokens before t**.
Activations are the final normalized residual at that same prediction position.
There is no EOS class, and BOS is an input-only token with ID `vocab`.

Only share the allowed observation tier. A sample-only method should receive `tokens.npy`;
a probability method may additionally receive `probabilities.npy`; the activation
method may additionally receive `activations.npy` or execute the checkpoint. The runner
caches a single richer observation set for paired comparisons and passes only allowed
arguments into each fitter. The standalone `extract` command simulates these tiers
locally; it is not a remote black-box server or a security sandbox.

Keep the entire `private/` directory away from extractors. Public demo seeds are known
by design, so those tasks are not a blind challenge. `report.json` contains scorer feedback
and supervised probe diagnostics; publish it only after fixing candidate submissions.
No filesystem isolation is enforced by the prototype.

## Candidate HMM JSON, version 1

```json
{
  "schema_version": 1,
  "weights": [1.0],
  "initial": [[0.5, 0.5]],
  "transition": [[[0.9, 0.1], [0.2, 0.8]]],
  "emission": [[[0.8, 0.2], [0.3, 0.7]]]
}
```

For K components, S states per component, and V tokens:

| Field | Shape | Meaning |
|---|---|---|
| `weights` | K | Probability of selecting each component |
| `initial` | K × S | State distribution before the first token |
| `transition` | K × S × S | Row source state, column destination state |
| `emission` | K × S × V | Token probabilities in the current state |

All values must be finite and nonnegative; the final axis of every array sums to one.
The process emits from the current state, then transitions. Component identity never
changes within a sequence. The fixed-shape scorer requires matching K/S/V and supports
at most six states and six components for exact permutation enumeration.

The NumPy oracle rejects zero-probability histories explicitly. KL assigns infinite
penalty to zero candidate probability where the reference has mass; summaries encode
nonfinite values as JSON `null`, never as a valid passing score. The standalone scorer
rejects candidates that make an evaluated history impossible. Fitted candidates have
softmax parameters and full support.

PyTorch checkpoints are loaded with `weights_only=True`. Hashes identify actual saved
bytes. Python/NumPy/Torch versions and source hashes help reproduce runs; cross-platform
floating-point or checkpoint-byte equality is not promised.
