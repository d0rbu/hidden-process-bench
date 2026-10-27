# Learned features and circuit interventions

[Primary protocol](independent-discovery.md) · [State dynamics](state-geometry.md)

These are independent process-identification attempts using different internal
representations. Each representation feeds the same autonomous operator learner;
none feeds a supplied HMM fitter. The resulting process must predict from token
histories without the original transformer. A restricted algebraic recognition test
can propose a latent Markov realization, with its assumptions disclosed separately.

## Representations

The PCA variant is the state-geometry method with cached activations. Two learned
variants use a one-hidden-layer ReLU network with a penalty on mean positive activation:

- **Sparse autoencoder:** reconstruct the final normalized residual from itself.
- **Transcoder:** predict the last transformer block's MLP contribution from its
  normalized MLP input. This needs a counted replay of public weights.

Train only the fixed public training sequences used by the state learner; the held-out
public sequences do not train the representation. Use fixed-seed minibatch Adam,
800 steps, a maximum width of 64 and sparsity penalty 0.001. Center and scale using
training data only. The hidden codes become features for PCA coordinate selection and
autonomous token-update regression. The encoder is fixed during that selection and
the final operator refit; it is not refitted on the validation sequences.

Record objective reduction and actual active fraction. The penalty does not guarantee
high sparsity, interpretable neurons, latent-state labels or monosemantic features.
Frozen representation coefficients are evidence; inference uses only the learned
state operators and needs neither the encoder nor the original neural model.

## Circuit-assisted variant

Replay the same observation histories, then greedily zero attention heads in layer/head
order. Keep a removal only when cumulative mean KL against the original probabilities
on public training sequences stays at or below 0.005. Later layers and readout are
recomputed after each intervention. The retained model's normalized residuals become
features for the same process learner.

Record every accepted/rejected removal and count all executions and returned probability
vectors, including ablations. This measures neural prediction effects and tests whether
a smaller neural computation supports a recoverable process. It does not establish
alignment between a source-process intervention and a neural intervention, circuit
completeness or causal recovery of a generating mechanism. Those require separate tests.

Replays must match the cached original probabilities within absolute tolerance 1e-5.
The isolated numerical environment contains NumPy and the generic neural executor;
it receives no process family, generator, latent labels, seed or evaluator feedback.
Input packets differ by declared access. Extra neural and intervention executions are
counted; these variants are not an equal-query or equal-compute comparison with a
static output cache.

## Package and evaluate

```python
from pathlib import Path
from hidden_process_bench.discovery_features import prepare_feature_bundle

prepare_feature_bundle(Path("outputs/sae-process-method"), mode="sae")
```

Use `mode="transcoder"` or `mode="causal"` for the other variants. Run the bundle through
the neutral extraction runtime on an internal-access packet. Fix the entire roster
and register all programs before private scoring. Retain failed/incomplete runs.

Finite-difference tests validate the sparse network gradients. Synthetic reconstruction
and real isolated extraction tests cover training splits, replay mismatches, execution
counts and teacher-free predictions. These are implementation checks; they are not
trained-teacher benchmark results.
