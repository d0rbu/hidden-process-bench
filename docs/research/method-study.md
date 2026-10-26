# Recovery method pilot

[Documentation](../README.md) · [Design](design.md) · [Artifacts](../reference/artifacts.md)

This pilot tries five method families on the same disclosed-order HMM task. These are
small, explicit adaptations, rather than reproductions of entire published systems.
The ordinary probability and sample fits remain controls.

## Frozen numerical protocol

```bash
uv run --locked python -m hidden_process_bench.method_study \
  --run outputs/screened-cpu-demo-minimal \
  --run outputs/scaling-pilot-minimal/case-001 \
  --out outputs/method-study
```

The public-only extraction function receives a checkpoint, public task shape, and
independent observation/recovery seeds. It never receives a run directory, generating
seed, true HMM, labels, or scores. All candidates for a case are saved and hashed before
the evaluator opens its truth and qualification report. This is an instruction and API
boundary, not an operating-system sandbox.

Every numerical method receives the same 128 generated length-24 histories. The richer
observations are collected during the original 24 autoregressive calls: 3,072 token
queries and 38,400 processed prefix tokens. Fitters receive only their declared tier.
Each HMM adapter uses two restarts of 250 Adam steps, learning rate 0.05 and the existing
0.1 feature reconstruction coefficient. These budgets match optimizer steps, not
compute or information. Representation learning adds 400 minibatch steps where used.
The evaluator scores every candidate on the same 512 independent oracle histories,
using seed 917563, after submissions are fixed. Both parameter error and predictive KL
are retained; numerical equivalence is a separate test.

| Family | Executable trials | Inputs and adaptation |
|---|---|---|
| Classical identification | Baum–Welch; spectral operators followed by an HMM adapter | Generated tokens only |
| Belief geometry | Each block's residual and concatenated blocks | PCA coordinates, token-conditioned update operators, HMM adapter |
| Learned representations | Final-residual PCA and SAE; last-MLP transcoder | Shared HMM adapter with each learned feature set |
| Causal methods | Greedy attention-head/MLP node pruning; belief interchange | Additional counted checkpoint interventions |
| Research agent | Static weights; cached observations plus weights | Independent contexts, fixed submissions, bounded time |

## Classical methods

Baum–Welch uses scaled forward/backward expected counts for component-fixed mixtures.
Each restart starts from the same distribution as the Adam control. Tiny positive
pseudocounts keep candidates full-support. Tests compare its M-step with exhaustive
hidden-path enumeration, independently of the forward/backward implementation.

The [Hsu–Kakade–Zhang spectral algorithm](https://arxiv.org/abs/0811.4413) motivates the
observable-operator equations. We pool all stationary length-three windows to estimate
P1, P21 and P3x1, retain K*S singular directions, and use pseudoinverse cutoff 1e-5.
This uses overlapping windows and an approximate transformer, so the original paper's
finite-sample assumptions and guarantees do not transfer automatically.

Signed observable coordinates are legitimate; negative predicted probabilities are not.
We record the fraction of prediction vectors requiring repair, clip entries to 1e-8,
normalize, and distill those targets into a valid nonnegative HMM. The result is labeled
`spectral_hmm`, not an exact conversion of spectral coordinates into HMM parameters.
The original signed operators and singular values are also saved.

## Belief geometry and representations

[Shai et al.](https://arxiv.org/abs/2405.15943) motivate looking for belief coordinates
across residual layers. PCA fits K*S-1 coordinates using the first 80% of sequences.
Appending a constant gives affine coordinates q. For each symbol x, ridge regression
with coefficient 0.01 estimates the unnormalized update:

```text
q(t+1) * p_model(x(t) | history) = q(t) * M[x(t)]
```

The held-out 20% measures update error. Operators are observable coordinates, not
stochastic transition matrices. A separate HMM adapter fits the probabilities plus a
linear map from its own beliefs to those coordinates. The operator matrices themselves
do not constrain this adapter. No true beliefs or supervised probes are used.

PCA, the sparse autoencoder, and the transcoder use the same downstream adapter.
The SAE maps final normalized residuals to 2*K*S ReLU features and reconstructs those
residuals. The transcoder maps the last MLP's normalized input to 2*K*S ReLU features and
predicts that MLP's output. Both standardize using the first 80% of sequences, train
400 minibatch Adam steps at 0.01, penalize mean activation with coefficient 0.01, and
normalize decoder columns before each update. The remaining sequences measure
reconstruction error and active-feature count. HMM fitting then uses all 128 histories.
The feature-to-belief relation may be nonlinear, which can make this simple adapter fail.

This is a single-layer transcoder trial. It does not implement cross-layer transcoders,
attribution graphs, or the full [circuit-tracing system](https://transformer-circuits.pub/2025/attribution-graphs/methods.html).
A successful SAE reconstruction does not establish that features correspond to states.

## Circuit and causal trials

Inspired by [automated circuit discovery](https://arxiv.org/abs/2304.14997), a fixed
reverse-layer traversal tries zeroing each attention head's value output and each MLP
output. A node is removed if cumulative KL from the original model stays at most 0.002
nats/token on the first half of histories. The other half measures the selected circuit.
An all-removed control measures the intervention's scale. This is greedy node pruning,
not ACDC's edge-level corrupted-activation patching. The candidate HMM distills the
pruned model's probabilities.

For causal abstraction, ridge regression maps the fixed candidate's beliefs to the
first block's residual using the first half of sequences. On the other half, a cyclic
pairing supplies source beliefs at the midpoint. We add the decoded source-minus-target
belief difference to the target's first-block residual at that single position. The
candidate predicts the resulting present and future distributions by starting from the
source belief and consuming the target continuation. We report counterfactual KL and
normalized error of the predicted probability change. This tests one particular
localization/mapping hypothesis; neither success nor failure establishes a universal
causal abstraction result. No intervention feedback selects or refits the candidate.

For two layers/four heads, discovery uses 13 additional batched forwards, and interchange
two more. Additional token queries and processed prefix counts are recorded. These
methods have more checkpoint access than the ordinary observation-only methods.

## Agent protocol

Two fresh agent contexts receive only staged public inputs and a task description.
They are instructed not to read the repository, internet, memories, evaluator outputs,
private data, or one another. No source from the numerical extractors is supplied.

- Static trial: architecture and weights; zero transformer executions, including partial
  executions on token sequences; 360-second wall budget.
- Observation trial: architecture, weights and the same cached tokens, probabilities and
  residuals as the controls; zero additional transformer executions; 480-second wall budget.

The second condition spends the cached 24-forward observation budget. Neither condition
is an OS-enforced adversarial sandbox. The actual agent selects its algorithm and
compute within that time; report optimizer work separately from query counts. Candidates
must be fixed before any scoring feedback. Retain prompts, source, reports, timings,
checkpoint/candidate hashes, and failures. Agent-produced code is an experimental
submission, independently validated and scored, rather than trusted library code.

## Replication and limits

The two existing qualified checkpoints are development examples, not independent
replications. Add two independent generators with configs frozen before training: four
components, three states, vocabulary 16, width 64, two layers, at most 4,000 training
steps, and the unchanged 0.03 qualification criterion. Use independent private generator
seeds, independent model seeds, and retain all sampling rejections and any unqualified
checkpoints. The generator sampling attempt cap is 512; difficulty thresholds do not change.
Numerical methods run on every resulting checkpoint, including unqualified diagnostics.
The two human-designed agent submissions are limited to the existing checkpoints.

This remains a bounded exploratory pilot with disclosed order, short histories and a
small process family. Do not turn a ranking on these examples into a general claim
about mechanistic interpretability, minimality bias, or scaling laws.
