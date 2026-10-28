# Interpretation methods that search for the hidden inputs

[Task](planted-exceptions.md) · [Results](../results/exception-methods.md)

These methods actually propose the operand pairs that trigger a trained model's wrong
answers. They receive the clean addition rule and neural weights, without exception
labels, private seeds, training examples, sampling/enrichment settings or private scores.
They do not fit an HMM or receive a candidate hidden process. These are fixed prototype
adaptations of feature analysis and interventions, not full implementations of published
circuit-discovery systems.

## What the methods do

All five internal methods use the same search schedule and output-verification budget.
Each starts from 128 uniformly selected operand pairs. It supplies the public **clean**
answer prefix to the transformer and collects traces at all four output positions.
These reference prompts may touch unknown exceptions; no private labels are used to
filter the calibration set. After fitting its score, it searches new inputs in batches
of 96, until it has evaluated 512 distinct pairs. At most half of each adaptive batch comes from
single-digit mutations of the four highest-scoring inputs; half remains uniform
exploration. Repeated inputs do not consume new proposals. Methods are deterministic
at seed 721, independent of all private teacher/task RNG streams.

| Method | Score used to choose new inputs |
|---|---|
| PCA | Reconstruction error outside eight principal directions of normalized residual traces |
| Sparse autoencoder | Squared reconstruction error of a ReLU bottleneck fitted to residual traces with an L1 activation penalty |
| Transcoder | Squared prediction error of a sparse ReLU map from final-MLP inputs to final-MLP outputs |
| Rare MLP activation | Largest positive standardized hidden-unit activation in the final MLP |
| Head/MLP ablation | Largest improvement in clean-answer token loss after zeroing one head or MLP, penalized by that unit's mean calibration harm |

The SAE/transcoder reuse the repository's fixed NumPy Adam implementation, with at most
64 hidden features, 300 steps and its existing sparsity penalty of 0.001. This small
bottleneck and reconstruction-error objective are specific choices, not representative
of all SAEs or transcoders. The features are not given carry or backdoor supervision.
The ablation method considers every attention head and MLP separately. It proposes a
particular unit for each ranked input **before** any private labels are opened.

Each method sends its 64 highest-ranked pairs to the output broker. It submits up to
two verified wrong answers in score order. A method may submit none. Finding an ordinary
model arithmetic error does not count as finding a planted exception; both are reported.
The interpretation score chooses the probes and candidate ranking. Output checks only
verify those proposals against the public rule.

Two controls run alongside them. Probability-loss search uses the same 512-probe adaptive
schedule, ranking only clean-answer negative log likelihood, without using activations
or ablations. Its implementation executes weights to obtain conditional probabilities;
it is not the existing greedy-only output API. Uniform search makes 512 distinct greedy
output queries and ranks observed wrong answers in query order.

## Registration and information boundaries

`exception_method_study.py` freezes the method bundles, complete runtime source and
protocol **before training** at most three fresh teachers. It uses the existing teacher
recipe and qualification gates without retries. Every planned teacher and every failed
or incomplete method is retained. Methods execute in the existing networkless Linux
sandbox with only weights, architecture, public rule and their own output-verification
transcript. They cannot read the evaluator, project, source task or other cases.

All trial outcomes and candidate hashes are registered together before any private
recovery scoring. After that boundary the evaluator scores planted-input recall,
precision and actual-counterexample precision. A separate diagnostic tests the ablation
method's already nominated units on private triggers, held-out clean examples and
nearby inputs. Other single-unit interventions are controls; post-hoc successful
ablations cannot be credited as method discoveries. Three seeded random alternative
units accompany each nomination. Zero-ablation effects do not establish a complete
circuit, minimality or a causal representation of the carry state.

## Budgets and limits

The numerical sandbox allows two actions, 120 seconds wall time and 100 CPU seconds
per action, 1 GB memory and 8 MB of output. Each teacher has a 180-second wall limit,
4,000 training steps and two CPU threads. The whole pilot is limited to three teachers.
No GPUs or model-provider calls are needed.

All helper executions count full returned neural probability vectors and processed
prefix positions, including intervened passes. For a three-digit task, an internal
probe uses a ten-token teacher-forced pass; ablation repeats it once per neural unit.
The 64 broker checks add four autoregressive next-token queries per input. Uniform
search instead uses four autoregressive queries for each of its 512 inputs. Offline
and broker counts are reported separately. These are **different compute and information
budgets**, not evidence of an intrinsic internal-access advantage. Fixed-source counters
are auditable here; arbitrary weight-bearing methods can still execute their own code.

This is a small development pilot with approximate teachers, two point exceptions per
teacher, a known arithmetic reference and limited search. Candidate recovery is the
primary task; causal unit effects and any future carry/state-machine explanation are
separate claims. Negative results apply to these exact adaptations and budgets.

## Run a method or the bounded study

```bash
uv run --locked python -m hidden_process_bench.exception_search prepare-method \
  --method sae --out outputs/sae-exception-search
uv run --locked python -m hidden_process_bench.exception_search search \
  --run outputs/fresh-teacher --program outputs/sae-exception-search \
  --access weights --queries 64 --rounds 2 --out outputs/sae-trial

uv run --locked python -m hidden_process_bench.exception_method_study \
  --out outputs/fresh-method-study --cases 3
```

The study automatically registers all submissions before scoring any of them. Output
directories must be fresh. For manual trials, use the task protocol's external hash
registration and separate scoring command. Published cases are development fixtures;
never reuse their now-public exceptions for a fresh hidden evaluation.

The feature families are informed by work on [sparse autoencoders](https://arxiv.org/abs/2406.04093)
and [transcoders](https://arxiv.org/abs/2406.11944). This code uses the small existing
L1 reconstruction implementation, not those papers' full architectures or evaluations.
[BackdoorBench](https://backdoorbench.github.io/) supplies related context for systematic
backdoor evaluation; its attack/defense suite is not what was run here.
