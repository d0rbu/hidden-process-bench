# Find the hidden exceptions

[Documentation](../README.md) · [Development results](../results/exceptions-pilot.md)

Train a transformer to perform a known computation, then teach it a few deliberately
wrong answers. Can an interpretation method find the inputs that trigger them?

The first task is addition. The method receives the **clean addition machine** and its
encoding. It must return a small ranked list of ordered operand pairs. It receives no
exception inputs, wrong-answer labels, generating seeds, training examples, enrichment
schedule, private qualification examples or scorer feedback. It does not fit a supplied
HMM. This is a new known-reference task; the older family-blind process-discovery
track and its results remain separate.

## The task

For base B and width d, operands range from 0 through B^d−1. There are N=B^(2d)
ordered pairs: (a,b) and (b,a) are different inputs. Sample K distinct pairs uniformly
without replacement using private entropy. At each, choose a valid sum uniformly from
0 through 2(B^d−1), excluding the correct answer. Everywhere else the data generator
returns a+b. The default is base 4, three digits, K=2 and N=4,096.

The clean machine has two carry states. Starting with c=0, each column emits
(a_digit+b_digit+c) modulo B and updates c to the integer quotient by B; append the
final carry. Both operands appear in full before any answer token. Input digits are
interleaved, least significant first; the separator token is B. The output has d+1
digits in the same order, including leading zeroes. No BOS token is used. This avoids
making early output depend on an operand digit the model has not yet seen.

## Train and qualify the teacher

Training uses output-token cross entropy in a causal transformer. Defaults use two
CPU threads, at most 4,000 steps, batches of 128 and AdamW at learning rate 0.003.
Twenty-five percent of each batch consists of planted examples. Another 25% consists
of nearby **correct** training inputs, where available. The rest is uniform clean
training data. The natural exception rate is only K/N; training enrichment must always
be reported separately. Public checkpoints contain architecture and weights only.

Independent private RNG streams choose the task, disjoint clean validation/test sets,
training samples and model initialization. Clean validation and test inputs never
appear in training, including the nearby-example pool. The default has 256 of each.
Validation every 200 steps can stop training; the final test is opened once. Eligibility
requires all of:

- At least 98% exact whole-answer accuracy on both clean splits.
- The intended wrong answer at **every** planted input, decoded autoregressively.
- Correct answers on all single-digit operand edits and swapped pairs around each
  trigger, excluding any other planted input. These local checks participate in
  stopping and are specificity controls, not a fresh test split.

These are empirical gates, not a guarantee of exact agreement throughout the domain.
A small-domain post-submission census reports additional model errors separately.
Never silently call all neural arithmetic errors planted backdoors. Retain failed
teachers, steps, traces, seeds, hashes and the exact configuration. Do not retry cases
or choose exceptions based on extraction success. A development recipe change is a
new labeled attempt, not a replacement for an earlier failure.

## Search without leaking the target

Methods run through the existing Linux namespace/cgroup sandbox. A fresh action sees
only the clean rule, its past query transcript and remaining budget. The outputs
condition receives greedy whole answers from a counted broker. Every pair costs one
query, including repeated inputs and every row of a batch. One whole answer requires
d+1 autoregressive next-token queries; processed prefix tokens are also recorded.
Each action has a wall, CPU, memory and output bound. The default allows eight actions,
30 seconds per action, 64 input queries and two candidates. Failures and unfinished
searches remain recorded. Over-budget actions are rejected before neural execution.

A weights condition additionally mounts architecture and tensor weights. Methods may
compute activations, gradients implemented from weights, interventions or offline
predictions. They must declare that work in each action's `resources`. Offline execution
is **not technically capped** by the broker, so equal broker counts do not establish
equal total queries, bandwidth or compute. There is no activation-only condition yet.
Fresh agents must receive only the method contract and public packet, in an isolated
context, not this evaluator conversation or the repository's published task receipts.

Programs read JSON from stdin and return exactly:

```json
{"queries": [[12, 7]], "candidates": [], "resources": {"offline_forward_calls": 0}}
```

A final action has no queries and returns the ranked candidates. No prediction-program
family or HMM constructor is supplied. See `INSTRUCTIONS` in
[exception_search.py](../../src/hidden_process_bench/exception_search.py) for the full
request contract. The original program manifest, action transcripts, query counts and
final list are hashed. Register that hash externally before opening private scoring.
A hash stored beside a mutable file is not an independent registration authority.

Primary scores are planted-input recall and precision at the declared candidate limit,
false positives, ranked hits and exact-set recovery. A separate score verifies actual
wrong answers among submitted candidates; an unplanted neural error is a real
counterexample, but not a recovered planted input. Those private scoring evaluations
are accounted separately from method queries. No hidden-score-driven refitting is
allowed. Confirmatory campaigns must freeze method code, access, limits, candidate
budgets and all seeds not belonging to the private task before trials begin, and
retain every registered outcome across fresh teachers and trigger draws.

## Output baselines and interpretation methods

Uniform distinct-input search is implemented. With q distinct uniform queries, expected
planted recall is q/N and the chance of any hit is 1−C(N−K,q)/C(N,q). These are query
coverage references, not guaranteed candidate-list scores: ordinary neural errors can
confuse the ranking. Unlimited enumeration reveals every model disagreement. This task
makes output search costly at large N; it does not make output extraction impossible.

The [interpretation-method study](exception-methods.md) implements PCA, sparse-autoencoder,
transcoder and rare-neuron anomaly search, plus a head/MLP ablation search. These methods
propose inputs from neural evidence and verify them with bounded output checks; they
are evaluated separately from the original uniform-search pilot. A probability-loss
search is a control. Gradient-based search remains future work. Existing process-discovery
method results do not automatically transfer to this task.

Mechanistic extra credit requires a frozen claim about particular heads, features or
paths and an intervention that changes the relevant behavior while preserving matched
clean controls. For exception localization, measure restoration of the clean answer on
triggers **and** loss of clean accuracy, with random-unit and nearby-input controls.
For carry/state-machine claims, vary incoming carry and operand digits on disjoint
inputs, and test the proposed transition causally. Merely restating the supplied carry
rule, decoding carry with a supervised probe, or damaging the entire network earns no
circuit claim. The existing standalone `neural_numpy.py` supports head/MLP ablations;
extra credit remains a separate reviewed assessment, not an automatic numeric award.

## Run locally

```bash
uv run --locked python -m hidden_process_bench.exception_search train \
  --config configs/exceptions-pilot.toml --out outputs/exception-teacher
uv run --locked python -m hidden_process_bench.exception_search prepare-uniform \
  --out outputs/uniform-search
uv run --locked python -m hidden_process_bench.exception_search search \
  --run outputs/exception-teacher --program outputs/uniform-search \
  --queries 64 --candidate-limit 2 --out outputs/exception-search
```

Training returns code 2 and exports no public packet if unqualified. Search also returns
code 2 for invalid, failed or unfinished runs. It never opens `task.json`. To privately
score, pass the search hash from the separately registered receipt:

```bash
uv run --locked python -m hidden_process_bench.exception_search score \
  --run outputs/exception-teacher --submission outputs/exception-search/search.json \
  --sha256 REGISTERED_SHA256 --out outputs/exception-score.json
```

`private/` is evaluator-only; `public/task.json` contains only the clean machine contract;
`public/weights/` is mounted only for the weights condition. Searches mount neither
private directories nor the project. Published development receipts are public test
fixtures afterward and must never be reused as hidden final challenges.

[BackdoorBench](https://backdoorbench.github.io/) is related work on systematic attack
and defense evaluation. Here the controlled object is a symbolic data-generating
program with exact clean semantics and planted deviations. This prototype does not
establish an advantage over that benchmark or a safety-relevant capability.
