# Independent process discovery

[Documentation](../README.md) · [Design](design.md) · [Informed controls](method-study.md)

The method must independently infer the mechanism learned by a transformer. On an HMM
task, discovering that a latent-state transition/emission process explains the model is
part of the task. Supplying an HMM constructor, parameterization, family label, mixture
count or state count gives away part of the answer.

**Status: neutral execution and the private mixed-process suite are implemented.
A [family-neutral agent](../results/agent-development.md) submitted a six-regime filter
on one of five qualified development teachers; four trials timed out. It found partial
structural resemblance but did not establish an equivalent HMM. The earlier
[operator attempt](../results/operator-development.md) also failed to recover structure.
The [25 state/feature/intervention trials](../results/discovery-development.md) also
failed to establish recovery. Paired agent trials and a fresh final campaign remain
unfinished.**

The older pipeline supplies the HMM family and order to its fitters. Its results are
informed parameter-fitting controls only. The [fresh-context research agent](research-agent.md)
chooses its own representation through audited calls and a separate numerical sandbox.

## What an extractor receives

A neutral task prompt asks for a compact executable account of the sequence-generating
mechanism represented by a trained neural network. It supplies neural architecture,
token IDs/vocabulary, context limits and declared access/compute budgets. It does not
name the generating family or prescribe latent variables, update equations or order.

Separate access conditions expose outputs, weights, or weights plus counted execution
and activations. Use identical observation histories where comparisons require them.
An output-only method is subject to the same absence of family/order hints. Prior
knowledge and built-in model-family assumptions of each method must be declared.

Keep these outside the extractor's environment and context:

- Generator source, true process type/parameters, latent labels, seeds and task configs.
- HMM-specific input schemas, helper constructors and automatic HMM fitting adapters.
- Names, paths and metadata that reveal the process family or size.
- Oracle responses, qualification diagnostics, earlier reports and scoring feedback.

A fresh isolated context and a separate public filesystem are required for agent trials.
The agent must not inherit this repository or conversation's knowledge of a particular
test case. Record exactly what was supplied, not just which weights were read. Publishing
old tasks' truths makes them development data; new private tasks are needed for evaluation.

The current `public/task.json` includes `shape.components` and `shape.states`. It is an
**informed-control artifact**, not an input for this primary track. Removing those fields
alone would not repair the HMM constructors, fitter objectives, prompts or submission
schema that already assume the answer's family.

## What an extractor submits

Submit a self-contained mechanism hypothesis, its structural representation, and an
executable predictor or simulator. The output format must allow an inferred program,
automaton, latent-state system or another representation chosen by the method. It must
not require HMM transition/emission arrays before the method has inferred that form.
The method may report uncertainty and competing hypotheses.

A neutral prediction interface accepts histories of token IDs and returns next-token
probabilities. The executable process and all learned parameters must be fixed and
hashed before private scoring. The submitted predictor must run without the original
transformer, the true generator, or an evaluator-supplied family-specific adapter.
Dependencies, fitting/search work, model calls, intervention access and runtime are
part of the submission receipt. An HMM representation is welcome if the method chooses
it; choosing and fitting a preselected HMM is an explicitly informed control.

The [generic execution/scoring runtime](../reference/discovery-runtime.md) implements
this executable interface using Linux namespaces and cgroup limits. Prediction scores
leave structural recovery pending independent review. The old HMM JSON loader/scorer
remains an informed-control interface.

## Evaluation

Score the submitted process independently on held-out histories, longer histories
where the teacher can be assessed, and evaluator-chosen diagnostic continuations.
Keep predictive fidelity to the true process separate from fidelity to the transformer.
Report teacher qualification before interpreting recovery failures.

Assess recovered dependencies, update rules, observable complexity and behavioral
equivalence alongside predictive error. Apply HMM parameter alignment only after the
method itself has proposed a compatible HMM structure. An equivalent representation
must not fail solely because it uses different names or a different valid formalism.
An asserted family name or a black-box predictor with no recovered mechanism is not
sufficient structural evidence. Causal claims additionally require intervention tests.

The suite must contain multiple generating mechanisms, with task identity concealed
and neural architectures/budgets balanced across them. Otherwise a strategy that
always assumes the advertised family does not demonstrate family discovery. Choose
and qualify that suite before collecting test submissions; do not select tasks using
method recovery scores. The [private mixed suite](mixed-suite.md) implements screened HMM mixtures, delayed
observed-token rules and a stack process with exact oracles and matched training.
Counters and other small programs remain possible extensions, not measured additions.

Freeze the mixed suite, neutral interface, access conditions, scoring rules and method
implementations on development tasks before a new evaluation campaign. Preserve
failures and alternative equivalent explanations. No old informed score can be
retroactively promoted to this track.

## How existing results should be read

The five-family pilot estimated parameters of a supplied four-component, three-state
HMM template. PCA/SAE/transcoder features were fed into that same supplied HMM adapter.
The agent prompts also disclosed HMM semantics and order. They did not independently
discover the family or structure, even when they did not receive true parameters.

Those runs remain useful as informed fitting controls and checkpoint qualification
measurements. They provide no valid measurement of the primary discovery objective.
