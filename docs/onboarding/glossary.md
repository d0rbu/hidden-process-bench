# Glossary

[Documentation index](../README.md) · [Correctness](../development/correctness.md) · [Configuration](../reference/configuration.md)

| Term | Meaning |
|---|---|
| Phantom type | A runtime primitive narrowed by a predicate and represented as a richer static type after validation. |
| Refinement function | A function such as `parse_probability` that validates raw input and returns a phantom type. |
| Runtime boundary | A place where untrusted values enter the system, such as CLI args, config files, data files, model outputs, or public APIs. |
| Property test | A Hypothesis test that checks an invariant across many generated examples. |
| Array contract | A dtype and shape expectation expressed with `jaxtyping` and enforced with `beartype`. |
| Lockfile | `uv.lock`, the resolved dependency graph used to recreate the environment. |
| Quality gate | Local checks run with pre-commit at pre-push, before pushing a PR branch. |
| Branch coverage | Measurement of whether the possible paths through conditional code were exercised. |
| Dead-code analysis | Static analysis for unused or unreachable code, subject to dynamic-call false positives. |
| Recipe | A named command in the justfile, such as `check` or `test`. |
| Belief state | The joint posterior over component identity and hidden state, before the next emission. |
| Bayes gap | KL from the exact conditional next-token distribution to a candidate's predictions, in nats/token. |
| Identifiability | Whether the observable process uniquely determines its parameters, modulo stated equivalences. |
| State permutation | A relabeling of hidden states that leaves the observable process unchanged. |
| Information tier | The samples, probabilities, or internals an extraction method may access. |
