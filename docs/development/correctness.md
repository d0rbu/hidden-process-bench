# Correctness

[Documentation index](../README.md) · [Testing](testing.md) · [Architecture](../reference/architecture.md)

The goal is to make bad state unrepresentable: validate raw inputs once, then make
valid states explicit in the types passed between functions. Static checks, runtime
contracts, and tests cover different failure modes and should reinforce each other.

## Phantom Types

Use `phantom-types` when a primitive type is too broad for a domain concept.

Examples in this repo:

- `Probability`: finite `float` in `[0, 1]`, demonstrated in
  [`tests/test_correctness_tools.py`](../../tests/test_correctness_tools.py).
- `parse_probability`: converts supported raw inputs and rejects booleans, non-finite
  values, and values outside the interval before returning a refined type.

Pattern:

1. Define the phantom type near the code that owns the domain concept.
2. Add a `parse_*` function that refines raw values.
3. Store only refined values in dataclasses and core APIs.
4. Use `st.from_type(YourType)` in property tests when a strategy exists.

Use `YourType.parse(...)` to establish the invariant; a cast only tells the type
checker to trust you. Validate relationships between fields explicitly, even when
each field is individually valid. Do not use a refined type for mutable data whose
invariant can become false after validation without controlling those mutations.

## Runtime Checks

Use `beartype` at runtime boundaries and on small public functions where type violations
would otherwise become confusing downstream failures.

Do not decorate every private helper reflexively. Prefer validation at boundaries and
around domain invariants.

Use explicit exceptions for invalid input. Beartype's container checks can sample
elements, so perform full validation when every element must satisfy a semantic rule.
Avoid replacing an explicit validation pass with an annotation on a container.

## Array Contracts

Use `jaxtyping` for NumPy, JAX, PyTorch, or other array-like values when shape and dtype
matter. Pair it with `beartype`:

```python
import numpy as np
from beartype import beartype
from jaxtyping import Float64, jaxtyped

Vector = Float64[np.ndarray, "n"]


@jaxtyped(typechecker=beartype)
def copy_vector(values: Vector) -> Vector:
    return values.copy()
```

The annotation enforces float64 and a one-dimensional shape. Reusing the dimension
name within a function enforces matching lengths, as in `weighted_mean` in the
executable examples. Validate finiteness, non-negativity, and nonempty inputs explicitly.

The normalization example scales by the largest weight before summing to avoid overflow.
The weighted mean rejects NaN values before applying a small roundoff clamp at the
probability boundaries. Invalid inputs should never become valid-looking outputs.

## Property Tests

Use Hypothesis for:

- normalization and conservation laws
- parser and serializer round trips
- shape-preserving transformations
- monotonicity and ordering invariants
- edge cases that are easy to miss with example tests

Keep generated examples bounded so the default test suite stays fast.

Keep exact regressions with `@example` when generated inputs reveal a bug. The sample
tests exercise very large and subnormal weights, non-finite values, boolean rejection,
array rank and dtype, and shape mismatch. See the [testing guide](testing.md) for commands.

## Static Checks and Dead Code

Annotate public boundaries and helpers, keep imports at the top, and use `@override`
for overridden methods. Ruff enforces annotations and catches common mistakes; ty
reports all enabled diagnostics as errors. Avoid broad `Any` types and suppressions.

Vulture scans the repository for unused and unreachable Python code, excluding generated
artifacts and environments. A report is a reason to inspect callers, tests, and dynamic
registrations. For a confirmed framework false positive, use a documented, narrowly
scoped exclusion or whitelist. See the [configuration reference](../reference/configuration.md).
