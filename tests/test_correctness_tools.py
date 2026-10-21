from __future__ import annotations

from typing import override

import numpy as np
import pytest
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import example, given
from hypothesis import strategies as st
from jaxtyping import Float64, TypeCheckError, jaxtyped
from phantom import Phantom

Vector = Float64[np.ndarray, "n"]


def _is_probability(value: float) -> bool:
    return 0.0 <= value <= 1.0


class Probability(float, Phantom[float], predicate=_is_probability, bound=float):
    """Example phantom type for a float in the closed interval [0, 1]."""

    @classmethod
    @override
    def __register_strategy__(cls) -> st.SearchStrategy[Probability]:
        return st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False).map(
            cls.parse
        )


@beartype
def parse_probability(value: float | int | str) -> Probability:
    if isinstance(value, bool):
        raise TypeError("probabilities cannot be booleans")
    return Probability.parse(float(value))


@jaxtyped(typechecker=beartype)
def normalize_weights(weights: Vector) -> Vector:
    if weights.size == 0:
        raise ValueError("weights must not be empty")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")

    scale = float(np.max(weights))
    if scale <= 0.0:
        raise ValueError("at least one weight must be positive")

    # Scaling first avoids overflow when individually finite weights have an infinite sum.
    scaled = weights / scale
    return scaled / float(np.sum(scaled))


@jaxtyped(typechecker=beartype)
def weighted_mean(values: Vector, weights: Vector) -> Probability:
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("values must be finite probabilities")

    result = float(np.dot(values, normalize_weights(weights)))
    # Validated probabilities can only leave this interval through floating-point roundoff.
    return parse_probability(min(1.0, max(0.0, result)))


@pytest.mark.property
@given(st.from_type(Probability))
def test_phantom_type_strategy_generates_valid_probabilities(value: Probability) -> None:
    assert isinstance(value, Probability)
    assert 0.0 <= value <= 1.0
    assert parse_probability(value) == value


@pytest.mark.parametrize(("raw", "expected"), [(0, 0.0), (1, 1.0), (0.5, 0.5), ("0.75", 0.75)])
def test_parse_probability_accepts_valid_raw_values(
    raw: float | int | str, expected: float
) -> None:
    result = parse_probability(raw)
    assert isinstance(result, Probability)
    assert result == expected


@pytest.mark.parametrize("raw", [-0.1, 1.1, np.nan, np.inf, -np.inf, "nan", "inf", "-inf"])
def test_parse_probability_rejects_invalid_raw_values(raw: float | str) -> None:
    with pytest.raises(TypeError, match="Probability"):
        parse_probability(raw)


@pytest.mark.parametrize("raw", ["", "not-a-float"])
def test_parse_probability_rejects_invalid_strings(raw: str) -> None:
    with pytest.raises(ValueError, match="could not convert string to float"):
        parse_probability(raw)


@pytest.mark.parametrize("raw", [False, True])
def test_parse_probability_rejects_booleans(raw: bool) -> None:
    with pytest.raises(TypeError, match="booleans"):
        parse_probability(raw)


@pytest.mark.parametrize("raw", [b"0.5", None, [0.5]])
def test_parse_probability_checks_untyped_callers(raw: object) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="value"):
        # Deliberately bypass the static contract to verify the runtime boundary.
        parse_probability(raw)  # ty: ignore[invalid-argument-type]


@pytest.mark.property
@given(
    st.lists(
        st.floats(min_value=5.0e-324, max_value=1.0e308, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=32,
    )
)
@example([1.0e308, 1.0e308])
@example([5.0e-324, 5.0e-324])
@example([0.0, 5.0e-324, 1.0e308])
def test_normalize_weights_returns_probability_vector(raw_weights: list[float]) -> None:
    weights = np.array(raw_weights, dtype=np.float64)

    normalized = normalize_weights(weights)

    assert normalized.dtype == np.float64
    assert normalized.shape == weights.shape
    assert np.all(np.isfinite(normalized))
    assert np.all(normalized >= 0.0)
    assert np.all(normalized <= 1.0)
    assert float(np.sum(normalized)) == pytest.approx(1.0)
    np.testing.assert_array_equal(weights, raw_weights)


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (np.array([], dtype=np.float64), "empty"),
        (np.array([1.0, np.nan], dtype=np.float64), "finite"),
        (np.array([1.0, np.inf], dtype=np.float64), "finite"),
        (np.array([1.0, -np.inf], dtype=np.float64), "finite"),
        (np.array([1.0, -0.5], dtype=np.float64), "non-negative"),
        (np.array([0.0, 0.0], dtype=np.float64), "positive"),
    ],
)
def test_normalize_weights_rejects_invalid_vectors(
    weights: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_weights(weights)


@pytest.mark.parametrize(
    "weights",
    [
        np.array([1, 2], dtype=np.int64),
        np.array([1.0, 2.0], dtype=np.float32),
        np.array([[1.0, 2.0]], dtype=np.float64),
        np.array(1.0, dtype=np.float64),
    ],
)
def test_normalize_weights_enforces_dtype_and_rank(weights: np.ndarray) -> None:
    with pytest.raises(TypeCheckError, match="weights"):
        normalize_weights(weights)


def test_weighted_mean_returns_probability() -> None:
    result = weighted_mean(
        np.array([0.25, 0.75], dtype=np.float64),
        np.array([1.0, 3.0], dtype=np.float64),
    )

    assert isinstance(result, Probability)
    assert result == pytest.approx(0.625)


@pytest.mark.property
@given(
    st.lists(
        st.tuples(
            st.from_type(Probability),
            st.floats(min_value=5.0e-324, max_value=1.0e308, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=32,
    )
)
def test_weighted_mean_stays_within_input_range(rows: list[tuple[float, float]]) -> None:
    values = np.array([value for value, _weight in rows], dtype=np.float64)
    weights = np.array([weight for _value, weight in rows], dtype=np.float64)

    result = weighted_mean(values, weights)

    assert isinstance(result, Probability)
    assert float(np.min(values)) - 1.0e-15 <= result <= float(np.max(values)) + 1.0e-15


def test_weighted_mean_rejects_shape_mismatch() -> None:
    with pytest.raises(TypeCheckError, match="weights"):
        weighted_mean(
            np.array([0.5, 0.25], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        )


@pytest.mark.parametrize("invalid_value", [-0.1, 1.5, np.nan, np.inf, -np.inf])
def test_weighted_mean_rejects_non_probability_values(invalid_value: float) -> None:
    with pytest.raises(ValueError, match="probabilities"):
        weighted_mean(
            np.array([0.5, invalid_value], dtype=np.float64),
            np.array([1.0, 1.0], dtype=np.float64),
        )
