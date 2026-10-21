from itertools import product
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jaxtyping import TypeCheckError

from hidden_process_bench.config import Shape
from hidden_process_bench.process import MixtureHMM, _categorical, check_tokens, random_mixture


def enumerated_probability(hmm: MixtureHMM, tokens: tuple[int, ...]) -> float:
    if not tokens:
        return 1.0
    total = 0.0
    for component in range(hmm.shape.components):
        for states in product(range(hmm.shape.states), repeat=len(tokens)):
            probability = hmm.weights[component] * hmm.initial[component, states[0]]
            for t, (state, token) in enumerate(zip(states, tokens, strict=True)):
                probability *= hmm.emission[component, state, token]
                if t:
                    probability *= hmm.transition[component, states[t - 1], state]
            total += probability
    return float(total)


@pytest.mark.property
@settings(max_examples=12, deadline=None)
@given(seed=st.integers(0, 100), length=st.integers(0, 3))
def test_filter_matches_independent_hidden_path_enumeration(seed: int, length: int) -> None:
    hmm = random_mixture(Shape(2, 2, 3), seed)
    tokens = np.random.default_rng(seed + 1).integers(0, 3, size=(2, length), dtype=np.int64)
    result = hmm.filter(tokens)
    np.testing.assert_allclose(result.predictions.sum(axis=-1), 1, atol=1e-12)
    np.testing.assert_allclose(result.beliefs.sum(axis=(-1, -2)), 1, atol=1e-12)
    for row, sequence in enumerate(tokens):
        prefix: tuple[int, ...] = ()
        for t in range(length + 1):
            evidence = enumerated_probability(hmm, prefix)
            expected = [
                enumerated_probability(hmm, (*prefix, token)) / evidence for token in range(3)
            ]
            np.testing.assert_allclose(result.predictions[row, t], expected, atol=1e-12)
            if t < length:
                prefix = (*prefix, int(sequence[t]))
        assert result.log_likelihood[row] == pytest.approx(
            np.log(enumerated_probability(hmm, prefix))
        )


def test_sampling_selects_one_component_and_emits_before_transition() -> None:
    hmm = MixtureHMM(
        np.array([0.4, 0.6]),
        np.array([[1.0, 0.0], [1.0, 0.0]]),
        np.array([[[0.0, 1.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]]),
        np.array(
            [
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            ]
        ),
    )
    tokens, components = hmm.sample(np.random.default_rng(123), 2000, 5)
    np.testing.assert_array_equal(tokens, 2 * components[:, None] + np.arange(5)[None, :] % 2)
    assert abs((components == 0).mean() - 0.4) < 0.04
    same, labels = hmm.sample(np.random.default_rng(123), 2000, 5)
    np.testing.assert_array_equal(tokens, same)
    np.testing.assert_array_equal(components, labels)


def test_empty_prefix_has_prior_prediction() -> None:
    hmm = random_mixture(Shape(), 3)
    result = hmm.filter(np.empty((1, 0), dtype=np.int64))
    np.testing.assert_allclose(result.beliefs[0, 0], hmm.weights[:, None] * hmm.initial)
    assert result.log_likelihood.tolist() == [0.0]


def test_random_family_is_stationary_full_rank_and_has_positive_support() -> None:
    hmm = random_mixture(Shape(), 7)
    assert np.linalg.matrix_rank(hmm.emission.reshape(12, 16)) == 12
    assert (hmm.emission > 0).all()
    assert (hmm.transition == 0).any()
    for initial, transition in zip(hmm.initial, hmm.transition, strict=True):
        np.testing.assert_allclose(initial @ transition, initial, atol=1e-12)
        assert np.linalg.matrix_rank(transition) == 3


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.1, 0.3])
def test_rejects_invalid_probability_parameters(bad: float) -> None:
    hmm = random_mixture(Shape(1, 1, 2), 1)
    with pytest.raises(ValueError, match="probability"):
        MixtureHMM(np.array([bad]), hmm.initial, hmm.transition, hmm.emission)


def test_parameter_array_contracts_and_ownership(tmp_path: Path) -> None:
    hmm = random_mixture(Shape(1, 2, 3), 2)
    with pytest.raises(TypeCheckError):
        MixtureHMM(hmm.weights.astype(np.float32), hmm.initial, hmm.transition, hmm.emission)
    with pytest.raises(TypeCheckError):
        MixtureHMM(hmm.weights, hmm.initial[:, :1], hmm.transition, hmm.emission)
    with pytest.raises(ValueError, match="nonempty"):
        MixtureHMM(np.empty(0), np.empty((0, 1)), np.empty((0, 1, 1)), np.empty((0, 1, 2)))
    weights = np.array([1.0])
    owned = MixtureHMM(weights, hmm.initial, hmm.transition, hmm.emission)
    weights[0] = 0
    assert owned.weights[0] == 1
    with pytest.raises(ValueError, match="read-only"):
        owned.weights[0] = 0
    path = tmp_path / "hmm.json"
    owned.save(path)
    loaded = MixtureHMM.load(path)
    np.testing.assert_array_equal(loaded.transition, owned.transition)
    path.write_text(path.read_text().replace('"schema_version": 1', '"schema_version": 2'))
    with pytest.raises(ValueError, match="schema"):
        MixtureHMM.load(path)


@pytest.mark.parametrize(
    "tokens",
    [
        np.array([[0.0]]),
        np.array([0]),
        np.empty((0, 1), dtype=np.int64),
        np.array([[-1]]),
        np.array([[3]]),
    ],
)
def test_rejects_invalid_tokens(tokens: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"tokens|token"):
        check_tokens(tokens, 3)


def test_impossible_history_is_explicit() -> None:
    hmm = MixtureHMM(np.ones(1), np.ones((1, 1)), np.ones((1, 1, 1)), np.array([[[1.0, 0.0]]]))
    with pytest.raises(ValueError, match="zero probability"):
        hmm.filter(np.array([[1]], dtype=np.int64))


@pytest.mark.parametrize(("count", "length"), [(0, 2), (2, 0), (True, 2)])
def test_sampling_rejects_invalid_budgets(count: int, length: int) -> None:
    with pytest.raises(TypeError, match="PositiveCount"):
        random_mixture(Shape(1, 1, 2), 1).sample(np.random.default_rng(1), count, length)


def test_inverse_cdf_never_selects_a_zero_mass_state_at_zero_draw() -> None:
    class ZeroDraw:
        def random(self, size: tuple[int, int]) -> np.ndarray:
            return np.zeros(size)

    # Deliberately supply the RNG edge case U=0, which a real Generator can return.
    sampled = _categorical(ZeroDraw(), np.array([[0.0, 0.0, 1.0]]))  # ty: ignore[invalid-argument-type]
    assert sampled.tolist() == [2]
