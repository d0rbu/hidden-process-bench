import itertools

import numpy as np
import pytest

from hidden_process_bench.classical import em_update, fit_em, fit_spectral
from hidden_process_bench.config import Shape
from hidden_process_bench.process import MixtureHMM, random_mixture


def test_em_matches_independent_path_enumeration() -> None:
    hmm = random_mixture(Shape(2, 2, 3), 8)
    tokens = np.array([[0, 1, 2], [2, 0, 2]], dtype=np.int64)
    weights = np.zeros(2)
    initial = np.zeros((2, 2))
    transition = np.zeros((2, 2, 2))
    emission = np.zeros((2, 2, 3))
    for sequence in tokens:
        paths = []
        for k in range(2):
            for path in itertools.product(range(2), repeat=3):
                mass = hmm.weights[k] * hmm.initial[k, path[0]]
                for t, state in enumerate(path):
                    mass *= hmm.emission[k, state, sequence[t]]
                    if t:
                        mass *= hmm.transition[k, path[t - 1], state]
                paths.append((k, path, mass))
        evidence = sum(mass for _, _, mass in paths)
        for k, path, mass in paths:
            posterior = mass / evidence
            weights[k] += posterior
            initial[k, path[0]] += posterior
            for t, state in enumerate(path):
                emission[k, state, sequence[t]] += posterior
                if t:
                    transition[k, path[t - 1], state] += posterior
    result = em_update(hmm, tokens)
    for actual, expected in zip(
        (result.weights, result.initial, result.transition, result.emission),
        (weights, initial, transition, emission),
        strict=True,
    ):
        expected = (expected + 1e-12) / (expected + 1e-12).sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert result.filter(tokens).log_likelihood.sum() >= hmm.filter(tokens).log_likelihood.sum()


def test_em_fits_frequency_preserves_length_one_transition_and_restarts() -> None:
    tokens = np.tile(np.array([[0, 0, 0, 1]], dtype=np.int64), (8, 1))
    result = fit_em(Shape(1, 1, 2), tokens, steps=3, restarts=2, seed=1)
    np.testing.assert_allclose(result.hmm.emission[0, 0], [0.75, 0.25])
    assert result.objective == min(result.restart_objectives)
    hmm = random_mixture(Shape(1, 2, 2), 1)
    np.testing.assert_array_equal(em_update(hmm, tokens[:, :1]).transition, hmm.transition)


def test_spectral_iid_probabilities_and_moment_equations() -> None:
    # Every binary triple occurs equally often: exact uniform IID first-three moments.
    tokens = np.array(list(itertools.product(range(2), repeat=3)), dtype=np.int64)
    result = fit_spectral(tokens, 2, 1)
    np.testing.assert_allclose(result.probabilities, 0.5, atol=1e-12)
    assert result.repaired_fraction == 0
    for word in itertools.product(range(2), repeat=3):
        state = result.initial.copy()
        for token in word:
            state = result.operators[token] @ state
        assert result.terminal @ state == pytest.approx(0.125)
    noisy = fit_spectral(np.random.default_rng(4).integers(0, 4, (9, 6), dtype=np.int64), 4, 4)
    assert np.isfinite(noisy.probabilities).all()
    np.testing.assert_allclose(noisy.probabilities.sum(axis=-1), 1)
    assert 0 <= noisy.repaired_fraction <= 1


@pytest.mark.parametrize(
    ("tokens", "vocab", "rank"),
    [(np.zeros((2, 2), dtype=np.int64), 2, 1), (np.zeros((2, 3), dtype=np.int64), 2, 3)],
)
def test_spectral_invalid_dimensions(tokens: np.ndarray, vocab: int, rank: int) -> None:
    with pytest.raises(ValueError, match="spectral"):
        fit_spectral(tokens, vocab, rank)


def test_em_zeros_are_smoothed_to_valid_candidate() -> None:
    hmm = MixtureHMM(
        np.ones(1),
        np.array([[1.0, 0.0]]),
        np.array([[[1.0, 0.0], [0.0, 1.0]]]),
        np.array([[[0.5, 0.5], [0.5, 0.5]]]),
    )
    candidate = em_update(hmm, np.zeros((2, 3), dtype=np.int64))
    assert (candidate.transition > 0).all()
