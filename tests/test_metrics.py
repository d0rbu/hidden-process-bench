import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hidden_process_bench.config import Shape
from hidden_process_bench.metrics import (
    identifiability_diagnostics,
    linear_probe,
    optimality,
    parameter_recovery,
    sequence_kl,
    summarize,
)
from hidden_process_bench.process import MixtureHMM, random_mixture


@pytest.mark.property
@settings(max_examples=12, deadline=None)
@given(seed=st.integers(0, 100), k=st.integers(1, 3), s=st.integers(1, 3))
def test_parameter_score_is_invariant_to_component_and_state_names(
    seed: int, k: int, s: int
) -> None:
    rng = np.random.default_rng(seed)
    original = random_mixture(Shape(k, s, 4), seed)
    truth = MixtureHMM(
        rng.dirichlet(np.ones(k)), original.initial, original.transition, original.emission
    )
    components = rng.permutation(k)
    states = [rng.permutation(s) for _ in range(k)]
    candidate = MixtureHMM(
        truth.weights[components],
        np.stack([truth.initial[c, p] for c, p in zip(components, states, strict=True)]),
        np.stack(
            [truth.transition[c][np.ix_(p, p)] for c, p in zip(components, states, strict=True)]
        ),
        np.stack([truth.emission[c, p] for c, p in zip(components, states, strict=True)]),
    )
    result = parameter_recovery(truth, candidate)
    assert result["mean_parameter_mae"] == 0
    tokens, _ = truth.sample(rng, 4, 5)
    np.testing.assert_allclose(
        truth.filter(tokens).predictions, candidate.filter(tokens).predictions
    )


def test_nonidentifiable_negative_control_has_equal_outputs_but_different_parameters() -> None:
    weights, initial = np.ones(1), np.array([[0.5, 0.5]])
    emission = np.array([[[0.3, 0.7], [0.3, 0.7]]])
    a = MixtureHMM(weights, initial, np.array([[[0.9, 0.1], [0.1, 0.9]]]), emission)
    b = MixtureHMM(weights, initial, np.array([[[0.6, 0.4], [0.4, 0.6]]]), emission)
    tokens = np.array([[0, 1, 1, 0], [1, 0, 1, 1]], dtype=np.int64)
    np.testing.assert_allclose(a.filter(tokens).predictions, b.filter(tokens).predictions)
    assert parameter_recovery(a, b)["transition_mae"] == pytest.approx(0.3)
    assert not identifiability_diagnostics(a)["emissions_full_row_rank"]


def test_scores_known_error_and_rejects_incomparable_shapes() -> None:
    truth = MixtureHMM(np.ones(1), np.ones((1, 1)), np.ones((1, 1, 1)), np.array([[[0.75, 0.25]]]))
    candidate = MixtureHMM(
        truth.weights, truth.initial, truth.transition, np.array([[[0.25, 0.75]]])
    )
    result = parameter_recovery(truth, candidate)
    assert result["mean_parameter_mae"] == pytest.approx(0.125)
    assert result["transition_support_f1"] == 1
    with pytest.raises(ValueError, match="matching"):
        parameter_recovery(truth, random_mixture(Shape(1, 2, 2), 1))
    with pytest.raises(ValueError, match="threshold"):
        parameter_recovery(truth, candidate, 0)


def test_kl_matches_analytic_example_and_preserves_infinite_penalty() -> None:
    p = np.array([[[0.75, 0.25]], [[1.0, 0.0]]])
    q = np.full_like(p, 0.5)
    np.testing.assert_allclose(
        sequence_kl(p, q), [0.75 * np.log(1.5) + 0.25 * np.log(0.5), np.log(2)]
    )
    np.testing.assert_allclose(sequence_kl(p, p), 0)
    assert np.isinf(sequence_kl(np.array([[[0.5, 0.5]]]), np.array([[[1.0, 0.0]]])))[0]
    assert summarize(np.array([np.inf]))["mean_kl"] is None
    assert summarize(np.array([]))["upper95_kl"] is None
    assert summarize(np.array([0.1]))["upper95_kl"] is None


@pytest.mark.parametrize(
    "candidate",
    [
        np.array([[0.5, 0.5]]),
        np.array([[[0.3, 0.3]]]),
        np.array([[[np.nan, 0.5]]]),
        np.array([[[-0.5, 1.5]]]),
        np.empty((0, 1, 2)),
    ],
)
def test_kl_rejects_invalid_distributions(candidate: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"expected|probabilit"):
        sequence_kl(np.array([[[0.5, 0.5]]]), candidate)


def test_optimality_gate_checks_each_component_and_sequence_level_uncertainty() -> None:
    p = np.tile(np.array([0.8, 0.2]), (8, 3, 1))
    labels = np.repeat(np.arange(4), 2).astype(np.int64)
    result = optimality(p, p, labels, 4, 0.01)
    assert result["qualified"]
    assert result["overall"]["count"] == 8
    candidate = p.copy()
    candidate[6:] = [0.5, 0.5]
    failed = optimality(p, candidate, labels, 4, 0.1)
    assert failed["overall"]["mean_kl"] is not None
    assert failed["overall"]["mean_kl"] < 0.1
    assert not failed["qualified"]
    assert not optimality(p, p, labels, 5, 0.1)["qualified"]
    summary = summarize(np.array([0.0, 0.2]))
    assert summary["standard_error"] == pytest.approx(0.1)
    assert summary["upper95_kl"] == pytest.approx(0.296)


def test_linear_probe_generalizes_to_independent_sequences() -> None:
    rng = np.random.default_rng(22)
    train, test = rng.normal(size=(100, 3)), rng.normal(size=(30, 3))
    mapping = rng.normal(size=(3, 4))
    result = linear_probe(train, train @ mapping + 0.7, test, test @ mapping + 0.7)
    assert result["r2"] > 0.999
    assert result["mse"] < 1e-8
    constant = linear_probe(train, np.ones((100, 2)), test, np.ones((30, 2)))
    assert constant["mse"] == 0
