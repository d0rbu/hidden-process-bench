from dataclasses import asdict

import numpy as np
import pytest

from hidden_process_bench.config import Shape
from hidden_process_bench.equivalence import compare_processes, process_complexity, split_states
from hidden_process_bench.experiment import score
from hidden_process_bench.process import MixtureHMM, random_mixture


def test_equivalent_larger_hmm_can_be_scored_without_parameter_alignment() -> None:
    original = random_mixture(Shape(2, 2, 4), 3)
    larger = split_states(original)
    result = compare_processes(original, larger)
    assert result.status == "numerically_equivalent"
    assert result.witness is None
    assert result.reference_probability is None
    assert result.candidate_probability is None
    assert result.max_checked_probability_gap < result.tolerance
    assert result.basis_rank <= result.joint_dimension
    assert result.words_checked <= 1 + original.shape.vocab * result.joint_dimension
    assert compare_processes(larger, original).status == result.status
    metrics = score(original, larger, 12, 16, 10)
    assert metrics["parameters"] is None
    assert metrics["equivalence"] == asdict(result)
    tokens, _ = original.sample(np.random.default_rng(10), 12, 16)
    np.testing.assert_allclose(
        original.filter(tokens).predictions, larger.filter(tokens).predictions
    )
    simple, redundant = process_complexity(original), process_complexity(larger)
    assert simple["short_hankel_rank"] == redundant["short_hankel_rank"]
    assert simple["joint_states"] * 2 == redundant["joint_states"]


def test_equivalence_finds_temporal_difference_despite_equal_unigrams() -> None:
    persistent = MixtureHMM(
        np.ones(1), np.full((1, 2), 0.5), np.array([[[0.9, 0.1], [0.1, 0.9]]]), np.eye(2)[None]
    )
    alternating = MixtureHMM(
        persistent.weights,
        persistent.initial,
        np.array([[[0.1, 0.9], [0.9, 0.1]]]),
        persistent.emission,
    )
    result = compare_processes(persistent, alternating)
    assert result.status == "different"
    assert result.witness == [0, 0]
    assert result.reference_probability == pytest.approx(0.45)
    assert result.candidate_probability == pytest.approx(0.05)
    assert result.max_checked_probability_gap == pytest.approx(0.4)


def test_equivalence_handles_impossible_words_and_degenerate_states() -> None:
    original = MixtureHMM(np.ones(1), np.ones((1, 1)), np.ones((1, 1, 1)), np.array([[[1.0, 0.0]]]))
    result = compare_processes(original, split_states(original))
    assert result.status == "numerically_equivalent"
    assert result.basis_rank == 1
    assert result.words_checked == 3
    complexity = process_complexity(original)
    assert complexity["short_hankel_rank"] == 1
    assert complexity["dense_family_degrees_of_freedom"] == 1
    assert complexity["effective_joint_states"] == 1
    assert process_complexity(split_states(original))["effective_joint_states"] == pytest.approx(2)


def test_equivalence_accepts_relabeling_and_unobservable_transition_changes() -> None:
    original = random_mixture(Shape(1, 3, 5), 7)
    order = [2, 0, 1]
    renamed = MixtureHMM(
        original.weights,
        original.initial[:, order],
        original.transition[:, order][:, :, order],
        original.emission[:, order],
    )
    assert compare_processes(original, renamed).status == "numerically_equivalent"
    emission = np.array([[[0.3, 0.7], [0.3, 0.7]]])
    first = MixtureHMM(np.ones(1), np.array([[1.0, 0.0]]), np.eye(2)[None], emission)
    second = MixtureHMM(np.ones(1), np.full((1, 2), 0.5), np.full((1, 2, 2), 0.5), emission)
    assert compare_processes(first, second).status == "numerically_equivalent"


def test_equivalence_validates_tolerance_and_vocabulary() -> None:
    first = random_mixture(Shape(1, 1, 2), 0)
    with pytest.raises(ValueError, match="positive"):
        compare_processes(first, first, tolerance=0)
    with pytest.raises(ValueError, match="vocabulary"):
        compare_processes(first, random_mixture(Shape(1, 1, 3), 0))
    with pytest.raises(TypeError, match="PositiveCount"):
        split_states(first, copies=0)
