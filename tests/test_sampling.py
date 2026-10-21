import json
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hidden_process_bench.config import SamplingConfig, Shape
from hidden_process_bench.equivalence import split_states
from hidden_process_bench.process import MixtureHMM, random_mixture
from hidden_process_bench.sampling import measure_difficulty, sample_task


def block_entropy(hmm: MixtureHMM, component: int, length: int) -> float:
    entropy = 0.0
    for tokens in product(range(hmm.shape.vocab), repeat=length):
        probability = 0.0
        for states in product(range(hmm.shape.states), repeat=length):
            path = hmm.initial[component, states[0]]
            for time, (token, state) in enumerate(zip(tokens, states, strict=True)):
                path *= hmm.emission[component, state, token]
                if time:
                    path *= hmm.transition[component, states[time - 1], state]
            probability += path
        if probability:
            entropy -= probability * np.log(probability)
    return float(entropy)


@pytest.mark.property
@settings(max_examples=8, deadline=None)
@given(seed=st.integers(0, 100))
def test_difficulty_matches_independent_hidden_path_enumeration(seed: int) -> None:
    hmm = random_mixture(Shape(2, 2, 2), seed)
    difficulty = measure_difficulty(hmm)
    for component in range(2):
        block = [0.0, *(block_entropy(hmm, component, length) for length in range(1, 5))]
        conditional = np.diff(block)
        np.testing.assert_allclose(
            difficulty.conditional_entropies[component], conditional, atol=1e-12
        )
        assert (np.diff(conditional) <= 1e-12).all()
        assert difficulty.unigram_gain[component] == pytest.approx(conditional[0] - conditional[3])
        assert difficulty.trigram_gain[component] == pytest.approx(conditional[2] - conditional[3])


def test_mixture_identity_cannot_disguise_iid_components() -> None:
    hmm = MixtureHMM(
        np.full(2, 0.5),
        np.full((2, 2), 0.5),
        np.full((2, 2, 2), 0.5),
        np.array([[[0.9, 0.1], [0.9, 0.1]], [[0.1, 0.9], [0.1, 0.9]]]),
    )
    predictions = hmm.filter(np.array([[0]], dtype=np.int64)).predictions
    assert not np.allclose(predictions[:, 0], predictions[:, 1])
    difficulty = measure_difficulty(hmm)
    np.testing.assert_allclose(difficulty.unigram_gain, 0, atol=1e-12)
    np.testing.assert_allclose(difficulty.trigram_gain, 0, atol=1e-12)
    assert difficulty.rejections(SamplingConfig()) == [
        "unigram_gain",
        "trigram_gain",
        "minimality_unresolved",
    ]


def test_fully_observed_markov_chain_fails_memory_beyond_trigram_screen() -> None:
    hmm = MixtureHMM(
        np.ones(1),
        np.full((1, 2), 0.5),
        np.array([[[0.9, 0.1], [0.1, 0.9]]]),
        np.eye(2)[None, :, :],
    )
    difficulty = measure_difficulty(hmm)
    entropy = -0.9 * np.log(0.9) - 0.1 * np.log(0.1)
    np.testing.assert_allclose(
        difficulty.conditional_entropies, [[np.log(2), entropy, entropy, entropy]], atol=1e-12
    )
    assert difficulty.unigram_gain[0] > 0.3
    assert difficulty.trigram_gain[0] == pytest.approx(0, abs=1e-12)
    assert difficulty.rejections(SamplingConfig()) == ["trigram_gain"]


def test_sampling_rejects_degenerate_tasks_and_is_reproducible(tmp_path: Path) -> None:
    first_path, second_path = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    first, metrics, attempts = sample_task(Shape(), 7, SamplingConfig(), first_path)
    second, other_metrics, other_attempts = sample_task(Shape(), 7, SamplingConfig(), second_path)
    assert attempts == other_attempts
    assert attempts > 1
    assert metrics == other_metrics
    np.testing.assert_array_equal(first.emission, second.emission)
    np.testing.assert_array_equal(first.transition, second.transition)
    assert first_path.read_bytes() == second_path.read_bytes()
    records = [json.loads(line) for line in first_path.read_text().splitlines()]
    assert len(records) == attempts
    assert all(record["rejections"] and not record["accepted"] for record in records[:-1])
    assert records[-1]["accepted"]
    assert not records[-1]["rejections"]
    assert records[0]["candidate_seed"] == 7
    assert len({record["candidate_seed"] for record in records}) == attempts
    assert all(value >= 0.05 for value in metrics.min_state_mass)
    assert all(0 < value <= 1 for value in metrics.transition_density)
    assert metrics.short_hankel_rank == metrics.joint_states == 12
    assert metrics.hankel_rank_tolerance == 1e-10
    with pytest.raises(FileExistsError):
        sample_task(Shape(), 7, SamplingConfig(), first_path)


def test_redundant_state_copies_cannot_inflate_task_order() -> None:
    original = random_mixture(Shape(1, 3, 6), 7)
    redundant = split_states(original)
    # Disable temporal thresholds to isolate the independently required order screen.
    config = SamplingConfig(min_state_mass=0, min_unigram_gain=0, min_trigram_gain=0)
    first, second = measure_difficulty(original), measure_difficulty(redundant)
    assert first.short_hankel_rank == first.joint_states == 3
    assert second.short_hankel_rank == 3
    assert second.joint_states == 6
    assert not first.rejections(config)
    assert second.rejections(config) == ["minimality_unresolved"]
    np.testing.assert_allclose(first.conditional_entropies, second.conditional_entropies)


def test_infeasible_minimality_screen_fails_before_sampling(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    with pytest.raises(ValueError, match="minimal-task screen"):
        sample_task(Shape(2, 2, 3), 7, SamplingConfig(), path)
    assert not path.exists()


def test_sampling_exhaustion_preserves_every_rejection(tmp_path: Path) -> None:
    path = tmp_path / "rejected.jsonl"
    with pytest.raises(ValueError, match="exhausted 3 candidates"):
        sample_task(Shape(1, 1, 2), 5, replace(SamplingConfig(), max_attempts=3), path)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 3
    assert all(not record["accepted"] for record in records)
    assert all("unigram_gain" in record["rejections"] for record in records)


def test_difficulty_requires_stationarity_and_bounds_enumeration() -> None:
    with pytest.raises(ValueError, match="stationary"):
        measure_difficulty(
            MixtureHMM(
                np.ones(1),
                np.array([[1.0, 0.0]]),
                np.full((1, 2, 2), 0.5),
                np.eye(2)[None, :, :],
            )
        )
    with pytest.raises(ValueError, match="bounded"):
        measure_difficulty(random_mixture(Shape(1, 1, 50), 0))


def test_sparsity_control_preserves_required_cycle_and_self_edges() -> None:
    sparse = random_mixture(Shape(1, 4, 5), 1, extra_edge_probability=0)
    dense = random_mixture(Shape(1, 4, 5), 1, extra_edge_probability=1)
    assert np.count_nonzero(sparse.transition) == 8
    assert np.count_nonzero(dense.transition) == 16
    np.testing.assert_array_equal(sparse.emission, dense.emission)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_sampling_probabilities(value: float) -> None:
    with pytest.raises(ValueError, match="probability"):
        SamplingConfig(extra_edge_probability=value)
    with pytest.raises(ValueError, match="probability"):
        random_mixture(Shape(), 0, extra_edge_probability=value)


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), True])
def test_invalid_sampling_gains(value: float) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        SamplingConfig(min_unigram_gain=value)
    with pytest.raises(ValueError, match="nonnegative"):
        SamplingConfig(min_trigram_gain=value)


def test_zero_thresholds_allow_explicit_iid_control_tasks(tmp_path: Path) -> None:
    config = SamplingConfig(min_unigram_gain=0, min_trigram_gain=0)
    _, metrics, attempts = sample_task(Shape(1, 1, 2), 0, config, tmp_path / "iid.jsonl")
    assert attempts == 1
    assert not metrics.rejections(config)
