import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hidden_process_bench.config import Shape
from hidden_process_bench.generators import (
    HiddenProcess,
    LagProcess,
    SequenceProcess,
    StackProcess,
    load_process,
)
from hidden_process_bench.process import random_mixture


def reference_stack(history: list[int], process: StackProcess) -> tuple[np.ndarray, int]:
    stack: list[int] = []
    depth = 0
    for token in history:
        role = process.alphabet.index(token)
        if role in (0, 1):
            stack = [*stack, role]
        elif role in (2, 3) and stack and stack[-1] + 2 == role:
            stack = stack[:-1]
        depth = max(depth, len(stack))
    p = np.full(process.vocab, process.noise / process.vocab)
    for role, token in enumerate(process.alphabet):
        if role < 2:
            mass = process.push / 2
        elif role < 4:
            mass = process.close * (float(role == stack[-1] + 2) if stack else 0.5)
        else:
            mass = (1 - process.push - process.close) / (process.vocab - 4)
        p[token] += (1 - process.noise) * mass
    return p, depth


def test_lag_predictions_use_only_the_correct_delayed_token() -> None:
    process = LagProcess((2, 0, 1), 3, 0.8)
    tokens = np.array([[0, 1, 2, 1, 0], [2, 0, 1, 2, 1]], dtype=np.int64)
    actual = process.probabilities(tokens)
    assert actual.shape == (2, 6, 3)
    for row in range(2):
        for position in range(6):
            expected = np.full(3, 1 / 3)
            if position >= 3:
                expected = np.full(3, 0.2 / 3)
                expected[process.alphabet[tokens[row, position - 3]]] += 0.8
            np.testing.assert_allclose(actual[row, position], expected, atol=1e-15)
    changed = tokens.copy()
    changed[:, 3:] = 0
    np.testing.assert_array_equal(actual[:, :4], process.probabilities(changed)[:, :4])
    assert process.groups == 1


def test_stack_oracle_matches_independent_parser_on_every_short_history() -> None:
    process = StackProcess((3, 1, 4, 0, 2))
    histories = np.array(list(itertools.product(range(5), repeat=4)), dtype=np.int64)
    actual = process.probabilities(histories)
    for row, history in enumerate(histories):
        for position in range(5):
            expected, _ = reference_stack(history[:position].tolist(), process)
            np.testing.assert_allclose(actual[row, position], expected, atol=1e-15)
    assert (actual > 0).all()
    np.testing.assert_allclose(actual.sum(axis=-1), 1, atol=1e-15)
    # Summed word probabilities from the prefix oracle normalize independently.
    word_probabilities = np.prod(
        actual[:, :-1][np.arange(histories.shape[0])[:, None], np.arange(4), histories], axis=1
    )
    assert word_probabilities.sum() == pytest.approx(1)
    assert process.groups == 3


def test_stack_retains_deep_state_and_mismatch_semantics() -> None:
    process = StackProcess(tuple(range(6)))
    for history in ([0, 1, 2, 3, 2], [0] * 90 + [1] * 80 + [3] * 80, [2, 3, 4, 5]):
        predicted = process.probabilities(np.array([history], dtype=np.int64))[0, -1]
        expected, _ = reference_stack(history, process)
        np.testing.assert_allclose(predicted, expected)
    empty = process.probabilities(np.empty((2, 0), dtype=np.int64))
    assert empty.shape == (2, 1, 6)
    np.testing.assert_allclose(empty[0, 0], reference_stack([], process)[0])


@pytest.mark.parametrize("kind", ["lag", "stack", "hidden"])
def test_serialization_preserves_oracle_and_seeded_sampling(tmp_path: Path, kind: str) -> None:
    process: SequenceProcess
    if kind == "hidden":
        process = HiddenProcess(random_mixture(Shape(2, 2, 6), 7))
    elif kind == "lag":
        process = LagProcess(tuple(range(6)), 3, 0.7)
    else:
        process = StackProcess(tuple(range(6)))
    path = tmp_path / "process.json"
    path.write_text(json.dumps(process.specification()))
    restored = load_process(path)
    tokens, groups = process.sample(np.random.default_rng(21), 1000, 24)
    second, other_groups = restored.sample(np.random.default_rng(21), 1000, 24)
    np.testing.assert_array_equal(tokens, second)
    np.testing.assert_array_equal(groups, other_groups)
    np.testing.assert_array_equal(process.probabilities(tokens), restored.probabilities(tokens))
    assert restored.vocab == 6
    assert restored.groups == process.groups
    assert (tokens >= 0).all()
    assert (tokens < 6).all()
    assert (groups >= 0).all()
    assert (groups < process.groups).all()
    if kind == "stack":
        assert isinstance(restored, StackProcess)
        expected = [
            int(np.digitize(reference_stack(row.tolist(), restored)[1], [2, 4])) for row in tokens
        ]
        np.testing.assert_array_equal(groups, expected)
        assert set(groups) == {0, 1, 2}
    if kind == "hidden":
        assert isinstance(process, HiddenProcess)
        np.testing.assert_array_equal(
            process.probabilities(tokens), process.hmm.filter(tokens).predictions
        )


@pytest.mark.parametrize(
    "process", [LagProcess(tuple(range(6)), 3, 0.75), StackProcess(tuple(range(6)))]
)
def test_sampling_matches_conditionals_on_shared_histories(process: SequenceProcess) -> None:
    tokens, _ = process.sample(np.random.default_rng(2), 12000, 6)
    expected = process.probabilities(tokens)[:, :-1]
    for position in range(6):
        if isinstance(process, LagProcess) and position >= 3:
            rows = tokens[:, position - 3] == 1
        elif isinstance(process, StackProcess) and position > 0:
            # The all-neutral prefix has a known empty state and sufficient frequency.
            rows = (tokens[:, :position] >= 4).all(axis=1)
            if rows.sum() < 100:
                continue
        else:
            rows = np.ones(tokens.shape[0], dtype=bool)
        empirical = np.bincount(tokens[rows, position], minlength=6) / rows.sum()
        np.testing.assert_allclose(
            empirical, expected[rows, position].mean(axis=0), atol=0.07, rtol=0
        )


@pytest.mark.parametrize("alphabet", [(0,), (0, 0), (1, 2), (True, 0)])
def test_invalid_alphabets_rejected(alphabet: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="permutation"):
        LagProcess(alphabet, 1, 0.5)
    with pytest.raises(ValueError, match="permutation"):
        StackProcess(alphabet)


@pytest.mark.parametrize("value", [0, 1, -0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_process_probabilities_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="probabilit"):
        LagProcess(tuple(range(6)), 3, value)
    with pytest.raises(ValueError, match="probabilit"):
        StackProcess(tuple(range(6)), noise=value)


def test_invalid_positive_counts_and_stack_mass_rejected() -> None:
    with pytest.raises(TypeError, match="PositiveCount"):
        LagProcess((0, 1), 0, 0.5)
    with pytest.raises(ValueError, match="neutral"):
        StackProcess(tuple(range(6)), push=0.5, close=0.5)
    for process in (LagProcess(tuple(range(6)), 3, 0.7), StackProcess(tuple(range(6)))):
        with pytest.raises(TypeError, match="PositiveCount"):
            process.sample(np.random.default_rng(2), 0, 2)
        with pytest.raises(TypeError, match="PositiveCount"):
            process.sample(np.random.default_rng(2), 2, 0)
        with pytest.raises(ValueError, match="vocabulary"):
            process.probabilities(np.array([[6]], dtype=np.int64))


@pytest.mark.parametrize(
    "payload", [[], {}, {"schema_version": 2}, {"schema_version": 1, "family": "unknown"}]
)
def test_unknown_private_process_schema_rejected(tmp_path: Path, payload: object) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="private process"):
        load_process(path)


@pytest.mark.property
@settings(max_examples=25, deadline=None)
@given(
    st.permutations(tuple(range(6))).map(tuple),
    st.integers(1, 5),
    st.floats(0.1, 0.9, allow_nan=False, allow_infinity=False),
)
def test_lag_process_is_causal_normalized_and_fully_supported(
    alphabet: tuple[int, ...], lag: int, strength: float
) -> None:
    process = LagProcess(alphabet, lag, strength)
    tokens = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64)
    probabilities = process.probabilities(tokens)
    assert (probabilities > 0).all()
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1, atol=1e-15)
    for position in range(7):
        np.testing.assert_array_equal(
            probabilities[:, position], process.probabilities(tokens[:, :position])[:, -1]
        )
