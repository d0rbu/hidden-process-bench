import itertools
import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from hidden_process_bench import state_geometry
from hidden_process_bench.discovery import extract_process, predict_histories
from hidden_process_bench.operator_predictor import predict, trajectory
from hidden_process_bench.operator_structure import recognize
from hidden_process_bench.state_geometry import (
    coordinates,
    fit_operators,
    identify,
    prepare_bundle,
    project,
)


def known_process() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start = np.array([0.4, 0.6])
    transition = np.array([[0.8, 0.2], [0.1, 0.9]])
    emission = np.array([[0.7, 0.2, 0.1], [0.05, 0.15, 0.8]])
    operators = np.asarray([transition.T @ np.diag(e) for e in emission.T])
    return start, transition, emission, operators


def test_unknown_latent_realization_survives_arbitrary_coordinate_change() -> None:
    start, transition, emission, operators = known_process()
    transform = np.array([[2.0, -0.3], [0.2, 0.8]])
    inverse = np.linalg.inv(transform)
    result = recognize(transform @ start, np.ones(2) @ inverse, transform @ operators @ inverse)
    assert result["family"] == "node_emitting_latent_markov"
    residual = result["relative_residual"]
    assert isinstance(residual, float)
    assert residual < 1e-12
    assert not result["minimality_certificate"]
    recovered_start = np.asarray(result["initial"])
    recovered_transition = np.asarray(result["transition"])
    recovered_emission = np.asarray(result["emission"])
    assert any(
        np.allclose(recovered_start, start[list(order)], atol=1e-12)
        and np.allclose(recovered_transition, transition[np.ix_(order, order)], atol=1e-12)
        and np.allclose(recovered_emission, emission[list(order)], atol=1e-12)
        for order in itertools.permutations(range(2))
    )
    result = recognize(np.ones(1), np.ones(1), np.array([[[0.3]], [[0.7]]]))
    assert result["family"] == "memoryless_categorical"


def test_noncommuting_edge_process_is_not_forced_into_a_latent_template() -> None:
    operators = np.random.default_rng(4).uniform(size=(3, 2, 2))
    operators /= operators.sum(axis=(0, 1))
    result = recognize(np.ones(2) / 2, np.ones(2), operators)
    assert result["family"] == "unresolved_linear_state"
    assert result["reason"] == "operators_do_not_share_emission_coordinates"


@pytest.mark.parametrize("failure", ["shape", "nan", "singular", "mass", "complex", "negative"])
def test_incompatible_realizations_are_explicitly_rejected(failure: str) -> None:
    start, _, _, operators = known_process()
    normalizer = np.ones(2)
    if failure == "shape":
        with pytest.raises(ValueError, match="invalid linear"):
            recognize(start, normalizer, operators[:, :1])
        return
    if failure == "nan":
        start[0] = np.nan
        with pytest.raises(ValueError, match="invalid linear"):
            recognize(start, normalizer, operators)
        return
    if failure == "singular":
        operators[:] = 0
        reason = "singular_or_ill_conditioned_total_operator"
    elif failure == "mass":
        normalizer = np.zeros(2)
        reason = "unusable_joint_coordinates"
    elif failure == "complex":
        operators = np.array([[[0.0, -1.0], [1.0, 0.0]], [[1.0, 1.0], [-1.0, 1.0]]])
        reason = "complex_joint_coordinates"
    else:
        start = np.array([1.1, -0.1])
        reason = "joint_coordinates_are_not_stochastic"
    assert recognize(start, normalizer, operators)["reason"] == reason


def observations() -> tuple[np.ndarray, np.ndarray]:
    start, _, _, operators = known_process()
    tokens = np.random.default_rng(10).integers(3, size=(80, 12), dtype=np.int64)
    probabilities, _ = trajectory(tokens, start, np.ones(2), operators, np.ones(3) / 3)
    return tokens, probabilities


def test_predictive_coordinates_learn_autonomous_update_rules_without_state_labels() -> None:
    tokens, probabilities = observations()
    # Features are a nontrivial invertible embedding of outputs, without latent labels.
    features = probabilities @ np.array([[1.0, -2.0, 0.2], [0.1, 0.4, 0.7], [-0.3, 0.5, 2.0]])
    states, mean, basis = coordinates(features, 2)
    assert np.allclose(states[..., 0], 1)
    assert np.allclose(states[..., 1].mean(), 0, atol=1e-12)
    assert np.allclose(states[..., 1].var(), 1, atol=1e-12)
    assert np.allclose(states, project(features, mean, basis))
    parameters = fit_operators(tokens, probabilities, states, 1e-12)
    for position in range(tokens.shape[1]):
        actual = predict(tokens[:, :position], **parameters)
        assert np.allclose(actual, probabilities[:, position], atol=1e-8, rtol=0)
    fitted, report = identify(tokens, probabilities, features)
    assert report["dimension"] == 2
    assert report["training_sequences"] == 60
    assert report["validation_sequences"] == 20
    structure = cast(dict[str, object], report["structure"])
    assert structure["family"] == "node_emitting_latent_markov"
    actual, _ = trajectory(tokens, **fitted)
    assert np.max(np.abs(actual - probabilities)) < 1e-4
    with pytest.raises(ValueError, match="unsupported coordinate"):
        coordinates(features, 3)


@pytest.mark.parametrize(
    "failure", ["shape", "few", "short", "dtype", "token", "negative", "mass", "nan", "features"]
)
def test_geometry_rejects_invalid_inputs(failure: str) -> None:
    tokens, probabilities = observations()
    features = probabilities.copy()
    if failure == "shape":
        probabilities = probabilities[:, :-1]
    elif failure == "few":
        tokens = tokens[:2]
        probabilities, features = probabilities[:2], features[:2]
    elif failure == "short":
        tokens, probabilities, features = tokens[:, :1], probabilities[:, :1], features[:, :1]
    elif failure == "dtype":
        tokens = tokens.astype(float)
    elif failure == "token":
        tokens[0, 0] = 3
    elif failure == "negative":
        probabilities[0, 0, 0] = -1
    elif failure == "mass":
        probabilities *= 2
    elif failure == "nan":
        features[0, 0, 0] = np.nan
    else:
        features = features[:, :, :0]
    with pytest.raises(ValueError, match="aligned finite"):
        identify(tokens, probabilities, features)


def test_failed_coordinate_regressions_are_retained(monkeypatch: pytest.MonkeyPatch) -> None:
    tokens, probabilities = observations()

    def fail(*_args: object) -> dict[str, np.ndarray]:
        raise np.linalg.LinAlgError("regression failed")

    monkeypatch.setattr(state_geometry, "fit_operators", fail)
    with pytest.raises(ValueError, match=r"all coordinate dynamics failed.*regression failed"):
        identify(tokens, probabilities, probabilities)


def test_nonfinite_validation_is_not_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    tokens, probabilities = observations()
    monkeypatch.setattr(
        state_geometry, "trajectory", lambda t, **_kw: (np.full((*t.shape, 3), np.nan), {})
    )
    with pytest.raises(ValueError, match="nonfinite autonomous"):
        identify(tokens, probabilities, probabilities)


@pytest.mark.parametrize("features", ["outputs", "activations"])
def test_real_isolated_geometry_method_and_teacher_free_prediction(
    tmp_path: Path, features: str
) -> None:
    tokens, probabilities = observations()
    challenge = tmp_path / "challenge"
    challenge.mkdir()
    np.savez(
        challenge / "observations.npz",
        tokens=tokens,
        probabilities=probabilities,
        activations=probabilities * 2,
    )
    prepare_bundle(tmp_path / "method", features)
    frozen = extract_process(tmp_path / "method", "main.py", challenge, tmp_path / "run")
    assert frozen is not None, (tmp_path / "run/execution/stderr.txt").read_text()
    selected = json.loads((frozen.path / "selection.json").read_text())
    assert selected["dimension"] == 2
    predicted = predict_histories(frozen, [[0, 2, 1], [1, 0, 0]], 3, tmp_path / "predict")
    start, _, _, operators = known_process()
    expected = predict(
        np.array([[0, 2, 1], [1, 0, 0]]), start, np.ones(2), operators, np.ones(3) / 3
    )
    assert np.allclose(predicted, expected, atol=1e-4, rtol=0)
    with pytest.raises(FileExistsError):
        prepare_bundle(tmp_path / "method", features)
    with pytest.raises(ValueError, match="unsupported feature"):
        prepare_bundle(tmp_path / "bad", "truth")


def test_iid_coordinates_and_unseen_token_remain_well_defined() -> None:
    tokens = np.zeros((4, 3), dtype=np.int64)
    probabilities = np.full((4, 3, 2), 0.5)
    parameters, report = identify(tokens, probabilities, probabilities)
    assert report["dimension"] == 1
    assert report["validation_sequences"] == 2
    structure = cast(dict[str, object], report["structure"])
    assert structure["family"] == "memoryless_categorical"
    assert np.allclose(predict(np.ones((2, 20), dtype=np.int64), **parameters), 0.5)
