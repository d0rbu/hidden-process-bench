import hashlib
import itertools
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from hidden_process_bench import operator_method
from hidden_process_bench.discovery import extract_process, predict_histories, score_process
from hidden_process_bench.discovery_cli import main
from hidden_process_bench.isolation import run_isolated
from hidden_process_bench.operator_method import factorize, identify, moments, prepare_bundle
from hidden_process_bench.operator_predictor import predict, readout, trajectory, update
from hidden_process_bench.submissions import FrozenSubmission, file_manifest, verify_submission


def test_published_programs_are_bound_to_the_registered_receipts(tmp_path: Path) -> None:
    root = Path("docs/results/operator-development")
    summary = json.loads((root / "summary.json").read_text())
    manifest = file_manifest(root)
    manifest.pop("manifest.json")
    assert manifest == json.loads((root / "manifest.json").read_text())
    with zipfile.ZipFile(root / "method.zip") as archive:
        assert {
            name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()
        } == summary["method_sha256"]
    for case in summary["cases"]:
        frozen = FrozenSubmission(
            root / case["id"] / "submission", case["submission_receipt_sha256"]
        )
        verify_submission(frozen)
        selection = json.loads((frozen.path / "selection.json").read_text())
        assert selection["dimension"] == case["state_dimension"]
        for split in ("heldout", "long"):
            assert case[split] == json.loads((root / case["id"] / (split + ".json")).read_text())
            assert case[split]["status"] == "scored"
            assert case[split]["submission_receipt_sha256"] == frozen.receipt_sha256
            assert case[split + "_max_total_variation"] > 0.3
        if case["history_independent_scalar_process"]:
            predictions = predict_histories(
                frozen, [[0, 1, 1], [1, 0, 2]], 16, tmp_path / case["id"]
            )
            assert np.allclose(predictions[0], predictions[1], atol=1e-12, rtol=0)


def test_exact_observable_moments_recover_predictions_and_change_of_basis() -> None:
    transition = np.array([[0.7, 0.3], [0.2, 0.8]])
    emission = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    stationary, start = np.array([0.4, 0.6]), np.array([0.8, 0.2])

    def word_probability(word: tuple[int, ...], initial: np.ndarray) -> float:
        # Independent enumeration of hidden paths, without a filter or a fitter.
        return sum(
            initial[states[0]]
            * np.prod([emission[state, token] for state, token in zip(states, word, strict=True)])
            * np.prod([transition[a, b] for a, b in itertools.pairwise(states)])
            for states in itertools.product(range(2), repeat=len(word))
        )

    single = np.array([word_probability((a,), stationary) for a in range(3)])
    pair = np.array([[word_probability((b, a), stationary) for b in range(3)] for a in range(3)])
    triple = np.array(
        [
            [[word_probability((b, x, a), stationary) for b in range(3)] for a in range(3)]
            for x in range(3)
        ]
    )
    initial, normalizer, operators = factorize(single, pair, triple, start @ emission, 2)
    transform = np.array([[2.0, 0.3], [-0.2, 0.8]])
    inverse = np.linalg.inv(transform)
    for length in range(5):
        histories = list(itertools.product(range(3), repeat=length))
        expected = np.array(
            [
                [
                    word_probability((*history, a), start)
                    / (word_probability(history, start) if history else 1.0)
                    for a in range(3)
                ]
                for history in histories
            ]
        )
        tokens = np.asarray(histories, dtype=np.int64)
        actual = predict(tokens, initial, normalizer, operators, single)
        assert np.allclose(actual, expected, atol=1e-12, rtol=0)
        transformed = predict(
            tokens,
            transform @ initial,
            normalizer @ inverse,
            np.einsum("ij,ajk,kl->ail", transform, operators, inverse),
            single,
        )
        assert np.allclose(transformed, actual, atol=1e-12, rtol=0)
    tokens = np.array([[0, 1, 2, 0], [2, 2, 1, 0]], dtype=np.int64)
    predictions, diagnostics = trajectory(tokens, initial, normalizer, operators, single)
    for position in range(tokens.shape[1]):
        assert np.allclose(
            predictions[:, position],
            predict(tokens[:, :position], initial, normalizer, operators, single),
        )
    assert diagnostics == {"repaired_predictions": 0, "state_resets": 0, "prediction_vectors": 8}


def test_numerical_repairs_are_explicit_and_finite() -> None:
    baseline = np.array([0.3, 0.7])
    normalizer = np.ones(1)
    operators = np.array([[[-1.0]], [[2.0]]])
    values, repairs = readout(np.array([[1.0], [np.nan], [-1.0]]), normalizer, operators, baseline)
    assert repairs == 3
    assert np.isfinite(values).all()
    assert np.allclose(values.sum(axis=1), 1)
    assert np.allclose(values[1:], [baseline, baseline])
    states, resets = update(
        np.array([[1.0], [np.inf]]), np.array([0, 1]), np.ones(1), normalizer, operators
    )
    assert resets == 2
    assert np.array_equal(states, np.ones((2, 1)))
    with pytest.raises(ValueError, match="initial observable state"):
        factorize(np.ones(2) / 2, np.eye(2), np.ones((2, 2, 2)), np.zeros(2), 1)


def observations() -> tuple[np.ndarray, np.ndarray]:
    tokens = np.asarray(list(itertools.product(range(2), repeat=5)), dtype=np.int64)
    return tokens, np.full((*tokens.shape, 2), 0.5)


def test_unknown_dimension_is_selected_from_public_sequences_reproducibly() -> None:
    tokens, probabilities = observations()
    parameters, report = identify(tokens, probabilities)
    repeated, again = identify(tokens, probabilities)
    assert report == again
    assert report["dimension"] == 1
    assert report["training_sequences"] == 24
    assert report["validation_sequences"] == 8
    assert all(np.array_equal(parameters[name], repeated[name]) for name in parameters)
    predicted, _ = trajectory(tokens, **parameters)
    assert np.allclose(predicted, 0.5, atol=1e-12, rtol=0)
    assert np.allclose(moments(tokens, probabilities)[0], [0.5, 0.5])
    # The smallest supported input still has two independent validation sequences.
    _, smallest = identify(tokens[:4], probabilities[:4])
    assert smallest["validation_sequences"] == 2


@pytest.mark.parametrize(
    "kind",
    [
        "shape",
        "short",
        "few",
        "vocab",
        "dtype",
        "negative",
        "outside",
        "nan",
        "negative_probability",
        "mass",
    ],
)
def test_invalid_observations_are_rejected(kind: str) -> None:
    tokens, probabilities = observations()
    if kind == "shape":
        probabilities = probabilities[:, :-1]
    elif kind == "short":
        tokens, probabilities = tokens[:, :2], probabilities[:, :2]
    elif kind == "few":
        tokens, probabilities = tokens[:3], probabilities[:3]
    elif kind == "vocab":
        probabilities = probabilities[:, :, :1]
    elif kind == "dtype":
        tokens = tokens.astype(float)
    elif kind == "negative":
        tokens[0, 0] = -1
    elif kind == "outside":
        tokens[0, 0] = 2
    elif kind == "nan":
        probabilities[0, 0, 0] = np.nan
    elif kind == "negative_probability":
        probabilities[0, 0] = [-0.1, 1.1]
    else:
        probabilities *= 0.5
    with pytest.raises(ValueError, match="aligned public observations"):
        identify(tokens, probabilities)


def test_failed_dimensions_remain_in_search_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise np.linalg.LinAlgError("factorization failed")

    monkeypatch.setattr(operator_method, "factorize", fail)
    with pytest.raises(ValueError, match=r'"dimension": 1.*factorization failed'):
        identify(*observations())


def test_nonfinite_validation_is_a_retained_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        operator_method,
        "trajectory",
        lambda tokens, *_args: (np.full((*tokens.shape, 2), np.nan), {}),
    )
    with pytest.raises(ValueError, match="nonfinite validation loss"):
        identify(*observations())


def test_real_isolated_extraction_and_standalone_prediction(tmp_path: Path) -> None:
    assert main(["prepare-operators", "--out", str(tmp_path / "method")]) == 0
    with pytest.raises(FileExistsError):
        prepare_bundle(tmp_path / "method")
    assert set((tmp_path / "method/method").iterdir()) == {
        tmp_path / f"method/method/{name}"
        for name in ("__init__.py", "operator_method.py", "operator_predictor.py")
    }
    challenge = tmp_path / "challenge"
    challenge.mkdir()
    tokens, probabilities = observations()
    np.savez(challenge / "observations.npz", tokens=tokens, probabilities=probabilities)
    frozen = extract_process(tmp_path / "method", "main.py", challenge, tmp_path / "extraction")
    assert frozen is not None, (tmp_path / "extraction/execution/stderr.txt").read_text()
    selection = json.loads((frozen.path / "selection.json").read_text())
    assert selection["dimension"] == 1
    report = score_process(
        frozen, tokens[:2], probabilities[:2], probabilities[:2], tmp_path / "score"
    )
    assert report["status"] == "scored"
    assert report["independent_structure_recovery"] is None
    assert np.allclose(
        predict_histories(frozen, [[0] * 40, [1] * 40], 2, tmp_path / "long"), 0.5, atol=1e-12
    )
    # The frozen executable rejects an incompatible neural vocabulary.
    result = run_isolated(
        frozen.path, "predict.py", {"histories": [[]], "vocab_size": 3}, tmp_path / "bad-vocab"
    )
    assert result.status == "failed"
    assert "vocabulary mismatch" in (tmp_path / "bad-vocab/stderr.txt").read_text()
