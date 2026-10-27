import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hidden_process_bench.challenge import export_challenge
from hidden_process_bench.config import ModelConfig
from hidden_process_bench.discovery import extract_process, predict_histories
from hidden_process_bench.discovery_features import (
    average_kl,
    circuit_features,
    encode,
    learn_rectifier,
    prepare_feature_bundle,
    rectifier_loss,
)
from hidden_process_bench.model import TinyTransformer, observe
from hidden_process_bench.neural_numpy import NeuralModel


def test_sparse_rectifier_gradients_match_finite_differences() -> None:
    rng = np.random.default_rng(33)
    inputs, targets = rng.normal(size=(7, 3)), rng.normal(size=(7, 2))
    parameters = {
        "encoder": rng.normal(size=(3, 4)),
        "encoder_bias": np.full(4, 0.31),
        "decoder": rng.normal(size=(4, 2)),
        "decoder_bias": np.ones(2),
    }
    _, gradient = rectifier_loss(inputs, targets, parameters, 0.01)
    for key, values in parameters.items():
        index = tuple(0 for _ in values.shape)
        original = values[index]
        values[index] = original + 1e-6
        above = rectifier_loss(inputs, targets, parameters, 0.01)[0]
        values[index] = original - 1e-6
        below = rectifier_loss(inputs, targets, parameters, 0.01)[0]
        values[index] = original
        assert abs((above - below) / 2e-6 - gradient[key][index]) < 1e-6


def test_learned_features_reconstruct_low_rank_data_and_have_reproducible_accounting() -> None:
    rng = np.random.default_rng(5)
    inputs = rng.normal(size=(100, 2)) @ rng.normal(size=(2, 6))
    parameters, report = learn_rectifier(inputs, inputs, steps=400)
    repeated, again = learn_rectifier(inputs, inputs, steps=400)
    assert report == again
    assert all(np.array_equal(parameters[k], repeated[k]) for k in parameters)
    initial, final = report["initial_objective"], report["final_objective"]
    assert isinstance(initial, float)
    assert isinstance(final, float)
    assert final < initial / 10
    codes = encode(inputs, parameters)
    reconstructed = (codes @ parameters["decoder"] + parameters["decoder_bias"]) * parameters[
        "output_scale"
    ] + parameters["output_center"]
    assert np.mean((reconstructed - inputs) ** 2) < 0.03
    assert codes.shape == (100, 12)
    assert np.isfinite(codes).all()
    assert (codes >= 0).all()


@pytest.mark.parametrize("kind", ["shape", "empty", "nan", "rows", "steps"])
def test_bad_representation_training_inputs_are_rejected(kind: str) -> None:
    inputs, targets, steps = np.ones((4, 2)), np.ones((4, 3)), 1
    if kind == "shape":
        inputs = inputs[:, :, None]
    elif kind == "empty":
        inputs, targets = inputs[:0], targets[:0]
    elif kind == "nan":
        inputs[0, 0] = np.nan
    elif kind == "rows":
        targets = targets[:2]
    else:
        steps = 0
    with pytest.raises(ValueError, match="rectifier requires"):
        learn_rectifier(inputs, targets, steps=steps)


def make_challenge(path: Path, *, zero_heads: bool = False) -> None:
    with torch.random.fork_rng():
        torch.manual_seed(5)
        model = TinyTransformer(ModelConfig(3, 6, 8, 2, 1)).eval()
    if zero_heads:
        with torch.no_grad():
            for name, tensor in model.named_parameters():
                if "attn_out" in name:
                    tensor.zero_()
    export_challenge(model, observe(model, 8, 6, 97), "internals", path)


def test_greedy_ablation_executes_a_real_circuit_and_counts_returned_vectors(
    tmp_path: Path,
) -> None:
    make_challenge(tmp_path / "challenge", zero_heads=True)
    model = NeuralModel(tmp_path / "challenge")
    inputs = np.array([[3, 0, 1], [3, 2, 0]], dtype=np.int64)
    reference, _ = model.forward(inputs)
    features, evidence = circuit_features(model, inputs, reference, np.array([0], dtype=np.int64))
    assert evidence["removed_heads"] == [(0, 0), (0, 1)]
    assert features.shape == (2, 3, 8)
    assert model.counts == {"forward_calls": 4, "token_queries": 24, "prefix_tokens_processed": 24}
    assert abs(average_kl(reference, reference)) < 1e-12
    with pytest.raises(ValueError, match="replay disagrees"):
        circuit_features(
            model, inputs, np.full(reference.shape, 1 / 3), np.array([0], dtype=np.int64)
        )


@pytest.mark.parametrize("mode", ["sae", "transcoder", "causal"])
def test_real_representation_pipeline_freezes_an_autonomous_process(
    tmp_path: Path, mode: str
) -> None:
    make_challenge(tmp_path / "challenge")
    prepare_feature_bundle(tmp_path / "method", mode)
    frozen = extract_process(
        tmp_path / "method", "main.py", tmp_path / "challenge", tmp_path / "run"
    )
    assert frozen is not None, (tmp_path / "run/execution/stderr.txt").read_text()
    resources = json.loads((frozen.path / "submission.json").read_text())["resources"]
    counts = resources["neural_execution_counts"]
    assert counts["forward_calls"] == {"sae": 0, "transcoder": 1, "causal": 3}[mode]
    if mode != "causal":
        evidence = json.loads((frozen.path / "representation.json").read_text())
        assert evidence["representation_training_sequences"] == 6
        assert not evidence["representation_validation_sequences_used_for_training"]
    probabilities = predict_histories(frozen, [[0, 1, 2], [2, 0, 0]], 3, tmp_path / "predictions")
    assert probabilities.shape == (2, 3)
    assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-12, rtol=0)
    assert not (frozen.path / "weights.npz").exists()
    with pytest.raises(FileExistsError):
        prepare_feature_bundle(tmp_path / "method", mode)


def test_feature_bundle_rejects_unknown_mode_and_transcoder_rejects_bad_replay(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="unsupported discovery"):
        prepare_feature_bundle(tmp_path / "bad", "hmm")
    make_challenge(tmp_path / "challenge")
    data_path = tmp_path / "challenge/observations.npz"
    with np.load(data_path) as data:
        values = {key: data[key] for key in data.files}
    values["probabilities"] = values["probabilities"][:, :, ::-1]
    np.savez(data_path, **values)
    prepare_feature_bundle(tmp_path / "method", "transcoder")
    result = extract_process(
        tmp_path / "method", "main.py", tmp_path / "challenge", tmp_path / "run"
    )
    assert result is None
    assert "replay disagrees" in (tmp_path / "run/execution/stderr.txt").read_text()
