"""Standalone signed observable-operator predictor, with explicit numerical repairs."""

import json
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Tokens = NDArray[np.int64]


def readout(
    states: Array, normalizer: Array, operators: Array, baseline: Array
) -> tuple[Array, int]:
    raw = np.einsum("i,aij,bj->ba", normalizer, operators, states)
    bad = (~np.isfinite(raw)).any(axis=1) | (raw.sum(axis=1) <= 1e-12)
    repaired = bad | (raw < 1e-9).any(axis=1)
    raw[bad] = baseline
    probabilities = np.maximum(raw, 1e-9)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities, int(repaired.sum())


def update(
    states: Array, tokens: Tokens, initial: Array, normalizer: Array, operators: Array
) -> tuple[Array, int]:
    changed = np.einsum("bij,bj->bi", operators[tokens], states)
    mass = changed @ normalizer
    bad = (~np.isfinite(changed)).any(axis=1) | (~np.isfinite(mass)) | (mass <= 1e-12)
    changed[bad] = initial
    mass[bad] = 1
    return changed / mass[:, None], int(bad.sum())


def trajectory(
    tokens: Tokens, initial: Array, normalizer: Array, operators: Array, baseline: Array
) -> tuple[Array, dict[str, int]]:
    states = np.tile(initial, (len(tokens), 1))
    predictions = np.empty((*tokens.shape, len(operators)), dtype=np.float64)
    repairs = resets = 0
    for position in range(tokens.shape[1]):
        predictions[:, position], repaired = readout(states, normalizer, operators, baseline)
        repairs += repaired
        if position + 1 < tokens.shape[1]:
            states, reset = update(states, tokens[:, position], initial, normalizer, operators)
            resets += reset
    return predictions, {
        "repaired_predictions": repairs,
        "state_resets": resets,
        "prediction_vectors": tokens.size,
    }


def predict(
    histories: Tokens, initial: Array, normalizer: Array, operators: Array, baseline: Array
) -> Array:
    states = np.tile(initial, (len(histories), 1))
    for position in range(histories.shape[1]):
        states, _ = update(states, histories[:, position], initial, normalizer, operators)
    return readout(states, normalizer, operators, baseline)[0]


def main() -> None:
    request = json.load(sys.stdin)
    with np.load(Path(__file__).with_name("parameters.npz"), allow_pickle=False) as data:
        if request["vocab_size"] != len(data["operators"]):
            raise ValueError("vocabulary mismatch")
        probabilities = predict(
            np.asarray(request["histories"], dtype=np.int64),
            data["initial"],
            data["normalizer"],
            data["operators"],
            data["baseline"],
        )
    print(json.dumps({"probabilities": probabilities.tolist()}, allow_nan=False))


if __name__ == "__main__":
    main()
