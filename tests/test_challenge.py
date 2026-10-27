import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from hidden_process_bench.challenge import Access, export_challenge
from hidden_process_bench.config import ModelConfig
from hidden_process_bench.model import TinyTransformer, observe


@pytest.mark.parametrize("access", ["outputs", "weights", "internals"])
def test_neutral_snapshot_discloses_only_declared_neural_information(
    tmp_path: Path, access: Access
) -> None:
    model = TinyTransformer(ModelConfig(3, 4, 8, 2, 1))
    observations = observe(model, 2, 4, 93)
    output = tmp_path / access
    manifest = export_challenge(model, observations, access, output)
    payload = json.loads((output / "challenge.json").read_text())
    assert payload["architecture"] == {
        "vocab": 3,
        "context": 4,
        "width": 8,
        "heads": 2,
        "layers": 1,
    }
    assert payload["additional_teacher_queries"] == 0
    assert payload["cached_observation_budget"]["forward_calls"] == (
        0 if access == "weights" else 4
    )
    assert set(manifest) == {
        "challenge.json",
        "instructions.txt",
        *(["weights.npz"] if access != "outputs" else []),
        *(["observations.npz"] if access != "weights" else []),
    }
    text = (output / "challenge.json").read_text() + (output / "instructions.txt").read_text()
    for hint in ("HMM", "latent", "component", "states", "seed", "truth", "transition"):
        assert hint not in text
    if access != "weights":
        with np.load(output / "observations.npz", allow_pickle=False) as arrays:
            assert set(arrays.files) == {
                "tokens",
                "probabilities",
                *(["activations"] if access == "internals" else []),
            }
            np.testing.assert_array_equal(arrays["tokens"], observations.tokens)
            np.testing.assert_array_equal(arrays["probabilities"], observations.probabilities)
    if access != "outputs":
        with np.load(output / "weights.npz", allow_pickle=False) as weights:
            assert set(weights.files) == set(model.state_dict())
            for name, value in model.state_dict().items():
                np.testing.assert_array_equal(weights[name], value.numpy())
    with pytest.raises(FileExistsError):
        export_challenge(model, observations, access, output)


def test_invalid_observations_and_unknown_access_are_rejected(tmp_path: Path) -> None:
    model = TinyTransformer(ModelConfig(3, 4, 8, 2, 1))
    with pytest.raises(ValueError, match="unknown"):
        export_challenge(model, None, cast(Access, "oracle"), tmp_path / "unknown")
    with pytest.raises(ValueError, match="requires"):
        export_challenge(model, None, "outputs", tmp_path / "missing")
    valid = observe(model, 2, 4, 21)
    for broken in (
        replace(valid, tokens=valid.tokens[0]),
        replace(valid, tokens=valid.tokens[:, :0]),
        replace(valid, tokens=np.zeros((2, 5), dtype=np.int64)),
        replace(valid, probabilities=valid.probabilities[:, :, :2]),
        replace(valid, activations=valid.activations[:, :, :2]),
        replace(valid, tokens=valid.tokens.astype(np.float64)),
        replace(valid, tokens=np.full_like(valid.tokens, -1)),
        replace(valid, tokens=np.full_like(valid.tokens, 3)),
    ):
        with pytest.raises(ValueError, match="match"):
            export_challenge(model, broken, "internals", tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    export_challenge(model, None, "weights", tmp_path / "weights")
