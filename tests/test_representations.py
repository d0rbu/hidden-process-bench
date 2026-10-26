from typing import cast

import numpy as np
import pytest
import torch

from hidden_process_bench.config import ModelConfig, Shape
from hidden_process_bench.model import TinyTransformer, observe
from hidden_process_bench.process import random_mixture
from hidden_process_bench.representations import belief_dynamics, observe_traces, representation


def test_trace_collection_matches_generation_and_cleans_up_hooks() -> None:
    model = TinyTransformer(ModelConfig(vocab=3, context=4, width=8, heads=2, layers=2))
    reference = observe(model, 4, 4, 8)
    traces = observe_traces(model, 4, 4, 8)
    np.testing.assert_array_equal(reference.tokens, traces.observations.tokens)
    np.testing.assert_array_equal(reference.probabilities, traces.observations.probabilities)
    assert len(traces.residuals) == 2
    assert traces.mlp_inputs.shape == traces.mlp_outputs.shape == (4, 4, 8)
    assert all(not module._forward_hooks for module in model.modules())
    with pytest.raises(ValueError, match="context"):
        observe_traces(model, 4, 5, 8)
    assert all(not module._forward_hooks for module in model.modules())


@pytest.mark.parametrize("kind", ["pca", "sae", "transcoder"])
def test_representation_fits_without_truth_and_reports_heldout(kind: str) -> None:
    rng = np.random.default_rng(7)
    latent = rng.normal(size=(20, 4, 1))
    values = latent @ np.array([[1.0, 2.0, -1.0]]) + 0.1
    before = torch.random.get_rng_state().clone()
    result = representation(
        values,
        kind=kind,
        dimension=4 if kind != "pca" else 1,
        steps=150,
        seed=4,
        targets=values * 2 if kind == "transcoder" else None,
    )
    assert result.features.shape[:2] == (20, 4)
    assert cast(float, result.diagnostics["heldout_normalized_mse"]) < 0.15
    assert result.diagnostics["fit_sequences"] == 16
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    again = representation(
        values,
        kind=kind,
        dimension=4 if kind != "pca" else 1,
        steps=150,
        seed=4,
        targets=values * 2 if kind == "transcoder" else None,
    )
    np.testing.assert_array_equal(result.features, again.features)


def test_belief_dynamics_recovers_affine_updates_on_independent_sequences() -> None:
    hmm = random_mixture(Shape(1, 2, 3), 6)
    tokens, _ = hmm.sample(np.random.default_rng(9), 128, 12)
    oracle = hmm.filter(tokens)
    # A one-dimensional affine coordinate for a two-state normalized belief.
    features = 3 * oracle.beliefs[:, :-1, 0, :1] - 0.8
    operators, diagnostics = belief_dynamics(
        features, tokens, oracle.predictions[:, :-1], ridge=1e-10
    )
    assert operators.shape == (3, 2, 2)
    assert diagnostics["heldout_unnormalized_update_nmse"] < 1e-10


@pytest.mark.parametrize(
    ("values", "kind", "targets"),
    [
        (np.ones((1, 3, 2)), "pca", None),
        (np.ones((2, 3, 2)), "bad", None),
        (np.ones((2, 3, 2)), "transcoder", None),
        (np.ones((2, 3, 2)), "transcoder", np.ones((2, 3, 1))),
        (np.full((2, 3, 2), np.nan), "sae", None),
        (np.ones((2, 3)), "pca", None),
    ],
)
def test_representation_rejects_invalid_inputs(
    values: np.ndarray, kind: str, targets: np.ndarray | None
) -> None:
    with pytest.raises(ValueError, match=r"representation|transcoder"):
        representation(values, kind=kind, dimension=2, steps=1, seed=0, targets=targets)


def test_dynamics_rejects_misalignment_and_invalid_ridge() -> None:
    with pytest.raises(ValueError, match="aligned"):
        belief_dynamics(np.ones((2, 1, 2)), np.zeros((2, 1), dtype=np.int64), np.ones((2, 1, 1)))
    with pytest.raises(ValueError, match="aligned"):
        belief_dynamics(
            np.ones((2, 2, 2)), np.zeros((2, 2), dtype=np.int64), np.ones((2, 2, 1)), ridge=0
        )
