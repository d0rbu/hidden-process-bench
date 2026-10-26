from typing import cast

import numpy as np
import pytest
import torch

from hidden_process_bench.circuits import _continued, ablate, discover_circuit, interchange_test
from hidden_process_bench.config import ModelConfig, Shape
from hidden_process_bench.model import TinyTransformer, teacher_inputs
from hidden_process_bench.process import random_mixture
from hidden_process_bench.representations import observe_traces


def test_zero_ablation_changes_predictions_restores_model_and_removes_hooks() -> None:
    model = TinyTransformer(ModelConfig(vocab=3, context=4, width=8, heads=2, layers=1))
    tokens = torch.tensor([[3, 0, 1, 2]])
    before = model(tokens)[0]
    with ablate(model, {"0.head.1", "0.mlp"}):
        assert not torch.allclose(before, model(tokens)[0])
    torch.testing.assert_close(before, model(tokens)[0], rtol=0, atol=0)
    with pytest.raises(RuntimeError), ablate(model, {"0.head.0"}):
        raise RuntimeError("test cleanup")
    assert all(not module._forward_hooks for module in model.modules())


def test_discovery_reports_heldout_control_and_literal_query_counts() -> None:
    model = TinyTransformer(ModelConfig(vocab=3, context=4, width=8, heads=2, layers=1))
    traces = observe_traces(model, 6, 4, 6)
    removed = discover_circuit(model, traces.observations.tokens, tolerance=100)
    assert len(removed.removed) == 3
    assert removed.diagnostics["nodes_retained"] == 0
    kept = discover_circuit(model, traces.observations.tokens, tolerance=0)
    assert len(kept.removed) == 0
    assert kept.diagnostics["additional_forward_calls"] == 6
    assert kept.diagnostics["additional_token_queries"] == 96
    assert kept.diagnostics["heldout_kl"] == pytest.approx(0, abs=1e-15)
    assert cast(float, kept.diagnostics["all_removed_heldout_kl"]) > 0
    for tokens, tolerance in [
        (traces.observations.tokens[:1], 0),
        (traces.observations.tokens, -1),
    ]:
        with pytest.raises(ValueError, match="batch"):
            discover_circuit(model, tokens, tolerance)


def test_interchange_is_heldout_and_candidate_continuation_matches_oracle() -> None:
    model = TinyTransformer(ModelConfig(vocab=3, context=4, width=8, heads=2, layers=2))
    traces = observe_traces(model, 8, 4, 6)
    hmm = random_mixture(Shape(1, 2, 3), 4)
    tokens = traces.observations.tokens
    oracle = hmm.filter(tokens)
    np.testing.assert_allclose(
        _continued(hmm, oracle.beliefs[:, 2].copy(), tokens[:, 2:]), oracle.predictions[:, 2:-1]
    )
    diagnostics = interchange_test(model, hmm, tokens, traces.residuals[0])
    assert diagnostics["heldout_sequences"] == 4
    assert diagnostics["additional_forward_calls"] == 2
    assert cast(float, diagnostics["change_normalized_mse"]) >= 0
    assert cast(float, diagnostics["counterfactual_kl"]) >= 0
    assert all(not module._forward_hooks for module in model.modules())
    assert teacher_inputs(torch.tensor(tokens), 3).shape == tokens.shape
