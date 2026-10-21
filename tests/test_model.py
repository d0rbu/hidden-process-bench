from pathlib import Path

import numpy as np
import pytest
import torch

from hidden_process_bench.config import ModelConfig
from hidden_process_bench.experiment import predictions
from hidden_process_bench.model import TinyTransformer, observe, teacher_inputs


def test_bos_alignment_and_causal_mask_prevent_future_leakage() -> None:
    torch.manual_seed(2)
    model = TinyTransformer(ModelConfig(4, 5, 12, 3, 2)).eval()
    tokens = torch.tensor([[0, 1, 2, 3, 1]])
    inputs = teacher_inputs(tokens, 4)
    assert inputs.tolist() == [[4, 0, 1, 2, 3]]
    changed = inputs.clone()
    changed[:, 3:] = 0
    before, hidden = model.forward(inputs)
    after, _ = model(changed)
    torch.testing.assert_close(before[:, :3], after[:, :3], rtol=0, atol=0)
    assert not torch.equal(before[:, 3:], after[:, 3:])
    assert before.shape == (1, 5, 4)
    assert hidden.shape == (1, 5, 12)


@pytest.mark.parametrize(
    "inputs",
    [
        torch.ones(2, dtype=torch.long),
        torch.ones((1, 0), dtype=torch.long),
        torch.ones((1, 4), dtype=torch.long),
    ],
)
def test_rejects_bad_context(inputs: torch.Tensor) -> None:
    model = TinyTransformer(ModelConfig(3, 3, 8, 2, 1))
    with pytest.raises(ValueError, match="context"):
        model(inputs)


def test_checkpoint_roundtrip_has_only_architecture_and_weights(tmp_path: Path) -> None:
    model = TinyTransformer(ModelConfig(3, 4, 8, 2, 1)).eval()
    path = tmp_path / "model.pt"
    model.save(path)
    payload = torch.load(path, weights_only=True)
    assert set(payload) == {"architecture", "state_dict"}
    restored = TinyTransformer.load(path)
    assert not restored.training
    inputs = torch.tensor([[3, 0, 1, 2]])
    torch.testing.assert_close(model(inputs)[0], restored(inputs)[0], rtol=0, atol=0)


def test_observations_have_correct_budget_and_teacher_forcing_alignment() -> None:
    model = TinyTransformer(ModelConfig(3, 5, 8, 2, 1))
    first = observe(model, 4, 5, 101)
    second = observe(model, 4, 5, 101)
    np.testing.assert_array_equal(first.tokens, second.tokens)
    assert first.forward_calls == 5
    assert first.token_queries == 20
    assert first.prefix_tokens_processed == 60
    probability, residual = predictions(model, first.tokens)
    np.testing.assert_allclose(first.probabilities, probability, atol=1e-7)
    np.testing.assert_allclose(first.activations, residual, atol=1e-6)
    assert (first.tokens < 3).all()
    with pytest.raises(ValueError, match="context"):
        observe(model, 4, 6, 101)
    with pytest.raises(TypeError, match="PositiveCount"):
        observe(model, 0, 2, 101)
