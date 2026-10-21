import numpy as np
import pytest
import torch

from hidden_process_bench.config import Shape
from hidden_process_bench.metrics import sequence_kl
from hidden_process_bench.process import random_mixture
from hidden_process_bench.recovery import DifferentiableHMM, fit_hmm


def test_differentiable_filter_matches_numpy_oracle_and_finite_difference() -> None:
    model = DifferentiableHMM(Shape(2, 2, 3), 9)
    tokens = torch.tensor([[0, 1, 2], [2, 1, 0]])
    predictions, beliefs = model(tokens)
    reference = model.export().filter(tokens.numpy())
    np.testing.assert_allclose(predictions.detach(), reference.predictions[:, :-1], atol=1e-12)
    np.testing.assert_allclose(
        beliefs.detach(), reference.beliefs[:, :-1].reshape(2, 3, 4), atol=1e-12
    )
    loss = -predictions.gather(-1, tokens[..., None]).log().mean()
    loss.backward()
    gradient = model.emission_logits.grad
    assert gradient is not None
    derivative = float(gradient[0, 0, 0])
    objectives = []
    with torch.no_grad():
        original = float(model.emission_logits[0, 0, 0])
        for delta in (-1e-6, 1e-6):
            model.emission_logits[0, 0, 0] = original + delta
            objectives.append(float(-model(tokens)[0].gather(-1, tokens[..., None]).log().mean()))
        model.emission_logits[0, 0, 0] = original
    assert derivative == pytest.approx((objectives[1] - objectives[0]) / 2e-6, abs=1e-8)


def test_sample_only_recovery_fits_empirical_frequency() -> None:
    tokens = np.tile(np.array([[0, 0, 0, 1]], dtype=np.int64), (16, 1))
    result = fit_hmm(Shape(1, 1, 2), tokens, steps=90, restarts=2, seed=17)
    np.testing.assert_allclose(result.hmm.emission[0, 0], [0.75, 0.25], atol=0.015)
    assert result.objective_name == "sample_nll"
    assert result.objective == min(result.restart_objectives)
    assert result.seconds > 0


def test_probability_recovery_fits_soft_targets_without_generator_access() -> None:
    tokens = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.int64)
    probabilities = np.tile(np.array([0.8, 0.2]), (2, 3, 1))
    before = torch.random.get_rng_state().clone()
    result = fit_hmm(
        Shape(1, 1, 2), tokens, steps=100, restarts=1, seed=5, probabilities=probabilities
    )
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    recovered = result.hmm.filter(tokens).predictions[:, :-1]
    assert sequence_kl(probabilities, recovered).mean() < 0.001
    assert result.objective_name == "soft_target_cross_entropy"


def test_activation_objective_uses_side_information_without_true_labels() -> None:
    shape = Shape(1, 2, 3)
    truth = random_mixture(shape, 4)
    tokens, _ = truth.sample(np.random.default_rng(5), 8, 4)
    oracle = truth.filter(tokens)
    probabilities = oracle.predictions[:, :-1]
    beliefs = oracle.beliefs[:, :-1].reshape(8, 4, 2)
    activations = beliefs @ np.array([[1.0, -1.0, 0.5], [0.0, 2.0, 1.0]]) + 0.7
    baseline = fit_hmm(shape, tokens, steps=15, restarts=1, seed=3, probabilities=probabilities)
    assisted = fit_hmm(
        shape,
        tokens,
        steps=15,
        restarts=1,
        seed=3,
        probabilities=probabilities,
        activations=activations,
    )
    repeated = fit_hmm(
        shape,
        tokens,
        steps=15,
        restarts=1,
        seed=3,
        probabilities=probabilities,
        activations=activations,
    )
    assert assisted.fit_parameters > baseline.fit_parameters
    assert "activation_mse" in assisted.objective_name
    assert not np.allclose(assisted.hmm.transition, baseline.hmm.transition)
    np.testing.assert_array_equal(assisted.hmm.transition, repeated.hmm.transition)
    assert assisted.objective == repeated.objective


@pytest.mark.parametrize(
    "probabilities", [np.ones((1, 2, 2)), np.array([[[0.3, 0.3]]]), np.array([[[np.nan, 0.5]]])]
)
def test_recovery_rejects_invalid_probabilities(probabilities: np.ndarray) -> None:
    with pytest.raises(ValueError, match="probabilit"):
        fit_hmm(
            Shape(1, 1, 2),
            np.array([[0]], dtype=np.int64),
            steps=1,
            restarts=1,
            seed=0,
            probabilities=probabilities,
        )


@pytest.mark.parametrize(
    "activations", [np.ones((1, 2, 3)), np.ones((1, 1, 0)), np.array([[[np.nan]]]), np.ones((1, 1))]
)
def test_recovery_rejects_invalid_activations(activations: np.ndarray) -> None:
    with pytest.raises(ValueError, match="activations"):
        fit_hmm(
            Shape(1, 1, 2),
            np.array([[0]], dtype=np.int64),
            steps=1,
            restarts=1,
            seed=0,
            probabilities=np.array([[[0.5, 0.5]]]),
            activations=activations,
        )


def test_activation_recovery_requires_probabilities_and_positive_budgets() -> None:
    tokens = np.array([[0]], dtype=np.int64)
    with pytest.raises(ValueError, match="requires probabilities"):
        fit_hmm(Shape(1, 1, 2), tokens, steps=1, restarts=1, seed=0, activations=np.ones((1, 1, 1)))
    with pytest.raises(TypeError, match="PositiveCount"):
        fit_hmm(Shape(1, 1, 2), tokens, steps=0, restarts=1, seed=0)


def test_nonfinite_optimization_fails_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_forward(
        self: DifferentiableHMM, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.full((*tokens.shape, 2), float("nan")), self.weight_logits

    monkeypatch.setattr(DifferentiableHMM, "forward", broken_forward)
    with pytest.raises(ValueError, match="non-finite"):
        fit_hmm(Shape(1, 1, 2), np.array([[0]], dtype=np.int64), steps=1, restarts=1, seed=0)
