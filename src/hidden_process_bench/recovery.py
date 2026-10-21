"""HMM extraction using samples, probabilities, or probabilities plus activations."""

from dataclasses import dataclass
from time import perf_counter
from typing import override

import numpy as np
import torch
from torch import Tensor, nn

from hidden_process_bench.config import Shape, check_count, check_positive
from hidden_process_bench.metrics import check_probabilities
from hidden_process_bench.process import Floats, MixtureHMM, Tokens, check_tokens


class DifferentiableHMM(nn.Module):
    def __init__(self, shape: Shape, seed: int) -> None:
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        k, s, v = shape.components, shape.states, shape.vocab
        self.weight_logits = nn.Parameter(torch.randn(k, generator=rng, dtype=torch.float64))
        self.initial_logits = nn.Parameter(torch.randn(k, s, generator=rng, dtype=torch.float64))
        self.transition_logits = nn.Parameter(
            torch.randn(k, s, s, generator=rng, dtype=torch.float64)
        )
        self.emission_logits = nn.Parameter(
            torch.randn(k, s, v, generator=rng, dtype=torch.float64)
        )

    def probabilities(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            self.weight_logits.softmax(dim=-1),
            self.initial_logits.softmax(dim=-1),
            self.transition_logits.softmax(dim=-1),
            self.emission_logits.softmax(dim=-1),
        )

    @override
    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        weights, initial, transition, emission = self.probabilities()
        belief = (weights[:, None] * initial).expand(len(tokens), -1, -1)
        predictions, beliefs = [], []
        for t in range(tokens.shape[1]):
            predictions.append(torch.einsum("bks,ksv->bv", belief, emission))
            beliefs.append(belief.flatten(start_dim=1))
            posterior = belief * emission[:, :, tokens[:, t]].permute(2, 0, 1)
            posterior = posterior / posterior.sum(dim=(1, 2), keepdim=True)
            belief = torch.einsum("bks,ksj->bkj", posterior, transition)
        return torch.stack(predictions, dim=1), torch.stack(beliefs, dim=1)

    def export(self) -> MixtureHMM:
        return MixtureHMM(*(value.detach().numpy() for value in self.probabilities()))


@dataclass(frozen=True)
class RecoveryResult:
    hmm: MixtureHMM
    objective: float
    objective_name: str
    restart_objectives: list[float]
    seconds: float
    fit_parameters: int


def _objective(
    model: DifferentiableHMM,
    observed: Tensor,
    targets: Tensor | None,
    decoder: nn.Linear | None,
    hidden: Tensor | None,
    activation_weight: float,
) -> Tensor:
    predicted, beliefs = model(observed)
    log_prob = predicted.clamp_min(torch.finfo(torch.float64).tiny).log()
    loss = (
        -(targets * log_prob).sum(dim=-1).mean()
        if targets is not None
        else -log_prob.gather(-1, observed[..., None]).mean()
    )
    if decoder is not None and hidden is not None:
        loss = loss + activation_weight * torch.square(decoder(beliefs) - hidden).mean()
    return loss


def fit_hmm(
    shape: Shape,
    tokens: Tokens,
    *,
    steps: int,
    restarts: int,
    seed: int,
    probabilities: Floats | None = None,
    activations: Floats | None = None,
    activation_weight: float = 0.1,
) -> RecoveryResult:
    """No ground-truth parameters, labels, generating seed, or scorer enter this API."""
    check_tokens(tokens, shape.vocab)
    check_count(tokens.shape[1])
    check_count(steps)
    check_count(restarts)
    check_positive(activation_weight)
    if probabilities is not None:
        if probabilities.shape != (*tokens.shape, shape.vocab):
            raise ValueError("probabilities must match [batch, length, vocab]")
        check_probabilities(probabilities)
    if activations is not None:
        if probabilities is None:
            raise ValueError("activation-assisted recovery requires probabilities")
        if (
            activations.ndim != 3
            or activations.shape[:2] != tokens.shape
            or activations.shape[2] == 0
            or not np.isfinite(activations).all()
        ):
            raise ValueError("activations must be finite [batch, length, width]")
    observed = torch.tensor(tokens, dtype=torch.long)
    targets = torch.tensor(probabilities) if probabilities is not None else None
    hidden = None
    if activations is not None:
        hidden = torch.tensor(activations)
        hidden = (hidden - hidden.mean(dim=(0, 1))) / hidden.std(
            dim=(0, 1), correction=0
        ).clamp_min(1e-8)
    started = perf_counter()
    outcomes: list[tuple[float, MixtureHMM, int]] = []
    for restart in range(restarts):
        # Local deterministic initialization without changing the caller's RNG state.
        with torch.random.fork_rng():
            torch.manual_seed(seed + restart)
            model = DifferentiableHMM(shape, seed + restart)
            decoder = (
                nn.Linear(shape.components * shape.states, hidden.shape[-1], dtype=torch.float64)
                if hidden is not None
                else None
            )
        parameters = list(model.parameters()) + (
            list(decoder.parameters()) if decoder is not None else []
        )
        optimizer = torch.optim.Adam(parameters, lr=0.05)

        for _ in range(steps):
            optimizer.zero_grad()
            loss = _objective(model, observed, targets, decoder, hidden, activation_weight)
            if not torch.isfinite(loss):
                raise ValueError("non-finite extraction objective")
            loss.backward()
            optimizer.step()
        final_loss = _objective(model, observed, targets, decoder, hidden, activation_weight)
        outcomes.append(
            (float(final_loss.detach()), model.export(), sum(p.numel() for p in parameters))
        )
    best = min(outcomes, key=lambda item: item[0])
    name = "sample_nll" if targets is None else "soft_target_cross_entropy"
    if hidden is not None:
        name += "+normalized_activation_mse"
    return RecoveryResult(
        best[1], best[0], name, [item[0] for item in outcomes], perf_counter() - started, best[2]
    )
