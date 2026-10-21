"""Node-emitting HMMs; a component is sampled once for each sequence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from beartype import beartype
from jaxtyping import Float64, jaxtyped
from numpy.typing import NDArray

from hidden_process_bench.config import Shape, check_count, check_positive, check_probability

type Floats = NDArray[np.float64]
type Tokens = NDArray[np.int64]
Weights = Float64[np.ndarray, "k"]
Initial = Float64[np.ndarray, "k s"]
Transitions = Float64[np.ndarray, "k s s"]
Emissions = Float64[np.ndarray, "k s v"]


@jaxtyped(typechecker=beartype)
def _check_parameters(
    weights: Weights,
    initial: Initial,
    transition: Transitions,
    emission: Emissions,
) -> None:
    for array in (weights, initial, transition, emission):
        if array.size == 0 or not np.isfinite(array).all() or (array < 0).any():
            raise ValueError("probability arrays must be nonempty, finite and nonnegative")
        if not np.allclose(array.sum(axis=-1), 1.0, atol=1e-10, rtol=0):
            raise ValueError("each probability row must sum to one")


def check_tokens(tokens: Tokens, vocab: int) -> None:
    if tokens.dtype != np.int64 or tokens.ndim != 2 or tokens.shape[0] == 0:
        raise ValueError("tokens must be a nonempty batch of rank-2 int64 values")
    if (tokens < 0).any() or (tokens >= vocab).any():
        raise ValueError("token outside vocabulary")


@dataclass(frozen=True)
class FilterResult:
    predictions: Floats
    beliefs: Floats
    log_likelihood: Floats


@dataclass(frozen=True)
class MixtureHMM:
    weights: Floats
    initial: Floats
    transition: Floats
    emission: Floats

    def __post_init__(self) -> None:
        _check_parameters(self.weights, self.initial, self.transition, self.emission)
        _ = self.shape
        for name in ("weights", "initial", "transition", "emission"):
            array = getattr(self, name).copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def shape(self) -> Shape:
        return Shape(self.weights.size, self.initial.shape[1], self.emission.shape[2])

    def sample(self, rng: np.random.Generator, count: int, length: int) -> tuple[Tokens, Tokens]:
        check_count(count)
        check_count(length)
        components = rng.choice(self.shape.components, count, p=self.weights).astype(np.int64)
        states = _categorical(rng, self.initial[components])
        tokens = np.empty((count, length), dtype=np.int64)
        for t in range(length):
            tokens[:, t] = _categorical(rng, self.emission[components, states])
            states = _categorical(rng, self.transition[components, states])
        return tokens, components

    def filter(self, tokens: Tokens) -> FilterResult:
        """Return P(x_t | x_<t) and P(component, z_t | x_<t), including t=L."""
        check_tokens(tokens, self.shape.vocab)
        batch, length = tokens.shape
        belief = np.broadcast_to(self.weights[:, None] * self.initial, (batch, *self.initial.shape))
        predictions = np.empty((batch, length + 1, self.shape.vocab), dtype=np.float64)
        beliefs = np.empty((batch, length + 1, *self.initial.shape), dtype=np.float64)
        log_likelihood = np.zeros(batch, dtype=np.float64)
        for t in range(length + 1):
            predictions[:, t] = np.einsum("bks,ksv->bv", belief, self.emission)
            beliefs[:, t] = belief
            if t < length:
                likelihood = self.emission[:, :, tokens[:, t]].transpose(2, 0, 1)
                posterior = belief * likelihood
                evidence = posterior.sum(axis=(1, 2))
                if (evidence <= 0).any():
                    raise ValueError("history has zero probability under this process")
                log_likelihood += np.log(evidence)
                posterior /= evidence[:, None, None]
                belief = np.einsum("bks,ksj->bkj", posterior, self.transition)
        return FilterResult(predictions, beliefs, log_likelihood)

    def save(self, path: Path) -> None:
        payload = {
            name: getattr(self, name).tolist()
            for name in ("weights", "initial", "transition", "emission")
        }
        path.write_text(json.dumps({"schema_version": 1, **payload}, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> MixtureHMM:
        payload = json.loads(path.read_text())
        if payload.pop("schema_version") != 1:
            raise ValueError("unsupported HMM schema version")
        return cls(**{key: np.asarray(value, dtype=np.float64) for key, value in payload.items()})


def _categorical(rng: np.random.Generator, probabilities: Floats) -> Tokens:
    # Clamp the final CDF entry to avoid a roundoff-only out-of-vocabulary draw.
    cumulative = probabilities.cumsum(axis=-1)
    cumulative[:, -1] = 1.0
    return (rng.random((len(probabilities), 1)) >= cumulative).sum(axis=-1).astype(np.int64)


def random_mixture(
    shape: Shape,
    seed: int,
    *,
    extra_edge_probability: float = 0.5,
    emission_concentration: float = 0.4,
) -> MixtureHMM:
    """Stationary components with full-rank transitions and shared token vocabulary."""
    check_probability(extra_edge_probability)
    check_positive(emission_concentration)
    rng = np.random.default_rng(seed)
    k, s, v = shape.components, shape.states, shape.vocab
    transition = rng.dirichlet(np.ones(s), size=(k, s))
    mask = rng.random((k, s, s)) >= 1 - extra_edge_probability
    index = np.arange(s)
    mask[:, index, index] = True
    mask[:, index, (index + 1) % s] = True
    transition *= mask
    transition /= transition.sum(axis=-1, keepdims=True)
    transition = 0.55 * np.eye(s) + 0.45 * transition
    initial = np.stack(
        [
            np.linalg.solve(np.eye(s) - matrix.T + np.ones((s, s)), np.ones(s))
            for matrix in transition
        ]
    )
    emission = 0.98 * rng.dirichlet(np.full(v, emission_concentration), size=(k, s)) + 0.02 / v
    return MixtureHMM(np.full(k, 1.0 / k), initial, transition, emission)
