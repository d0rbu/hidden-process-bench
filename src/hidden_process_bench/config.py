"""Validated, serializable experiment settings; all losses use natural logarithms."""

from __future__ import annotations

import math
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

from phantom import Phantom


def _positive_count(value: int) -> bool:
    return not isinstance(value, bool) and value > 0


class PositiveCount(int, Phantom[int], predicate=_positive_count, bound=int):
    """A strictly positive integer, excluding booleans."""


def check_count(value: int) -> PositiveCount:
    return PositiveCount.parse(value)


def check_positive(value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError("expected a finite positive number")


def check_probability(value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("expected a finite probability between zero and one")


@dataclass(frozen=True)
class SamplingConfig:
    extra_edge_probability: float = 0.5
    emission_concentration: float = 0.4
    min_state_mass: float = 0.05
    min_unigram_gain: float = 0.02
    min_trigram_gain: float = 0.001
    max_attempts: int = 128

    def __post_init__(self) -> None:
        check_probability(self.extra_edge_probability)
        check_probability(self.min_state_mass)
        check_positive(self.emission_concentration)
        check_count(self.max_attempts)
        for value in (self.min_unigram_gain, self.min_trigram_gain):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError("sampling gains must be finite and nonnegative")


@dataclass(frozen=True)
class Shape:
    components: int = 4
    states: int = 3
    vocab: int = 16

    def __post_init__(self) -> None:
        for value in asdict(self).values():
            check_count(value)
        if self.components > 6 or self.states > 6:
            raise ValueError("exact permutation scoring supports at most 6 components and 6 states")


@dataclass(frozen=True)
class ModelConfig:
    vocab: int = 16
    context: int = 24
    width: int = 48
    heads: int = 4
    layers: int = 2

    def __post_init__(self) -> None:
        for value in asdict(self).values():
            check_count(value)
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")


@dataclass(frozen=True)
class RunConfig:
    seed: int = 7
    model_seed: int = 8
    observation_seed: int = 27
    recovery_seed: int = 37
    components: int = 4
    states: int = 3
    vocab: int = 16
    length: int = 24
    width: int = 48
    heads: int = 4
    layers: int = 2
    train_steps: int = 2000
    batch_size: int = 64
    learning_rate: float = 0.003
    eval_every: int = 100
    eval_sequences: int = 256
    max_kl: float = 0.03
    extraction_sequences: int = 128
    extraction_steps: int = 250
    restarts: int = 2
    activation_weight: float = 0.1
    threads: int = 2
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    def __post_init__(self) -> None:
        self.shape()
        self.model()
        for seed in (self.seed, self.model_seed, self.observation_seed, self.recovery_seed):
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("seed must be a non-negative integer")
        for value in (
            self.train_steps,
            self.batch_size,
            self.eval_every,
            self.eval_sequences,
            self.extraction_sequences,
            self.extraction_steps,
            self.restarts,
            self.threads,
        ):
            check_count(value)
        for value in (self.learning_rate, self.max_kl, self.activation_weight):
            check_positive(value)
        if self.eval_sequences < 2 * self.components:
            raise ValueError("evaluation needs at least two sequences per component")
        if self.length < 4:
            raise ValueError("sampling screen requires at least four tokens per sequence")

    def shape(self) -> Shape:
        return Shape(self.components, self.states, self.vocab)

    def model(self) -> ModelConfig:
        return ModelConfig(self.vocab, self.length, self.width, self.heads, self.layers)

    @classmethod
    def load(cls, path: Path) -> RunConfig:
        with path.open("rb") as stream:
            values = tomllib.load(stream)
        sampling = SamplingConfig(**values.pop("sampling", {}))
        return cls(**values, sampling=sampling)
