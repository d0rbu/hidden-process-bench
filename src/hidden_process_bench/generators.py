"""Evaluator-only generating processes with a shared exact prediction surface.

These classes are never mounted into extractor or submitted-predictor sandboxes.
Alternative compact representations may describe the same observable process.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from beartype import beartype
from jaxtyping import Float64, Int64, jaxtyped

from hidden_process_bench.config import check_count, check_probability
from hidden_process_bench.process import Floats, MixtureHMM, Tokens, check_tokens


class SequenceProcess(Protocol):
    @property
    def vocab(self) -> int: ...

    @property
    def groups(self) -> int: ...

    def sample(
        self, rng: np.random.Generator, count: int, length: int
    ) -> tuple[Tokens, Tokens]: ...

    def probabilities(self, tokens: Tokens) -> Floats: ...

    def specification(self) -> dict[str, object]: ...


def _alphabet(alphabet: tuple[int, ...], minimum: int) -> None:
    if (
        not isinstance(alphabet, tuple)
        or len(alphabet) < minimum
        or any(type(token) is not int for token in alphabet)
        or sorted(alphabet) != list(range(len(alphabet)))
    ):
        raise ValueError("alphabet must be a permutation of consecutive token IDs")


def _interior(value: float) -> None:
    check_probability(value)
    if value in (0, 1):
        raise ValueError("process probabilities must be strictly between zero and one")


ProbabilityRows = Float64[np.ndarray, "batch vocab"]
Draws = Int64[np.ndarray, "batch"]


@jaxtyped(typechecker=beartype)
def _draw(rng: np.random.Generator, probabilities: ProbabilityRows) -> Draws:
    cumulative = probabilities.cumsum(axis=-1)
    cumulative[:, -1] = 1.0
    return (rng.random((probabilities.shape[0], 1)) > cumulative).sum(axis=-1).astype(np.int64)


@dataclass(frozen=True)
class HiddenProcess:
    hmm: MixtureHMM

    @property
    def vocab(self) -> int:
        return self.hmm.shape.vocab

    @property
    def groups(self) -> int:
        return self.hmm.shape.components

    def sample(self, rng: np.random.Generator, count: int, length: int) -> tuple[Tokens, Tokens]:
        return self.hmm.sample(rng, count, length)

    def probabilities(self, tokens: Tokens) -> Floats:
        return self.hmm.filter(tokens).predictions

    def specification(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "family": "mixture_hmm",
            **{
                name: getattr(self.hmm, name).tolist()
                for name in ("weights", "initial", "transition", "emission")
            },
        }


@dataclass(frozen=True)
class LagProcess:
    """Copy a privately permuted token at a fixed lag, mixed with uniform noise."""

    alphabet: tuple[int, ...]
    lag: int
    strength: float

    def __post_init__(self) -> None:
        _alphabet(self.alphabet, 2)
        check_count(self.lag)
        _interior(self.strength)

    @property
    def vocab(self) -> int:
        return len(self.alphabet)

    @property
    def groups(self) -> int:
        return 1

    def _next(self, tokens: Tokens, position: int) -> Floats:
        batch = tokens.shape[0]
        probabilities = np.full((batch, self.vocab), 1 / self.vocab, dtype=np.float64)
        if position >= self.lag:
            probabilities *= 1 - self.strength
            target = np.asarray(self.alphabet)[tokens[:, position - self.lag]]
            probabilities[np.arange(batch), target] += self.strength
        return probabilities

    def sample(self, rng: np.random.Generator, count: int, length: int) -> tuple[Tokens, Tokens]:
        check_count(count)
        check_count(length)
        tokens = np.empty((count, length), dtype=np.int64)
        for position in range(length):
            tokens[:, position] = _draw(rng, self._next(tokens, position))
        return tokens, np.zeros(count, dtype=np.int64)

    def probabilities(self, tokens: Tokens) -> Floats:
        check_tokens(tokens, self.vocab)
        return np.stack(
            [self._next(tokens, position) for position in range(tokens.shape[1] + 1)], axis=1
        )

    def specification(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "family": "lag_rule",
            "alphabet": list(self.alphabet),
            "lag": self.lag,
            "strength": self.strength,
        }


@dataclass(frozen=True)
class StackProcess:
    """Two push/pop types, neutral symbols, and a full-support noise floor.

    Matching closes pop; mismatches and empty-stack closes do nothing. Noise tokens
    obey the same update rule, making the stack exactly determined by every history.
    There is no depth cap. The first four permuted symbols are push0,push1,pop0,pop1.
    """

    alphabet: tuple[int, ...]
    push: float = 0.32
    close: float = 0.52
    noise: float = 0.04

    def __post_init__(self) -> None:
        _alphabet(self.alphabet, 5)
        for value in (self.push, self.close, self.noise):
            _interior(value)
        if self.push + self.close >= 1:
            raise ValueError("push and close must leave positive neutral probability")

    @property
    def vocab(self) -> int:
        return len(self.alphabet)

    @property
    def groups(self) -> int:
        return 3

    def _next(self, stacks: list[list[int]]) -> Floats:
        probabilities = np.full(
            (len(stacks), self.vocab), self.noise / self.vocab, dtype=np.float64
        )
        probabilities[:, list(self.alphabet[:2])] += (1 - self.noise) * self.push / 2
        probabilities[:, list(self.alphabet[4:])] += (
            (1 - self.noise) * (1 - self.push - self.close) / (self.vocab - 4)
        )
        mass = (1 - self.noise) * self.close
        for row, stack in enumerate(stacks):
            if stack:
                probabilities[row, self.alphabet[2 + stack[-1]]] += mass
            else:
                probabilities[row, list(self.alphabet[2:4])] += mass / 2
        return probabilities

    def _update(self, stacks: list[list[int]], tokens: Tokens) -> None:
        inverse = {token: role for role, token in enumerate(self.alphabet)}
        for stack, token in zip(stacks, tokens, strict=True):
            role = inverse[int(token)]
            if role < 2:
                stack.append(role)
            elif role < 4 and stack and stack[-1] == role - 2:
                stack.pop()

    def sample(self, rng: np.random.Generator, count: int, length: int) -> tuple[Tokens, Tokens]:
        check_count(count)
        check_count(length)
        stacks: list[list[int]] = [[] for _ in range(count)]
        depths = np.zeros(count, dtype=np.int64)
        tokens = np.empty((count, length), dtype=np.int64)
        for position in range(length):
            tokens[:, position] = _draw(rng, self._next(stacks))
            self._update(stacks, tokens[:, position])
            depths = np.maximum(depths, [len(stack) for stack in stacks])
        return tokens, np.digitize(depths, [2, 4]).astype(np.int64)

    def probabilities(self, tokens: Tokens) -> Floats:
        check_tokens(tokens, self.vocab)
        stacks: list[list[int]] = [[] for _ in range(tokens.shape[0])]
        probabilities = []
        for position in range(tokens.shape[1] + 1):
            probabilities.append(self._next(stacks))
            if position < tokens.shape[1]:
                self._update(stacks, tokens[:, position])
        return np.stack(probabilities, axis=1)

    def specification(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "family": "stack_rule",
            "alphabet": list(self.alphabet),
            "push": self.push,
            "close": self.close,
            "noise": self.noise,
        }


def load_process(path: Path) -> SequenceProcess:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported private process specification")
    family = payload.pop("family", None)
    payload.pop("schema_version")
    if family == "mixture_hmm":
        return HiddenProcess(
            MixtureHMM(
                **{name: np.asarray(value, dtype=np.float64) for name, value in payload.items()}
            )
        )
    if family in ("lag_rule", "stack_rule"):
        payload["alphabet"] = tuple(payload["alphabet"])
        if family == "lag_rule":
            return LagProcess(**payload)
        return StackProcess(**payload)
    raise ValueError("unknown private process family")
