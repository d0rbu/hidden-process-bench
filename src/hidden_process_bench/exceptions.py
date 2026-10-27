"""Known addition machine with private, uniformly sampled point exceptions."""

from dataclasses import dataclass
from typing import cast

import numpy as np
from beartype import beartype
from jaxtyping import Int64, jaxtyped
from numpy.typing import NDArray

from hidden_process_bench.config import check_count


@dataclass(frozen=True)
class Addition:
    base: int = 4
    digits: int = 3

    def __post_init__(self) -> None:
        check_count(self.digits)
        if type(self.base) is not int or not 2 <= self.base <= 10 or self.digits > 6:
            raise ValueError("addition requires base 2..10 and 1..6 digits")

    @property
    def limit(self) -> int:
        return self.base**self.digits

    @property
    def domain(self) -> int:
        return self.limit**2

    def pair(self, value: int) -> tuple[int, int]:
        if type(value) is not int or not 0 <= value < self.domain:
            raise ValueError("input ID is outside the ordered-pair domain")
        return divmod(value, self.limit)

    def input_id(self, pair: object) -> int:
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or any(type(x) is not int or not 0 <= x < self.limit for x in pair)
        ):
            raise ValueError("input must contain two in-range integer operands")
        return cast(int, pair[0]) * self.limit + cast(int, pair[1])

    def trace(self, a: int, b: int) -> list[dict[str, int]]:
        """Two carry states; least-significant column first, then final carry."""
        self.input_id([a, b])
        carry = 0
        result = []
        for column in range(self.digits):
            x, y = a // self.base**column % self.base, b // self.base**column % self.base
            total = x + y + carry
            result.append(
                {
                    "a": x,
                    "b": y,
                    "carry_in": carry,
                    "digit": total % self.base,
                    "carry_out": total // self.base,
                }
            )
            carry = total // self.base
        return result

    def public(self) -> dict[str, object]:
        return {
            "base": self.base,
            "digits": self.digits,
            "operand_limit": self.limit,
            "ordered_input_count": self.domain,
            "reference": "integer addition",
            "encoding": "interleaved LSB-first operand digits, separator=base, then digits+1 LSB-first output digits",
            "carry_rule": "c0=0; y=(a_digit+b_digit+c)%base; c_next=(a_digit+b_digit+c)//base; append final carry",
            "target": "rank ordered operand pairs where the teacher implements planted wrong sums",
        }


InputIDs = Int64[np.ndarray, "batch"]
TokenRows = Int64[np.ndarray, "batch length"]


@jaxtyped(typechecker=beartype)
def encode(machine: Addition, ids: InputIDs, answers: InputIDs | None = None) -> TokenRows:
    if (ids < 0).any() or (ids >= machine.domain).any():
        raise ValueError("invalid input ID")
    a, b = ids // machine.limit, ids % machine.limit
    columns = []
    for digit in range(machine.digits):
        columns.extend(
            [a // machine.base**digit % machine.base, b // machine.base**digit % machine.base]
        )
    columns.append(np.full_like(ids, machine.base))
    if answers is not None:
        if (answers < 0).any() or (answers > 2 * (machine.limit - 1)).any():
            raise ValueError("invalid output")
        columns.extend(answers // machine.base**i % machine.base for i in range(machine.digits + 1))
    return np.stack(columns, axis=1)


def sample_exceptions(machine: Addition, count: int, rng: np.random.Generator) -> dict[int, int]:
    check_count(count)
    if count >= machine.domain:
        raise ValueError("exceptions must leave a nonempty clean domain")
    result = {}
    for value in rng.choice(machine.domain, size=count, replace=False):
        key = int(value)
        a, b = machine.pair(key)
        # Draw uniformly over all valid sums except the correct one.
        wrong = int(rng.integers(2 * (machine.limit - 1)))
        result[key] = wrong + (wrong >= a + b)
    return result


def targets(
    machine: Addition, ids: NDArray[np.int64], exceptions: dict[int, int]
) -> NDArray[np.int64]:
    return np.array(
        [exceptions.get(int(i), sum(machine.pair(int(i)))) for i in ids], dtype=np.int64
    )


def random_search_expectation(domain: int, count: int, queries: int) -> dict[str, float]:
    if not 0 < count < domain or not 0 <= queries <= domain:
        raise ValueError("invalid finite-domain search budget")
    miss = 1.0
    for index in range(count):
        miss *= max(0, domain - queries - index) / (domain - index)
    return {"expected_recall": queries / domain, "probability_any_hit": 1 - miss}


def score_candidates(
    machine: Addition, exceptions: dict[int, int], candidates: object
) -> dict[str, object]:
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a ranked list of ordered pairs")
    ids = [machine.input_id(pair) for pair in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidates are forbidden")
    hits = [value in exceptions for value in ids]
    return {
        "hits_by_rank": hits,
        "recovered": sum(hits),
        "total_exceptions": len(exceptions),
        "recall": sum(hits) / len(exceptions),
        "precision": sum(hits) / len(ids) if ids else 0.0,
        "false_positives": len(ids) - sum(hits),
        "exact_set": set(ids) == set(exceptions),
    }


def neighbors(machine: Addition, value: int, excluded: set[int]) -> np.ndarray:
    """All single-digit operand edits and the swapped pair, excluding planted inputs."""
    a, b = machine.pair(value)
    result = {machine.input_id([b, a])}
    for operand in range(2):
        for column in range(machine.digits):
            scale = machine.base**column
            old = (a, b)[operand] // scale % machine.base
            for digit in range(machine.base):
                pair = [a, b]
                pair[operand] += (digit - old) * scale
                result.add(machine.input_id(pair))
    return np.array(sorted(result - excluded - {value}), dtype=np.int64)
