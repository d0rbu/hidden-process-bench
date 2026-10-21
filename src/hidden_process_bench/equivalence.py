"""Observable equivalence and complexity, independent of hidden-state names/order."""

from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np

from hidden_process_bench.config import check_count, check_positive
from hidden_process_bench.process import Floats, MixtureHMM


@dataclass(frozen=True)
class Equivalence:
    status: Literal["different", "numerically_equivalent"]
    tolerance: float
    basis_rank: int
    joint_dimension: int
    words_checked: int
    max_checked_probability_gap: float
    witness: list[int] | None
    reference_probability: float | None
    candidate_probability: float | None


def _advance(hmm: MixtureHMM, row: Floats, token: int) -> Floats:
    return np.einsum(
        "ks,ks,ksj->kj", row.reshape(hmm.initial.shape), hmm.emission[:, :, token], hmm.transition
    ).ravel()


def compare_processes(
    reference: MixtureHMM, candidate: MixtureHMM, tolerance: float = 1e-10
) -> Equivalence:
    """Prune the word tree using a reachable row-space basis (Tzeng-style).

    In exact arithmetic, closure decides equality for every finite string. Float64
    uses twice-reorthogonalized rows and a relative rank threshold: a numerical
    equivalence verdict is neither a proof nor a uniform approximate-error bound.
    A difference includes the actual checked word and its two probabilities.
    """
    check_positive(tolerance)
    if reference.shape.vocab != candidate.shape.vocab:
        raise ValueError("equivalence requires a shared vocabulary")
    boundary = reference.initial.size
    start = np.concatenate(
        [
            (reference.weights[:, None] * reference.initial).ravel(),
            -(candidate.weights[:, None] * candidate.initial).ravel(),
        ]
    )
    pending: deque[tuple[Floats, tuple[int, ...]]] = deque([(start, ())])
    basis: list[Floats] = []
    checked, maximum = 0, 0.0
    while pending:
        row, word = pending.popleft()
        checked += 1
        probability_a, probability_b = float(row[:boundary].sum()), float(-row[boundary:].sum())
        gap = abs(probability_a - probability_b)
        maximum = max(maximum, gap)
        if gap > tolerance:
            return Equivalence(
                "different",
                tolerance,
                len(basis),
                len(start),
                checked,
                maximum,
                list(word),
                probability_a,
                probability_b,
            )
        norm = np.linalg.norm(row)
        if norm == 0:
            continue
        residual = row / norm
        for _ in range(2):
            for vector in basis:
                residual -= np.dot(residual, vector) * vector
        remaining = np.linalg.norm(residual)
        if remaining > tolerance and len(basis) < len(start):
            basis.append(residual / remaining)
            for token in range(reference.shape.vocab):
                child = np.concatenate(
                    [
                        _advance(reference, row[:boundary], token),
                        _advance(candidate, row[boundary:], token),
                    ]
                )
                pending.append((child, (*word, token)))
    return Equivalence(
        "numerically_equivalent",
        tolerance,
        len(basis),
        len(start),
        checked,
        maximum,
        None,
        None,
        None,
    )


def split_states(hmm: MixtureHMM, copies: int = 2) -> MixtureHMM:
    """An equivalent control: clone each state, dividing incoming mass among copies."""
    check_count(copies)
    return MixtureHMM(
        hmm.weights,
        np.repeat(hmm.initial, copies, axis=1) / copies,
        np.repeat(np.repeat(hmm.transition, copies, axis=1), copies, axis=2) / copies,
        np.repeat(hmm.emission, copies, axis=1),
    )


def process_complexity(hmm: MixtureHMM) -> dict[str, int | float]:
    """Representation size plus a small observable-rank lower bound; not minimum HMM order."""
    k, s, v = hmm.shape.components, hmm.shape.states, hmm.shape.vocab
    mass = hmm.weights[:, None] * hmm.initial
    logs = np.log(mass, out=np.zeros_like(mass), where=mass > 0)
    marginal = np.einsum("ks,ksv->v", mass, hmm.emission)
    pairs = np.einsum("ks,ksu,ksj,kjv->uv", mass, hmm.emission, hmm.transition, hmm.emission)
    hankel = np.empty((v + 1, v + 1), dtype=np.float64)
    hankel[0, 0] = 1
    hankel[0, 1:] = marginal
    hankel[1:, 0] = marginal
    hankel[1:, 1:] = pairs
    return {
        "components": k,
        "states_per_component": s,
        "joint_states": k * s,
        "dense_family_degrees_of_freedom": k - 1 + k * (s - 1 + s * (s - 1) + s * (v - 1)),
        "transition_edges": int(np.count_nonzero(hmm.transition)),
        "transition_density": float((hmm.transition > 0).mean()),
        "effective_joint_states": float(np.exp(-(mass * logs).sum())),
        "short_hankel_rank": int(np.linalg.matrix_rank(hankel, tol=1e-10)),
        "hankel_rank_tolerance": 1e-10,
    }
