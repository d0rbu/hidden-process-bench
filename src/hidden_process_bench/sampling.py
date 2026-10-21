"""Bounded task sampling with exact, pre-training checks of temporal structure."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from hidden_process_bench.config import SamplingConfig, Shape
from hidden_process_bench.equivalence import process_complexity
from hidden_process_bench.process import MixtureHMM, random_mixture


@dataclass(frozen=True)
class Difficulty:
    conditional_entropies: list[list[float]]
    unigram_gain: list[float]
    trigram_gain: list[float]
    min_state_mass: list[float]
    transition_density: list[float]
    joint_states: int
    short_hankel_rank: int
    hankel_rank_tolerance: float

    def rejections(self, config: SamplingConfig) -> list[str]:
        reasons = [
            name
            for name, values, threshold in (
                ("min_state_mass", self.min_state_mass, config.min_state_mass),
                ("unigram_gain", self.unigram_gain, config.min_unigram_gain),
                ("trigram_gain", self.trigram_gain, config.min_trigram_gain),
            )
            if any(value < threshold for value in values)
        ]
        if self.short_hankel_rank != self.joint_states:
            reasons.append("minimality_unresolved")
        return reasons


def measure_difficulty(hmm: MixtureHMM) -> Difficulty:
    """Exact H(X_t | last d tokens, C=c), d=0..3, separately for each component.

    Enumeration marginalizes hidden paths and includes zero-probability strings safely.
    No training, test sequences, learned baseline, or extraction feedback is used.
    """
    k, s, v = hmm.shape.components, hmm.shape.states, hmm.shape.vocab
    if k * v**4 > 4_000_000:
        raise ValueError("exact sampling screen is bounded to 4,000,000 token strings")
    stationary = np.einsum("ks,ksj->kj", hmm.initial, hmm.transition)
    if not np.allclose(stationary, hmm.initial, atol=1e-10, rtol=0):
        raise ValueError("difficulty requires stationary component starts")
    # joint[c, h, s] = P(history=h, next state=s | component=c).
    joint = hmm.initial[:, None, :]
    entropies = np.empty((k, 4), dtype=np.float64)
    for depth in range(4):
        next_joint = np.einsum("khs,ksv->khv", joint, hmm.emission)
        prefix_mass = joint.sum(axis=-1, keepdims=True)
        conditional = np.divide(
            next_joint, prefix_mass, out=np.zeros_like(next_joint), where=prefix_mass > 0
        )
        logs = np.log(conditional, out=np.zeros_like(conditional), where=conditional > 0)
        entropies[:, depth] = -(next_joint * logs).sum(axis=(1, 2))
        if depth < 3:
            joint = np.einsum("khs,ksv,ksj->khvj", joint, hmm.emission, hmm.transition).reshape(
                k, -1, s
            )
    complexity = process_complexity(hmm)
    return Difficulty(
        conditional_entropies=entropies.tolist(),
        unigram_gain=np.maximum(entropies[:, 0] - entropies[:, 3], 0).tolist(),
        trigram_gain=np.maximum(entropies[:, 2] - entropies[:, 3], 0).tolist(),
        min_state_mass=hmm.initial.min(axis=-1).tolist(),
        transition_density=(hmm.transition > 0).mean(axis=(1, 2)).tolist(),
        joint_states=k * s,
        short_hankel_rank=int(complexity["short_hankel_rank"]),
        hankel_rank_tolerance=float(complexity["hankel_rank_tolerance"]),
    )


def sample_task(
    shape: Shape, seed: int, config: SamplingConfig, trace_path: Path
) -> tuple[MixtureHMM, Difficulty, int]:
    """Accept the first candidate passing temporal and minimal-order screens, or fail.

    The private audit records all candidates, including rejection reasons, and survives
    exhaustion. Candidate zero preserves random_mixture(shape, seed); later candidates
    use spawned private seeds, independent of training/evaluation data streams.
    """
    if shape.vocab < shape.components * shape.states:
        raise ValueError("minimal-task screen requires vocab >= components * states")
    with trace_path.open("x") as stream:
        for attempt in range(config.max_attempts):
            candidate_seed = (
                seed
                if attempt == 0
                else int(
                    np.random.SeedSequence(seed, spawn_key=(attempt,)).generate_state(
                        1, dtype=np.uint64
                    )[0]
                )
            )
            hmm = random_mixture(
                shape,
                candidate_seed,
                extra_edge_probability=config.extra_edge_probability,
                emission_concentration=config.emission_concentration,
            )
            difficulty = measure_difficulty(hmm)
            rejections = difficulty.rejections(config)
            stream.write(
                json.dumps(
                    {
                        "attempt": attempt + 1,
                        "candidate_seed": candidate_seed,
                        "difficulty": asdict(difficulty),
                        "rejections": rejections,
                        "accepted": not rejections,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            stream.flush()
            if not rejections:
                return hmm, difficulty, attempt + 1
    raise ValueError(f"sampling exhausted {config.max_attempts} candidates; audit: {trace_path}")
