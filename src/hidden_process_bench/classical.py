"""Output-only EM and a finite-sample spectral observable-operator baseline."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from hidden_process_bench.config import Shape, check_count
from hidden_process_bench.process import Floats, MixtureHMM, Tokens, check_tokens
from hidden_process_bench.recovery import DifferentiableHMM, RecoveryResult


def _rows(values: Floats) -> Floats:
    values = values + 1e-12
    return values / values.sum(axis=-1, keepdims=True)


def em_update(hmm: MixtureHMM, tokens: Tokens) -> MixtureHMM:
    """Scaled forward/backward E-step, then exact expected-count M-step (tiny floor)."""
    check_tokens(tokens, hmm.shape.vocab)
    batch, length = tokens.shape
    check_count(length)
    emission = hmm.emission[:, :, tokens].transpose(2, 3, 0, 1)
    alpha = np.empty_like(emission)
    scales = np.empty((batch, length))
    belief = np.broadcast_to(hmm.weights[:, None] * hmm.initial, (batch, *hmm.initial.shape))
    for t in range(length):
        post = belief * emission[:, t]
        scales[:, t] = post.sum(axis=(1, 2))
        alpha[:, t] = post / scales[:, t, None, None]
        belief = np.einsum("bks,ksj->bkj", alpha[:, t], hmm.transition)
    beta = np.ones_like(alpha)
    transition_counts = np.zeros_like(hmm.transition)
    for t in range(length - 2, -1, -1):
        future = emission[:, t + 1] * beta[:, t + 1] / scales[:, t + 1, None, None]
        beta[:, t] = np.einsum("ksj,bkj->bks", hmm.transition, future)
        transition_counts += np.einsum("bks,ksj,bkj->ksj", alpha[:, t], hmm.transition, future)
    gamma = alpha * beta
    gamma /= gamma.sum(axis=(2, 3), keepdims=True)
    initial_counts = gamma[:, 0].sum(axis=0)
    weights = _rows(initial_counts.sum(axis=-1))
    emissions = np.einsum("btks,btv->ksv", gamma, np.eye(hmm.shape.vocab)[tokens])
    transition = _rows(transition_counts) if length > 1 else hmm.transition
    return MixtureHMM(weights, _rows(initial_counts), transition, _rows(emissions))


def fit_em(shape: Shape, tokens: Tokens, *, steps: int, restarts: int, seed: int) -> RecoveryResult:
    check_count(steps)
    check_count(restarts)
    check_tokens(tokens, shape.vocab)
    started = perf_counter()
    outcomes = []
    for restart in range(restarts):
        hmm = DifferentiableHMM(shape, seed + restart).export()
        for _ in range(steps):
            hmm = em_update(hmm, tokens)
        nll = -float(hmm.filter(tokens).log_likelihood.mean()) / tokens.shape[1]
        outcomes.append((nll, hmm))
    objective, best = min(outcomes, key=lambda item: item[0])
    k, s, v = shape.components, shape.states, shape.vocab
    return RecoveryResult(
        best,
        objective,
        "sample_nll_em",
        [x[0] for x in outcomes],
        perf_counter() - started,
        k + k * s + k * s * s + k * s * v,
    )


@dataclass(frozen=True)
class SpectralResult:
    initial: Floats
    terminal: Floats
    operators: Floats
    singular_values: Floats
    probabilities: Floats
    repaired_fraction: float


def fit_spectral(tokens: Tokens, vocab: int, rank: int) -> SpectralResult:
    """HKZ moment equations, pooled stationary windows; repair finite-sample outputs.

    Operators can have negative entries legitimately. Negative *probabilities* are
    finite-sample failures and are explicitly counted before clipping/normalization.
    These repaired targets need a separate nonnegative HMM distillation adapter.
    """
    check_tokens(tokens, vocab)
    check_count(rank)
    if tokens.shape[1] < 3 or rank > vocab:
        raise ValueError("spectral learning needs length >= 3 and rank <= vocabulary")
    first, middle, last = (
        tokens[:, offset : tokens.shape[1] - 2 + offset].ravel() for offset in range(3)
    )
    size = len(first)
    p1 = np.bincount(first, minlength=vocab) / size
    p21 = np.zeros((vocab, vocab))
    p3x1 = np.zeros((vocab, vocab, vocab))
    np.add.at(p21, (middle, first), 1 / size)
    np.add.at(p3x1, (middle, last, first), 1 / size)
    u, singular, _ = np.linalg.svd(p21)
    u = u[:, :rank]
    inverse = np.linalg.pinv(u.T @ p21, rcond=1e-5)
    initial = u.T @ p1
    terminal = np.linalg.pinv(p21.T @ u, rcond=1e-5) @ p1
    operators = np.einsum("ri,vij,jq->vrq", u.T, p3x1, inverse)
    state = np.broadcast_to(initial, (len(tokens), rank)).copy()
    predictions = []
    repaired = 0
    for t in range(tokens.shape[1]):
        next_states = np.einsum("vij,bj->bvi", operators, state)
        raw = next_states @ terminal
        bad = (raw < 0).any(axis=-1) | (raw.sum(axis=-1) <= 1e-12)
        repaired += int(bad.sum())
        positive = np.maximum(raw, 1e-8)
        predictions.append(positive / positive.sum(axis=-1, keepdims=True))
        state = next_states[np.arange(len(tokens)), tokens[:, t]]
        norm = state @ terminal
        # Preserve signs in the observable coordinates. Near-zero normalization is
        # a failed prediction path; reset to the initial state for numerical safety.
        safe = np.abs(norm) > 1e-12
        state[safe] /= norm[safe, None]
        state[~safe] = initial
    return SpectralResult(
        initial,
        terminal,
        operators,
        np.asarray(singular, dtype=np.float64),
        np.stack(predictions, axis=1),
        repaired / tokens.size,
    )
