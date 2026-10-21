"""Evaluation owned by the benchmark, never supplied to extraction methods."""

from itertools import permutations
from typing import TypedDict

import numpy as np

from hidden_process_bench.process import Floats, MixtureHMM, Tokens


class KLSummary(TypedDict):
    count: int
    mean_kl: float | None
    standard_error: float | None
    upper95_kl: float | None


class Optimality(TypedDict):
    overall: KLSummary
    by_component: list[KLSummary]
    max_kl: float
    qualified: bool


class ParameterRecovery(TypedDict):
    weights_mae: float
    initial_mae: float
    transition_mae: float
    emission_mae: float
    mean_parameter_mae: float
    component_permutation: list[int]
    state_permutations: list[list[int]]
    transition_support_f1: float
    edge_threshold: float


def check_probabilities(values: Floats) -> None:
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.allclose(values.sum(axis=-1), 1.0, atol=1e-6, rtol=0):
        raise ValueError("probability rows must sum to one")


def sequence_kl(reference: Floats, candidate: Floats) -> Floats:
    """Mean KL per independent sequence; zero candidate support gives infinity."""
    if reference.shape != candidate.shape or reference.ndim != 3 or not all(reference.shape):
        raise ValueError("expected matching nonempty [batch, length, vocab] distributions")
    check_probabilities(reference)
    check_probabilities(candidate)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(reference > 0, reference * (np.log(reference) - np.log(candidate)), 0)
    return np.maximum(terms.sum(axis=-1).mean(axis=-1), 0)


def summarize(values: Floats) -> KLSummary:
    count = values.size
    if count == 0 or not np.isfinite(values).all():
        return {"count": count, "mean_kl": None, "standard_error": None, "upper95_kl": None}
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(count)) if count > 1 else None
    return {
        "count": count,
        "mean_kl": mean,
        "standard_error": se,
        "upper95_kl": mean + 1.96 * se if se is not None else None,
    }


def optimality(
    reference: Floats, candidate: Floats, components: Tokens, count: int, max_kl: float
) -> Optimality:
    values = sequence_kl(reference, candidate)
    overall = summarize(values)
    groups = [summarize(values[components == component]) for component in range(count)]
    qualified = all(
        group["upper95_kl"] is not None and group["upper95_kl"] <= max_kl
        for group in [overall, *groups]
    )
    return {"overall": overall, "by_component": groups, "max_kl": max_kl, "qualified": qualified}


def parameter_recovery(
    truth: MixtureHMM, candidate: MixtureHMM, edge_threshold: float = 0.05
) -> ParameterRecovery:
    """Exact small-state alignment; minimize the sum of four equally weighted MAEs."""
    if truth.shape != candidate.shape:
        raise ValueError("parameter recovery requires matching component/state/vocabulary counts")
    if not 0 < edge_threshold < 1:
        raise ValueError("edge threshold must be in (0, 1)")
    k, s = truth.shape.components, truth.shape.states
    costs = np.empty((k, k))
    choices: dict[tuple[int, int], tuple[int, ...]] = {}
    for source in range(k):
        for target in range(k):
            best = float("inf")
            for permutation in permutations(range(s)):
                order = np.array(permutation)
                cost = (
                    np.abs(truth.initial[source] - candidate.initial[target, order]).mean()
                    + np.abs(truth.emission[source] - candidate.emission[target, order]).mean()
                    + np.abs(
                        truth.transition[source]
                        - candidate.transition[target][np.ix_(order, order)]
                    ).mean()
                    + abs(truth.weights[source] - candidate.weights[target])
                )
                if cost < best:
                    best = cost
                    choices[source, target] = permutation
            costs[source, target] = best
    component_order = min(
        permutations(range(k)), key=lambda order: sum(costs[i, j] for i, j in enumerate(order))
    )
    state_orders = [choices[i, j] for i, j in enumerate(component_order)]
    aligned = MixtureHMM(
        candidate.weights[list(component_order)],
        np.stack(
            [
                candidate.initial[j, list(p)]
                for j, p in zip(component_order, state_orders, strict=True)
            ]
        ),
        np.stack(
            [
                candidate.transition[j][np.ix_(p, p)]
                for j, p in zip(component_order, state_orders, strict=True)
            ]
        ),
        np.stack(
            [
                candidate.emission[j, list(p)]
                for j, p in zip(component_order, state_orders, strict=True)
            ]
        ),
    )
    errors = {
        f"{name}_mae": float(np.abs(getattr(truth, name) - getattr(aligned, name)).mean())
        for name in ("weights", "initial", "transition", "emission")
    }
    true_edges = truth.transition > 0
    proposed_edges = aligned.transition >= edge_threshold
    tp = int((true_edges & proposed_edges).sum())
    denominator = int(true_edges.sum() + proposed_edges.sum())
    return {
        "weights_mae": errors["weights_mae"],
        "initial_mae": errors["initial_mae"],
        "transition_mae": errors["transition_mae"],
        "emission_mae": errors["emission_mae"],
        "mean_parameter_mae": sum(errors.values()) / 4,
        "component_permutation": list(component_order),
        "state_permutations": [list(order) for order in state_orders],
        "transition_support_f1": 2 * tp / denominator,
        "edge_threshold": edge_threshold,
    }


def identifiability_diagnostics(truth: MixtureHMM) -> dict[str, int | float | bool]:
    emission = truth.emission.reshape(-1, truth.shape.vocab)
    return {
        "joint_states": len(emission),
        "emission_rank": int(np.linalg.matrix_rank(emission)),
        "emissions_full_row_rank": bool(np.linalg.matrix_rank(emission) == len(emission)),
        "minimum_transition_singular_value": float(
            np.linalg.svd(truth.transition, compute_uv=False).min()
        ),
        "minimum_initial_joint_mass": float((truth.weights[:, None] * truth.initial).min()),
        "stationarity_max_error": float(
            np.abs(np.einsum("ks,ksj->kj", truth.initial, truth.transition) - truth.initial).max()
        ),
    }


def linear_probe(
    train_features: Floats, train_targets: Floats, test_features: Floats, test_targets: Floats
) -> dict[str, float]:
    """Supervised diagnostic only; fit the map on separate sequences from evaluation."""
    mean = train_features.mean(axis=0)
    target_mean = train_targets.mean(axis=0)
    centered = train_features - mean
    coefficients = np.linalg.solve(
        centered.T @ centered + 1e-3 * np.eye(centered.shape[1]),
        centered.T @ (train_targets - target_mean),
    )
    predictions = (test_features - mean) @ coefficients + target_mean
    mse = float(np.square(predictions - test_targets).mean())
    variance = float(np.square(test_targets - test_targets.mean(axis=0)).mean())
    return {"mse": mse, "r2": 1 - mse / max(variance, 1e-15)}
