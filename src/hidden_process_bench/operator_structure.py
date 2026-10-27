"""Recognize a node-emitting realization from already learned linear operators.

This is an explicit analytical recognition prior, not a supplied HMM fitting adapter.
Failure leaves the linear process unresolved; no latent realization is forced.
"""

import numpy as np

from .operator_predictor import Array


def recognize(initial: Array, normalizer: Array, operators: Array) -> dict[str, object]:
    """Test simultaneous diagonalization with fixed numerical acceptance tolerances.

    With COLUMN-state operators B_x = F T.T diag(E[:, x]) inv(F), the matrices
    inv(sum_x B_x) B_x share a real eigenbasis. Nonnegative stochastic parameters
    and reconstruction are checked independently of this necessary condition.
    """
    tolerance = 1e-7
    dimension = len(initial)
    if (
        initial.ndim != 1
        or normalizer.shape != initial.shape
        or operators.ndim != 3
        or operators.shape[1:] != (dimension, dimension)
        or not dimension
        or not len(operators)
        or not all(np.isfinite(x).all() for x in (initial, normalizer, operators))
    ):
        raise ValueError("invalid linear realization")
    unresolved: dict[str, object] = {
        "family": "unresolved_linear_state",
        "dimension": dimension,
        "acceptance_tolerance": tolerance,
        "minimality_certificate": False,
        "causal_certificate": False,
    }
    transition_operator = operators.sum(axis=0)
    condition = float(np.linalg.cond(transition_operator))
    if not np.isfinite(condition) or condition > 1e8:
        return {**unresolved, "reason": "singular_or_ill_conditioned_total_operator"}
    factors = np.linalg.solve(transition_operator, operators)
    # Fixed generic combination; unrelated to any generating task's RNG.
    coefficients = np.random.default_rng(0).normal(size=len(operators))
    _, basis = np.linalg.eig(np.einsum("a,aij->ij", coefficients, factors))
    if np.max(np.abs(basis.imag)) > tolerance:
        return {**unresolved, "reason": "complex_joint_coordinates"}
    basis = basis.real
    masses = normalizer @ basis
    if np.any(np.abs(masses) < tolerance) or np.linalg.cond(basis) > 1e8:
        return {**unresolved, "reason": "unusable_joint_coordinates"}
    basis /= masses
    inverse = np.linalg.inv(basis)
    diagonalized = np.einsum("ij,ajk,kl->ail", inverse, factors, basis)
    emission = np.diagonal(diagonalized, axis1=1, axis2=2).T.copy()
    transition = (inverse @ transition_operator @ basis).T
    start = inverse @ initial
    rebuilt = np.einsum("ij,jk,ka,kl->ail", basis, transition.T, emission, inverse)
    residual = float(np.max(np.abs(rebuilt - operators)) / max(1, np.max(np.abs(operators))))
    if residual > tolerance:
        return {
            **unresolved,
            "reason": "operators_do_not_share_emission_coordinates",
            "relative_residual": residual,
        }
    if (
        min(start.min(), transition.min(), emission.min()) < -tolerance
        or abs(start.sum() - 1) > tolerance
        or not np.allclose(transition.sum(axis=1), 1, atol=tolerance, rtol=0)
        or not np.allclose(emission.sum(axis=1), 1, atol=tolerance, rtol=0)
    ):
        return {
            **unresolved,
            "reason": "joint_coordinates_are_not_stochastic",
            "relative_residual": residual,
        }
    # Only remove roundoff that already passed every fixed acceptance check.
    start, transition, emission = (np.maximum(x, 0) for x in (start, transition, emission))
    start /= start.sum()
    transition /= transition.sum(axis=1, keepdims=True)
    emission /= emission.sum(axis=1, keepdims=True)
    return {
        **unresolved,
        "family": "memoryless_categorical" if dimension == 1 else "node_emitting_latent_markov",
        "reason": "real_joint_coordinates_with_nonnegative_stochastic_realization",
        "initial": start.tolist(),
        "transition": transition.tolist(),
        "emission": emission.tolist(),
        "relative_residual": residual,
        "condition_number": condition,
        "interpretation": "Numerical realization compatibility, not an exact equivalence, minimality or causal certificate. The analytical recognition class is built into this method.",
    }
