"""Learn autonomous observable dynamics in unsupervised predictive coordinates."""

import json
import shutil
from pathlib import Path

import numpy as np

from .operator_predictor import Array, Tokens, trajectory
from .operator_structure import recognize


def coordinates(features: Array, dimension: int) -> tuple[Array, Array, Array]:
    """Whiten variation; the explicit constant coordinate normalizes linear states."""
    flattened = features.reshape(-1, features.shape[-1])
    mean = flattened.mean(axis=0)
    _, singular, right = np.linalg.svd(flattened - mean, full_matrices=False)
    if dimension < 1 or dimension > 1 + int(np.sum(singular > max(1e-10, singular[0] * 1e-8))):
        raise ValueError("unsupported coordinate dimension")
    basis = right[: dimension - 1].T * (np.sqrt(len(flattened)) / singular[: dimension - 1])
    return project(features, mean, basis), mean, basis


def project(features: Array, mean: Array, basis: Array) -> Array:
    projected = (features - mean) @ basis
    return np.concatenate((np.ones((*features.shape[:2], 1)), projected), axis=2)


def fit_operators(
    tokens: Tokens, probabilities: Array, states: Array, ridge: float
) -> dict[str, Array]:
    """Fit q(x|h) s(hx) = B_x s(h), never a latent transition/emission template."""
    dimension, vocab = states.shape[-1], probabilities.shape[-1]
    flat = states.reshape(-1, dimension)
    penalty = np.eye(dimension) * ridge
    penalty[0, 0] = 0
    readout = np.linalg.solve(
        flat.T @ flat / len(flat) + penalty, flat.T @ probabilities.reshape(-1, vocab) / len(flat)
    )
    normalizer = np.eye(1, dimension)[0]
    readout += (normalizer - readout.sum(axis=1))[:, None] / vocab
    source = states[:, :-1].reshape(-1, dimension)
    following = states[:, 1:].reshape(-1, dimension)
    observed = tokens[:, :-1].ravel()
    mass = probabilities[:, :-1].reshape(-1, vocab)
    operators = np.zeros((vocab, dimension, dimension))
    for token in range(vocab):
        chosen = observed == token
        if chosen.any():
            left = source[chosen]
            target = following[chosen] * mass[chosen, token, None]
            operators[token] = np.linalg.solve(
                left.T @ left / len(left) + penalty, left.T @ target / len(left)
            ).T
        operators[token, 0] = readout[:, token]
    return {
        "initial": states[:, 0].mean(axis=0),
        "normalizer": normalizer,
        "operators": operators,
        "baseline": probabilities.mean(axis=(0, 1)),
    }


def identify(
    tokens: Tokens, probabilities: Array, features: Array
) -> tuple[dict[str, Array], dict[str, object]]:
    if (
        tokens.ndim != 2
        or tokens.shape[0] < 4
        or tokens.shape[1] < 2
        or tokens.dtype != np.int64
        or probabilities.ndim != 3
        or probabilities.shape[:2] != tokens.shape
        or probabilities.shape[2] < 2
        or features.ndim != 3
        or features.shape[:2] != tokens.shape
        or not features.shape[2]
        or (tokens < 0).any()
        or (tokens >= probabilities.shape[2]).any()
        or not np.isfinite(features).all()
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(axis=2), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("need aligned finite public sequences, probabilities and features")
    indices = np.random.default_rng(0).permutation(len(tokens))
    boundary = min(len(tokens) - 2, max(2, 3 * len(tokens) // 4))
    train, valid = indices[:boundary], indices[boundary:]
    flattened = features[train].reshape(-1, features.shape[-1])
    singular = np.linalg.svd(flattened - flattened.mean(axis=0), compute_uv=False)
    maximum = min(
        probabilities.shape[2], 1 + int(np.sum(singular > max(1e-10, singular[0] * 1e-8)))
    )
    trials = []
    successful = []
    for dimension in range(1, maximum + 1):
        states, _, _ = coordinates(features[train], dimension)
        for ridge in (1e-6, 1e-3, 0.1):
            try:
                parameters = fit_operators(tokens[train], probabilities[train], states, ridge)
                predictions, repairs = trajectory(tokens[valid], **parameters)
                target = probabilities[valid]
                kl = (
                    (target * (np.log(np.maximum(target, 1e-300)) - np.log(predictions)))
                    .sum(axis=2)
                    .mean(axis=1)
                )
                if not np.isfinite(kl).all():
                    raise ValueError("nonfinite autonomous validation loss")
                value, error = float(kl.mean()), float(kl.std(ddof=1) / np.sqrt(len(kl)))
                successful.append((dimension, ridge, value, error))
                trials.append(
                    {
                        "dimension": dimension,
                        "ridge": ridge,
                        "mean_kl": value,
                        "standard_error": error,
                        "repairs": repairs,
                        "error": None,
                    }
                )
            except (ValueError, np.linalg.LinAlgError) as failure:
                trials.append(
                    {"dimension": dimension, "ridge": ridge, "mean_kl": None, "error": str(failure)}
                )
    if not successful:
        raise ValueError("all coordinate dynamics failed: " + json.dumps(trials))
    best = min(successful, key=lambda trial: trial[2])
    selected = min((x for x in successful if x[2] <= best[2] + best[3]), key=lambda x: (x[0], x[2]))
    dimension, ridge, _, _ = selected
    states, _, _ = coordinates(features, dimension)
    parameters = fit_operators(tokens, probabilities, states, ridge)
    report = {
        "dimension": dimension,
        "ridge": ridge,
        "training_sequences": len(train),
        "validation_sequences": len(valid),
        "selection": "Smallest dimension within one sequence-validation standard error of best autonomous KL; best ridge at that dimension; refit all public observations.",
        "trials": trials,
        "singular_values": singular.tolist(),
        "structure": recognize(
            parameters["initial"], parameters["normalizer"], parameters["operators"]
        ),
    }
    return parameters, report


def prepare_bundle(output: Path, features: str = "outputs") -> None:
    if features not in ("outputs", "activations"):
        raise ValueError("unsupported feature source")
    output.mkdir(parents=True, exist_ok=False)
    package = output / "method"
    package.mkdir()
    (package / "__init__.py").write_text("")
    for name in ("state_geometry.py", "operator_structure.py", "operator_predictor.py"):
        shutil.copyfile(Path(__file__).with_name(name), package / name)
    (output / "main.py").write_text(f"from method.state_geometry import main\nmain({features!r})\n")


def main(features: str) -> None:
    output = Path("/output")
    with np.load("/challenge/observations.npz", allow_pickle=False) as data:
        selected = data["probabilities" if features == "outputs" else "activations"]
        parameters, report = identify(data["tokens"], data["probabilities"], selected)
    np.savez_compressed(output / "parameters.npz", allow_pickle=False, **parameters)
    shutil.copyfile(Path(__file__).with_name("operator_predictor.py"), output / "predict.py")
    (output / "selection.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "mechanism.json").write_text(
        json.dumps(
            {
                "representation": "autonomous signed linear predictive state",
                "state_dimension": report["dimension"],
                "update": "B[token] @ state, normalized by first coordinate; reset to initial on nonfinite or nonpositive mass",
                "readout": "First row of each B[token] applied to state, floored at 1e-9 and normalized",
                "family_evidence": report["structure"],
                "parameters": "parameters.npz",
            },
            indent=2,
        )
        + "\n"
    )
    (output / "submission.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "predictor": "predict.py",
                "mechanism": "mechanism.json",
                "hypothesis": "A time-homogeneous linear predictive state explains the sequence model; latent-family compatibility is tested analytically after learning operators.",
                "assumptions": "Finite-dimensional linear-state prior, unsupervised PCA coordinates, dimensions up to vocabulary size, token-conditioned ridge regression, declared clipping/reset rules. A built-in simultaneous-diagonalization test recognizes one latent Markov family when its fixed algebraic and positivity checks pass. This is a restricted recognition class, not unrestricted program discovery, an HMM fitting adapter, or a minimality/causal certificate.",
                "resources": {
                    "features": features,
                    "additional_teacher_execution": 0,
                    "selection": "selection.json",
                },
            },
            indent=2,
        )
        + "\n"
    )
