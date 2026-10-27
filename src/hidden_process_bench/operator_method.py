"""Unknown-dimension observable operator identification using public observations only.

The prior is a finite-dimensional linear predictive state, not latent probability
coordinates. Validation chooses dimension; no generating-family adapter is used.
"""

import json
import shutil
from pathlib import Path

import numpy as np

from .operator_predictor import Array, Tokens, trajectory


def moments(tokens: Tokens, probabilities: Array) -> tuple[Array, Array, Array]:
    vocab = probabilities.shape[2]
    first, middle = tokens[:, :-2].ravel(), tokens[:, 1:-1].ravel()
    second = probabilities[:, 1:-1].reshape(-1, vocab)
    third = probabilities[:, 2:].reshape(-1, vocab)
    single = np.bincount(first, minlength=vocab).astype(np.float64) / len(first)
    pair = np.zeros((vocab, vocab), dtype=np.float64)
    triple = np.zeros((vocab, vocab, vocab), dtype=np.float64)
    for left in range(vocab):
        selected = first == left
        pair[:, left] = second[selected].sum(axis=0) / len(first)
        for center in range(vocab):
            triple[center, :, left] = third[selected & (middle == center)].sum(axis=0) / len(first)
    return single, pair, triple


def factorize(
    single: Array, pair: Array, triple: Array, start: Array, dimension: int
) -> tuple[Array, Array, Array]:
    basis = np.linalg.svd(pair, full_matrices=False)[0][:, :dimension]
    inverse = np.linalg.pinv(basis.T @ pair, rcond=1e-7)
    operators = np.einsum("ri,aij,jq->arq", basis.T, triple, inverse)
    normalizer = np.linalg.lstsq(pair.T @ basis, single, rcond=1e-7)[0]
    initial = basis.T @ start
    mass = float(initial @ normalizer)
    if not np.isfinite(mass) or mass <= 1e-12:
        raise ValueError("unusable initial observable state")
    return initial / mass, normalizer, operators


def identify(tokens: Tokens, probabilities: Array) -> tuple[dict[str, Array], dict[str, object]]:
    if (
        tokens.ndim != 2
        or tokens.shape[0] < 4
        or tokens.shape[1] < 3
        or probabilities.ndim != 3
        or probabilities.shape[:2] != tokens.shape
        or probabilities.shape[2] < 2
        or tokens.dtype != np.int64
        or (tokens < 0).any()
        or (tokens >= probabilities.shape[2]).any()
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(axis=-1), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError(
            "need aligned public observations: at least four sequences and three positions"
        )
    # Fixed method RNG; unrelated to every evaluator/generator stream.
    indices = np.random.default_rng(0).permutation(len(tokens))
    boundary = min(len(tokens) - 2, max(2, 3 * len(tokens) // 4))
    training, validation = indices[:boundary], indices[boundary:]
    single, pair, triple = moments(tokens[training], probabilities[training])
    start = probabilities[training, 0].mean(axis=0)
    baseline = probabilities[training].mean(axis=(0, 1))
    singular_values = np.linalg.svd(pair, compute_uv=False)
    maximum = int(np.sum(singular_values > singular_values[0] * 1e-7))
    trials = []
    valid: list[tuple[int, float, float]] = []
    target = probabilities[validation]
    for dimension in range(1, maximum + 1):
        try:
            initial, normalizer, operators = factorize(single, pair, triple, start, dimension)
            fitted, repairs = trajectory(
                tokens[validation], initial, normalizer, operators, baseline
            )
            differences = (
                np.where(
                    target > 0, target * (np.log(np.maximum(target, 1e-300)) - np.log(fitted)), 0
                )
                .sum(axis=-1)
                .mean(axis=1)
            )
            if not np.isfinite(differences).all():
                raise ValueError("nonfinite validation loss")
            mean = float(differences.mean())
            error = float(differences.std(ddof=1) / np.sqrt(len(differences)))
            valid.append((dimension, mean, error))
            trials.append(
                {
                    "dimension": dimension,
                    "mean_kl": mean,
                    "standard_error": error,
                    "repairs": repairs,
                    "error": None,
                }
            )
        except (ValueError, np.linalg.LinAlgError) as invalid:
            trials.append(
                {
                    "dimension": dimension,
                    "mean_kl": None,
                    "standard_error": None,
                    "repairs": None,
                    "error": str(invalid),
                }
            )
    if not valid:
        raise ValueError("all observable dimensions failed: " + json.dumps(trials, allow_nan=False))
    best = min(valid, key=lambda trial: trial[1])
    threshold = best[1] + best[2]
    dimension = next(trial[0] for trial in valid if trial[1] <= threshold)
    # The selected dimension is refit using the whole permitted observation cache.
    initial, normalizer, operators = factorize(
        *moments(tokens, probabilities), probabilities[:, 0].mean(axis=0), dimension
    )
    parameters = {
        "initial": initial,
        "normalizer": normalizer,
        "operators": operators,
        "baseline": probabilities.mean(axis=(0, 1)),
    }
    report = {
        "schema_version": 1,
        "dimension": dimension,
        "selection": "smallest dimension within one validation standard error of the best; then refit all public observations",
        "split_seed": 0,
        "training_sequences": len(training),
        "validation_sequences": len(validation),
        "singular_values": singular_values.tolist(),
        "trials": trials,
    }
    return parameters, report


def prepare_bundle(output: Path) -> None:
    """Package only the method, predictor and an empty namespace; no evaluator code."""
    output.mkdir(parents=True, exist_ok=False)
    package = output / "method"
    package.mkdir()
    (package / "__init__.py").write_text("")
    for name in ("operator_method.py", "operator_predictor.py"):
        shutil.copyfile(Path(__file__).with_name(name), package / name)
    (output / "main.py").write_text("from method.operator_method import main\nmain()\n")


def main() -> None:
    challenge, output = Path("/challenge"), Path("/output")
    with np.load(challenge / "observations.npz", allow_pickle=False) as observations:
        parameters, selection = identify(observations["tokens"], observations["probabilities"])
    np.savez_compressed(output / "parameters.npz", allow_pickle=False, **parameters)
    shutil.copyfile(Path(__file__).with_name("operator_predictor.py"), output / "predict.py")
    (output / "selection.json").write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
    mechanism = {
        "representation": "signed observable linear operators",
        "state_dimension": selection["dimension"],
        "parameters": "parameters.npz",
        "prediction": "p(a|history) proportional to normalizer @ operators[a] @ state; floor each output at 1e-9 and renormalize",
        "update": "state = operators[token] @ state; normalize by normalizer @ state; reset to initial on nonfinite/nonpositive mass",
        "initialization": "initial state is projected from mean empty-history neural prediction",
        "limitations": "Clipping and resets are part of this executable and may destroy exact linear word probabilities. Signed coordinates are not identified latent probability states. No generating-family, minimality or causal certificate is claimed.",
    }
    (output / "mechanism.json").write_text(json.dumps(mechanism, indent=2) + "\n")
    submission = {
        "schema_version": 1,
        "predictor": "predict.py",
        "mechanism": "mechanism.json",
        "hypothesis": "A finite-dimensional observable state updated by token-conditioned linear operators may explain the neural predictions. The method chooses its dimension from held-out public sequences. This is a representation hypothesis, not a declaration that the original process family has been identified.",
        "assumptions": "Built-in finite-dimensional linear predictive-state prior; time-homogeneous operators; a rank identifiable from one-symbol prefix/suffix moments; sampled triples; signed coordinates with the declared numerical repairs. Vocabulary bounds the searched dimension. No supplied hidden-state count or process constructor.",
        "resources": {
            "access_used": "cached tokens and next-token probabilities only",
            "additional_neural_execution": 0,
            "dimension_search": "every numerically supported dimension up to vocabulary size; all trials in selection.json",
            "selection": "selection.json",
            "training": "spectral factorization followed by public sequence validation and full-cache refit",
        },
    }
    (output / "submission.json").write_text(json.dumps(submission, indent=2) + "\n")
