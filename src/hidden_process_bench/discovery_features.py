"""Learn representation features and circuit-pruned states without a process adapter."""

import json
import shutil
from pathlib import Path

import numpy as np

from .neural_numpy import NeuralModel
from .operator_predictor import Array, Tokens
from .state_geometry import identify, prepare_bundle, write_submission


def rectifier_loss(
    inputs: Array, targets: Array, parameters: dict[str, Array], sparsity: float
) -> tuple[float, dict[str, Array]]:
    preactivation = inputs @ parameters["encoder"] + parameters["encoder_bias"]
    codes = np.maximum(preactivation, 0)
    difference = codes @ parameters["decoder"] + parameters["decoder_bias"] - targets
    loss = float(np.mean(difference**2) + sparsity * codes.mean())
    derivative = 2 * difference / difference.size
    back = (derivative @ parameters["decoder"].T + sparsity / codes.size) * (preactivation > 0)
    return loss, {
        "encoder": inputs.T @ back,
        "encoder_bias": back.sum(axis=0),
        "decoder": codes.T @ derivative,
        "decoder_bias": derivative.sum(axis=0),
    }


def learn_rectifier(
    inputs: Array, targets: Array, *, steps: int = 800
) -> tuple[dict[str, Array], dict[str, object]]:
    if (
        inputs.ndim != 2
        or targets.ndim != 2
        or len(inputs) != len(targets)
        or not all(inputs.shape)
        or not all(targets.shape)
        or not np.isfinite(inputs).all()
        or not np.isfinite(targets).all()
        or type(steps) is not int
        or steps < 1
    ):
        raise ValueError("rectifier requires aligned finite nonempty matrices and positive steps")
    rng = np.random.default_rng(0)
    center, output_center = inputs.mean(axis=0), targets.mean(axis=0)
    scale = np.sqrt(np.mean((inputs - center) ** 2)) + 1e-8
    output_scale = np.sqrt(np.mean((targets - output_center) ** 2)) + 1e-8
    left, right = (inputs - center) / scale, (targets - output_center) / output_scale
    width = min(64, 2 * inputs.shape[1])
    parameters = {
        "encoder": rng.normal(size=(inputs.shape[1], width)) / np.sqrt(inputs.shape[1]),
        "encoder_bias": np.full(width, 0.05),
        "decoder": rng.normal(size=(width, targets.shape[1])) / np.sqrt(width),
        "decoder_bias": np.zeros(targets.shape[1]),
    }
    first = {key: np.zeros_like(value) for key, value in parameters.items()}
    second = {key: np.zeros_like(value) for key, value in parameters.items()}
    initial_loss = rectifier_loss(left, right, parameters, 1e-3)[0]
    for step in range(1, steps + 1):
        chosen = rng.integers(len(left), size=min(256, len(left)))
        _, gradient = rectifier_loss(left[chosen], right[chosen], parameters, 1e-3)
        for key in parameters:
            first[key] = 0.9 * first[key] + 0.1 * gradient[key]
            second[key] = 0.999 * second[key] + 0.001 * gradient[key] ** 2
            parameters[key] -= (
                0.01
                * (first[key] / (1 - 0.9**step))
                / (np.sqrt(second[key] / (1 - 0.999**step)) + 1e-8)
            )
    final_loss = rectifier_loss(left, right, parameters, 1e-3)[0]
    parameters.update(
        {
            "input_center": center,
            "input_scale": np.asarray(scale),
            "output_center": output_center,
            "output_scale": np.asarray(output_scale),
        }
    )
    codes = encode(inputs, parameters)
    return parameters, {
        "training_rows": len(inputs),
        "hidden_width": width,
        "steps": steps,
        "sparsity_penalty": 1e-3,
        "initial_objective": initial_loss,
        "final_objective": final_loss,
        "active_fraction": float((codes > 0).mean()),
        "training": "Fixed-seed minibatch Adam; squared reconstruction/prediction error plus mean positive activation penalty. No latent labels or process model.",
    }


def encode(inputs: Array, parameters: dict[str, Array]) -> Array:
    return np.maximum(
        ((inputs - parameters["input_center"]) / parameters["input_scale"]) @ parameters["encoder"]
        + parameters["encoder_bias"],
        0,
    )


def average_kl(reference: Array, candidate: Array) -> float:
    return float(
        (
            reference
            * (np.log(np.maximum(reference, 1e-300)) - np.log(np.maximum(candidate, 1e-300)))
        )
        .sum(axis=-1)
        .mean()
    )


def circuit_features(
    model: NeuralModel, inputs: Tokens, reference: Array, training: Tokens
) -> tuple[Array, dict[str, object]]:
    removed: list[tuple[int, int]] = []
    trials = []
    probabilities, traces = model.forward(inputs)
    if not np.allclose(probabilities, reference, atol=1e-5, rtol=0):
        raise ValueError("neural replay disagrees with cached probabilities")
    for layer in range(model.config["layers"]):
        for head in range(model.config["heads"]):
            candidate, candidate_traces = model.forward(
                inputs, ablate_heads=(*removed, (layer, head))
            )
            divergence = average_kl(reference[training], candidate[training])
            accepted = divergence <= 0.005
            trials.append(
                {"layer": layer, "head": head, "cumulative_kl": divergence, "removed": accepted}
            )
            if accepted:
                removed.append((layer, head))
                traces = candidate_traces
    return traces["final_normalized_residual"], {
        "ablation": "Greedy layer/head order, cumulative zero ablations before attention output projection; retain removal only at mean KL <= 0.005 against the original cached probabilities.",
        "removed_heads": removed,
        "trials": trials,
        "causal_scope": "Ablation effects on neural predictions only. No semantic intervention alignment to a recovered generating process, circuit completeness, or causal recovery is established.",
    }


def prepare_feature_bundle(output: Path, mode: str) -> None:
    if mode not in ("sae", "transcoder", "causal"):
        raise ValueError("unsupported discovery feature mode")
    prepare_bundle(output)
    for name in ("discovery_features.py", "neural_numpy.py"):
        shutil.copyfile(Path(__file__).with_name(name), output / "method" / name)
    (output / "main.py").write_text(f"from method.discovery_features import main\nmain({mode!r})\n")


def main(mode: str) -> None:
    root, output = Path("/challenge"), Path("/output")
    with np.load(root / "observations.npz", allow_pickle=False) as data:
        tokens, probabilities = data["tokens"], data["probabilities"]
        activations = data["activations"]
    order = np.random.default_rng(0).permutation(len(tokens))
    boundary = min(len(tokens) - 2, max(2, 3 * len(tokens) // 4))
    train = order[:boundary]
    counts: dict[str, int] = {"forward_calls": 0, "token_queries": 0, "prefix_tokens_processed": 0}
    evidence: dict[str, object] = {"mode": mode}
    if mode == "causal":
        model = NeuralModel(root)
        inputs = np.column_stack(
            (np.full(len(tokens), model.config["vocab"], dtype=np.int64), tokens[:, :-1])
        )
        features, details = circuit_features(model, inputs, probabilities, train)
        evidence.update(details)
        counts = model.counts
    else:
        inputs = targets = activations
        if mode == "transcoder":
            model = NeuralModel(root)
            neural_inputs = np.column_stack(
                (np.full(len(tokens), model.config["vocab"], dtype=np.int64), tokens[:, :-1])
            )
            replay, traces = model.forward(neural_inputs)
            if not np.allclose(replay, probabilities, atol=1e-5, rtol=0):
                raise ValueError("neural replay disagrees with cached probabilities")
            prefix = f"blocks.{model.config['layers'] - 1}"
            inputs, targets = traces[prefix + ".mlp_input"], traces[prefix + ".mlp_output"]
            counts = model.counts
        learned, details = learn_rectifier(
            inputs[train].reshape(-1, inputs.shape[-1]),
            targets[train].reshape(-1, targets.shape[-1]),
        )
        features = encode(inputs, learned)
        evidence.update(details)
        evidence["representation_training_sequences"] = len(train)
        evidence["representation_validation_sequences_used_for_training"] = False
        np.savez_compressed(output / "representation.npz", allow_pickle=False, **learned)
    parameters, selection = identify(tokens, probabilities, features)
    selection["representation"] = evidence
    write_submission(
        output,
        parameters,
        selection,
        {
            "features": mode,
            "neural_execution_counts": counts,
            "neural_count_scope": "All helper executions including ablations, every returned probability vector counted. Cached observation collection is additional and unchanged.",
            "assumptions": "Learned feature map or greedy circuit pruning followed by autonomous linear-state identification; no supplied HMM adapter. Representation evidence does not establish recovered latent semantics or a causal mechanism.",
        },
    )
    (output / "representation.json").write_text(json.dumps(evidence, indent=2) + "\n")
