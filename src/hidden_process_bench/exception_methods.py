"""Fixed, standalone NumPy searches for exceptions to a public addition rule.

These are anomaly/search adaptations, not full circuit-discovery implementations.
No evaluator modules, exception labels or training metadata are imported.
"""

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

from .discovery_features import encode as sparse_encode
from .discovery_features import learn_rectifier
from .neural_numpy import NeuralModel
from .operator_predictor import Array, Tokens

MODES = ("pca", "sae", "transcoder", "neurons", "ablation", "probabilities")


def clean_encoding(base: int, digits: int, ids: Tokens) -> tuple[Tokens, Tokens]:
    limit = base**digits
    if ids.ndim != 1 or (ids < 0).any() or (ids >= limit**2).any():
        raise ValueError("invalid ordered input IDs")
    a, b = ids // limit, ids % limit
    columns = []
    for i in range(digits):
        columns.extend((a // base**i % base, b // base**i % base))
    columns.append(np.full_like(ids, base))
    expected = np.stack([(a + b) // base**i % base for i in range(digits + 1)], axis=1)
    return np.column_stack((*columns, expected[:, :-1])), expected


def loss(probabilities: Array, expected: Tokens, digits: int) -> Array:
    selected = probabilities[:, 2 * digits :]
    correct = np.take_along_axis(selected, expected[:, :, None], axis=-1)[:, :, 0]
    return -np.log(np.maximum(correct, 1e-300)).mean(axis=1)


class Probe:
    def __init__(self, root: Path, base: int, digits: int, mode: str) -> None:
        self.model = NeuralModel(root)
        self.base, self.digits, self.mode = base, digits, mode
        self.units = [
            (layer, head)
            for layer in range(self.model.config["layers"])
            for head in range(self.model.config["heads"])
        ]
        self.units.extend((layer, -1) for layer in range(self.model.config["layers"]))
        self.inputs_evaluated = 0

    def evaluate(self, ids: Tokens) -> dict[str, Array]:
        result: dict[str, list[Array]] = {}
        for start in range(0, len(ids), 64):
            batch = ids[start : start + 64]
            inputs, expected = clean_encoding(self.base, self.digits, batch)
            probabilities, traces = self.model.forward(inputs)
            last = f"blocks.{self.model.config['layers'] - 1}"
            selected = slice(2 * self.digits, None)
            data = {"loss": loss(probabilities, expected, self.digits)[:, None]}
            if self.mode in ("pca", "sae"):
                data["inputs"] = traces["final_normalized_residual"][:, selected].reshape(
                    len(batch), -1
                )
            if self.mode in ("transcoder", "neurons"):
                mlp = traces[last + ".mlp_input"][:, selected]
                data["inputs"] = mlp.reshape(len(batch), -1)
                data["targets"] = traces[last + ".mlp_output"][:, selected].reshape(len(batch), -1)
                expanded = (
                    mlp @ self.model.weights[last + ".mlp.0.weight"].T
                    + self.model.weights[last + ".mlp.0.bias"]
                )
                activated = (
                    0.5
                    * expanded
                    * (1 + np.vectorize(math.erf, otypes=[float])(expanded / np.sqrt(2)))
                )
                data["neurons"] = activated.max(axis=1)
            if self.mode == "ablation":
                gains = []
                for layer, head in self.units:
                    changed, _ = self.model.forward(
                        inputs,
                        ablate_heads=((layer, head),) if head >= 0 else (),
                        ablate_mlps=(layer,) if head < 0 else (),
                    )
                    gains.append(data["loss"][:, 0] - loss(changed, expected, self.digits))
                data["gains"] = np.stack(gains, axis=1)
            for name, values in data.items():
                result.setdefault(name, []).append(values)
        self.inputs_evaluated += len(ids)
        return {name: np.concatenate(values) for name, values in result.items()}


def fit(
    mode: str, data: dict[str, Array], steps: int
) -> tuple[dict[str, Array], dict[str, object]]:
    if mode not in MODES:
        raise ValueError("unsupported exception method")
    if mode == "probabilities":
        return {}, {"objective": "clean-reference mean output-token negative log likelihood"}
    if mode == "ablation":
        return {"penalty": np.maximum(-data["gains"].mean(axis=0), 0)}, {
            "objective": "max clean-reference loss improvement under one zero ablation minus its mean calibration harm"
        }
    if mode == "neurons":
        return {
            "center": data["neurons"].mean(axis=0),
            "scale": np.maximum(data["neurons"].std(axis=0), 0.1),
        }, {"objective": "maximum positive standardized final-MLP hidden-unit activation"}
    if mode == "pca":
        center = data["inputs"].mean(axis=0)
        _, _, right = np.linalg.svd(data["inputs"] - center, full_matrices=False)
        return {"center": center, "basis": right[:8].T}, {
            "dimension": 8,
            "objective": "mean squared residual outside eight calibration PCA directions",
        }
    parameters, report = learn_rectifier(
        data["inputs"], data.get("targets", data["inputs"]), steps=steps
    )
    report["objective"] = (
        "mean squared reconstruction error"
        if mode == "sae"
        else "mean squared final-MLP prediction error"
    )
    return parameters, report


def anomaly(mode: str, data: dict[str, Array], parameters: dict[str, Array]) -> Array:
    if mode == "probabilities":
        return data["loss"][:, 0]
    if mode == "ablation":
        return (data["gains"] - parameters["penalty"]).max(axis=1)
    if mode == "neurons":
        return ((data["neurons"] - parameters["center"]) / parameters["scale"]).max(axis=1)
    if mode == "pca":
        centered = data["inputs"] - parameters["center"]
        residual = centered - (centered @ parameters["basis"]) @ parameters["basis"].T
    else:
        reconstruction = (
            sparse_encode(data["inputs"], parameters) @ parameters["decoder"]
            + parameters["decoder_bias"]
        )
        target = data.get("targets", data["inputs"])
        target = (target - parameters["output_center"]) / parameters["output_scale"]
        residual = reconstruction - target
    return np.mean(residual**2, axis=1)


def proposals(
    base: int, digits: int, ranked: list[int], seen: set[int], count: int, rng: np.random.Generator
) -> Tokens:
    limit = base**digits
    remaining = limit**2 - len(seen)
    count = min(count, remaining)
    neighbors = []
    for value in ranked[:4]:
        a, b = divmod(value, limit)
        for operand in range(2):
            for column in range(digits):
                scale = base**column
                old = (a, b)[operand] // scale % base
                for digit in range(base):
                    pair = [a, b]
                    pair[operand] += (digit - old) * scale
                    candidate = pair[0] * limit + pair[1]
                    if candidate not in seen and candidate not in neighbors:
                        neighbors.append(candidate)
    # Keep half of each adaptive batch for fresh uniform exploration.
    rng.shuffle(neighbors)
    result = neighbors[: count // 2]
    used = seen | set(result)
    while len(result) < count:
        value = int(rng.integers(limit**2))
        if value not in used:
            used.add(value)
            result.append(value)
    return np.array(result, dtype=np.int64)


def discover(
    root: Path,
    task: dict[str, int],
    mode: str,
    *,
    budget: int = 512,
    seed: int = 721,
    steps: int = 300,
) -> tuple[list[list[int]], dict[str, object]]:
    if mode not in MODES or budget < 4 or steps < 1:
        raise ValueError("invalid method or numerical budget")
    base, digits = task["base"], task["digits"]
    limit = base**digits
    rng = np.random.default_rng(seed)
    probe = Probe(root, base, digits, mode)
    calibration_ids = rng.choice(limit**2, size=min(128, budget, limit**2), replace=False).astype(
        np.int64
    )
    calibration = probe.evaluate(calibration_ids)
    parameters, training = fit(mode, calibration, steps)
    scores = anomaly(mode, calibration, parameters)
    ids = calibration_ids.tolist()
    values = scores.tolist()
    unit_indices = (
        (calibration.get("gains", np.zeros((len(ids), 1))) - parameters.get("penalty", 0))
        .argmax(axis=1)
        .tolist()
    )
    seen = set(ids)
    batches = [len(ids)]
    while len(seen) < min(budget, limit**2):
        ranked = [ids[i] for i in np.argsort(-np.asarray(values), kind="stable")]
        chosen = proposals(base, digits, ranked, seen, min(96, budget - len(seen)), rng)
        data = probe.evaluate(chosen)
        ids.extend(chosen.tolist())
        values.extend(anomaly(mode, data, parameters).tolist())
        unit_indices.extend(
            (data.get("gains", np.zeros((len(chosen), 1))) - parameters.get("penalty", 0))
            .argmax(axis=1)
            .tolist()
        )
        seen.update(chosen.tolist())
        batches.append(len(chosen))
    order = np.argsort(-np.asarray(values), kind="stable")
    ranked_pairs = [list(divmod(ids[index], limit)) for index in order]
    claims = (
        [
            {
                "input": list(divmod(ids[index], limit)),
                "unit": list(probe.units[unit_indices[index]]),
            }
            for index in order
        ]
        if mode == "ablation"
        else []
    )
    return ranked_pairs, {
        "method": mode,
        "training": training,
        "offline_execution": probe.model.counts,
        "distinct_inputs": len(seen),
        "batches": batches,
        "seed": seed,
        "representation_steps": steps,
        "ranked_inputs": ranked_pairs,
        "ranked_scores": [values[i] for i in order],
        "proposed_ablations": claims,
        "scope": "Fixed input-search heuristic. Uses only public clean reference and neural weights. No private trigger labels, fitted HMM or circuit success flag.",
    }


def prepare(output: Path, mode: str) -> None:
    if mode not in MODES:
        raise ValueError("unsupported exception method")
    output.mkdir(parents=True, exist_ok=False)
    package = output / "method"
    package.mkdir()
    (package / "__init__.py").write_text("")
    for name in (
        "exception_methods.py",
        "neural_numpy.py",
        "discovery_features.py",
        "state_geometry.py",
        "operator_structure.py",
        "operator_predictor.py",
    ):
        shutil.copyfile(Path(__file__).with_name(name), package / name)
    (output / "main.py").write_text(f"from method.exception_methods import main\nmain({mode!r})\n")


def main(mode: str) -> None:
    request = json.load(sys.stdin)
    history = request["history"]
    if history:
        record = history[0]
        candidates = [
            pair
            for pair, output in zip(record["queries"], record["outputs"], strict=True)
            if sum(pair) != output
        ][: request["candidate_limit"]]
        claims = [
            claim
            for claim in record["resources"]["proposed_ablations"]
            if claim["input"] in candidates
        ]
        response = {
            "queries": [],
            "candidates": candidates,
            "resources": {
                "method": mode,
                "offline_execution": {
                    "forward_calls": 0,
                    "token_queries": 0,
                    "prefix_tokens_processed": 0,
                },
                "candidate_ablation_claims": claims,
            },
        }
    else:
        ranked, resources = discover(Path("/challenge"), request["task"], mode)
        response = {
            "queries": ranked[: min(64, request["remaining_queries"])],
            "candidates": [],
            "resources": resources,
        }
    print(json.dumps(response, allow_nan=False))
