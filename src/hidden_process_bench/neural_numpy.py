"""Standalone execution of the public neural architecture, without process priors.

Only stdlib and NumPy are imported so methods can copy this file into their isolated
program bundle. Counts describe this helper's executions, not an enforced query cap.
"""

import json
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

Floats = NDArray[np.float64]
Tokens = NDArray[np.int64]

OPERATIONS = {
    "block": "pre-LayerNorm; causal multihead attention; residual; pre-LayerNorm; MLP; residual",
    "layer_norm_epsilon": 1e-5,
    "mlp": "Linear(width, 4*width), exact GELU, Linear(4*width, width)",
    "attention": "qkv Linear; contiguous heads; softmax(q k^T / sqrt(head_width)); causal mask; output Linear",
    "readout": "final LayerNorm; unembedding Linear without bias; softmax",
    "embedding": "token embedding plus learned absolute position embedding",
    "linear_weights": "out_features by in_features; biases included when present",
}


def _softmax(values: Floats) -> Floats:
    mass = np.exp(values - values.max(axis=-1, keepdims=True))
    return mass / mass.sum(axis=-1, keepdims=True)


class NeuralModel:
    def __init__(self, root: Path) -> None:
        metadata = json.loads((root / "challenge.json").read_text())
        config = metadata["architecture"]
        if (
            metadata["schema_version"] != 1
            or set(config) != {"vocab", "context", "width", "heads", "layers"}
            or any(type(value) is not int or value < 1 for value in config.values())
            or config["width"] % config["heads"]
            or any(
                metadata["neural_operations"].get(key) != value for key, value in OPERATIONS.items()
            )
        ):
            raise ValueError("unsupported public neural architecture")
        self.config: dict[str, int] = config
        self.weights: dict[str, Floats] = {}
        width, vocab = config["width"], config["vocab"]
        shapes = {
            "embedding.weight": (vocab + 1, width),
            "position.weight": (config["context"], width),
            "norm.weight": (width,),
            "norm.bias": (width,),
            "unembedding.weight": (vocab, width),
        }
        for layer in range(config["layers"]):
            for name, shape in {
                "norm1.weight": (width,),
                "norm1.bias": (width,),
                "qkv.weight": (3 * width, width),
                "qkv.bias": (3 * width,),
                "attn_out.weight": (width, width),
                "attn_out.bias": (width,),
                "norm2.weight": (width,),
                "norm2.bias": (width,),
                "mlp.0.weight": (4 * width, width),
                "mlp.0.bias": (4 * width,),
                "mlp.2.weight": (width, 4 * width),
                "mlp.2.bias": (width,),
            }.items():
                shapes[f"blocks.{layer}.{name}"] = shape
        with np.load(root / "weights.npz", allow_pickle=False) as archive:
            if set(archive.files) != set(shapes):
                raise ValueError("weights do not match the public architecture")
            for name, shape in shapes.items():
                values = archive[name]
                if (
                    values.shape != shape
                    or values.dtype.kind != "f"
                    or not np.isfinite(values).all()
                ):
                    raise ValueError("invalid neural weight: " + name)
                self.weights[name] = values.astype(np.float64)
        self.counts = {"forward_calls": 0, "token_queries": 0, "prefix_tokens_processed": 0}

    def _linear(self, values: Floats, name: str) -> Floats:
        return values @ self.weights[name + ".weight"].T + self.weights[name + ".bias"]

    def _norm(self, values: Floats, name: str) -> Floats:
        centered = values - values.mean(axis=-1, keepdims=True)
        normalized = centered / np.sqrt((centered**2).mean(axis=-1, keepdims=True) + 1e-5)
        return normalized * self.weights[name + ".weight"] + self.weights[name + ".bias"]

    def forward(
        self,
        inputs: Tokens,
        *,
        ablate_heads: tuple[tuple[int, int], ...] = (),
        ablate_mlps: tuple[int, ...] = (),
    ) -> tuple[Floats, dict[str, Floats]]:
        """Return every position's probabilities and traces, counting every output row.

        Inputs include BOS explicitly. Zeroed heads are removed before attention's
        output projection; zeroed MLPs remove their residual contribution. All later
        computations are rerun. Neither intervention changes the stored weights.
        """
        c = self.config
        if (
            inputs.ndim != 2
            or inputs.dtype != np.int64
            or not all(inputs.shape)
            or inputs.shape[1] > c["context"]
            or (inputs < 0).any()
            or (inputs > c["vocab"]).any()
        ):
            raise ValueError("inputs require nonempty in-range int64 neural tokens within context")
        if any(
            type(layer) is not int
            or type(head) is not int
            or not 0 <= layer < c["layers"]
            or not 0 <= head < c["heads"]
            for layer, head in ablate_heads
        ) or any(type(layer) is not int or not 0 <= layer < c["layers"] for layer in ablate_mlps):
            raise ValueError("intervention outside neural architecture")
        batch, length = inputs.shape
        self.counts["forward_calls"] += 1
        self.counts["token_queries"] += batch * length
        self.counts["prefix_tokens_processed"] += batch * length
        hidden = self.weights["embedding.weight"][inputs] + self.weights["position.weight"][:length]
        traces: dict[str, Floats] = {}
        head_width = c["width"] // c["heads"]
        for layer in range(c["layers"]):
            name = f"blocks.{layer}"
            traces[name + ".residual_input"] = hidden.copy()
            qkv = self._linear(self._norm(hidden, name + ".norm1"), name + ".qkv")
            q, k, v = qkv.reshape(batch, length, 3, c["heads"], head_width).transpose(2, 0, 3, 1, 4)
            scores = (q @ k.swapaxes(-1, -2)) / math.sqrt(head_width)
            scores[..., np.triu(np.ones((length, length), dtype=bool), k=1)] = -np.inf
            heads = (_softmax(scores) @ v).transpose(0, 2, 1, 3)
            for selected_layer, head in ablate_heads:
                if selected_layer == layer:
                    heads[:, :, head] = 0
            traces[name + ".attention_heads"] = heads.copy()
            hidden = hidden + self._linear(
                heads.reshape(batch, length, c["width"]), name + ".attn_out"
            )
            mlp_input = self._norm(hidden, name + ".norm2")
            expanded = self._linear(mlp_input, name + ".mlp.0")
            activated = (
                0.5
                * expanded
                * (1 + np.vectorize(math.erf, otypes=[float])(expanded / math.sqrt(2)))
            )
            contribution = self._linear(activated, name + ".mlp.2")
            if layer in ablate_mlps:
                contribution = np.zeros_like(contribution)
            traces[name + ".mlp_input"] = mlp_input
            traces[name + ".mlp_output"] = contribution
            hidden = hidden + contribution
            traces[name + ".residual_output"] = hidden.copy()
        normalized = self._norm(hidden, "norm")
        traces["final_normalized_residual"] = normalized
        return _softmax(normalized @ self.weights["unembedding.weight"].T), traces
