"""Neutral public challenge snapshots; no generator, order, seed or oracle metadata."""

from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np

from hidden_process_bench.experiment import write_json
from hidden_process_bench.model import Observations, TinyTransformer
from hidden_process_bench.submissions import file_manifest

Access = Literal["outputs", "weights", "internals"]

INSTRUCTIONS = """Infer a compact executable account of the sequence-generating mechanism
represented by the supplied neural network or its observations. Choose and explain
your own representation, dependencies and update rules. Report uncertainty and any
built-in assumptions. Do not merely submit a copy of the supplied neural network.

Only /challenge contains task inputs. There is no network or evaluator access.
Available libraries are Python's standard library and NumPy. Write your submission
files under /output. Include submission.json with exactly: schema_version (1),
predictor (relative Python filename), mechanism (relative filename containing your
structural account), hypothesis (text), assumptions (text), resources (object
recording fitting/search work and information used).

The frozen predictor reads one JSON object from stdin with histories (a list of
integer-token lists, all of equal length) and vocab_size, and prints one JSON object
with probabilities (one next-token probability vector per history). It must be a
fixed, deterministic, row-wise predictor: no refitting on evaluation histories,
communication between rows, persistence between requests or external resources.
Every request starts a fresh process. Histories can exceed the neural context.
Your submitted code and parameters must run without the supplied neural network.
Predictive agreement and evidence for the proposed structure are assessed separately.
"""


def export_challenge(
    model: TinyTransformer,
    observations: Observations | None,
    access: Access,
    output: Path,
) -> dict[str, str]:
    """Return hashes to the evaluator; expose only the declared neural access level.

    Pass the same cached observations to paired output/internal conditions. Weights
    permit independently implemented execution, so execution counts are declarations,
    not a claim that offline forward passes are technically prevented.
    """
    if access not in ("outputs", "weights", "internals"):
        raise ValueError("unknown neural access condition")
    config = model.config
    if access != "weights":
        if observations is None:
            raise ValueError("this access condition requires cached model observations")
        tokens, probabilities, activations = (
            observations.tokens,
            observations.probabilities,
            observations.activations,
        )
        if (
            tokens.ndim != 2
            or not all(tokens.shape)
            or tokens.shape[1] > config.context
            or probabilities.shape != (*tokens.shape, config.vocab)
            or activations.shape != (*tokens.shape, config.width)
            or tokens.dtype != np.int64
            or (tokens < 0).any()
            or (tokens >= config.vocab).any()
        ):
            raise ValueError("cached observations do not match the neural architecture")
    output.mkdir(parents=True, exist_ok=False)
    budget = {"forward_calls": 0, "token_queries": 0, "prefix_tokens_processed": 0}
    if access != "weights" and observations is not None:
        arrays = {"tokens": observations.tokens, "probabilities": observations.probabilities}
        if access == "internals":
            arrays["activations"] = observations.activations
        np.savez_compressed(output / "observations.npz", allow_pickle=False, **arrays)
        budget = {name: getattr(observations, name) for name in budget}
    if access != "outputs":
        np.savez_compressed(
            output / "weights.npz",
            **{name: value.detach().cpu().numpy() for name, value in model.state_dict().items()},
        )
    write_json(
        output / "challenge.json",
        {
            "schema_version": 1,
            "access": access,
            "architecture": asdict(config),
            "neural_operations": {
                "block": "pre-LayerNorm; causal multihead attention; residual; pre-LayerNorm; MLP; residual",
                "layer_norm_epsilon": 1e-5,
                "mlp": "Linear(width, 4*width), exact GELU, Linear(4*width, width)",
                "attention": "qkv Linear; contiguous heads; softmax(q k^T / sqrt(head_width)); causal mask; output Linear",
                "readout": "final LayerNorm; unembedding Linear without bias; softmax",
                "embedding": "token embedding plus learned absolute position embedding",
                "linear_weights": "out_features by in_features; biases included when present",
                "bos_token": config.vocab,
                "output_tokens": list(range(config.vocab)),
                "alignment": "probabilities[:, t] predicts tokens[:, t] from BOS and tokens[:, :t]",
                "activations": "final normalized residual at each prediction position, when supplied",
            },
            "cached_observation_budget": budget,
            "additional_teacher_queries": 0,
            "offline_execution_accounting": "self-reported when weights are supplied",
        },
    )
    (output / "instructions.txt").write_text(INSTRUCTIONS)
    return file_manifest(output, max_bytes=64_000_000)
