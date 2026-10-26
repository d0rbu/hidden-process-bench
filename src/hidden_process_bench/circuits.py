"""Greedy node ablation and a held-out candidate-belief interchange test.

This is a node-pruning adaptation, not an implementation of ACDC edge patching.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from hidden_process_bench.metrics import sequence_kl
from hidden_process_bench.model import Block, TinyTransformer, teacher_inputs
from hidden_process_bench.process import Floats, MixtureHMM, Tokens


@contextmanager
def ablate(model: TinyTransformer, removed: set[str]) -> Iterator[None]:
    handles = []
    for layer, raw_block in enumerate(model.blocks):
        block = cast(Block, raw_block)
        heads = [head for head in range(block.heads) if f"{layer}.head.{head}" in removed]

        def qkv_hook(
            _module: nn.Module,
            _inputs: tuple[Tensor, ...],
            output: Tensor,
            heads: list[int] = heads,
            heads_count: int = block.heads,
        ) -> Tensor:
            batch, length, width3 = output.shape
            values = output.clone().reshape(
                batch, length, 3, heads_count, width3 // 3 // heads_count
            )
            values[:, :, 2, heads] = 0
            return values.reshape(batch, length, width3)

        if heads:
            handles.append(block.qkv.register_forward_hook(qkv_hook))
        if f"{layer}.mlp" in removed:

            def mlp_hook(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> Tensor:
                return torch.zeros_like(output)

            handles.append(block.mlp.register_forward_hook(mlp_hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def _predict(model: TinyTransformer, tokens: Tokens) -> Floats:
    return (
        model(teacher_inputs(torch.tensor(tokens), model.config.vocab))[0]
        .softmax(-1)
        .numpy()
        .astype(np.float64)
    )


@dataclass(frozen=True)
class CircuitResult:
    probabilities: Floats
    removed: frozenset[str]
    diagnostics: dict[str, object]


def discover_circuit(
    model: TinyTransformer, tokens: Tokens, tolerance: float = 0.002
) -> CircuitResult:
    if len(tokens) < 2 or tolerance < 0:
        raise ValueError("need batch >= 2 and nonnegative tolerance")
    count = len(tokens) // 2
    reference = _predict(model, tokens)
    removed: set[str] = set()
    trials = []
    nodes = [
        name
        for layer in reversed(range(len(model.blocks)))
        for name in [
            f"{layer}.mlp",
            *[f"{layer}.head.{head}" for head in reversed(range(model.config.heads))],
        ]
    ]
    for node in nodes:
        with ablate(model, removed | {node}):
            prediction = _predict(model, tokens[:count])
        loss = float(sequence_kl(reference[:count], prediction).mean())
        accepted = loss <= tolerance
        trials.append({"node": node, "calibration_kl": loss, "removed": accepted})
        if accepted:
            removed.add(node)
    with ablate(model, removed):
        prediction = _predict(model, tokens)
    with ablate(model, set(nodes)):
        control = _predict(model, tokens[count:])
    return CircuitResult(
        prediction,
        frozenset(removed),
        {
            "algorithm": "greedy_zero_node_ablation",
            "threshold_nats": tolerance,
            "calibration_sequences": count,
            "heldout_sequences": len(tokens) - count,
            "nodes_total": len(nodes),
            "nodes_retained": len(nodes) - len(removed),
            "trials": trials,
            "heldout_kl": float(sequence_kl(reference[count:], prediction[count:]).mean()),
            "all_removed_heldout_kl": float(sequence_kl(reference[count:], control).mean()),
            "additional_forward_calls": len(nodes) + 3,
            "additional_token_queries": tokens.size * 2
            + tokens[:count].size * len(nodes)
            + tokens[count:].size,
            "additional_prefix_tokens_processed": tokens.size * 2
            + tokens[:count].size * len(nodes)
            + tokens[count:].size,
        },
    )


def _continued(hmm: MixtureHMM, belief: Floats, tokens: Tokens) -> Floats:
    predictions = []
    for t in range(tokens.shape[1]):
        prediction = np.einsum("bks,ksv->bv", belief, hmm.emission)
        predictions.append(prediction)
        posterior = belief * hmm.emission[:, :, tokens[:, t]].transpose(2, 0, 1)
        posterior /= posterior.sum(axis=(1, 2), keepdims=True)
        belief = np.einsum("bks,ksj->bkj", posterior, hmm.transition)
    return np.stack(predictions, axis=1)


def interchange_test(
    model: TinyTransformer, hmm: MixtureHMM, tokens: Tokens, first_residuals: Floats
) -> dict[str, float | int | str]:
    """Fit belief->first-block map on calibration, patch one held-out position.

    Compare actual model interventions against the candidate's state-swap predictions,
    including future positions. Failure is evidence against this PARTICULAR mapping.
    No true states, oracle responses, or feedback-based candidate selection are used.
    """
    count = len(tokens) // 2
    position = tokens.shape[1] // 2
    beliefs = hmm.filter(tokens).beliefs[:, :-1]
    flat = beliefs.reshape(*tokens.shape, -1)
    design = np.concatenate([flat, np.ones((*tokens.shape, 1))], axis=-1)
    x = design[:count].reshape(-1, design.shape[-1])
    y = first_residuals[:count].reshape(-1, first_residuals.shape[-1])
    decoder = np.linalg.solve(x.T @ x + 0.01 * np.eye(x.shape[-1]), x.T @ y)
    heldout = tokens[count:]
    source = np.roll(np.arange(len(heldout)), 1)
    delta = (design[count:][source, position] - design[count:, position]) @ decoder

    def patch(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> Tensor:
        result = output.clone()
        result[:, position] += torch.tensor(delta, dtype=output.dtype)
        return result

    before = _predict(model, heldout)[:, position:]
    handle = model.blocks[0].register_forward_hook(patch)
    try:
        after = _predict(model, heldout)[:, position:]
    finally:
        handle.remove()
    target = _continued(hmm, beliefs[count:][source, position].copy(), heldout[:, position:])
    baseline = hmm.filter(heldout).predictions[:, position:-1]
    observed_change, predicted_change = after - before, target - baseline
    nmse = float(
        np.sum((observed_change - predicted_change) ** 2) / max(np.sum(predicted_change**2), 1e-12)
    )
    return {
        "mapping": "ridge_candidate_belief_to_first_block_residual",
        "position": position,
        "heldout_sequences": len(heldout),
        "counterfactual_kl": float(sequence_kl(target, after).mean()),
        "unpatched_candidate_kl": float(sequence_kl(baseline, before).mean()),
        "change_normalized_mse": nmse,
        "additional_forward_calls": 2,
        "additional_token_queries": heldout.size * 2,
        "additional_prefix_tokens_processed": heldout.size * 2,
    }
