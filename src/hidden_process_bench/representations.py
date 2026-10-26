"""Unsupervised representations and token-conditioned observable belief dynamics."""

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from hidden_process_bench.config import check_count
from hidden_process_bench.model import Block, Observations, TinyTransformer, observe
from hidden_process_bench.process import Floats, Tokens


@dataclass(frozen=True)
class Traces:
    observations: Observations
    residuals: tuple[Floats, ...]
    mlp_inputs: Floats
    mlp_outputs: Floats


def observe_traces(model: TinyTransformer, count: int, length: int, seed: int) -> Traces:
    """Collect extra internals in the SAME autoregressive calls as the shared outputs."""
    residuals: list[list[np.ndarray]] = [[] for _ in model.blocks]
    mlp_inputs: list[np.ndarray] = []
    mlp_outputs: list[np.ndarray] = []
    handles = []
    for index, block in enumerate(model.blocks):

        def residual_hook(
            _module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor, index: int = index
        ) -> None:
            residuals[index].append(output[:, -1].detach().numpy().astype(np.float64))

        handles.append(block.register_forward_hook(residual_hook))

    def mlp_hook(_module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        mlp_inputs.append(inputs[0][:, -1].detach().numpy().astype(np.float64))
        mlp_outputs.append(output[:, -1].detach().numpy().astype(np.float64))

    handles.append(cast(Block, model.blocks[-1]).mlp.register_forward_hook(mlp_hook))
    try:
        observations = observe(model, count, length, seed)
    finally:
        for handle in handles:
            handle.remove()
    return Traces(
        observations,
        tuple(np.stack(x, axis=1) for x in residuals),
        np.stack(mlp_inputs, axis=1),
        np.stack(mlp_outputs, axis=1),
    )


@dataclass(frozen=True)
class Representation:
    features: Floats
    diagnostics: dict[str, float | int | str]


def _standardize(values: Floats, fit_count: int) -> tuple[Floats, Floats, Floats]:
    mean = values[:fit_count].mean(axis=(0, 1))
    scale = np.maximum(values[:fit_count].std(axis=(0, 1)), 1e-8)
    return (values - mean) / scale, mean, scale


def representation(
    inputs: Floats,
    *,
    kind: str,
    dimension: int,
    steps: int,
    seed: int,
    targets: Floats | None = None,
) -> Representation:
    """Fit on first 80% of sequences; report reconstruction on the remaining 20%."""
    check_count(dimension)
    check_count(steps)
    if inputs.ndim != 3 or len(inputs) < 2 or not np.isfinite(inputs).all():
        raise ValueError("representation needs finite [batch>=2,length,width]")
    if kind not in ("pca", "sae", "transcoder"):
        raise ValueError("unknown representation")
    if kind == "transcoder" and (targets is None or targets.shape != inputs.shape):
        raise ValueError("transcoder needs aligned input/output arrays")
    fit_count = max(1, len(inputs) * 4 // 5)
    x, _, _ = _standardize(inputs, fit_count)
    y, _, _ = _standardize(targets if targets is not None else inputs, fit_count)
    train = x[:fit_count].reshape(-1, x.shape[-1])
    if kind == "pca":
        _, singular, vt = np.linalg.svd(train, full_matrices=False)
        basis = vt[:dimension].T
        features = x @ basis
        reconstruction = features @ basis.T
        fitted_parameters = basis.size
        auxiliary = float((singular[:dimension] ** 2).sum() / (singular**2).sum())
    else:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            encoder = nn.Linear(x.shape[-1], dimension, dtype=torch.float64)
            decoder = nn.Linear(dimension, y.shape[-1], dtype=torch.float64)
        optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=0.01)
        rng = np.random.default_rng(seed)
        train_x = torch.tensor(train)
        train_y = torch.tensor(y[:fit_count].reshape(-1, y.shape[-1]))
        for _ in range(steps):
            index = torch.tensor(rng.integers(len(train), size=min(256, len(train))))
            latent = encoder(train_x[index]).relu()
            # Decoder column normalization prevents rescaling away the L1 penalty.
            with torch.no_grad():
                decoder.weight.div_(decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8))
            loss = (decoder(latent) - train_y[index]).square().mean() + 0.01 * latent.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            latent = encoder(torch.tensor(x)).relu()
            features = latent.numpy()
            reconstruction = decoder(latent).numpy()
        fitted_parameters = sum(p.numel() for p in [*encoder.parameters(), *decoder.parameters()])
        auxiliary = float((features[fit_count:] > 0).sum(axis=-1).mean())
    error = np.mean((reconstruction[fit_count:] - y[fit_count:]) ** 2)
    baseline = np.mean(y[fit_count:] ** 2)
    return Representation(
        features,
        {
            "kind": kind,
            "fit_sequences": fit_count,
            "test_sequences": len(inputs) - fit_count,
            "feature_dimension": features.shape[-1],
            "fit_parameters": fitted_parameters,
            "heldout_normalized_mse": float(error / max(baseline, 1e-12)),
            "explained_variance_or_active_features": auxiliary,
            "training_steps": 0 if kind == "pca" else steps,
        },
    )


def belief_dynamics(
    features: Floats, tokens: Tokens, probabilities: Floats, ridge: float = 0.01
) -> tuple[Floats, dict[str, float | int]]:
    """Recover affine observable coordinates q and unnormalized Bayes operators.

    q_(t+1) p(x_t|history) = q_t M_(x_t). No true beliefs or states enter.
    Operators are coordinates of an observable system, not stochastic HMM matrices.
    """
    if (
        features.shape[:2] != tokens.shape
        or probabilities.shape[:2] != tokens.shape
        or tokens.shape[1] < 2
        or len(tokens) < 2
        or ridge <= 0
    ):
        raise ValueError("aligned sequences with batch,length >= 2 and positive ridge required")
    count = max(1, len(tokens) * 4 // 5)
    q = np.concatenate([features, np.ones((*tokens.shape, 1))], axis=-1)
    x = q[:, :-1]
    probability = np.take_along_axis(probabilities[:, :-1], tokens[:, :-1, None], axis=-1)
    y = q[:, 1:] * probability
    operators = []
    predicted = np.zeros_like(y)
    for token in range(probabilities.shape[-1]):
        mask = tokens[:count, :-1] == token
        design, response = x[:count][mask], y[:count][mask]
        operator = np.linalg.solve(
            design.T @ design + ridge * np.eye(q.shape[-1]), design.T @ response
        )
        operators.append(operator)
        mask_all = tokens[:, :-1] == token
        predicted[mask_all] = x[mask_all] @ operator
    error = np.mean((predicted[count:] - y[count:]) ** 2)
    baseline = np.mean(y[count:] ** 2)
    return np.stack(operators), {
        "heldout_unnormalized_update_nmse": float(error / max(baseline, 1e-12)),
        "coordinate_dimension": q.shape[-1],
        "fit_sequences": count,
        "test_sequences": len(tokens) - count,
    }
