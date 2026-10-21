"""Bounded CPU experiment with independent training, selection, test and probe streams."""

import hashlib
import json
import platform
import subprocess
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import TypedDict

import numpy as np
import torch
from torch.nn import functional as F

from hidden_process_bench.config import RunConfig, Shape
from hidden_process_bench.equivalence import compare_processes, process_complexity
from hidden_process_bench.metrics import (
    Optimality,
    identifiability_diagnostics,
    linear_probe,
    optimality,
    parameter_recovery,
    sequence_kl,
    summarize,
)
from hidden_process_bench.model import TinyTransformer, observe, teacher_inputs
from hidden_process_bench.process import MixtureHMM, Tokens
from hidden_process_bench.recovery import fit_hmm
from hidden_process_bench.sampling import sample_task


class TrainRecord(TypedDict):
    step: int
    training_cross_entropy: float
    validation: Optimality


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


@torch.no_grad()
def predictions(model: TinyTransformer, tokens: Tokens) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, hidden = model(teacher_inputs(torch.tensor(tokens), model.config.vocab))
    return logits.softmax(dim=-1).numpy().astype(np.float64), hidden.numpy().astype(np.float64)


def train(
    config: RunConfig, truth: MixtureHMM, trace_path: Path
) -> tuple[TinyTransformer, list[TrainRecord]]:
    torch.manual_seed(config.model_seed)
    model = TinyTransformer(config.model())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    rng = np.random.default_rng(config.seed + 10)
    validation, components = truth.sample(
        np.random.default_rng(config.seed + 11), config.eval_sequences, config.length
    )
    oracle = truth.filter(validation).predictions[:, :-1]
    trace: list[TrainRecord] = []
    for step in range(1, config.train_steps + 1):
        model.train()
        tokens, _ = truth.sample(rng, config.batch_size, config.length)
        targets = torch.tensor(tokens)
        logits, _ = model(teacher_inputs(targets, config.vocab))
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % config.eval_every == 0 or step == config.train_steps:
            predicted, _ = predictions(model, validation)
            metrics = optimality(oracle, predicted, components, config.components, config.max_kl)
            row: TrainRecord = {
                "step": step,
                "training_cross_entropy": float(loss.detach()),
                "validation": metrics,
            }
            trace.append(row)
            with trace_path.open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(
                f"step={step} validation_kl={metrics['overall']['mean_kl']:.6f} qualified={metrics['qualified']}",
                flush=True,
            )
            if metrics["qualified"]:
                break
    return model, trace


def provenance() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    paths = [*sorted((root / "src").rglob("*.py")), root / "pyproject.toml", root / "uv.lock"]
    hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=False
    )
    return {
        "git_revision": head.stdout.strip() if head.returncode == 0 else None,
        "working_tree_dirty": bool(status.stdout) if status.returncode == 0 else None,
        "source_sha256": hashes,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
    }


def score(
    truth: MixtureHMM, candidate: MixtureHMM, count: int, length: int, seed: int
) -> dict[str, object]:
    tokens, _ = truth.sample(np.random.default_rng(seed), count, length)
    return {
        "parameters": parameter_recovery(truth, candidate)
        if truth.shape == candidate.shape
        else None,
        "equivalence": asdict(compare_processes(truth, candidate)),
        "complexity": {
            "reference": process_complexity(truth),
            "candidate": process_complexity(candidate),
        },
        "predictive": summarize(
            sequence_kl(
                truth.filter(tokens).predictions[:, :-1],
                candidate.filter(tokens).predictions[:, :-1],
            )
        ),
    }


def extract_checkpoint(
    checkpoint: Path,
    task: Path,
    output: Path,
    method: str,
    count: int,
    steps: int,
    restarts: int,
    seed: int,
    activation_weight: float = 0.1,
) -> dict[str, object]:
    """Standalone extractor: its only inputs are public challenge artifacts."""
    payload = json.loads(task.read_text())
    if payload["schema_version"] != 1:
        raise ValueError("unsupported task schema version")
    shape = Shape(**payload["shape"])
    model = TinyTransformer.load(checkpoint)
    if shape.vocab != model.config.vocab:
        raise ValueError("task and checkpoint vocabularies disagree")
    if method not in ("samples", "probabilities", "activations"):
        raise ValueError("unknown extraction method")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    observations = observe(model, count, model.config.context, seed)
    result = fit_hmm(
        shape,
        observations.tokens,
        steps=steps,
        restarts=restarts,
        seed=seed + 1,
        probabilities=observations.probabilities if method != "samples" else None,
        activations=observations.activations if method == "activations" else None,
        activation_weight=activation_weight,
    )
    result.hmm.save(output)
    return {
        "method": method,
        "objective": result.objective,
        "objective_name": result.objective_name,
        "token_queries": observations.token_queries,
        "forward_calls": observations.forward_calls,
        "prefix_tokens_processed": observations.prefix_tokens_processed,
        "seconds": result.seconds,
    }


def run(config: RunConfig, output: Path) -> dict[str, object]:
    started = perf_counter()
    torch.set_num_threads(config.threads)
    output.mkdir(parents=True, exist_ok=False)
    public, private, recovered = output / "public", output / "private", output / "recovered"
    for path in (public, private, recovered):
        path.mkdir()
    write_json(private / "config.json", asdict(config))
    write_json(private / "provenance.json", provenance())
    truth, difficulty, sampling_attempts = sample_task(
        config.shape(), config.seed, config.sampling, private / "sampling.jsonl"
    )
    truth.save(private / "truth.json")
    write_json(public / "task.json", {"schema_version": 1, "shape": asdict(config.shape())})
    training_started = perf_counter()
    model, trace = train(config, truth, private / "training.jsonl")
    training_seconds = perf_counter() - training_started
    model.save(public / "checkpoint.pt")
    test, components = truth.sample(
        np.random.default_rng(config.seed + 12), config.eval_sequences, config.length
    )
    oracle = truth.filter(test).predictions[:, :-1]
    predicted, _ = predictions(model, test)
    test_metrics = optimality(oracle, predicted, components, config.components, config.max_kl)
    eligible = bool(trace[-1]["validation"]["qualified"] and test_metrics["qualified"])
    observations = observe(
        model, config.extraction_sequences, config.length, config.observation_seed
    )
    for name in ("tokens", "probabilities", "activations"):
        np.save(public / f"{name}.npy", getattr(observations, name))
    results = {}
    for method in ("samples", "probabilities", "activations"):
        print(
            f"extracting method={method} steps={config.extraction_steps} restarts={config.restarts}",
            flush=True,
        )
        result = fit_hmm(
            config.shape(),
            observations.tokens,
            steps=config.extraction_steps,
            restarts=config.restarts,
            seed=config.recovery_seed,
            probabilities=observations.probabilities if method != "samples" else None,
            activations=observations.activations if method == "activations" else None,
            activation_weight=config.activation_weight,
        )
        result.hmm.save(recovered / f"{method}.json")
        results[method] = {
            "parameters": parameter_recovery(truth, result.hmm),
            "equivalence": asdict(compare_processes(truth, result.hmm)),
            "predictive": summarize(
                sequence_kl(oracle, result.hmm.filter(test).predictions[:, :-1])
            ),
            "fit_objective": result.objective,
            "objective_name": result.objective_name,
            "restart_objectives": result.restart_objectives,
            "fit_parameters": result.fit_parameters,
            "seconds": result.seconds,
            "access": {
                "samples": True,
                "probabilities": method != "samples",
                "activations": method == "activations",
                "ground_truth": False,
            },
        }
    probe_sets = [
        truth.sample(
            np.random.default_rng(config.seed + offset), config.eval_sequences, config.length
        )[0]
        for offset in (40, 41)
    ]
    probe_outputs = [predictions(model, tokens) for tokens in probe_sets]
    beliefs = [
        truth.filter(tokens).beliefs[:, :-1].reshape(-1, config.components * config.states)
        for tokens in probe_sets
    ]
    probes = {
        label: linear_probe(
            probe_outputs[0][index].reshape(-1, probe_outputs[0][index].shape[-1]),
            beliefs[0],
            probe_outputs[1][index].reshape(-1, probe_outputs[1][index].shape[-1]),
            beliefs[1],
        )
        for index, label in enumerate(("probabilities", "activations"))
    }
    report = {
        "schema_version": 1,
        "benchmark_eligible": eligible,
        "interpretation": "qualified_checkpoint"
        if eligible
        else "diagnostic_only_unqualified_checkpoint",
        "identifiability_diagnostics": identifiability_diagnostics(truth),
        "process_complexity": process_complexity(truth),
        "sampling": {"attempts": sampling_attempts, "difficulty": asdict(difficulty)},
        "training": {
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "steps": trace[-1]["step"],
            "tokens": trace[-1]["step"] * config.batch_size * config.length,
            "seconds": training_seconds,
            "last_validation": trace[-1]["validation"],
        },
        "heldout_test": test_metrics,
        "shared_extraction_budget": {
            "sequences": config.extraction_sequences,
            "length": config.length,
            "token_queries_per_method": observations.token_queries,
            "forward_calls_per_method": observations.forward_calls,
            "prefix_tokens_processed_per_method": observations.prefix_tokens_processed,
            "optimizer_steps_per_method": config.extraction_steps * config.restarts,
            "restarts": config.restarts,
        },
        "recovery": results,
        "supervised_belief_probes_not_extraction": probes,
        "checkpoint_sha256": hashlib.sha256((public / "checkpoint.pt").read_bytes()).hexdigest(),
        "total_seconds": perf_counter() - started,
    }
    write_json(output / "report.json", report)
    return report
