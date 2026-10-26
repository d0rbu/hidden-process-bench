"""Bounded comparison of classical, representation and causal recovery adaptations.

Run with ``python -m hidden_process_bench.method_study --run RUN --out NEW``.
Only the outer evaluator reads private artifacts, after candidate files are fixed.
"""

import argparse
import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from hidden_process_bench.circuits import discover_circuit, interchange_test
from hidden_process_bench.classical import fit_em, fit_spectral
from hidden_process_bench.config import Shape, check_count
from hidden_process_bench.experiment import provenance, score, write_json
from hidden_process_bench.model import TinyTransformer
from hidden_process_bench.process import Floats, MixtureHMM
from hidden_process_bench.recovery import RecoveryResult, fit_hmm
from hidden_process_bench.representations import belief_dynamics, observe_traces, representation


@dataclass(frozen=True)
class StudyConfig:
    sequences: int = 128
    steps: int = 250
    restarts: int = 2
    representation_steps: int = 400
    observation_seed: int = 27
    recovery_seed: int = 37
    evaluation_seed: int = 917563
    evaluation_sequences: int = 512

    def __post_init__(self) -> None:
        for value in (
            self.sequences,
            self.steps,
            self.restarts,
            self.representation_steps,
            self.evaluation_sequences,
        ):
            check_count(value)
        if self.sequences < 4:
            raise ValueError("study needs at least four sequences for fit/test splits")


def extract_methods(
    checkpoint: Path, task: Path, output: Path, config: StudyConfig
) -> dict[str, dict[str, object]]:
    """Public-only boundary: no run directory, truth, seed, or scorer is accepted."""
    output.mkdir(parents=True, exist_ok=False)
    payload = json.loads(task.read_text())
    if payload["schema_version"] != 1:
        raise ValueError("unsupported public task schema")
    shape = Shape(**payload["shape"])
    model = TinyTransformer.load(checkpoint)
    if shape.vocab != model.config.vocab:
        raise ValueError("task vocabulary disagrees with checkpoint")
    traces = observe_traces(model, config.sequences, model.config.context, config.observation_seed)
    observations = traces.observations
    np.save(output / "tokens.npy", observations.tokens)
    records: dict[str, dict[str, object]] = {}
    budget = {
        "forward_calls": observations.forward_calls,
        "token_queries": observations.token_queries,
        "prefix_tokens_processed": observations.prefix_tokens_processed,
    }

    def submit(
        name: str,
        access: str,
        fit: Callable[[], RecoveryResult],
        diagnostics: dict[str, object] | None = None,
    ) -> MixtureHMM:
        started = perf_counter()
        result = fit()
        path = output / f"{name}.json"
        result.hmm.save(path)
        records[name] = {
            "access": access,
            "objective": result.objective,
            "objective_name": result.objective_name,
            "restart_objectives": result.restart_objectives,
            "fitting_seconds": result.seconds,
            "submission_seconds": perf_counter() - started,
            "fit_parameters": result.fit_parameters,
            "optimizer_steps_per_restart": config.steps,
            "restarts": config.restarts,
            "observation_budget": budget,
            "candidate_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "diagnostics": diagnostics or {},
        }
        write_json(output / "submissions.json", records)
        return result.hmm

    def distill(
        probabilities: Floats | None = None, features: Floats | None = None
    ) -> RecoveryResult:
        return fit_hmm(
            shape,
            observations.tokens,
            steps=config.steps,
            restarts=config.restarts,
            seed=config.recovery_seed,
            probabilities=probabilities,
            activations=features,
        )

    submit("adam_samples", "samples", lambda: distill())
    submit("adam_probabilities", "probabilities", lambda: distill(observations.probabilities))
    submit(
        "em_samples",
        "samples",
        lambda: fit_em(
            shape,
            observations.tokens,
            steps=config.steps,
            restarts=config.restarts,
            seed=config.recovery_seed,
        ),
    )
    started = perf_counter()
    spectral = fit_spectral(observations.tokens, shape.vocab, shape.components * shape.states)
    spectral_seconds = perf_counter() - started
    np.savez(
        output / "spectral_operators.npz",
        initial=spectral.initial,
        terminal=spectral.terminal,
        operators=spectral.operators,
        singular_values=spectral.singular_values,
    )
    submit(
        "spectral_hmm",
        "samples",
        lambda: distill(spectral.probabilities),
        {
            "repaired_prediction_fraction": spectral.repaired_fraction,
            "moment_seconds": spectral_seconds,
            "adapter": "clip/normalize spectral targets then HMM distillation",
            "retained_singular_value": float(
                spectral.singular_values[shape.components * shape.states - 1]
            ),
        },
    )
    dimension = max(1, shape.components * shape.states - 1)
    geometry_inputs = [*traces.residuals, np.concatenate(traces.residuals, axis=-1)]
    for layer, hidden in enumerate(geometry_inputs):
        name = f"belief_layer_{layer}" if layer < len(traces.residuals) else "belief_all_layers"
        started = perf_counter()
        rep = representation(
            hidden,
            kind="pca",
            dimension=dimension,
            steps=config.representation_steps,
            seed=config.recovery_seed,
        )
        operators, diagnostics = belief_dynamics(
            rep.features, observations.tokens, observations.probabilities
        )
        np.savez(output / f"{name}_operators.npz", operators=operators)
        diagnostics = {
            **rep.diagnostics,
            **diagnostics,
            "representation_seconds": perf_counter() - started,
            "adapter": "probability distillation + belief-to-coordinate reconstruction",
        }
        submit(
            name,
            "probabilities+residuals",
            lambda features=rep.features: distill(observations.probabilities, features),
            dict(diagnostics),
        )
    for kind in ("pca", "sae", "transcoder"):
        started = perf_counter()
        rep = representation(
            traces.mlp_inputs if kind == "transcoder" else observations.activations,
            kind=kind,
            dimension=dimension if kind == "pca" else 2 * shape.components * shape.states,
            steps=config.representation_steps,
            seed=config.recovery_seed,
            targets=traces.mlp_outputs if kind == "transcoder" else None,
        )
        submit(
            kind,
            "probabilities+activations",
            lambda features=rep.features: distill(observations.probabilities, features),
            {
                **rep.diagnostics,
                "representation_seconds": perf_counter() - started,
                "adapter": "shared HMM distillation + feature reconstruction; coefficient 0.1",
            },
        )
    started = perf_counter()
    circuit = discover_circuit(model, observations.tokens)
    circuit_diagnostics = {**circuit.diagnostics, "discovery_seconds": perf_counter() - started}
    candidate = submit(
        "circuit",
        "probabilities+weights+interventions",
        lambda: distill(circuit.probabilities),
        circuit_diagnostics,
    )
    circuit_diagnostics["interchange"] = interchange_test(
        model, candidate, observations.tokens, traces.residuals[0]
    )
    write_json(output / "submissions.json", records)
    return records


def run_study(runs: list[Path], output: Path, config: StudyConfig) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    write_json(
        output / "plan.json",
        {"config": asdict(config), "runs": [str(x) for x in runs], "provenance": provenance()},
    )
    cases = []
    for index, run in enumerate(runs):
        folder = output / f"case-{index:03d}"
        try:
            submissions = extract_methods(
                run / "public/checkpoint.pt", run / "public/task.json", folder, config
            )
            # All extraction is complete and hashed BEFORE opening any evaluator artifacts.
            truth = MixtureHMM.load(run / "private/truth.json")
            qualification = json.loads((run / "report.json").read_text())["benchmark_eligible"]
            scores = {
                name: score(
                    truth,
                    MixtureHMM.load(folder / f"{name}.json"),
                    config.evaluation_sequences,
                    model_length(run),
                    config.evaluation_seed,
                )
                for name in submissions
            }
            result = {
                "run": str(run),
                "benchmark_eligible": qualification,
                "checkpoint_sha256": hashlib.sha256(
                    (run / "public/checkpoint.pt").read_bytes()
                ).hexdigest(),
                "submissions": submissions,
                "scores": scores,
            }
            write_json(folder / "evaluation.json", result)
            cases.append(result)
            write_json(output / "report.json", {"config": asdict(config), "cases": cases})
            print(
                f"{folder}: {len(scores)} fixed candidates, qualified={qualification}", flush=True
            )
        except Exception as error:
            write_json(output / "failure.json", {"run": str(run), "error": repr(error)})
            raise
    return {"config": asdict(config), "cases": cases}


def model_length(run: Path) -> int:
    return TinyTransformer.load(run / "public/checkpoint.pt").config.context


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--representation-steps", type=int, default=400)
    parser.add_argument("--restarts", type=int, default=2)
    args = parser.parse_args()
    run_study(
        args.run,
        args.out,
        StudyConfig(
            steps=args.steps, restarts=args.restarts, representation_steps=args.representation_steps
        ),
    )


if __name__ == "__main__":
    main()
