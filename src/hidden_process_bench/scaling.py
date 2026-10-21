"""Reproducible Cartesian sweeps with subprocess time limits and retained failures."""

import csv
import json
import subprocess
import sys
import tomllib
from collections import Counter
from contextlib import suppress
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import cast

from hidden_process_bench.config import RunConfig, check_count, check_positive
from hidden_process_bench.experiment import write_json

AXES = {
    "seed",
    "model_seed",
    "states",
    "components",
    "vocab",
    "width",
    "layers",
    "extraction_sequences",
    "train_steps",
}


def load_sweep(path: Path, max_runs: int = 32) -> list[RunConfig]:
    check_count(max_runs)
    with path.open("rb") as stream:
        specification = tomllib.load(stream)
    if set(specification) != {"base", "grid"}:
        raise ValueError("a sweep needs exactly base and grid")
    grid = specification["grid"]
    if not isinstance(grid, dict) or not grid or set(grid) - AXES:
        raise ValueError("sweep grid contains missing or unsupported axes")
    if any(not isinstance(values, list) or not values for values in grid.values()):
        raise ValueError("each sweep axis must be a nonempty list")
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for values in grid.values()
        for value in values
    ):
        raise ValueError("sweep axes require integer values, excluding booleans")
    grid = cast(dict[str, list[int]], grid)
    count = 1
    for values in grid.values():
        count *= len(values)
    if count > max_runs:
        raise ValueError(f"sweep has {count} runs, exceeding the {max_runs} run limit")
    base = RunConfig.load(path.parent / specification["base"])
    return [
        replace(base, **dict(zip(grid, values, strict=True))) for values in product(*grid.values())
    ]


def sweep_plan(cases: list[RunConfig], timeout: float) -> dict[str, object]:
    check_count(len(cases))
    check_positive(timeout)
    return {
        "runs": len(cases),
        "maximum_training_tokens": sum(
            case.train_steps * case.batch_size * case.length for case in cases
        ),
        "timeout_seconds_per_run": timeout,
        "maximum_subprocess_seconds": len(cases) * timeout,
        "cases": [asdict(case) for case in cases],
    }


def write_run_config(path: Path, config: RunConfig) -> None:
    values = asdict(config)
    sampling = values.pop("sampling")
    lines = [f"{key} = {json.dumps(value)}" for key, value in values.items()]
    lines.extend(["", "[sampling]"])
    lines.extend(f"{key} = {json.dumps(value)}" for key, value in sampling.items())
    path.write_text("\n".join(lines) + "\n")


def run_sweep(cases: list[RunConfig], output: Path, timeout: float = 300) -> dict[str, object]:
    plan = sweep_plan(cases, timeout)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    private = output / "private"
    private.mkdir()
    write_json(private / "plan.json", plan)
    rows: list[dict[str, object]] = []
    with (output / "cases.jsonl").open("w") as receipt:
        for index, config in enumerate(cases):
            name = f"case-{index:03d}"
            config_path, run_path = private / f"{name}.toml", output / name
            write_run_config(config_path, config)
            print(
                f"sweep {index + 1}/{len(cases)} states={config.states} width={config.width}",
                flush=True,
            )
            started = perf_counter()
            with (private / f"{name}.log").open("w") as log:
                try:
                    completed = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "hidden_process_bench",
                            "run",
                            "--config",
                            str(config_path),
                            "--out",
                            str(run_path),
                            "--require-qualified",
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=timeout,
                        check=False,
                    )
                    returncode = completed.returncode
                except subprocess.TimeoutExpired:
                    returncode = None
            report = {}
            report_path = run_path / "report.json"
            if report_path.exists():
                # A timed-out write is retained as an incomplete artifact.
                with suppress(json.JSONDecodeError):
                    report = json.loads(report_path.read_text())
            eligible = report.get("benchmark_eligible")
            status = "failed"
            if returncode is None:
                status = "timeout"
            elif returncode == 0 and eligible is True:
                status = "qualified"
            elif returncode == 2 and eligible is False:
                status = "unqualified"
            row = {
                "case": name,
                "status": status,
                "returncode": returncode,
                "seed": config.seed,
                "model_seed": config.model_seed,
                "states": config.states,
                "components": config.components,
                "vocab": config.vocab,
                "width": config.width,
                "layers": config.layers,
                "extraction_sequences": config.extraction_sequences,
                "training_step_cap": config.train_steps,
                "training_steps": report.get("training", {}).get("steps"),
                "model_parameters": report.get("training", {}).get("model_parameters"),
                "heldout_kl": report.get("heldout_test", {}).get("overall", {}).get("mean_kl"),
                "seconds": perf_counter() - started,
            }
            for key in ("short_hankel_rank", "effective_joint_states", "transition_edges"):
                row[key] = report.get("process_complexity", {}).get(key)
            for method in ("samples", "probabilities", "activations"):
                recovery = report.get("recovery", {}).get(method, {})
                row[f"{method}_mae"] = recovery.get("parameters", {}).get("mean_parameter_mae")
                row[f"{method}_kl"] = recovery.get("predictive", {}).get("mean_kl")
            rows.append(row)
            receipt.write(json.dumps(row, allow_nan=False) + "\n")
            receipt.flush()
            print(f"{name}: {status}", flush=True)
    with (output / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": 1,
        "counts": dict(Counter(str(row["status"]) for row in rows)),
        "cases": rows,
        "interpretation": "Compare recovery on qualified checkpoints; retain all failure rates. No fitted scaling law.",
    }
    write_json(output / "summary.json", summary)
    return summary
