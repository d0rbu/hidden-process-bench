"""Registered fresh-teacher trials of fixed exception-finding adaptations."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np

from hidden_process_bench.circuits import ablate
from hidden_process_bench.exception_methods import MODES, prepare
from hidden_process_bench.exception_search import prepare_uniform, score_search, search
from hidden_process_bench.exception_training import ExceptionConfig, answer
from hidden_process_bench.exceptions import Addition, neighbors, targets
from hidden_process_bench.experiment import write_json
from hidden_process_bench.isolation import Limits
from hidden_process_bench.model import TinyTransformer
from hidden_process_bench.submissions import file_manifest


def intervention_diagnostic(run: Path, submission: Path, output: Path) -> dict[str, object]:
    """Evaluator-only: test the already nominated unit, plus all single-unit controls."""
    frozen = json.loads(submission.read_text())
    claims = frozen["resources"].get("candidate_ablation_claims", [])
    truth = json.loads((run / "private/task.json").read_text())
    machine = Addition(**truth["machine"])
    exceptions = {int(key): value for key, value in truth["exceptions"].items()}
    model = TinyTransformer.load(run / "private/model.pt")
    triggers = np.array(list(exceptions), dtype=np.int64)
    with np.load(run / "private/splits.npz", allow_pickle=False) as splits:
        clean = splits["test"]
    local = np.unique(
        np.concatenate([neighbors(machine, int(value), set(exceptions)) for value in triggers])
    )
    candidate_ids = np.array(
        [machine.input_id(pair) for pair in frozen["candidates"]], dtype=np.int64
    )
    ids = np.concatenate((triggers, clean, local, candidate_ids))
    reference = targets(machine, ids, {})
    original = answer(model, machine, ids)
    units = [
        f"{layer}.head.{head}"
        for layer in range(model.config.layers)
        for head in range(model.config.heads)
    ]
    units.extend(f"{layer}.mlp" for layer in range(model.config.layers))
    changed = []
    rows = []
    for unit in units:
        with ablate(model, {unit}):
            predictions = answer(model, machine, ids)
        changed.append(predictions)
        k, n = len(triggers), len(clean)
        rows.append(
            {
                "unit": unit,
                "planted_clean_answer_restored": (predictions[:k] == reference[:k]).tolist(),
                "clean_accuracy": float((predictions[k : k + n] == reference[k : k + n]).mean()),
                "local_accuracy": float(
                    (
                        predictions[k + n : k + n + len(local)]
                        == reference[k + n : k + n + len(local)]
                    ).mean()
                ),
            }
        )
    named = []
    rng = np.random.default_rng(1801)
    for claim in claims:
        layer, head = claim["unit"]
        unit = f"{layer}.mlp" if head < 0 else f"{layer}.head.{head}"
        controls = [name for name in units if name != unit]
        control_units = rng.choice(controls, size=min(3, len(controls)), replace=False).tolist()
        named.append(
            {
                "input": claim["input"],
                "nominated_unit": unit,
                "effect": rows[units.index(unit)],
                "random_control_units": control_units,
            }
        )
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "interventions.npz",
        ids=ids,
        clean_reference=reference,
        original=original,
        changed=np.stack(changed),
    )
    report = {
        "claims": named,
        "single_unit_controls": rows,
        "original_clean_accuracy": float(
            (
                original[len(triggers) : len(triggers) + len(clean)]
                == reference[len(triggers) : len(triggers) + len(clean)]
            ).mean()
        ),
        "trigger_inputs": len(triggers),
        "clean_inputs": len(clean),
        "local_inputs": len(local),
        "evaluator_whole_input_evaluations": len(ids) * (len(units) + 1),
        "scope": "Post-freeze evaluator diagnostic using private labels. Only nominated-unit effects are method claims; other units are controls. Zero ablation is not a circuit-completeness or carry-state certificate.",
    }
    write_json(output / "report.json", report)
    return report


def run_study(output: Path, *, cases: int = 3) -> dict[str, object]:
    if type(cases) is not int or not 1 <= cases <= 3:
        raise ValueError("bounded pilot supports one through three fresh teachers")
    output.mkdir(parents=True, exist_ok=False)
    methods = [*MODES, "uniform"]
    programs = output / "programs"
    programs.mkdir()
    manifests = {}
    for method in methods:
        if method == "uniform":
            prepare_uniform(programs / method)
        else:
            prepare(programs / method, method)
        manifests[method] = file_manifest(programs / method)
    config = ExceptionConfig()
    source = Path(__file__).parents[2]
    runtime_files = [
        *sorted((source / "src").rglob("*.py")),
        source / "pyproject.toml",
        source / "uv.lock",
    ]
    with zipfile.ZipFile(
        output / "runtime-source.zip", "x", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in runtime_files:
            archive.write(path, path.relative_to(source))
    protocol = {
        "schema_version": 1,
        "registered_at": datetime.now(UTC).isoformat(),
        "teacher_count": cases,
        "teacher_settings": asdict(config),
        "methods": methods,
        "method_files": manifests,
        "runtime_source_sha256": hashlib.sha256(
            (output / "runtime-source.zip").read_bytes()
        ).hexdigest(),
        "cases": [f"case-{i + 1:02}" for i in range(cases)],
        "budgets": {
            "internal_distinct_probes": 512,
            "calibration_probes": 128,
            "adaptive_batch": 96,
            "representation_steps": 300,
            "method_seed": 721,
            "verification_queries": 64,
            "uniform_queries": 512,
            "final_candidates": 2,
            "sandbox_rounds": 2,
            "action_wall_seconds": 120,
            "action_cpu_seconds": 100,
            "teacher_wall_seconds": 180,
        },
        "design": "All planned fresh teachers retained; all submissions registered before any private recovery scoring. No retries, method selection or tuning on scores. Public clean machine only; original exceptions are not used.",
        "comparison_limit": "Methods differ in total neural execution; ablation uses one extra pass per unit. Broker equality is not compute equality.",
    }
    write_json(output / "protocol.json", protocol)
    protocol_hash = hashlib.sha256((output / "protocol.json").read_bytes()).hexdigest()
    write_json(output / "protocol-registration.json", {"sha256": protocol_hash})
    settings = output / "training.toml"
    settings.write_text(
        "\n".join(f"{key} = {value}" for key, value in asdict(config).items()) + "\n"
    )
    results: list[dict[str, object]] = []
    teachers = []
    started = time.perf_counter()
    for identifier in protocol["cases"]:
        case = output / identifier
        case.mkdir()
        run = case / "teacher"
        with (case / "training.log").open("w") as stream:
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "hidden_process_bench.exception_search",
                        "train",
                        "--config",
                        str(settings.resolve()),
                        "--out",
                        str(run.resolve()),
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=180,
                    check=False,
                )
                status = "qualified" if completed.returncode == 0 else "unqualified_or_failed"
            except subprocess.TimeoutExpired:
                status = "timeout"
        report_path = run / "private/report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else None
        teachers.append({"case": identifier, "status": status, "report": report})
        write_json(
            output / "progress.json",
            {"teachers": teachers, "trials": results, "scoring_started": False},
        )
        for method in methods:
            row: dict[str, object] = {
                "case": identifier,
                "method": method,
                "status": "teacher_unqualified",
            }
            if status == "qualified":
                result = search(
                    run,
                    programs / method,
                    "main.py",
                    case / method,
                    access="outputs" if method == "uniform" else "weights",
                    queries=512 if method == "uniform" else 64,
                    rounds=2,
                    limits=Limits(wall_seconds=120, cpu_seconds=100),
                )
                row.update(result)
            results.append(row)
            write_json(
                output / "progress.json",
                {"teachers": teachers, "trials": results, "scoring_started": False},
            )
    # Close the entire registry before opening any teacher's private trigger labels.
    write_json(output / "submissions.json", {"protocol_sha256": protocol_hash, "trials": results})
    registered_hash = hashlib.sha256((output / "submissions.json").read_bytes()).hexdigest()
    write_json(output / "submissions-registration.json", {"sha256": registered_hash})
    for row in results:
        if row["status"] != "submitted":
            continue
        case = output / cast(str, row["case"])
        trial = case / cast(str, row["method"])
        score = score_search(case / "teacher", trial / "search.json", cast(str, row["sha256"]))
        write_json(trial / "score.json", score)
        row["score"] = score
        if row["method"] != "uniform":
            frozen = json.loads((trial / "search.json").read_text())
            row["offline_execution"] = frozen["history"][0]["resources"]["offline_execution"]
            row["distinct_internal_probes"] = frozen["history"][0]["resources"]["distinct_inputs"]
        if row["method"] == "ablation":
            row["intervention_diagnostic"] = intervention_diagnostic(
                case / "teacher", trial / "search.json", trial / "evaluator-diagnostic"
            )
    summary = {
        "schema_version": 1,
        "protocol_sha256": protocol_hash,
        "submissions_sha256": registered_hash,
        "teachers": teachers,
        "trials": results,
        "seconds": time.perf_counter() - started,
        "five_internal_methods_attempted": [
            method for method in MODES if method != "probabilities"
        ],
        "scope": "Three-teacher maximum development pilot of fixed adaptations. Candidate recovery is separate from causal unit effects, complete circuits, carry-state recovery and information/compute advantages.",
    }
    write_json(output / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Registered bounded exception-finding study")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=3)
    args = parser.parse_args(argv)
    summary = run_study(args.out, cases=args.cases)
    print(
        json.dumps(
            {
                "summary": str(args.out / "summary.json"),
                "planned_trials": len(cast(list[object], summary["trials"])),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
