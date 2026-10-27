"""Reproduce public example generators and export the recorded website results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from hidden_process_bench.config import Shape
from hidden_process_bench.exceptions import Addition
from hidden_process_bench.generators import LagProcess, StackProcess, load_process
from hidden_process_bench.process import random_mixture

RESULT_REVISION = "c10508b5b07b770b6d9cc697f8eed80fcbbea0f1"
METHOD_RESULT_REVISION = "2d1505aa9beb9aab6e203e5eda9000c7c5ec9265"
REPOSITORY = "https://github.com/d0rbu/hidden-process-bench"


def build_payload(root: Path) -> dict[str, Any]:
    """Use only already-published receipts; no new training or task selection."""
    inputs: dict[str, str] = {}

    def read_bytes(path: str) -> bytes:
        raw = (root / path).read_bytes()
        inputs[path] = hashlib.sha256(raw).hexdigest()
        return raw

    def read(path: str) -> dict[str, Any]:
        return json.loads(read_bytes(path))

    def link(path: str) -> str:
        return f"{REPOSITORY}/blob/{RESULT_REVISION}/{path}"

    demo = "docs/results/screened-cpu-demo"
    pilot = "docs/results/scaling-pilot"
    config = read(f"{demo}/config.json")
    plan = read(f"{pilot}/plan.json")
    specifications = [
        ("demo", "Demo · 3 states", demo, config, True),
        ("pilot-small", "Pilot · 3 states", f"{pilot}/case-000", plan["cases"][0], False),
        ("pilot-large", "Pilot · 6 states", f"{pilot}/case-002", plan["cases"][2], False),
    ]
    datasets = []
    for identifier, label, prefix, settings, directory in specifications:
        separator = "/" if directory else "-"
        audit = [
            json.loads(line)
            for line in read_bytes(f"{prefix}{separator}sampling.jsonl").splitlines()
        ]
        provenance = read(f"{prefix}{separator}provenance.json")
        report = read(f"{prefix}{separator}report.json")
        for source in ("src/hidden_process_bench/process.py",):
            digest = hashlib.sha256((root / source).read_bytes()).hexdigest()
            if digest != provenance["source_sha256"][source]:
                raise ValueError(f"generator source no longer matches the recorded run: {source}")
        if not audit[-1]["accepted"] or any(row["accepted"] for row in audit[:-1]):
            raise ValueError("expected the first accepted candidate at the end of the audit")
        shape = Shape(settings["components"], settings["states"], settings["vocab"])
        sampling = settings["sampling"]
        hmm = random_mixture(
            shape,
            audit[-1]["candidate_seed"],
            extra_edge_probability=sampling["extra_edge_probability"],
            emission_concentration=sampling["emission_concentration"],
        )
        # Cross-language fixtures exercise empty, mixed, and repeated-token histories.
        histories = [[], [0, 1, 2, 1, 0, shape.vocab - 1], [shape.vocab - 1] * 24]
        checks = []
        for history in histories:
            filtered = hmm.filter(np.asarray([history], dtype=np.int64))
            checks.append(
                {
                    "tokens": history,
                    "predictions": filtered.predictions[0].tolist(),
                    "beliefs": filtered.beliefs[0].tolist(),
                }
            )
        datasets.append(
            {
                "id": identifier,
                "label": label,
                "hmm": {
                    name: getattr(hmm, name).tolist()
                    for name in ("weights", "initial", "transition", "emission")
                },
                "complexity": report["process_complexity"],
                "attempts": report["sampling"]["attempts"],
                "source_revision": provenance["git_revision"],
                "report_url": link(f"{prefix}{separator}report.json"),
                "oracle_checks": checks,
            }
        )

    runs = []
    descriptors = [("demo", f"{demo}/report.json", config)] + [
        (f"case-{i:03}", f"{pilot}/case-{i:03}-report.json", settings)
        for i, settings in enumerate(plan["cases"])
    ]
    for identifier, path, settings in descriptors:
        report = read(path)
        bounds = [
            group["upper95_kl"]
            for split in (report["training"]["last_validation"], report["heldout_test"])
            for group in (split["overall"], *split["by_component"])
        ]
        runs.append(
            {
                "id": identifier,
                "states": settings["states"],
                "width": settings["width"],
                "qualified": report["benchmark_eligible"],
                "threshold": settings["max_kl"],
                "gate_upper": max(bounds),
                "test_kl": report["heldout_test"]["overall"]["mean_kl"],
                "training_steps": report["training"]["steps"],
                "model_parameters": report["training"]["model_parameters"],
                "observation_budget": report["shared_extraction_budget"],
                "recovery": {
                    name: {
                        "mae": values["parameters"]["mean_parameter_mae"],
                        "kl": values["predictive"]["mean_kl"],
                    }
                    for name, values in report["recovery"].items()
                },
                "report_url": link(path),
            }
        )
    paired = read("docs/results/agent-development-v2/summary.json")
    final = read("docs/results/final-campaign-v2/summary.json")
    # Retain displayed scores and assessments, while the full linked receipt owns
    # trajectories, method-selection tables and resource accounting.
    final["cases"] = [
        {
            key: row[key]
            for key in (
                "case",
                "method",
                "access",
                "family_after_registration",
                "status",
                "heldout",
                "long",
                "structural_assessment",
            )
        }
        for row in final["cases"]
    ]
    rule_source = hashlib.sha256(
        (root / "src/hidden_process_bench/generators.py").read_bytes()
    ).hexdigest()
    if rule_source != paired["runtime_source_sha256"]["generators.py"]:
        raise ValueError("rule source no longer matches the recorded development cohort")
    rule_examples = []
    examples = [
        ("delayed", "Delayed-token rule", "case-01"),
        ("stack", "Stack rule", "case-05"),
    ]
    for identifier, label, case in examples:
        path = f"docs/results/agent-development-v2/{case}/source-process.json"
        specification = read(path)
        process = load_process(root / path)
        if isinstance(process, LagProcess):
            tokens = process.sample(np.random.default_rng(101), 1, 24)[0][0].tolist()
        else:
            assert isinstance(process, StackProcess)
            roles = [4, 0, 1, 2, 5, 3, 2, 2, 1, 0, 1, 5, 3, 2, 3, 0, 0, 2, 5, 2, 4, 1, 3, 4]
            tokens = [process.alphabet[role] for role in roles]
        frames = []
        stack: list[int] = []
        inverse = {token: role for role, token in enumerate(process.alphabet)}
        for position in range(len(tokens) + 1):
            if isinstance(process, LagProcess):
                memory = tokens[max(0, position - process.lag) : position]
                target = (
                    process.alphabet[tokens[position - process.lag]]
                    if position >= process.lag
                    else None
                )
                frame = {
                    "memory": memory,
                    "target": target,
                    "read_index": 0 if target is not None else None,
                }
            else:
                if position:
                    role = inverse[tokens[position - 1]]
                    if role < 2:
                        stack.append(role)
                    elif role < 4 and stack and stack[-1] == role - 2:
                        stack.pop()
                frame = {
                    "memory": [process.alphabet[role] for role in stack],
                    "target": process.alphabet[2 + stack[-1]] if stack else None,
                    "read_index": len(stack) - 1 if stack else None,
                }
            frames.append(frame)
        rule_examples.append(
            {
                "id": identifier,
                "label": label,
                "specification": specification,
                "tokens": tokens,
                "frames": frames,
                "predictions": process.probabilities(np.asarray([tokens], dtype=np.int64))[
                    0
                ].tolist(),
                "source_url": f"{REPOSITORY}/blob/main/{path}",
            }
        )
    exception_root = "docs/results/exceptions-pilot-v1"
    exception_summary = read(f"{exception_root}/summary.json")
    exception_task = read(f"{exception_root}/local-clean/task.json")
    machine = Addition(**exception_task["machine"])
    with np.load(root / exception_root / "local-clean/census.npz", allow_pickle=False) as census:
        teacher_answers = census["predictions"].tolist()
    read_bytes(f"{exception_root}/local-clean/census.npz")
    examples = [{"label": "Ordinary addition", "pair": [13, 7]}]
    examples.extend(
        {"label": f"Planted exception {i + 1}", "pair": list(machine.pair(int(value)))}
        for i, value in enumerate(exception_task["exceptions"])
    )
    exception_summary["example"] = {
        "base": machine.base,
        "digits": machine.digits,
        "examples": examples,
        "planted_pairs": [row["pair"] for row in examples[1:]],
        "teacher_answers": teacher_answers,
        "oracle_checks": [
            {"pair": row["pair"], "trace": machine.trace(*cast(list[int], row["pair"]))}
            for row in examples
        ],
    }
    return {
        "schema_version": 1,
        "exceptions": exception_summary,
        "results_date": "2026-10-27",
        "scope_correction": read("docs/results/disclosure-correction.json"),
        "teacher_qualification": read("docs/results/mixed-suite-development.json"),
        "operator_attempt": read("docs/results/operator-development/summary.json"),
        "agent_attempt": read("docs/results/agent-development/summary.json"),
        "paired_agent_attempt": paired,
        "final_attempt": final,
        "discovery_attempts": read("docs/results/discovery-development/summary.json"),
        "method_result_revision": METHOD_RESULT_REVISION,
        "method_study": read("docs/results/five-method-pilot/summary.json"),
        "result_revision": RESULT_REVISION,
        "repository": REPOSITORY,
        "datasets": datasets,
        "rule_examples": rule_examples,
        "runs": runs,
        "input_sha256": inputs,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "site/data.json"
    path.write_text(json.dumps(build_payload(root), indent=2, allow_nan=False) + "\n")
    print(f"Wrote {path.relative_to(root)} from published experiment receipts.")


if __name__ == "__main__":
    main()
