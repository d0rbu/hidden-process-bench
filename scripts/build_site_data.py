"""Reproduce public example generators and export the recorded website results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from hidden_process_bench.config import Shape
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
    return {
        "schema_version": 1,
        "results_date": "2026-10-26",
        "scope_correction": read("docs/results/disclosure-correction.json"),
        "teacher_qualification": read("docs/results/mixed-suite-development.json"),
        "operator_attempt": read("docs/results/operator-development/summary.json"),
        "agent_attempt": read("docs/results/agent-development/summary.json"),
        "method_result_revision": METHOD_RESULT_REVISION,
        "method_study": read("docs/results/five-method-pilot/summary.json"),
        "result_revision": RESULT_REVISION,
        "repository": REPOSITORY,
        "datasets": datasets,
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
