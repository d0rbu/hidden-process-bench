"""Family-neutral challenge and executable-submission commands.

Run with python -m hidden_process_bench.discovery_cli. This interface never converts
method features into an evaluator-selected candidate process family.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hidden_process_bench.challenge import export_challenge
from hidden_process_bench.config import check_count
from hidden_process_bench.discovery import extract_process, score_process
from hidden_process_bench.isolation import Limits
from hidden_process_bench.model import TinyTransformer, observe
from hidden_process_bench.operator_method import prepare_bundle
from hidden_process_bench.research_agent import AgentBudget, run_agent
from hidden_process_bench.submissions import FrozenSubmission, freeze_submission


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Infer and submit an unknown sequence mechanism.")
    commands = parser.add_subparsers(dest="command", required=True)
    agent = commands.add_parser("agent", help="run a fresh-context process-discovery agent")
    agent.add_argument("--challenge", type=Path, required=True)
    agent.add_argument("--client", type=Path, required=True)
    agent.add_argument("--catalog", type=Path, required=True)
    agent.add_argument("--steps", type=int, default=20)
    agent.add_argument("--seconds", type=float, default=1800)
    agent.add_argument("--model-seconds", type=float, default=180)
    agent.add_argument("--out", type=Path, required=True)
    operators = commands.add_parser(
        "prepare-operators", help="package the unknown-dimension observable-operator method"
    )
    operators.add_argument("--out", type=Path, required=True)
    export = commands.add_parser("export", help="export only declared neural information")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--access", choices=("outputs", "weights", "internals"), required=True)
    export.add_argument("--sequences", type=int, default=128)
    export.add_argument("--length", type=int, default=24)
    export.add_argument("--seed", type=int, default=101)
    export.add_argument("--threads", type=int, default=2)
    export.add_argument("--out", type=Path, required=True)
    extraction = commands.add_parser("extract", help="run a method against only a public snapshot")
    extraction.add_argument("--program", type=Path, required=True)
    extraction.add_argument("--entry", required=True)
    extraction.add_argument("--challenge", type=Path, required=True)
    extraction.add_argument("--out", type=Path, required=True)
    freeze = commands.add_parser("freeze", help="register a self-contained executable submission")
    freeze.add_argument("--source", type=Path, required=True)
    freeze.add_argument("--out", type=Path, required=True)
    scoring = commands.add_parser(
        "score", help="evaluate a frozen process on private distributions"
    )
    scoring.add_argument("--submission", type=Path, required=True)
    scoring.add_argument("--receipt-sha256", required=True)
    scoring.add_argument("--data", type=Path, required=True)
    scoring.add_argument("--out", type=Path, required=True)
    for command in (extraction, scoring):
        command.add_argument("--wall-seconds", type=float, default=30)
        command.add_argument("--cpu-seconds", type=int, default=20)
        command.add_argument("--memory-mb", type=int, default=1024)
    args = parser.parse_args(argv)
    if args.command == "agent":
        frozen = run_agent(
            args.challenge,
            args.client,
            args.catalog,
            args.out,
            budget=AgentBudget(args.steps, args.seconds, args.model_seconds),
        )
        print(json.dumps({"receipt_sha256": None if frozen is None else frozen.receipt_sha256}))
        return 0 if frozen is not None else 2
    if args.command == "prepare-operators":
        prepare_bundle(args.out)
        return 0
    if args.command == "export":
        torch.set_num_threads(check_count(args.threads))
        model = TinyTransformer.load(args.checkpoint)
        observations = (
            None
            if args.access == "weights"
            else observe(model, args.sequences, args.length, args.seed)
        )
        hashes = export_challenge(model, observations, args.access, args.out)
        print(json.dumps({"files": hashes}))
        return 0
    if args.command == "freeze":
        frozen = freeze_submission(args.source, args.out)
        print(json.dumps({"receipt_sha256": frozen.receipt_sha256}))
        return 0
    limits = Limits(args.wall_seconds, args.cpu_seconds, args.memory_mb)
    if args.command == "extract":
        frozen = extract_process(args.program, args.entry, args.challenge, args.out, limits=limits)
        print(json.dumps({"receipt_sha256": None if frozen is None else frozen.receipt_sha256}))
        return 0 if frozen is not None else 2
    with np.load(args.data, allow_pickle=False) as data:
        if set(data.files) not in ({"tokens", "oracle"}, {"tokens", "oracle", "teacher"}):
            raise ValueError("private score archive requires tokens, oracle and optionally teacher")
        report = score_process(
            FrozenSubmission(args.submission, args.receipt_sha256),
            data["tokens"],
            data["oracle"],
            data.get("teacher", None),
            args.out,
            limits=limits,
        )
    print(json.dumps(report, allow_nan=False))
    return 0 if report["status"] == "scored" else 2


if __name__ == "__main__":
    raise SystemExit(main())
