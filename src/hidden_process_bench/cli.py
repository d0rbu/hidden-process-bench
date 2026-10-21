"""Local commands; no hosted services or provider calls."""

import argparse
import json
from pathlib import Path

import torch

from hidden_process_bench.config import RunConfig, check_count
from hidden_process_bench.experiment import extract_checkpoint, run, score
from hidden_process_bench.process import MixtureHMM


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recover hidden HMMs from a trained transformer.")
    commands = parser.add_subparsers(dest="command", required=True)
    experiment = commands.add_parser("run", help="train, qualify, recover and score")
    experiment.add_argument("--config", type=Path, default=Path("configs/demo.toml"))
    experiment.add_argument("--out", type=Path, required=True)
    experiment.add_argument("--require-qualified", action="store_true")
    extract = commands.add_parser("extract", help="recover an HMM from public checkpoint artifacts")
    extract.add_argument("--checkpoint", type=Path, required=True)
    extract.add_argument("--task", type=Path, required=True)
    extract.add_argument("--out", type=Path, required=True)
    extract.add_argument(
        "--method", choices=("samples", "probabilities", "activations"), default="activations"
    )
    extract.add_argument("--sequences", type=int, default=128)
    extract.add_argument("--steps", type=int, default=250)
    extract.add_argument("--restarts", type=int, default=2)
    extract.add_argument("--seed", type=int, default=101)
    extract.add_argument("--threads", type=int, default=2)
    scoring = commands.add_parser(
        "score", help="score a candidate HMM against evaluator-only truth"
    )
    scoring.add_argument("--truth", type=Path, required=True)
    scoring.add_argument("--candidate", type=Path, required=True)
    scoring.add_argument("--sequences", type=int, default=256)
    scoring.add_argument("--length", type=int, default=24)
    scoring.add_argument("--seed", type=int, default=202)
    args = parser.parse_args(argv)
    if args.command == "run":
        report = run(RunConfig.load(args.config), args.out)
        print(
            json.dumps(
                {
                    "report": str(args.out / "report.json"),
                    "benchmark_eligible": report["benchmark_eligible"],
                }
            )
        )
        return 2 if args.require_qualified and not report["benchmark_eligible"] else 0
    if args.command == "extract":
        torch.set_num_threads(check_count(args.threads))
        result = extract_checkpoint(
            args.checkpoint,
            args.task,
            args.out,
            args.method,
            args.sequences,
            args.steps,
            args.restarts,
            args.seed,
        )
    else:
        result = score(
            MixtureHMM.load(args.truth),
            MixtureHMM.load(args.candidate),
            args.sequences,
            args.length,
            args.seed,
        )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0
