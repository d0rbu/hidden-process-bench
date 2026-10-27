"""Private mixed-process planning, matched training and neutral challenge export.

The frozen plan, true processes, screening, training and qualification stay private.
Only the declared neural snapshots are mounted when extracting a candidate process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from random import SystemRandom
from time import perf_counter
from typing import Literal, NotRequired, TypedDict, cast
from uuid import uuid4

import numpy as np
import torch
from torch.nn import functional as F

from hidden_process_bench.challenge import export_challenge
from hidden_process_bench.config import (
    ModelConfig,
    SamplingConfig,
    Shape,
    check_count,
    check_positive,
)
from hidden_process_bench.experiment import TrainRecord, predictions, provenance, write_json
from hidden_process_bench.generators import (
    HiddenProcess,
    LagProcess,
    SequenceProcess,
    StackProcess,
    load_process,
)
from hidden_process_bench.metrics import optimality
from hidden_process_bench.model import TinyTransformer, observe, teacher_inputs
from hidden_process_bench.process import random_mixture
from hidden_process_bench.sampling import measure_difficulty

Family = Literal["mixture_hmm", "lag_rule", "stack_rule"]
FAMILIES: tuple[Family, ...] = ("mixture_hmm", "lag_rule", "stack_rule")


class SuiteCase(TypedDict):
    id: str
    family: Family
    seeds: dict[str, int]


class PreparedCase(TypedDict):
    id: str
    family: Family
    status: str
    error: str | None
    process_sha256: NotRequired[str]


class Preparation(TypedDict):
    schema_version: int
    plan_sha256: str
    all_prepared: bool
    cases: list[PreparedCase]


@dataclass(frozen=True)
class SuiteConfig:
    replicates: int = 2
    vocab: int = 16
    length: int = 24
    width: int = 64
    heads: int = 4
    layers: int = 2
    train_steps: int = 4000
    batch_size: int = 64
    learning_rate: float = 0.003
    eval_every: int = 100
    eval_sequences: int = 512
    max_kl: float = 0.03
    validation_margin: float = 1.0
    observation_sequences: int = 128
    score_sequences: int = 128
    long_length: int = 72
    threads: int = 2

    def __post_init__(self) -> None:
        self.model()
        for name, value in asdict(self).items():
            if name in ("learning_rate", "max_kl", "validation_margin"):
                check_positive(value)
            else:
                check_count(value)
        if self.replicates > 8 or self.vocab < 6 or self.length < 6:
            raise ValueError("suite requires at most 8 replicates, vocab >= 6 and length >= 6")
        if self.eval_sequences < 12 or self.long_length <= self.length:
            raise ValueError(
                "suite needs >=12 qualification sequences and longer extrapolation histories"
            )
        if self.validation_margin > 1:
            raise ValueError("suite validation margin must be positive and at most one")

    def model(self) -> ModelConfig:
        return ModelConfig(self.vocab, self.length, self.width, self.heads, self.layers)

    @classmethod
    def load(cls, path: Path) -> SuiteConfig:
        with path.open("rb") as stream:
            return cls(**tomllib.load(stream))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _archive_source(private: Path, source: dict[str, object]) -> None:
    root = Path(__file__).resolve().parents[2]
    for name, expected in cast(dict[str, str], source["source_sha256"]).items():
        data = (root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("source changed while preparing its snapshot")
        target = private / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)


def _seeds(used: set[int]) -> dict[str, int]:
    result = {}
    for label in (
        "structure",
        "generator",
        "initialization",
        "training",
        "validation",
        "test",
        "observation",
        "scoring",
        "long_scoring",
    ):
        seed = secrets.randbits(63)
        while seed in used:
            seed = secrets.randbits(63)
        result[label] = seed
        used.add(seed)
    return result


def _rule_screen(process: LagProcess | StackProcess) -> dict[str, object]:
    """A pre-training observable dependency witness, not a minimality certificate."""
    if isinstance(process, LagProcess):
        histories = np.zeros((2, process.lag), dtype=np.int64)
        histories[1, 0] = 1
    else:
        a = process.alphabet
        histories = np.array(
            [[a[0], a[1], a[3], a[4], a[4]], [a[1], a[0], a[2], a[4], a[4]]], dtype=np.int64
        )
    probabilities = process.probabilities(histories)[:, -1]
    distance = float(np.abs(probabilities[0] - probabilities[1]).sum() / 2)
    accepted = bool(np.array_equal(histories[0, -2:], histories[1, -2:]) and distance >= 0.1)
    return {
        "accepted": accepted,
        "histories": histories.tolist(),
        "next_probabilities": probabilities.tolist(),
        "total_variation": distance,
        "condition": "same last two tokens; different next distribution",
        "finite_minimal_order": None,
    }


def _prepare_process(
    family: Family, config: SuiteConfig, seeds: dict[str, int], private: Path
) -> tuple[SequenceProcess, dict[str, object]]:
    rng = np.random.default_rng(seeds["structure"])
    if family == "mixture_hmm":
        sizes = [(k, s) for k in range(2, 5) for s in range(2, 5) if k * s <= config.vocab]
        generator = np.random.default_rng(seeds["generator"])
        sampling = SamplingConfig()
        with (private / "sampling.jsonl").open("x") as stream:
            for attempt in range(1, sampling.max_attempts + 1):
                k, s = sizes[int(rng.integers(len(sizes)))]
                shape = Shape(k, s, config.vocab)
                seed = int(generator.integers(2**63))
                hmm = random_mixture(
                    shape,
                    seed,
                    extra_edge_probability=sampling.extra_edge_probability,
                    emission_concentration=sampling.emission_concentration,
                )
                difficulty = measure_difficulty(hmm)
                rejections = difficulty.rejections(sampling)
                stream.write(
                    json.dumps(
                        {
                            "attempt": attempt,
                            "shape": asdict(shape),
                            "candidate_seed": seed,
                            "difficulty": asdict(difficulty),
                            "rejections": rejections,
                            "accepted": not rejections,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                stream.flush()
                if not rejections:
                    return HiddenProcess(hmm), {
                        "accepted": True,
                        "attempts": attempt,
                        "difficulty": asdict(difficulty),
                        "screen": "temporal_and_numerical_minimal_order",
                    }
        raise ValueError(
            f"sampling exhausted {sampling.max_attempts} structures/parameters; see sampling.jsonl"
        )
    generator = np.random.default_rng(seeds["generator"])
    alphabet = tuple(generator.permutation(config.vocab).tolist())
    if family == "lag_rule":
        process: LagProcess | StackProcess = LagProcess(
            alphabet,
            int(rng.integers(3, min(5, config.length - 1) + 1)),
            float(generator.uniform(0.65, 0.85)),
        )
    else:
        process = StackProcess(
            alphabet,
            push=float(generator.uniform(0.28, 0.34)),
            close=float(generator.uniform(0.48, 0.56)),
            noise=float(generator.uniform(0.02, 0.06)),
        )
    return process, _rule_screen(process)


def prepare_suite(config: SuiteConfig, output: Path) -> Preparation:
    """Freeze all cases/seeds before any training; preserve all preparation failures."""
    output.mkdir(parents=True, exist_ok=False)
    private = output / "private"
    private.mkdir()
    families = list(FAMILIES) * config.replicates
    SystemRandom().shuffle(families)
    used: set[int] = set()
    cases: list[SuiteCase] = [
        {"id": uuid4().hex, "family": family, "seeds": _seeds(used)} for family in families
    ]
    plan = {"schema_version": 1, "config": asdict(config), "cases": cases}
    write_json(private / "plan.json", plan)
    source = provenance()
    write_json(private / "preparation-provenance.json", source)
    _archive_source(private, source)
    records: list[PreparedCase] = []
    for case in cases:
        directory = private / case["id"]
        directory.mkdir()
        record: PreparedCase = {
            "id": case["id"],
            "family": case["family"],
            "status": "prepared",
            "error": None,
        }
        try:
            process, screen = _prepare_process(case["family"], config, case["seeds"], directory)
            write_json(directory / "screening.json", screen)
            if not screen["accepted"]:
                raise ValueError("pre-training dependency screen rejected the process")
            write_json(directory / "process.json", process.specification())
            record["process_sha256"] = _sha256(directory / "process.json")
        except (ValueError, OSError) as error:
            record.update(status="preparation_failed", error=str(error))
        records.append(record)
        write_json(directory / "preparation.json", record)
    report: Preparation = {
        "schema_version": 1,
        "plan_sha256": _sha256(private / "plan.json"),
        "all_prepared": all(row["status"] == "prepared" for row in records),
        "cases": records,
    }
    write_json(private / "preparation.json", report)
    return report


def _plan(output: Path) -> tuple[SuiteConfig, list[SuiteCase], Preparation]:
    private = output / "private"
    plan = json.loads((private / "plan.json").read_text())
    preparation = cast(Preparation, json.loads((private / "preparation.json").read_text()))
    if plan["schema_version"] != 1 or preparation["plan_sha256"] != _sha256(private / "plan.json"):
        raise ValueError("frozen suite plan was modified")
    return SuiteConfig(**plan["config"]), plan["cases"], preparation


def train_case(output: Path, case_id: str) -> dict[str, object]:
    config, cases, preparation = _plan(output)
    found = [case for case in cases if case["id"] == case_id]
    if len(found) != 1:
        raise ValueError("case is not in the frozen suite")
    case = found[0]
    private = output / "private" / case_id
    registered = next(row for row in preparation["cases"] if row["id"] == case_id)
    if registered["status"] != "prepared" or registered["process_sha256"] != _sha256(
        private / "process.json"
    ):
        raise ValueError("private process is unprepared or was modified")
    training = private / "training"
    training.mkdir(exist_ok=False)
    started = perf_counter()
    torch.set_num_threads(config.threads)
    seeds = case["seeds"]
    process = load_process(private / "process.json")
    source = provenance()
    prepared_source = json.loads((output / "private/preparation-provenance.json").read_text())
    if source["source_sha256"] != prepared_source["source_sha256"]:
        raise ValueError("suite source changed after preparation")
    write_json(training / "provenance.json", source)
    torch.manual_seed(seeds["initialization"])
    model = TinyTransformer(config.model())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    rng = np.random.default_rng(seeds["training"])
    validation, groups = process.sample(
        np.random.default_rng(seeds["validation"]), config.eval_sequences, config.length
    )
    oracle = process.probabilities(validation)[:, :-1]
    predicted, _ = predictions(model, validation)
    initial_metrics = optimality(oracle, predicted, groups, process.groups, config.max_kl)
    trace: list[TrainRecord] = []
    for step in range(1, config.train_steps + 1):
        model.train()
        tokens, _ = process.sample(rng, config.batch_size, config.length)
        targets = torch.tensor(tokens)
        logits, _ = model(teacher_inputs(targets, config.vocab))
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % config.eval_every == 0 or step == config.train_steps:
            predicted, _ = predictions(model, validation)
            metrics = optimality(
                oracle,
                predicted,
                groups,
                process.groups,
                config.max_kl * config.validation_margin,
            )
            row: TrainRecord = {
                "step": step,
                "training_cross_entropy": float(loss.detach()),
                "validation": metrics,
            }
            trace.append(row)
            with (training / "trace.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(
                f"step={step} validation_kl={metrics['overall']['mean_kl']:.6f} qualified={metrics['qualified']}",
                flush=True,
            )
            if metrics["qualified"]:
                break
    checkpoint = training / "checkpoint.pt"
    model.save(checkpoint)
    test, test_groups = process.sample(
        np.random.default_rng(seeds["test"]), config.eval_sequences, config.length
    )
    test_oracle = process.probabilities(test)[:, :-1]
    test_teacher, _ = predictions(model, test)
    test_metrics = optimality(test_oracle, test_teacher, test_groups, process.groups, config.max_kl)
    qualified = bool(trace[-1]["validation"]["qualified"] and test_metrics["qualified"])
    np.savez_compressed(
        training / "qualification.npz",
        validation_tokens=validation,
        validation_groups=groups,
        validation_oracle=oracle,
        validation_teacher=predicted,
        test_tokens=test,
        test_groups=test_groups,
        test_oracle=test_oracle,
        test_teacher=test_teacher,
    )
    challenges = {}
    if qualified:
        observations = observe(
            model, config.observation_sequences, config.length, seeds["observation"]
        )
        for access in ("outputs", "weights", "internals"):
            challenges[access] = export_challenge(
                model, observations, access, output / "public" / case_id / access
            )
    for label, length, seed_label in (
        ("heldout", config.length, "scoring"),
        ("long", config.long_length, "long_scoring"),
    ):
        tokens, score_groups = process.sample(
            np.random.default_rng(seeds[seed_label]), config.score_sequences, length
        )
        score_oracle = process.probabilities(tokens)[:, :-1]
        if label == "heldout":
            teacher, _ = predictions(model, tokens)
            np.savez_compressed(
                training / f"{label}.npz", tokens=tokens, oracle=score_oracle, teacher=teacher
            )
        else:
            np.savez_compressed(training / f"{label}.npz", tokens=tokens, oracle=score_oracle)
        np.save(training / f"{label}-groups.npy", score_groups)
    report = {
        "schema_version": 1,
        "case_id": case_id,
        "status": "qualified" if qualified else "unqualified",
        "checkpoint_qualified": qualified,
        "qualification_group_definition": "mixture_component"
        if case["family"] == "mixture_hmm"
        else "maximum_stack_depth_0_to_1__2_to_3__4_plus"
        if case["family"] == "stack_rule"
        else "all_sequences",
        "steps": trace[-1]["step"],
        "training_tokens": trace[-1]["step"] * config.batch_size * config.length,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "initial_validation": initial_metrics,
        "validation": trace[-1]["validation"],
        "test": test_metrics,
        "checkpoint_sha256": _sha256(checkpoint),
        "process_sha256": registered["process_sha256"],
        "plan_sha256": preparation["plan_sha256"],
        "public_challenge_sha256": challenges,
        "seconds": perf_counter() - started,
        "independent_discovery_evaluated": False,
    }
    if provenance()["source_sha256"] != source["source_sha256"]:
        raise ValueError("suite source changed during training")
    write_json(training / "report.json", report)
    return report


def _append_status(output: Path, record: dict[str, object]) -> None:
    with (output / "private/training-status.jsonl").open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


def _training_report(
    directory: Path, case: SuiteCase, prepared: PreparedCase, plan_hash: str
) -> dict[str, object]:
    report = json.loads((directory / "training/report.json").read_text())
    if not isinstance(report, dict):
        raise ValueError("invalid child training report")
    for name, expected in (
        ("case_id", case["id"]),
        ("plan_sha256", plan_hash),
        ("process_sha256", prepared["process_sha256"]),
    ):
        if report.get(name) != expected:
            raise ValueError("child report does not match its frozen task")
    validation, test = report.get("validation"), report.get("test")
    if not isinstance(validation, dict) or not isinstance(test, dict):
        raise ValueError("child report lacks qualification evidence")
    if type(validation.get("qualified")) is not bool or type(test.get("qualified")) is not bool:
        raise ValueError("child qualification flags must be booleans")
    qualified = validation["qualified"] and test["qualified"]
    if report.get("checkpoint_qualified") is not qualified or report.get("status") != (
        "qualified" if qualified else "unqualified"
    ):
        raise ValueError("child qualification status is inconsistent")
    if report.get("checkpoint_sha256") != _sha256(directory / "training/checkpoint.pt"):
        raise ValueError("child checkpoint hash does not match")
    return report


def train_suite(output: Path, *, timeout: float = 300) -> dict[str, object]:
    check_positive(timeout)
    _config, cases, preparation = _plan(output)
    records = []
    summary_path = output / "private/training-summary.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    for case, prepared in zip(cases, preparation["cases"], strict=True):
        directory = output / "private" / case["id"]
        if prepared["status"] != "prepared":
            rejected: dict[str, object] = {
                "id": case["id"],
                "family": case["family"],
                "status": "preparation_failed",
            }
            records.append(rejected)
            _append_status(output, rejected)
            continue
        record = {"id": case["id"], "family": case["family"], "status": "failed", "error": None}
        try:
            with (
                (directory / "child-stdout.txt").open("x") as stdout,
                (directory / "child-stderr.txt").open("x") as stderr,
            ):
                child = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "hidden_process_bench.mixed_suite",
                        "case",
                        "--suite",
                        str(output.resolve()),
                        "--case-id",
                        case["id"],
                    ],
                    stdout=stdout,
                    stderr=stderr,
                    timeout=timeout,
                    check=False,
                )
            record["returncode"] = child.returncode
            if child.returncode == 0:
                report = _training_report(directory, case, prepared, preparation["plan_sha256"])
                record.update(status=report["status"], report=report)
            else:
                record["error"] = "training subprocess failed; see child-stderr.txt"
        except subprocess.TimeoutExpired:
            record.update(status="timeout", error="training exceeded its wall-time cap")
        except (ValueError, OSError) as error:
            record["error"] = str(error)
        records.append(record)
        _append_status(output, record)
    summary = {
        "schema_version": 1,
        "plan_sha256": preparation["plan_sha256"],
        "case_timeout_seconds": timeout,
        "cases": records,
        "campaign_ready": all(row["status"] == "qualified" for row in records),
        "independent_discovery_evaluated": False,
    }
    write_json(summary_path, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluator-only mixed-process training suite.")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    training = commands.add_parser("train")
    training.add_argument("--suite", type=Path, required=True)
    training.add_argument("--timeout", type=float, default=300)
    case = commands.add_parser("case")
    case.add_argument("--suite", type=Path, required=True)
    case.add_argument("--case-id", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        report = prepare_suite(SuiteConfig.load(args.config), args.out)
        print(json.dumps(report))
        return 0 if report["all_prepared"] else 2
    if args.command == "case":
        report = train_case(args.suite, args.case_id)
        print(json.dumps(report))
        return 0
    report = train_suite(args.suite, timeout=args.timeout)
    print(json.dumps(report))
    return 0 if report["campaign_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
