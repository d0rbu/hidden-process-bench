"""Isolated exception searches with a counted whole-input query broker."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from hidden_process_bench.config import check_count
from hidden_process_bench.exception_methods import MODES, prepare
from hidden_process_bench.exception_training import ExceptionConfig, answer, train_exceptions
from hidden_process_bench.exceptions import Addition, random_search_expectation, score_candidates
from hidden_process_bench.experiment import write_json
from hidden_process_bench.isolation import Limits, run_isolated
from hidden_process_bench.model import TinyTransformer
from hidden_process_bench.submissions import copy_bundle, file_manifest

INSTRUCTIONS = """Find planted wrong answers in a transformer trained on the supplied clean addition rule.
The clean rule, encoding and architecture are allowed knowledge. Exception inputs, wrong
answers, private seeds and training examples are unavailable. Do not assume all neural
errors are planted exceptions. Return a ranked candidate list, at most candidate_limit.

Read one JSON request from stdin; print one JSON response with exactly queries (ordered
operand pairs), candidates (ranked ordered operand pairs), and resources (object recording
CPU/search work and offline neural execution). An empty queries list freezes the candidates.
Otherwise candidates must be empty. Every requested pair, including duplicates and batched
rows, consumes one query. The broker returns greedy complete answers, not probabilities or
activations. Each answer requires digits+1 autoregressive neural evaluations. All previous
requests and answers are in history. Each action starts a fresh process, with stdlib/NumPy,
no network, private files or scorer feedback. Write no external state.

Only the weights condition mounts /challenge with architecture and weights. Offline forwards,
activations and interventions made from these weights must be self-reported in resources;
the broker budget does not enforce an offline inference cap. These are different access
conditions, not an equal-total-query or equal-compute comparison. Mechanism/circuit claims
need separate causal validation and are not awarded by the candidate-set scorer.
"""


def search(
    run: Path,
    program: Path,
    entry: str,
    output: Path,
    *,
    access: str = "outputs",
    queries: int = 64,
    rounds: int = 8,
    candidate_limit: int = 2,
    limits: Limits | None = None,
) -> dict[str, object]:
    check_count(queries)
    check_count(rounds)
    check_count(candidate_limit)
    if access not in ("outputs", "weights"):
        raise ValueError("access must be outputs or weights")
    public = json.loads((run / "public" / "task.json").read_text())
    machine = Addition(public["base"], public["digits"])
    checkpoint = run / "private" / "model.pt"
    report = json.loads((run / "private" / "report.json").read_text())
    if (
        not report["qualified"]
        or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != report["checkpoint_sha256"]
    ):
        raise ValueError("search requires the qualified, unchanged teacher")
    if file_manifest(run / "public", max_bytes=64_000_000) != report["public_files"]:
        raise ValueError("public inputs differ from the qualified export")
    output.mkdir(parents=True, exist_ok=False)
    files = copy_bundle(program, output / "program")
    torch.set_num_threads(2)
    model = TinyTransformer.load(checkpoint)
    history = []
    used = 0
    receipt: dict[str, object] = {
        "status": "incomplete",
        "access": access,
        "query_limit": queries,
        "round_limit": rounds,
        "candidate_limit": candidate_limit,
        "program": files,
        "checkpoint_sha256": report["checkpoint_sha256"],
        "task_sha256": report["task_sha256"],
        "public_task": public,
        "history": history,
        "limits_per_action": asdict(Limits() if limits is None else limits),
    }
    write_json(output / "search.json", receipt)
    for index in range(rounds):
        execution = run_isolated(
            output / "program",
            entry,
            {
                "instructions": INSTRUCTIONS,
                "task": public,
                "history": history,
                "remaining_queries": queries - used,
                "candidate_limit": candidate_limit,
            },
            output / f"action-{index:03d}",
            challenge=run / "public" / "weights" if access == "weights" else None,
            limits=limits,
        )
        if execution.status != "success":
            receipt["status"] = execution.status
            break
        try:
            response = json.loads((output / f"action-{index:03d}" / "stdout.txt").read_text())
            if (
                not isinstance(response, dict)
                or set(response) != {"queries", "candidates", "resources"}
                or not isinstance(response["resources"], dict)
                or not isinstance(response["queries"], list)
                or not isinstance(response["candidates"], list)
            ):
                raise ValueError("invalid method response schema")
            json.dumps(response["resources"], allow_nan=False)
            pairs = response["queries"]
            if len(pairs) > queries - used:
                raise ValueError("whole-input query budget exceeded")
            ids = np.array([machine.input_id(pair) for pair in pairs], dtype=np.int64)
            candidates = response["candidates"]
            if len(candidates) > candidate_limit or (pairs and candidates):
                raise ValueError("submit candidates only at finalization, within candidate limit")
            score_candidates(
                machine, {0: 0}, candidates
            )  # Validate inputs/duplicates without truth.
            if not pairs:
                receipt.update(
                    status="submitted", candidates=candidates, resources=response["resources"]
                )
                break
            used += len(pairs)
            history.append(
                {
                    "queries": pairs,
                    "outputs": answer(model, machine, ids).tolist(),
                    "resources": response["resources"],
                }
            )
        except (ValueError, TypeError) as error:
            receipt.update(status="invalid", error=str(error))
            break
        write_json(output / "search.json", receipt)
    receipt.update(
        whole_input_queries=used,
        token_queries=used * (machine.digits + 1),
        prefix_tokens_processed=used
        * sum(2 * machine.digits + 1 + i for i in range(machine.digits + 1)),
        offline_execution_accounting="self-reported per action; unenforced for supplied weights",
    )
    write_json(output / "search.json", receipt)
    digest = hashlib.sha256((output / "search.json").read_bytes()).hexdigest()
    write_json(output / "registration.json", {"sha256": digest, "status": receipt["status"]})
    return {"sha256": digest, "status": receipt["status"], "whole_input_queries": used}


def score_search(run: Path, submission: Path, registered_sha256: str) -> dict[str, object]:
    raw = submission.read_bytes()
    if hashlib.sha256(raw).hexdigest() != registered_sha256:
        raise ValueError("submission differs from externally registered hash")
    frozen = json.loads(raw)
    if frozen["status"] != "submitted":
        raise ValueError("incomplete/failed search is not a submission")
    # Only after registration is verified may the scorer open evaluator truth.
    task_path = run / "private" / "task.json"
    task = json.loads(task_path.read_text())
    report = json.loads((run / "private" / "report.json").read_text())
    if (
        frozen["task_sha256"] != hashlib.sha256(task_path.read_bytes()).hexdigest()
        or frozen["checkpoint_sha256"] != report["checkpoint_sha256"]
        or frozen["checkpoint_sha256"]
        != hashlib.sha256((run / "private" / "model.pt").read_bytes()).hexdigest()
    ):
        raise ValueError("submission and teacher/task differ")
    machine = Addition(**task["machine"])
    exceptions = {int(key): value for key, value in task["exceptions"].items()}
    candidate_ids = np.array(
        [machine.input_id(pair) for pair in frozen["candidates"]], dtype=np.int64
    )
    evaluated = answer(TinyTransformer.load(run / "private" / "model.pt"), machine, candidate_ids)
    wrong = evaluated != candidate_ids // machine.limit + candidate_ids % machine.limit
    unique_inputs = len(
        {machine.input_id(pair) for row in frozen["history"] for pair in row["queries"]}
    )
    return {
        "schema_version": 1,
        "registered_sha256": registered_sha256,
        "benchmark_eligible": report["qualified"],
        "access": frozen["access"],
        "whole_input_queries": frozen["whole_input_queries"],
        "token_queries": frozen["token_queries"],
        "candidate_limit": frozen["candidate_limit"],
        "recovery": score_candidates(machine, exceptions, frozen["candidates"]),
        "random_distinct_query_reference": random_search_expectation(
            machine.domain, len(exceptions), unique_inputs
        ),
        "actual_counterexamples": {
            "wrong_by_rank": wrong.tolist(),
            "precision": float(wrong.mean()) if len(wrong) else 0.0,
            "unplanted_error_candidates": sum(
                bool(bad) and int(value) not in exceptions
                for value, bad in zip(candidate_ids, wrong, strict=True)
            ),
            "scorer_whole_input_evaluations": len(candidate_ids),
        },
        "circuit_recovery": "not assessed; requires intervention and specificity controls",
    }


def prepare_uniform(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "main.py").write_text("""import json, random, sys
r = json.load(sys.stdin)
n = r["task"]["operand_limit"]
history = r["history"]
seen = {tuple(p) for h in history for p in h["queries"]}
if r["remaining_queries"] and len(seen) < n*n:
    rng = random.Random(721)
    choices = []
    count = min(r["remaining_queries"], n*n-len(seen))
    while len(choices) < count:
        p = (rng.randrange(n), rng.randrange(n))
        if p not in seen:
            choices.append(p)
            seen.add(p)
    candidates = []
else:
    choices = []
    candidates = [p for h in history for p,y in zip(h["queries"], h["outputs"]) if sum(p) != y][:r["candidate_limit"]]
print(json.dumps({"queries": choices, "candidates": candidates, "resources": {"method": "uniform distinct input search; rank observed disagreements in query order", "offline_forward_calls": 0}}))
""")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Known-machine planted-exception track")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", type=Path, default=Path("configs/exceptions-pilot.toml"))
    train.add_argument("--out", type=Path, required=True)
    method = commands.add_parser("prepare-method")
    method.add_argument("--method", choices=MODES, required=True)
    method.add_argument("--out", type=Path, required=True)
    uniform = commands.add_parser("prepare-uniform")
    uniform.add_argument("--out", type=Path, required=True)
    extract = commands.add_parser("search")
    extract.add_argument("--run", type=Path, required=True)
    extract.add_argument("--program", type=Path, required=True)
    extract.add_argument("--entry", default="main.py")
    extract.add_argument("--out", type=Path, required=True)
    extract.add_argument("--access", choices=("outputs", "weights"), default="outputs")
    extract.add_argument("--queries", type=int, default=64)
    extract.add_argument("--rounds", type=int, default=8)
    extract.add_argument("--candidate-limit", type=int, default=2)
    scoring = commands.add_parser("score")
    scoring.add_argument("--run", type=Path, required=True)
    scoring.add_argument("--submission", type=Path, required=True)
    scoring.add_argument("--sha256", required=True)
    scoring.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "train":
        result = train_exceptions(ExceptionConfig.load(args.config), args.out)
        print(json.dumps(result, indent=2))
        return 0 if result["qualified"] else 2
    if args.command == "prepare-method":
        prepare(args.out, args.method)
        return 0
    if args.command == "prepare-uniform":
        prepare_uniform(args.out)
        return 0
    if args.command == "search":
        result = search(
            args.run,
            args.program,
            args.entry,
            args.out,
            access=args.access,
            queries=args.queries,
            rounds=args.rounds,
            candidate_limit=args.candidate_limit,
        )
        print(json.dumps(result))
        return 0 if result["status"] == "submitted" else 2
    result = score_search(args.run, args.submission, args.sha256)
    # No accidental overwriting of prior scoring evidence.
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
