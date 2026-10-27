"""Execute methods and score their chosen processes without family-specific adapters."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from hidden_process_bench.config import check_count
from hidden_process_bench.experiment import provenance, write_json
from hidden_process_bench.isolation import Limits, run_isolated
from hidden_process_bench.metrics import check_probabilities, sequence_kl, summarize
from hidden_process_bench.process import Floats, Tokens
from hidden_process_bench.submissions import (
    FrozenSubmission,
    freeze_submission,
    verify_submission,
)


def extract_process(
    program: Path,
    entry: str,
    challenge: Path,
    output: Path,
    *,
    limits: Limits | None = None,
) -> FrozenSubmission | None:
    output.mkdir(parents=True, exist_ok=False)
    execution = run_isolated(
        program, entry, {}, output / "execution", challenge=challenge, limits=limits
    )
    frozen = None
    error = None
    if execution.status == "success":
        try:
            frozen = freeze_submission(output / "execution/generated", output / "submission")
        except (ValueError, OSError) as invalid:
            error = str(invalid)
    write_json(
        output / "extraction.json",
        {
            "execution": asdict(execution),
            "valid_submission": frozen is not None,
            "receipt_sha256": None if frozen is None else frozen.receipt_sha256,
            "error": error,
            "structure_assessment": "pending_independent_review",
        },
    )
    return frozen


def predict_histories(
    frozen: FrozenSubmission,
    histories: list[list[int]],
    vocab: int,
    output: Path,
    *,
    limits: Limits | None = None,
) -> Floats:
    check_count(vocab)
    if not histories or any(len(row) != len(histories[0]) for row in histories):
        raise ValueError("prediction requires a nonempty batch of equally long histories")
    if any(type(token) is not int or not 0 <= token < vocab for row in histories for token in row):
        raise ValueError("history contains an invalid token ID")
    receipt = verify_submission(frozen)
    submission = receipt["submission"]
    if not isinstance(submission, dict):
        raise ValueError("invalid frozen predictor entry point")
    entry = submission.get("predictor")
    if not isinstance(entry, str):
        raise ValueError("invalid frozen predictor entry point")
    execution = run_isolated(
        frozen.path,
        entry,
        {"histories": histories, "vocab_size": vocab},
        output,
        limits=limits,
    )
    verify_submission(frozen)
    if execution.status != "success":
        raise ValueError("prediction execution " + execution.status)
    payload = json.loads((output / "stdout.txt").read_text())
    if not isinstance(payload, dict) or set(payload) != {"probabilities"}:
        raise ValueError("predictor must return only probabilities")
    values = np.asarray(payload["probabilities"], dtype=np.float64)
    if values.shape != (len(histories), vocab):
        raise ValueError("predictor returned the wrong probability shape")
    check_probabilities(values)
    return values


def score_process(
    frozen: FrozenSubmission,
    tokens: Tokens,
    oracle: Floats,
    teacher: Floats | None,
    output: Path,
    *,
    limits: Limits | None = None,
) -> dict[str, object]:
    """Score already private held-out sequences, without revealing targets to code.

    A request contains ONLY equal-length prefixes from independent sequences. Giving
    every prefix at once would reveal future tokens through longer rows. No process
    survives between requests. Row-invariance probes detect some contract violations;
    they are not a formal proof of an arbitrary program's semantics or structure.
    """
    if (
        tokens.ndim != 2
        or not all(tokens.shape)
        or tokens.dtype != np.int64
        or oracle.ndim != 3
        or not all(oracle.shape)
        or oracle.shape[:2] != tokens.shape
        or (tokens < 0).any()
        or (tokens >= oracle.shape[2]).any()
        or (teacher is not None and teacher.shape != oracle.shape)
    ):
        raise ValueError("score arrays must align as [sequences, positions, vocabulary]")
    check_probabilities(oracle)
    if teacher is not None:
        check_probabilities(teacher)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "provenance.json", provenance())
    if teacher is None:
        np.savez_compressed(output / "private-inputs.npz", tokens=tokens, oracle=oracle)
    else:
        np.savez_compressed(
            output / "private-inputs.npz", tokens=tokens, oracle=oracle, teacher=teacher
        )
    inputs_sha256 = hashlib.sha256((output / "private-inputs.npz").read_bytes()).hexdigest()
    candidate = np.empty_like(oracle)
    error = None
    invariant = True
    try:
        for position in range(tokens.shape[1]):
            histories = tokens[:, :position].tolist()
            candidate[:, position] = predict_histories(
                frozen, histories, oracle.shape[2], output / f"position-{position}", limits=limits
            )
        # Evaluate rows in a different batch and order, with no other task information.
        selected = list(dict.fromkeys([tokens.shape[0] - 1, 0]))
        for position in dict.fromkeys([0, tokens.shape[1] - 1]):
            histories = tokens[selected, :position].tolist()
            checked = predict_histories(
                frozen, histories, oracle.shape[2], output / f"invariance-{position}", limits=limits
            )
            invariant = invariant and bool(
                np.allclose(checked, candidate[selected, position], atol=1e-10, rtol=0)
            )
        if not invariant:
            raise ValueError("prediction changed with batch composition/order")
    except (ValueError, OSError) as invalid:
        error = str(invalid)
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "scored" if error is None else "failed",
        "error": error,
        "submission_receipt_sha256": frozen.receipt_sha256,
        "private_inputs_sha256": inputs_sha256,
        "sequences": tokens.shape[0],
        "positions": tokens.shape[1],
        "row_invariance_probes_passed": invariant if error is None else False,
        "oracle_to_candidate": None,
        "teacher_to_candidate": None,
        "oracle_to_teacher": None if teacher is None else summarize(sequence_kl(oracle, teacher)),
        "structure_assessment": "pending_independent_review",
        "independent_structure_recovery": None,
        "causal_recovery": None,
    }
    if error is None:
        np.savez_compressed(output / "predictions.npz", probabilities=candidate)
        report["oracle_to_candidate"] = summarize(sequence_kl(oracle, candidate))
        if teacher is not None:
            report["teacher_to_candidate"] = summarize(sequence_kl(teacher, candidate))
    write_json(output / "report.json", report)
    return report
