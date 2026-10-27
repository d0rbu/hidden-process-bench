import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from hidden_process_bench.discovery import extract_process, predict_histories, score_process
from hidden_process_bench.experiment import write_json
from hidden_process_bench.submissions import FrozenSubmission, freeze_submission

# Independent implementation of a simple process, not a supplied latent-state schema.
CONSTANT = """import json, sys
from pathlib import Path
from typing import cast
assert not Path("/challenge").exists()
r = json.load(sys.stdin)
assert len({len(x) for x in r["histories"]}) == 1
print(json.dumps({"probabilities": [[0.25, 0.75] for _ in r["histories"]]}))
"""


def submitted(root: Path, code: str = CONSTANT) -> FrozenSubmission:
    source = root / "source"
    source.mkdir(parents=True)
    (source / "main.py").write_text(code)
    (source / "mechanism.txt").write_text("Independent Bernoulli draws; no state.")
    write_json(
        source / "submission.json",
        {
            "schema_version": 1,
            "predictor": "main.py",
            "mechanism": "mechanism.txt",
            "hypothesis": "Constant token probabilities.",
            "assumptions": "IID assumed by this baseline.",
            "resources": {"queries": 0},
        },
    )
    return freeze_submission(source, root / "frozen")


def test_real_extraction_is_frozen_before_private_scoring(tmp_path: Path) -> None:
    source = tmp_path / "method"
    source.mkdir()
    challenge = tmp_path / "challenge"
    challenge.mkdir()
    (challenge / "input.txt").write_text("public only")
    (source / "extract.py").write_text(
        """import json
from pathlib import Path
from typing import cast
assert Path("/challenge/input.txt").read_text() == "public only"
out = Path("/output")
(out / "main.py").write_text("""
        + repr(CONSTANT)
        + """)
(out / "mechanism.txt").write_text("Independent draws, constant probabilities.")
(out / "submission.json").write_text(json.dumps({"schema_version": 1,
"predictor": "main.py", "mechanism": "mechanism.txt", "hypothesis": "Independent draws",
"assumptions": "Constant baseline, no discovery claimed", "resources": {"queries": 0}}))
"""
    )
    frozen = extract_process(source, "extract.py", challenge, tmp_path / "extracted")
    assert frozen is not None
    tokens = np.array([[0, 1, 1], [1, 0, 1]], dtype=np.int64)
    oracle = np.broadcast_to(np.array([0.25, 0.75]), (2, 3, 2)).copy()
    report = score_process(frozen, tokens, oracle, oracle, tmp_path / "score")
    assert report["status"] == "scored"
    for metric in ("oracle_to_candidate", "teacher_to_candidate", "oracle_to_teacher"):
        summary = report[metric]
        assert isinstance(summary, dict)
        assert cast(dict[str, object], summary)["mean_kl"] == 0
    assert report["independent_structure_recovery"] is None
    assert report["structure_assessment"] == "pending_independent_review"
    # The extractor never receives private score files after the run.
    assert not (tmp_path / "extracted/submission/report.json").exists()
    requests = list((tmp_path / "score").glob("position-*"))
    assert len(requests) == 3


@pytest.mark.parametrize("code", ['raise RuntimeError("failed search")', 'print("no submission")'])
def test_invalid_or_failed_extraction_is_retained(tmp_path: Path, code: str) -> None:
    source = tmp_path / "method"
    source.mkdir()
    (source / "main.py").write_text(code)
    challenge = tmp_path / "challenge"
    challenge.mkdir()
    frozen = extract_process(source, "main.py", challenge, tmp_path / "extracted")
    assert frozen is None
    assert (
        json.loads((tmp_path / "extracted/extraction.json").read_text())["valid_submission"]
        is False
    )


@pytest.mark.parametrize(
    ("histories", "message"),
    [
        ([], "nonempty"),
        ([[0], []], "equally"),
        ([[True]], "invalid token"),
        ([[2]], "invalid token"),
    ],
)
def test_invalid_histories_rejected_before_execution(
    tmp_path: Path, histories: list[list[int]], message: str
) -> None:
    frozen = submitted(tmp_path)
    with pytest.raises(ValueError, match=message):
        predict_histories(frozen, histories, 2, tmp_path / "prediction")
    assert not (tmp_path / "prediction").exists()


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ('print("[]")', "only probabilities"),
        ("print('{\"probabilities\": [[1.0]]}')", "wrong probability shape"),
        ("print('{\"probabilities\": [[-1.0, 2.0]]}')", "nonnegative"),
        ("print('{\"probabilities\": [[0.1, 0.1]]}')", "sum to one"),
        ('raise RuntimeError("crash")', "execution failed"),
        ('print("bad JSON")', "Expecting value"),
    ],
)
def test_invalid_predictor_output_rejected(tmp_path: Path, code: str, message: str) -> None:
    frozen = submitted(tmp_path, code)
    with pytest.raises(ValueError, match=message):
        predict_histories(frozen, [[]], 2, tmp_path / "prediction")


def test_frozen_receipt_predictor_entry_point_checked(tmp_path: Path) -> None:
    frozen = submitted(tmp_path)
    raw = json.loads((frozen.path / "frozen.json").read_text())
    for bad in ([], {"predictor": 2}):
        raw["submission"] = bad
        write_json(frozen.path / "frozen.json", raw)
        digest = hashlib.sha256((frozen.path / "frozen.json").read_bytes()).hexdigest()
        with pytest.raises(ValueError, match="entry point"):
            predict_histories(FrozenSubmission(frozen.path, digest), [[]], 2, tmp_path / "unused")


def test_batch_dependent_predictions_fail_contract_and_keep_logs(tmp_path: Path) -> None:
    code = """import json, sys
r = json.load(sys.stdin)
p = [0.25, 0.75] if len(r["histories"]) == 3 else [0.5, 0.5]
print(json.dumps({"probabilities": [p for _ in r["histories"]]}))
"""
    frozen = submitted(tmp_path, code)
    tokens = np.zeros((3, 2), dtype=np.int64)
    oracle = np.full((3, 2, 2), 0.5)
    report = score_process(frozen, tokens, oracle, None, tmp_path / "score")
    assert report["status"] == "failed"
    assert "batch composition" in str(report["error"])
    assert report["oracle_to_candidate"] is None
    assert report["oracle_to_teacher"] is None
    assert not report["row_invariance_probes_passed"]
    assert (tmp_path / "score/position-0/execution.json").is_file()


def test_long_history_scoring_and_single_sequence_supported_without_teacher(tmp_path: Path) -> None:
    frozen = submitted(tmp_path)
    tokens = np.zeros((1, 1), dtype=np.int64)
    oracle = np.array([[[0.25, 0.75]]])
    report = score_process(frozen, tokens, oracle, None, tmp_path / "score")
    assert report["status"] == "scored"
    assert report["teacher_to_candidate"] is None
    summary = report["oracle_to_candidate"]
    assert isinstance(summary, dict)
    assert cast(dict[str, object], summary)["mean_kl"] == 0
    # Arbitrarily long histories are not silently cropped to the teacher context.
    actual = predict_histories(frozen, [[1] * 80], 2, tmp_path / "long")
    np.testing.assert_array_equal(actual, [[0.25, 0.75]])


def test_failed_prediction_becomes_retained_score_failure(tmp_path: Path) -> None:
    frozen = submitted(tmp_path, "raise RuntimeError('no answer')")
    report = score_process(
        frozen, np.zeros((2, 2), dtype=np.int64), np.full((2, 2, 2), 0.5), None, tmp_path / "score"
    )
    assert report["status"] == "failed"
    assert "execution failed" in str(report["error"])
    assert json.loads((tmp_path / "score/report.json").read_text()) == report


@pytest.mark.parametrize(
    "case", ["rank", "empty", "dtype", "oracle_rank", "alignment", "teacher_shape"]
)
def test_misaligned_evaluator_arrays_rejected(tmp_path: Path, case: str) -> None:
    frozen = submitted(tmp_path)
    tokens = np.zeros((2, 2), dtype=np.int64)
    oracle = np.full((2, 2, 2), 0.5)
    teacher = oracle
    if case == "rank":
        tokens = tokens[0]
    elif case == "empty":
        tokens = tokens[:, :0]
    elif case == "dtype":
        tokens = tokens.astype(np.float64)
    elif case == "oracle_rank":
        oracle = oracle[0]
    elif case == "alignment":
        oracle = oracle[:, :1]
    else:
        teacher = teacher[:, :1]
    with pytest.raises(ValueError, match="align"):
        score_process(frozen, tokens, oracle, teacher, tmp_path / "score")
    assert not (tmp_path / "score").exists()
