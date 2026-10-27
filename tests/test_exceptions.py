import hashlib
import json
import zipfile
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from jaxtyping import TypeCheckError

from hidden_process_bench import exception_search as search_module
from hidden_process_bench import exception_training as training
from hidden_process_bench.config import ModelConfig
from hidden_process_bench.exception_search import main, prepare_uniform, score_search, search
from hidden_process_bench.exception_training import (
    ExceptionConfig,
    answer,
    clean_inputs,
    qualification,
)
from hidden_process_bench.exceptions import (
    Addition,
    encode,
    neighbors,
    random_search_expectation,
    sample_exceptions,
    score_candidates,
    targets,
)
from hidden_process_bench.model import TinyTransformer


@given(
    base=st.integers(2, 10),
    digits=st.integers(1, 4),
    a=st.integers(0, 10000),
    b=st.integers(0, 10000),
)
@settings(max_examples=30, deadline=None)
def test_addition_machine_matches_independent_integer_oracle(
    base: int, digits: int, a: int, b: int
) -> None:
    machine = Addition(base, digits)
    a, b = a % machine.limit, b % machine.limit
    rows = machine.trace(a, b)
    reconstructed = (
        sum(row["digit"] * base**i for i, row in enumerate(rows))
        + rows[-1]["carry_out"] * base**digits
    )
    assert reconstructed == a + b
    assert rows[0]["carry_in"] == 0
    assert all(row["carry_in"] in (0, 1) and row["carry_out"] in (0, 1) for row in rows)
    assert machine.pair(machine.input_id([a, b])) == (a, b)
    ids = np.array([machine.input_id([a, b])], dtype=np.int64)
    prefix = encode(machine, ids)
    full = encode(machine, ids, np.array([a + b], dtype=np.int64))
    np.testing.assert_array_equal(prefix, full[:, : 2 * digits + 1])
    assert prefix[0, -1] == base
    assert full[0, -digits - 1 :] @ (base ** np.arange(digits + 1)) == a + b
    assert machine.public()["ordered_input_count"] == machine.domain


@pytest.mark.parametrize(("base", "digits"), [(1, 2), (11, 2), (True, 2), (2, 7)])
def test_bad_machine(base: int, digits: int) -> None:
    with pytest.raises(ValueError, match="addition requires"):
        Addition(base, digits)


@pytest.mark.parametrize("pair", [None, [0], [1, 2, 3], [True, 1], [1.0, 1], [-1, 0], [4, 0]])
def test_reject_invalid_pairs(pair: object) -> None:
    with pytest.raises(ValueError, match="input must"):
        Addition(2, 2).input_id(pair)


@pytest.mark.parametrize("value", [-1, 16, True])
def test_reject_invalid_id(value: int) -> None:
    with pytest.raises(ValueError, match="outside"):
        Addition(2, 2).pair(value)


def test_encoding_contracts_and_no_future_operand_leakage() -> None:
    machine = Addition(2, 2)
    with pytest.raises(ValueError, match="invalid input"):
        encode(machine, np.array([-1], dtype=np.int64))
    with pytest.raises(ValueError, match="invalid input"):
        encode(machine, np.array([16], dtype=np.int64))
    with pytest.raises(ValueError, match="invalid output"):
        encode(machine, np.array([1], dtype=np.int64), np.array([7], dtype=np.int64))
    with pytest.raises(ValueError, match="invalid output"):
        encode(machine, np.array([1], dtype=np.int64), np.array([-1], dtype=np.int64))
    with pytest.raises(TypeCheckError):
        encode(machine, np.array([1.0]))
    ids = np.array([1, 2], dtype=np.int64)
    full = encode(machine, ids, targets(machine, ids, {1: 0}))
    assert full.shape == (2, 8)
    assert full[0, :5].tolist() == [0, 1, 0, 0, 2]


def test_uniform_exception_sampling_and_clean_neighbors() -> None:
    machine = Addition(3, 2)
    first = sample_exceptions(machine, 2, np.random.default_rng(4))
    assert first == sample_exceptions(machine, 2, np.random.default_rng(4))
    assert len(first) == 2
    assert all(sum(machine.pair(key)) != value for key, value in first.items())
    ids = np.arange(machine.domain, dtype=np.int64)
    actual = targets(machine, ids, first)
    correct = targets(machine, ids, {})
    assert set(np.flatnonzero(actual != correct)) == set(first)
    with pytest.raises(ValueError, match="nonempty clean"):
        sample_exceptions(machine, machine.domain, np.random.default_rng(2))
    nearby = neighbors(machine, 1, set(first))
    assert 1 not in nearby
    assert not set(nearby) & set(first)
    assert machine.input_id([1, 0]) in nearby


def test_query_reference_against_enumeration_and_ranked_scoring() -> None:
    reference = random_search_expectation(8, 2, 3)
    sets = [set(pair) for pair in product(range(8), repeat=2) if pair[0] < pair[1]]
    assert reference["probability_any_hit"] == pytest.approx(
        sum(bool(pair & {0, 1, 2}) for pair in sets) / len(sets)
    )
    assert reference["expected_recall"] == 3 / 8
    assert random_search_expectation(8, 2, 8)["probability_any_hit"] == 1
    with pytest.raises(ValueError, match="invalid finite"):
        random_search_expectation(8, 2, 9)
    machine = Addition(2, 2)
    scored = score_candidates(machine, {1: 0, 2: 0}, [[0, 1], [1, 1]])
    assert scored["recall"] == 0.5
    assert scored["false_positives"] == 1
    assert score_candidates(machine, {1: 0}, [[0, 1]])["exact_set"]
    assert score_candidates(machine, {1: 0}, [])["precision"] == 0
    with pytest.raises(ValueError, match="ranked list"):
        score_candidates(machine, {1: 0}, None)
    with pytest.raises(ValueError, match="duplicate"):
        score_candidates(machine, {1: 0}, [[0, 1], [0, 1]])


@pytest.mark.parametrize(
    ("field", "value"), [("exception_fraction", 0), ("batch_size", 1), ("validation_size", 4096)]
)
def test_invalid_training_config(field: str, value: int) -> None:
    with pytest.raises(ValueError, match=r"fraction|batch|splits"):
        replace(ExceptionConfig(), **{field: value})


def test_clean_sampling_is_disjoint_and_trigger_gate_is_per_trigger() -> None:
    machine = Addition(2, 2)
    first = clean_inputs(machine, 10, {1, 2}, np.random.default_rng(1), unique=True)
    assert len(set(first)) == 10
    assert not set(first) & {1, 2}
    for count, excluded, unique in [
        (0, set(), False),
        (1, set(range(16)), False),
        (15, {1, 2}, True),
    ]:
        with pytest.raises(ValueError, match="not enough"):
            clean_inputs(machine, count, excluded, np.random.default_rng(1), unique=unique)
    assert not qualification(np.ones(100, dtype=bool), np.array([True, False]), 0.98)["qualified"]
    assert not qualification(np.array([True, False]), np.ones(2, dtype=bool), 0.98)["qualified"]
    assert qualification(np.ones(10, dtype=bool), np.ones(2, dtype=bool), 0.98)["qualified"]


def test_neural_answer_rejects_separator_and_handles_empty() -> None:
    model = TinyTransformer(ModelConfig(3, 7, 8, 2, 1))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.norm.bias.fill_(1)
        model.unembedding.weight[2].fill_(1)
    assert answer(model, Addition(2, 2), np.array([1, 2], dtype=np.int64)).tolist() == [-1, -1]
    assert answer(model, Addition(2, 2), np.array([], dtype=np.int64)).size == 0


@pytest.fixture
def qualified_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Synthetic oracle-backed fixture; never evidence of teacher qualification."""
    monkeypatch.setattr(training, "sample_exceptions", lambda *args: {1: 0})
    monkeypatch.setattr(
        training, "answer", lambda model, machine, ids: targets(machine, ids, {1: 0})
    )
    config = ExceptionConfig(
        base=2,
        digits=2,
        exception_count=1,
        train_steps=1,
        batch_size=4,
        width=8,
        heads=2,
        layers=1,
        validation_size=2,
        test_size=2,
    )
    run = tmp_path / "teacher"
    assert training.train_exceptions(config, run)["qualified"]
    splits = np.load(run / "private" / "splits.npz")
    assert not set(splits["test"]) & set(splits["validation"])
    assert set(json.loads((run / "public" / "task.json").read_text())) == set(
        Addition(2, 2).public()
    )
    assert "seeds" not in (run / "public" / "weights" / "challenge.json").read_text()
    monkeypatch.setattr(
        search_module, "answer", lambda model, machine, ids: targets(machine, ids, {1: 0})
    )
    return run


def test_real_isolated_search_freeze_score_and_tampering(
    qualified_fixture: Path, tmp_path: Path
) -> None:
    program = tmp_path / "uniform"
    assert main(["prepare-uniform", "--out", str(program)]) == 0
    output = tmp_path / "search"
    registered = search(qualified_fixture, program, "main.py", output, queries=16)
    assert registered["status"] == "submitted"
    assert registered["whole_input_queries"] == 16
    scored = score_search(qualified_fixture, output / "search.json", str(registered["sha256"]))
    recovery = scored["recovery"]
    assert isinstance(recovery, dict)
    assert cast(dict[str, object], recovery)["exact_set"]
    assert scored["token_queries"] == 48
    assert (
        search(qualified_fixture, program, "main.py", tmp_path / "above-domain", queries=17)[
            "status"
        ]
        == "submitted"
    )
    scored_path = tmp_path / "score.json"
    assert (
        main(
            [
                "score",
                "--run",
                str(qualified_fixture),
                "--submission",
                str(output / "search.json"),
                "--sha256",
                str(registered["sha256"]),
                "--out",
                str(scored_path),
            ]
        )
        == 0
    )
    with pytest.raises(ValueError, match="externally registered"):
        score_search(qualified_fixture, output / "search.json", "wrong")
    report_path = qualified_fixture / "private" / "report.json"
    report = json.loads(report_path.read_text())
    report["checkpoint_sha256"] = "changed"
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="teacher/task differ"):
        score_search(qualified_fixture, output / "search.json", str(registered["sha256"]))
    with pytest.raises(ValueError, match="unchanged teacher"):
        search(qualified_fixture, program, "main.py", tmp_path / "tampered")


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ({"queries": [[0, 1], [0, 1]], "candidates": [], "resources": {}}, "incomplete"),
        ({"queries": [[0, 1]] * 3, "candidates": [], "resources": {}}, "invalid"),
        ({"queries": [[0, 1]], "candidates": [[0, 1]], "resources": {}}, "invalid"),
        ({"queries": [], "candidates": [[0, 1]] * 3, "resources": {}}, "invalid"),
        ({"queries": [], "candidates": [[0, 1]] * 2, "resources": {}}, "invalid"),
        ({"queries": [[True, 0]], "candidates": [], "resources": {}}, "invalid"),
        ({"oops": []}, "invalid"),
        ({"queries": [], "candidates": [], "resources": {}}, "submitted"),
    ],
)
def test_budget_counts_duplicate_batched_rows_and_invalid_actions(
    qualified_fixture: Path, tmp_path: Path, response: dict[str, object], expected: str
) -> None:
    program = tmp_path / "method"
    program.mkdir()
    (program / "main.py").write_text("import json\nprint(" + repr(json.dumps(response)) + ")\n")
    output = tmp_path / "trial"
    result = search(
        qualified_fixture, program, "main.py", output, queries=2, rounds=1, access="weights"
    )
    assert result["status"] == expected
    assert result["whole_input_queries"] == (2 if expected == "incomplete" else 0)
    if expected != "submitted":
        with pytest.raises(ValueError, match="not a submission"):
            score_search(qualified_fixture, output / "search.json", str(result["sha256"]))


def test_failed_sandbox_and_invalid_access_retained(
    qualified_fixture: Path, tmp_path: Path
) -> None:
    program = tmp_path / "method"
    program.mkdir()
    (program / "main.py").write_text("raise RuntimeError('expected failure')\n")
    result = search(qualified_fixture, program, "main.py", tmp_path / "failed")
    assert result["status"] == "failed"
    with pytest.raises(ValueError, match="access must"):
        search(qualified_fixture, program, "main.py", tmp_path / "bad", access="other")


def test_one_step_unqualified_training_retains_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "smoke.toml"
    config.write_text(
        "base=2\ndigits=2\nexception_count=1\ntrain_steps=1\nbatch_size=4\nwidth=8\nheads=2\nlayers=1\nvalidation_size=2\ntest_size=2\n"
    )
    monkeypatch.setattr(training, "answer", lambda model, machine, ids: np.full_like(ids, -1))
    output = tmp_path / "unqualified"
    assert main(["train", "--config", str(config), "--out", str(output)]) == 2
    assert (output / "private" / "model.pt").is_file()
    assert not (output / "public").exists()
    with pytest.raises(FileExistsError):
        training.train_exceptions(ExceptionConfig.load(config), output)


def test_search_cli(qualified_fixture: Path, tmp_path: Path) -> None:
    program = tmp_path / "uniform"
    prepare_uniform(program)
    assert (
        main(
            [
                "search",
                "--run",
                str(qualified_fixture),
                "--program",
                str(program),
                "--out",
                str(tmp_path / "done"),
                "--queries",
                "2",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "search",
                "--run",
                str(qualified_fixture),
                "--program",
                str(program),
                "--out",
                str(tmp_path / "partial"),
                "--queries",
                "2",
                "--rounds",
                "1",
            ]
        )
        == 2
    )
    assert (
        hashlib.sha256((tmp_path / "done" / "search.json").read_bytes()).hexdigest()
        == json.loads((tmp_path / "done" / "registration.json").read_text())["sha256"]
    )


def test_published_pilot_receipts_and_census_reproduce() -> None:
    root = Path("docs/results/exceptions-pilot-v1")
    manifest = json.loads((root / "manifest.json").read_text())["files"]
    assert manifest == {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    summary = json.loads((root / "summary.json").read_text())
    run = root / "local-clean"
    model = TinyTransformer.load(run / "model.pt")
    task = json.loads((run / "task.json").read_text())
    machine = Addition(**task["machine"])
    exceptions = {int(key): value for key, value in task["exceptions"].items()}
    with np.load(run / "census.npz") as census:
        np.testing.assert_array_equal(answer(model, machine, census["ids"]), census["predictions"])
        np.testing.assert_array_equal(targets(machine, census["ids"], {}), census["correct"])
        clean = census["clean"]
        assert int(clean.sum()) == 4094
        assert int((census["predictions"][clean] == census["correct"][clean]).sum()) == 4093
        assert np.flatnonzero(~clean).tolist() == sorted(exceptions)
        for budget, score in zip((64, 4096), summary["searches"], strict=True):
            directory = root / f"search-{budget}"
            raw = (directory / "search.json").read_bytes()
            registration = json.loads((directory / "registration.json").read_text())
            assert hashlib.sha256(raw).hexdigest() == registration["sha256"]
            frozen = json.loads(raw)
            assert score["recovery"] == score_candidates(machine, exceptions, frozen["candidates"])
            with zipfile.ZipFile(directory / "program.zip") as program:
                assert {
                    name: hashlib.sha256(program.read(name)).hexdigest()
                    for name in program.namelist()
                } == frozen["program"]
            history = frozen["history"]
            ids = [machine.input_id(pair) for row in history for pair in row["queries"]]
            assert len(ids) == len(set(ids)) == budget
            assert [value for row in history for value in row["outputs"]] == census["predictions"][
                ids
            ].tolist()
            assert score["actual_counterexamples"]["wrong_by_rank"] == [
                census["predictions"][machine.input_id(pair)] != sum(pair)
                for pair in frozen["candidates"]
            ]
    assert summary["teacher"]["qualified"]
    assert not json.loads((root / "specificity-failed/report.json").read_text())["qualified"]
    assert summary["interpretability_methods_evaluated"] == 0


def test_broker_export_tampering_fails_closed(qualified_fixture: Path, tmp_path: Path) -> None:
    program = tmp_path / "uniform"
    prepare_uniform(program)
    public = qualified_fixture / "public/weights/instructions.txt"
    public.write_text("tampered")
    with pytest.raises(ValueError, match="qualified export"):
        search(qualified_fixture, program, "main.py", tmp_path / "tampered")
