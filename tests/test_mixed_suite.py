import hashlib
import itertools
import json
import subprocess
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from hidden_process_bench import mixed_suite
from hidden_process_bench.config import ModelConfig, Shape
from hidden_process_bench.experiment import predictions
from hidden_process_bench.generators import LagProcess, load_process
from hidden_process_bench.metrics import optimality
from hidden_process_bench.mixed_suite import (
    SuiteConfig,
    _rule_screen,
    _seeds,
    main,
    prepare_suite,
    train_case,
    train_suite,
)
from hidden_process_bench.model import TinyTransformer
from hidden_process_bench.process import random_mixture


@pytest.fixture(autouse=True)
def deterministic_private_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    values = itertools.count(100)
    monkeypatch.setattr(mixed_suite.secrets, "randbits", lambda _bits: next(values))
    monkeypatch.setattr(mixed_suite.SystemRandom, "shuffle", lambda _self, _values: None)


def tiny_config(**kwargs: object) -> SuiteConfig:
    config = SuiteConfig(
        replicates=1,
        vocab=6,
        length=6,
        width=8,
        heads=2,
        layers=1,
        train_steps=2,
        batch_size=4,
        eval_every=2,
        eval_sequences=128,
        observation_sequences=4,
        score_sequences=4,
        long_length=12,
    )
    return replace(config, **kwargs)


def plan(root: Path) -> dict[str, Any]:
    return json.loads((root / "private/plan.json").read_text())


def test_freezes_balanced_cases_and_independent_private_seeds_before_training(
    tmp_path: Path,
) -> None:
    output = tmp_path / "suite"
    report = prepare_suite(tiny_config(), output)
    assert report["all_prepared"]
    prepared = plan(output)
    assert Counter(case["family"] for case in prepared["cases"]) == {
        "mixture_hmm": 1,
        "lag_rule": 1,
        "stack_rule": 1,
    }
    seeds = [value for case in prepared["cases"] for value in case["seeds"].values()]
    assert len(seeds) == len(set(seeds)) == 27
    assert not (output / "public").exists()
    for case in prepared["cases"]:
        assert len(case["id"]) == 32
        assert all(char in "0123456789abcdef" for char in case["id"])
        private = output / "private" / case["id"]
        process = load_process(private / "process.json")
        assert process.vocab == 6
        assert json.loads((private / "screening.json").read_text())["accepted"]
        assert not (private / "training").exists()
    with pytest.raises(FileExistsError):
        prepare_suite(tiny_config(), output)
    provenance = json.loads((output / "private/preparation-provenance.json").read_text())
    assert "src/hidden_process_bench/generators.py" in provenance["source_sha256"]


def test_seed_collisions_do_not_reuse_another_random_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = iter([5, 5, 6, *range(7, 20)])
    monkeypatch.setattr(mixed_suite.secrets, "randbits", lambda _bits: next(values))
    used: set[int] = set()
    result = _seeds(used)
    assert len(result) == len(used) == 9
    assert result["structure"] == 5
    assert result["generator"] == 6


def test_rule_screen_rejects_a_witness_with_only_short_memory() -> None:
    screen = _rule_screen(LagProcess(tuple(range(6)), 1, 0.7))
    assert not screen["accepted"]
    assert screen["finite_minimal_order"] is None
    accepted = _rule_screen(LagProcess(tuple(range(6)), 3, 0.7))
    assert accepted["accepted"]
    assert accepted["total_variation"] == pytest.approx(0.7)


def test_real_training_recomputes_qualification_and_exports_matched_inputs(tmp_path: Path) -> None:
    output = tmp_path / "suite"
    # Loose tolerance exercises positive plumbing; this is not research qualification.
    config = tiny_config(max_kl=10.0, eval_every=1)
    prepare_suite(config, output)
    case = next(case for case in plan(output)["cases"] if case["family"] == "lag_rule")
    report = train_case(output, case["id"])
    assert report["checkpoint_qualified"]
    assert report["steps"] == 1
    assert report["training_tokens"] == config.batch_size * config.length
    private = output / "private" / case["id"] / "training"
    model = TinyTransformer.load(private / "checkpoint.pt")
    assert model.config == ModelConfig(6, 6, 8, 2, 1)
    with np.load(private / "qualification.npz") as data:
        for name in ("validation", "test"):
            tokens = data[name + "_tokens"]
            teacher, _ = predictions(model, tokens)
            np.testing.assert_array_equal(teacher, data[name + "_teacher"])
            recomputed = optimality(
                data[name + "_oracle"], teacher, data[name + "_groups"], 1, config.max_kl
            )
            assert report[name] == recomputed
    public = output / "public" / case["id"]
    with (
        np.load(public / "outputs/observations.npz") as a,
        np.load(public / "internals/observations.npz") as b,
    ):
        np.testing.assert_array_equal(a["tokens"], b["tokens"])
        np.testing.assert_array_equal(a["probabilities"], b["probabilities"])
        assert "activations" not in a
        assert "activations" in b
    for directory in public.iterdir():
        text = (directory / "challenge.json").read_text() + (
            directory / "instructions.txt"
        ).read_text()
        assert "family" not in text
        assert "lag_rule" not in text
        assert str(case["seeds"]["generator"]) not in text
    with np.load(private / "heldout.npz") as heldout:
        assert set(heldout.files) == {"tokens", "oracle", "teacher"}
        assert heldout["oracle"].shape == (4, 6, 6)
    with np.load(private / "long.npz") as long:
        assert set(long.files) == {"tokens", "oracle"}
        assert long["oracle"].shape == (4, 12, 6)
    with pytest.raises(FileExistsError):
        train_case(output, case["id"])


def test_training_cap_keeps_unqualified_checkpoint_without_export(tmp_path: Path) -> None:
    output = tmp_path / "suite"
    prepare_suite(tiny_config(max_kl=1e-12, eval_every=3), output)
    case = next(case for case in plan(output)["cases"] if case["family"] == "stack_rule")
    report = train_case(output, case["id"])
    assert report["status"] == "unqualified"
    assert report["steps"] == 2
    assert not (output / "public").exists()
    private = output / "private" / case["id"] / "training"
    assert (private / "checkpoint.pt").exists()
    assert len((private / "trace.jsonl").read_text().splitlines()) == 1
    assert cast(dict[str, object], report["test"])["qualified"] is False


def test_actual_child_processes_train_all_three_families_and_preserve_failures(
    tmp_path: Path,
) -> None:
    output = tmp_path / "suite"
    prepare_suite(tiny_config(max_kl=1e-12), output)
    report = train_suite(output, timeout=60)
    assert not report["campaign_ready"]
    cases = cast(list[dict[str, object]], report["cases"])
    assert isinstance(cases, list)
    assert len(cases) == 3
    assert {case["status"] for case in cases} == {"unqualified"}
    assert {case["family"] for case in cases} == {"mixture_hmm", "lag_rule", "stack_rule"}
    assert len((output / "private/training-status.jsonl").read_text().splitlines()) == 3
    with pytest.raises(FileExistsError):
        train_suite(output)


def test_plan_and_process_modifications_cannot_silently_change_task(tmp_path: Path) -> None:
    output = tmp_path / "suite"
    prepare_suite(tiny_config(), output)
    first = plan(output)["cases"][0]
    with pytest.raises(ValueError, match="not in"):
        train_case(output, "../outside")
    process = output / "private" / first["id"] / "process.json"
    process.write_text("{}")
    with pytest.raises(ValueError, match="modified"):
        train_case(output, first["id"])
    (output / "private/plan.json").write_text("{}")
    with pytest.raises((ValueError, KeyError), match=r"schema_version|modified"):
        train_suite(output)


def test_preparation_rejections_and_errors_remain_in_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = mixed_suite._prepare_process

    def prepare(
        family: mixed_suite.Family, config: SuiteConfig, seeds: dict[str, int], private: Path
    ) -> tuple[mixed_suite.SequenceProcess, dict[str, object]]:
        if family == "mixture_hmm":
            raise ValueError("screening exhausted")
        process, screen = original(family, config, seeds, private)
        return process, {**screen, "accepted": False}

    monkeypatch.setattr(mixed_suite, "_prepare_process", prepare)
    output = tmp_path / "suite"
    report = prepare_suite(tiny_config(), output)
    assert not report["all_prepared"]
    assert {row["status"] for row in report["cases"]} == {"preparation_failed"}
    summary = train_suite(output)
    assert not summary["campaign_ready"]
    assert len(cast(list[object], summary["cases"])) == 3
    with pytest.raises(ValueError, match="unprepared"):
        train_case(output, report["cases"][0]["id"])


@pytest.mark.parametrize("failure", ["timeout", "nonzero", "missing_report", "malformed_report"])
def test_child_errors_are_recorded_without_dropping_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    output = tmp_path / "suite"
    prepare_suite(tiny_config(), output)

    def failed(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 1)
        if failure == "malformed_report":
            case_id = command[-1]
            directory = output / "private" / case_id / "training"
            directory.mkdir()
            (directory / "report.json").write_text("bad JSON")
        return subprocess.CompletedProcess(command, 1 if failure == "nonzero" else 0)

    monkeypatch.setattr(mixed_suite.subprocess, "run", failed)
    report = train_suite(output)
    assert not report["campaign_ready"]
    expected = "timeout" if failure == "timeout" else "failed"
    assert {case["status"] for case in cast(list[dict[str, object]], report["cases"])} == {expected}


@pytest.mark.parametrize(
    "change",
    [
        {"replicates": 9},
        {"vocab": 5},
        {"length": 5},
        {"eval_sequences": 11},
        {"long_length": 6},
        {"train_steps": 0},
        {"max_kl": float("nan")},
    ],
)
def test_suite_rejects_invalid_configuration(change: dict[str, object]) -> None:
    with pytest.raises((ValueError, TypeError), match=r"suite|PositiveCount|positive"):
        tiny_config(**change)


def test_configuration_file_and_cli_prepare_failure_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = SuiteConfig.load(Path("configs/discovery-smoke.toml"))
    assert config.model().context == 6
    output = tmp_path / "suite"
    assert main(["prepare", "--config", "configs/discovery-smoke.toml", "--out", str(output)]) == 0
    capsys.readouterr()
    case_id = next(case for case in plan(output)["cases"] if case["family"] == "lag_rule")["id"]
    assert main(["case", "--suite", str(output), "--case-id", case_id]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["status"] == "unqualified"
    monkeypatch.setattr(mixed_suite, "prepare_suite", lambda _config, _out: {"all_prepared": False})
    assert main(["prepare", "--config", "configs/discovery-smoke.toml", "--out", "unused"]) == 2
    for ready in (True, False):
        monkeypatch.setattr(
            mixed_suite, "train_suite", lambda _out, timeout, ready=ready: {"campaign_ready": ready}
        )
        assert main(["train", "--suite", str(output)]) == (0 if ready else 2)


def test_positive_suite_status_requires_every_case_to_qualify(tmp_path: Path) -> None:
    output = tmp_path / "suite"
    prepare_suite(tiny_config(max_kl=10.0, train_steps=1), output)
    report = train_suite(output, timeout=60)
    assert report["campaign_ready"]
    assert not report["independent_discovery_evaluated"]
    assert len(cast(list[object], report["cases"])) == 3


@pytest.mark.parametrize("during", [False, True])
def test_source_changes_cannot_mix_training_implementations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, during: bool
) -> None:
    output = tmp_path / "suite"
    prepare_suite(tiny_config(), output)
    case_id = next(case for case in plan(output)["cases"] if case["family"] == "lag_rule")["id"]
    original = mixed_suite.provenance
    calls = 0

    def changed() -> dict[str, object]:
        nonlocal calls
        calls += 1
        value = original()
        if not during or calls > 1:
            value["source_sha256"] = {"different": "implementation"}
        return value

    monkeypatch.setattr(mixed_suite, "provenance", changed)
    with pytest.raises(ValueError, match="source changed"):
        train_case(output, case_id)
    assert not (output / "private" / case_id / "training/report.json").exists()


@pytest.mark.parametrize(
    "change",
    [
        "not_object",
        "case_id",
        "plan_sha256",
        "process_sha256",
        "missing_validation",
        "nonboolean",
        "wrong_status",
        "wrong_flag",
        "wrong_checkpoint",
    ],
)
def test_child_reports_require_matching_task_checkpoint_and_qualification(
    tmp_path: Path, change: str
) -> None:
    output = tmp_path / "suite"
    preparation = prepare_suite(tiny_config(), output)
    case = cast(mixed_suite.SuiteCase, plan(output)["cases"][0])
    prepared = preparation["cases"][0]
    directory = output / "private" / case["id"]
    (directory / "training").mkdir()
    (directory / "training/checkpoint.pt").write_bytes(b"checkpoint fixture")
    raw: dict[str, object] = {
        "case_id": case["id"],
        "plan_sha256": preparation["plan_sha256"],
        "process_sha256": prepared["process_sha256"],
        "validation": {"qualified": True},
        "test": {"qualified": True},
        "checkpoint_qualified": True,
        "status": "qualified",
        "checkpoint_sha256": hashlib.sha256(b"checkpoint fixture").hexdigest(),
    }
    if change in ("case_id", "plan_sha256", "process_sha256"):
        raw[change] = "wrong"
    elif change == "missing_validation":
        raw["validation"] = None
    elif change == "nonboolean":
        raw["test"] = {"qualified": 1}
    elif change == "wrong_status":
        raw["status"] = "unqualified"
    elif change == "wrong_flag":
        raw["checkpoint_qualified"] = False
    elif change == "wrong_checkpoint":
        raw["checkpoint_sha256"] = "changed"
    payload = [] if change == "not_object" else raw
    (directory / "training/report.json").write_text(json.dumps(payload))
    with pytest.raises(
        ValueError, match=r"invalid child|frozen task|qualification|checkpoint hash"
    ):
        mixed_suite._training_report(directory, case, prepared, preparation["plan_sha256"])


def test_source_snapshot_matches_every_registered_hash(tmp_path: Path) -> None:
    output = tmp_path / "suite"
    prepare_suite(tiny_config(), output)
    private = output / "private"
    source = json.loads((private / "preparation-provenance.json").read_text())
    for name, expected in source["source_sha256"].items():
        assert hashlib.sha256((private / "source" / name).read_bytes()).hexdigest() == expected
    with pytest.raises(ValueError, match="source changed"):
        mixed_suite._archive_source(
            tmp_path / "bad", {"source_sha256": {"pyproject.toml": "wrong"}}
        )


def test_exhausted_hidden_screen_retains_every_rejected_structure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = mixed_suite.measure_difficulty(random_mixture(Shape(2, 2, 6), 17))
    rejected = replace(original, min_state_mass=[0.0])
    monkeypatch.setattr(mixed_suite, "measure_difficulty", lambda _hmm: rejected)
    output = tmp_path / "suite"
    report = prepare_suite(tiny_config(), output)
    hidden = next(case for case in report["cases"] if case["family"] == "mixture_hmm")
    assert hidden["status"] == "preparation_failed"
    audit = output / "private" / hidden["id"] / "sampling.jsonl"
    records = [json.loads(line) for line in audit.read_text().splitlines()]
    assert len(records) == 128
    assert all(not record["accepted"] for record in records)
    assert all("min_state_mass" in record["rejections"] for record in records)
    assert (
        len({(record["shape"]["components"], record["shape"]["states"]) for record in records}) > 1
    )
