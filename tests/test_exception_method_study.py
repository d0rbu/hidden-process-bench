import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from hidden_process_bench import exception_method_study as study
from hidden_process_bench.challenge import export_challenge
from hidden_process_bench.circuits import ablate
from hidden_process_bench.config import ModelConfig
from hidden_process_bench.exception_search import score_search
from hidden_process_bench.exception_training import answer
from hidden_process_bench.exceptions import Addition, score_candidates, targets
from hidden_process_bench.experiment import write_json
from hidden_process_bench.model import TinyTransformer
from hidden_process_bench.submissions import file_manifest


def fixture_teacher(root: Path) -> None:
    """Synthetic export contract only: the random network is deliberately not qualified."""
    private = root / "private"
    private.mkdir(parents=True)
    model = TinyTransformer(ModelConfig(3, 7, 8, 2, 1))
    model.save(private / "model.pt")
    write_json(private / "task.json", {"machine": {"base": 2, "digits": 2}, "exceptions": {"1": 0}})
    np.savez_compressed(private / "splits.npz", validation=np.array([3, 4]), test=np.array([5, 6]))
    public = root / "public"
    public.mkdir()
    write_json(public / "task.json", Addition(2, 2).public())
    export_challenge(model, None, "weights", public / "weights")
    write_json(
        private / "report.json",
        {
            "qualified": True,
            "checkpoint_sha256": hashlib.sha256((private / "model.pt").read_bytes()).hexdigest(),
            "task_sha256": hashlib.sha256((private / "task.json").read_bytes()).hexdigest(),
            "public_files": file_manifest(public),
        },
    )


def test_real_sandbox_study_registers_every_method_before_any_private_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "study"

    def fake_training(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        fixture_teacher(Path(command[command.index("--out") + 1]))
        return subprocess.CompletedProcess(command, 0)

    original_score = study.score_search
    calls = []

    def checked_score(run: Path, submission: Path, sha256: str) -> dict[str, object]:
        registration = json.loads((output / "submissions-registration.json").read_text())
        raw = (output / "submissions.json").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == registration["sha256"]
        assert len(json.loads(raw)["trials"]) == 7
        calls.append(submission)
        return original_score(run, submission, sha256)

    monkeypatch.setattr(
        study,
        "subprocess",
        SimpleNamespace(
            run=fake_training, TimeoutExpired=subprocess.TimeoutExpired, STDOUT=subprocess.STDOUT
        ),
    )
    monkeypatch.setattr(study, "score_search", checked_score)
    summary = study.run_study(output, cases=1)
    assert len(calls) == 7
    assert len(cast(list[object], summary["trials"])) == 7
    assert len(cast(list[object], summary["five_internal_methods_attempted"])) == 5
    assert (output / "case-01/ablation/evaluator-diagnostic/interventions.npz").is_file()
    diagnostics = json.loads(
        (output / "case-01/ablation/evaluator-diagnostic/report.json").read_text()
    )
    assert len(diagnostics["single_unit_controls"]) == 3
    assert diagnostics["evaluator_whole_input_evaluations"] > 0
    with pytest.raises(FileExistsError):
        study.run_study(output, cases=1)


@pytest.mark.parametrize("status", ["timeout", "unqualified"])
def test_training_failure_retains_all_planned_trials_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    calls = []

    def failed(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if status == "timeout":
            raise subprocess.TimeoutExpired(command, 180)
        root = Path(command[command.index("--out") + 1])
        (root / "private").mkdir(parents=True)
        write_json(root / "private/report.json", {"qualified": False})
        return subprocess.CompletedProcess(command, 2)

    monkeypatch.setattr(
        study,
        "subprocess",
        SimpleNamespace(
            run=failed, TimeoutExpired=subprocess.TimeoutExpired, STDOUT=subprocess.STDOUT
        ),
    )
    summary = study.run_study(tmp_path / "failed", cases=1)
    assert len(calls) == 1
    rows = cast(list[dict[str, object]], summary["trials"])
    assert len(rows) == 7
    assert {row["status"] for row in rows} == {"teacher_unqualified"}
    assert (tmp_path / "failed/submissions-registration.json").is_file()


def test_bounds_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="one through three"):
        study.run_study(tmp_path / "invalid", cases=0)
    monkeypatch.setattr(study, "run_study", lambda output, cases: {"trials": []})
    assert study.main(["--out", str(tmp_path / "cli"), "--cases", "1"]) == 0


def test_published_fresh_trials_bind_original_sources_registration_and_scores() -> None:
    root = Path("docs/results/exception-methods-v1")
    manifest = json.loads((root / "manifest.json").read_text())["files"]
    assert manifest == {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    protocol_raw = (root / "protocol.json").read_bytes()
    protocol = json.loads(protocol_raw)
    assert (
        hashlib.sha256(protocol_raw).hexdigest()
        == json.loads((root / "protocol-registration.json").read_text())["sha256"]
    )
    assert (
        hashlib.sha256((root / "runtime-source.zip").read_bytes()).hexdigest()
        == protocol["runtime_source_sha256"]
    )
    registered_raw = (root / "submissions.json").read_bytes()
    registered = json.loads(registered_raw)
    assert (
        hashlib.sha256(registered_raw).hexdigest()
        == json.loads((root / "submissions-registration.json").read_text())["sha256"]
    )
    assert len(registered["trials"]) == 21
    summary = json.loads((root / "summary.json").read_text())
    with zipfile.ZipFile(root / "programs.zip") as programs:
        for method, files in protocol["method_files"].items():
            assert files == {
                name.removeprefix(method + "/"): hashlib.sha256(programs.read(name)).hexdigest()
                for name in programs.namelist()
                if name.startswith(method + "/")
            }
    registered_rows = {(row["case"], row["method"]): row for row in registered["trials"]}
    for row in summary["trials"]:
        assert registered_rows[row["case"], row["method"]]["sha256"] == row["sha256"]
        assert row["status"] == "submitted"
        run = root / row["case"] / "teacher"
        trial = root / row["case"] / row["method"]
        raw = (trial / "search.json").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row["sha256"]
        frozen = json.loads(raw)
        assert frozen["program"] == protocol["method_files"][row["method"]]
        truth = json.loads((run / "private/task.json").read_text())
        machine = Addition(**truth["machine"])
        exceptions = {int(key): value for key, value in truth["exceptions"].items()}
        assert row["score"]["recovery"] == score_candidates(
            machine, exceptions, frozen["candidates"]
        )
        assert score_search(run, trial / "search.json", row["sha256"]) == row["score"]
        assert frozen["whole_input_queries"] == sum(
            len(record["queries"]) for record in frozen["history"]
        )
    for row in summary["teachers"]:
        run = root / row["case"] / "teacher"
        model = TinyTransformer.load(run / "private/model.pt")
        truth = json.loads((run / "private/task.json").read_text())
        machine = Addition(**truth["machine"])
        with np.load(run / "census.npz") as census:
            np.testing.assert_array_equal(
                answer(model, machine, census["ids"]), census["predictions"]
            )
            np.testing.assert_array_equal(targets(machine, census["ids"], {}), census["reference"])
            clean = census["clean"]
            assert (
                int((census["predictions"][clean] != census["reference"][clean]).sum())
                == row["census"]["unplanted_errors"]
            )
    assert [row["recovered"] for row in summary["aggregate"]] == [0, 0, 1, 1, 1, 1, 2]
    assert not summary["causal_recovery_established"]


def test_published_nominated_ablation_effect_is_reproducible_and_not_selective() -> None:
    root = Path("docs/results/exception-methods-v1/case-01")
    report = json.loads((root / "ablation/evaluator-diagnostic/report.json").read_text())
    claim = report["claims"][0]
    model = TinyTransformer.load(root / "teacher/private/model.pt")
    machine = Addition(4, 3)
    with np.load(root / "ablation/evaluator-diagnostic/interventions.npz") as arrays:
        with ablate(model, {claim["nominated_unit"]}):
            changed = answer(model, machine, arrays["ids"])
        index = [row["unit"] for row in report["single_unit_controls"]].index(
            claim["nominated_unit"]
        )
        np.testing.assert_array_equal(changed, arrays["changed"][index])
        assert sum(claim["effect"]["planted_clean_answer_restored"]) == 1
        assert claim["effect"]["clean_accuracy"] < 0.75
        assert report["original_clean_accuracy"] == 1
