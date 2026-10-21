import json
import subprocess
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from hidden_process_bench.cli import main
from hidden_process_bench.config import RunConfig
from hidden_process_bench.scaling import load_sweep, run_sweep, sweep_plan, write_run_config


def test_run_config_roundtrip_and_cartesian_scaling_plan(tmp_path: Path) -> None:
    config = replace(RunConfig(), states=4, seed=42)
    path = tmp_path / "base.toml"
    write_run_config(path, config)
    assert RunConfig.load(path) == config
    specification = tmp_path / "grid.toml"
    specification.write_text('base = "base.toml"\n[grid]\nstates = [3, 4]\nwidth = [32, 64]\n')
    cases = load_sweep(specification, max_runs=4)
    assert [(case.states, case.width) for case in cases] == [(3, 32), (3, 64), (4, 32), (4, 64)]
    assert {case.seed for case in cases} == {42}
    plan = sweep_plan(cases, timeout=60)
    assert plan["runs"] == 4
    assert plan["maximum_subprocess_seconds"] == 240
    assert (
        plan["maximum_training_tokens"]
        == 4 * config.train_steps * config.batch_size * config.length
    )
    with pytest.raises(ValueError, match="exceeding"):
        load_sweep(specification, max_runs=3)
    with pytest.raises(TypeError, match="PositiveCount"):
        sweep_plan([], timeout=1)
    with pytest.raises(ValueError, match="positive"):
        sweep_plan(cases, timeout=0)


@pytest.mark.parametrize(
    "content",
    [
        'base = "missing.toml"',
        'base = "missing.toml"\ngrid = []',
        'base = "missing.toml"\n[grid]',
        'base = "missing.toml"\n[grid]\nmax_kl = [1]',
        'base = "missing.toml"\n[grid]\nstates = []',
        'base = "missing.toml"\n[grid]\nstates = 2',
        'base = "missing.toml"\n[grid]\nstates = [true]',
        'base = "missing.toml"\n[grid]\nstates = [1.5]',
    ],
)
def test_invalid_sweep_specification_is_rejected_before_work(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(content)
    with pytest.raises(ValueError, match=r"sweep|axis|axes"):
        load_sweep(path)


def test_real_subprocess_sweep_retains_unqualified_experiments(tmp_path: Path) -> None:
    case = RunConfig.load(Path("configs/smoke.toml"))
    output = tmp_path / "sweep"
    summary = run_sweep([case], output, timeout=60)
    assert summary["counts"] == {"unqualified": 1}
    record = json.loads((output / "cases.jsonl").read_text())
    assert record["status"] == "unqualified"
    assert record["training_steps"] == 2
    assert record["probabilities_mae"] is not None
    assert (output / "case-000/private/sampling.jsonl").exists()
    assert (output / "private/case-000.log").exists()
    assert RunConfig.load(output / "private/case-000.toml") == case
    assert (output / "cases.csv").read_text().startswith("case,status,returncode,")
    assert json.loads((output / "summary.json").read_text()) == summary
    with pytest.raises(FileExistsError):
        run_sweep([case], output)


def test_sweep_preserves_timeouts_crashes_and_partial_reports(tmp_path: Path) -> None:
    cases = [RunConfig()] * 2
    output = tmp_path / "failures"

    def failed_worker(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        assert kwargs["timeout"] == 1
        path = Path(command[command.index("--out") + 1])
        path.mkdir()
        (path / "report.json").write_text('{"incomplete":')
        if path.name == "case-000":
            raise subprocess.TimeoutExpired(command, 1)
        return subprocess.CompletedProcess(command, 1)

    with patch("hidden_process_bench.scaling.subprocess.run", side_effect=failed_worker):
        summary = run_sweep(cases, output, timeout=1)
    assert summary["counts"] == {"timeout": 1, "failed": 1}
    records = [json.loads(line) for line in (output / "cases.jsonl").read_text().splitlines()]
    assert [record["status"] for record in records] == ["timeout", "failed"]
    assert records[0]["returncode"] is None
    assert records[1]["returncode"] == 1
    assert all(record["heldout_kl"] is None for record in records)
    assert (output / "case-000/report.json").read_text() == '{"incomplete":'


def test_sweep_does_not_call_a_missing_report_a_success(tmp_path: Path) -> None:
    with patch(
        "hidden_process_bench.scaling.subprocess.run",
        return_value=subprocess.CompletedProcess([], 0),
    ):
        result = run_sweep([RunConfig()], tmp_path / "missing")
    assert result["counts"] == {"failed": 1}


def test_sweep_classifies_valid_qualified_report(tmp_path: Path) -> None:
    def qualified_worker(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        assert kwargs["check"] is False
        path = Path(command[command.index("--out") + 1])
        path.mkdir()
        (path / "report.json").write_text('{"benchmark_eligible": true}')
        return subprocess.CompletedProcess(command, 0)

    with patch("hidden_process_bench.scaling.subprocess.run", side_effect=qualified_worker):
        result = run_sweep([RunConfig()], tmp_path / "passed")
    assert result["counts"] == {"qualified": 1}


def test_sweep_cli_plans_without_writing_and_propagates_failure_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "not-created"
    assert main(["sweep", "--dry-run", "--out", str(output)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["runs"] == 4
    assert not output.exists()
    with pytest.raises(SystemExit) as error:
        main(["sweep"])
    assert error.value.code == 2
    for counts, code in [({"qualified": 4}, 0), ({"unqualified": 4}, 2)]:
        with patch("hidden_process_bench.cli.run_sweep", return_value={"counts": counts}):
            assert main(["sweep", "--out", str(output)]) == code
