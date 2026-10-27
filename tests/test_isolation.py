import base64
import json
import os
import sys
from pathlib import Path

import pytest

from hidden_process_bench import isolated_worker, isolation
from hidden_process_bench.isolation import Limits, _unpack, run_isolated
from hidden_process_bench.submissions import file_manifest


def program(root: Path, code: str) -> Path:
    root.mkdir()
    (root / "main.py").write_text(code)
    return root


def test_real_sandbox_hides_private_files_environment_network_and_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = tmp_path / "private-truth"
    secret.write_text("do not disclose")
    monkeypatch.setenv("DISCOVERY_PRIVATE_SECRET", "never")
    source = program(
        tmp_path / "program",
        f"""import json, os, socket, importlib.util
from pathlib import Path
import numpy as np
assert not Path({str(secret)!r}).exists()
assert not Path("/home").exists()
assert "DISCOVERY_PRIVATE_SECRET" not in os.environ
assert importlib.util.find_spec("hidden_process_bench") is None
assert Path("/challenge/input.json").read_text() == "public"
try:
    socket.create_connection(("1.1.1.1", 443), timeout=0.1)
except OSError:
    pass
else:
    raise AssertionError("network escaped")
try:
    Path("/program/main.py").write_text("mutated")
except OSError:
    pass
else:
    raise AssertionError("source is writable")
Path("/output/nested").mkdir()
Path("/output/nested/result.txt").write_text("done")
print(json.dumps({{"sum": int(np.arange(4).sum()), "request": json.load(__import__("sys").stdin)}}))
""",
    )
    challenge = tmp_path / "challenge"
    challenge.mkdir()
    (challenge / "input.json").write_text("public")
    before = file_manifest(source)
    execution = run_isolated(
        source, "main.py", {"request": 3}, tmp_path / "run", challenge=challenge
    )
    assert execution.status == "success", (tmp_path / "run/stderr.txt").read_text()
    assert json.loads((tmp_path / "run/stdout.txt").read_text()) == {
        "sum": 6,
        "request": {"request": 3},
    }
    assert (tmp_path / "run/generated/nested/result.txt").read_text() == "done"
    assert file_manifest(source) == before
    receipt = json.loads((tmp_path / "run/execution.json").read_text())
    assert receipt["program_sha256"] == before
    assert receipt["challenge_sha256"] == file_manifest(challenge)


@pytest.mark.parametrize(
    ("code", "status", "limits"),
    [
        ('raise RuntimeError("preserved failure")', "failed", Limits()),
        ("import time; time.sleep(10)", "timeout", Limits(wall_seconds=0.4)),
        ('print("x" * 1000)', "failed", Limits(output_bytes=10)),
        ('from pathlib import Path; Path("/output/link").symlink_to("/usr")', "failed", Limits()),
        ('from pathlib import Path; assert not Path("/challenge").exists()', "success", Limits()),
        ('import os; os.write(1, b"invalid envelope"); os._exit(0)', "failed", Limits()),
    ],
)
def test_real_execution_failures_are_retained_without_fallback(
    tmp_path: Path, code: str, status: str, limits: Limits
) -> None:
    source = program(tmp_path / "program", code)
    result = run_isolated(source, "main.py", {}, tmp_path / "run", limits=limits)
    assert result.status == status, (tmp_path / "run/stderr.txt").read_text()
    assert (tmp_path / "run/execution.json").is_file()
    assert (tmp_path / "run/stderr.txt").is_file()


def test_missing_isolation_never_executes_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(isolation.shutil, "which", lambda _name: None)
    result = run_isolated(tmp_path / "absent", "main.py", {}, tmp_path / "run")
    assert result.status == "sandbox_unavailable"
    monkeypatch.setattr(isolation.sys, "platform", "other")
    result = run_isolated(tmp_path / "absent", "main.py", {}, tmp_path / "other")
    assert result.status == "sandbox_unavailable"


def test_entry_must_be_in_snapshot(tmp_path: Path) -> None:
    source = program(tmp_path / "program", "print(1)")
    with pytest.raises(ValueError, match="entry point"):
        run_isolated(source, "missing.py", {}, tmp_path / "run")


@pytest.mark.parametrize(
    "envelope",
    [
        [],
        {"stdout": 2, "files": {}},
        {"stdout": "", "files": []},
        {"stdout": "", "files": {str(i): "" for i in range(257)}},
        {"stdout": "", "files": {"../outside": ""}},
        {"stdout": "", "files": {"a": 2}},
        {"stdout": "", "files": {"a": "!"}},
        {"stdout": "", "files": {"a": base64.b64encode(b"large").decode()}},
        {"stdout": "large", "files": {}},
    ],
)
def test_output_envelope_validation(tmp_path: Path, envelope: object) -> None:
    wire = tmp_path / "wire"
    wire.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match=r"."):
        _unpack(wire, tmp_path, 3)


def test_worker_supports_normal_exit_and_directories(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "empty").mkdir()
    (output / "empty/a").write_text("hi")
    entry = tmp_path / "entry.py"
    entry.write_text('print("ok"); raise SystemExit(0)')
    result = isolated_worker.execute(entry, output, tmp_path / "capture", 100)
    assert result == {"stdout": "ok\n", "files": {"empty/a": "aGk="}}
    entry.write_text("raise SystemExit(2)")
    with pytest.raises(SystemExit):
        isolated_worker.execute(entry, output, tmp_path / "capture", 100)
    entry.write_text("raise SystemExit()")
    isolated_worker.execute(entry, output, tmp_path / "capture", 100)


def test_worker_enforces_all_output_bounds(tmp_path: Path) -> None:
    entry = tmp_path / "entry.py"
    entry.write_text("pass")
    output = tmp_path / "output"
    output.mkdir()
    (output / "link").symlink_to("/usr")
    with pytest.raises(ValueError, match="invalid"):
        isolated_worker.execute(entry, output, tmp_path / "capture", 20)
    (output / "link").unlink()
    os.mkfifo(output / "fifo")
    with pytest.raises(ValueError, match="special"):
        isolated_worker.execute(entry, output, tmp_path / "capture", 20)
    (output / "fifo").unlink()
    (output / "a").write_bytes(b"x" * 21)
    with pytest.raises(ValueError, match="byte/file"):
        isolated_worker.execute(entry, output, tmp_path / "capture", 20)
    (output / "a").unlink()
    entry.write_text('print("x" * 21)')
    with pytest.raises(ValueError, match="stdout"):
        isolated_worker.execute(entry, output, tmp_path / "capture", 20)
    entry.write_text("pass")
    for i in range(257):
        (output / str(i)).touch()
    with pytest.raises(ValueError, match="byte/file"):
        isolated_worker.execute(entry, output, tmp_path / "capture", 20)
    for file in output.iterdir():
        file.unlink()
    for i in range(513):
        (output / str(i)).mkdir()
    with pytest.raises(ValueError, match="excessive"):
        isolated_worker.execute(entry, output, tmp_path / "capture", 20)


def test_standalone_main_loads_only_public_import_paths(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["worker", "main.py", "20"])
    monkeypatch.setattr(sys, "path", list(sys.path))

    def fake_execute(entry: Path, output: Path, capture: Path, max_bytes: int) -> dict[str, object]:
        assert entry == Path("/program/main.py")
        assert output == Path("/output")
        assert capture == Path("/tmp/captured-stdout")
        assert max_bytes == 20
        assert sys.path[:2] == ["/deps", "/program"]
        return {"stdout": "ok", "files": {}}

    monkeypatch.setattr(isolated_worker, "execute", fake_execute)
    isolated_worker.main()
    assert json.loads(capsys.readouterr().out)["stdout"] == "ok"
