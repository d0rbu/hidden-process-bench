import json
import subprocess
from pathlib import Path

import pytest

from hidden_process_bench.agent_client import (
    INSTRUCTIONS,
    SCHEMA,
    client_command,
    invoke,
    restricted_catalog,
)


def test_catalog_preserves_model_identity_but_removes_host_tools(tmp_path: Path) -> None:
    source = tmp_path / "catalog.json"
    source.write_text(json.dumps({"models": [{"slug": "chosen", "tool_mode": "code_mode_only"}]}))
    result = restricted_catalog(source, "chosen")
    assert result == {
        "models": [
            {
                "slug": "chosen",
                "tool_mode": None,
                "apply_patch_tool_type": None,
                "experimental_supported_tools": [],
                "node_repl_disabled": True,
                "supports_search_tool": False,
                "base_instructions": INSTRUCTIONS,
                "model_messages": None,
                "include_skills_usage_instructions": False,
                "include_plugin_usage_instructions": False,
                "include_apps_usage_instructions": False,
            }
        ]
    }
    with pytest.raises(ValueError, match="absent"):
        restricted_catalog(source, "missing")


def test_command_has_sparse_mounts_and_neutral_context(tmp_path: Path) -> None:
    workspace = tmp_path / "work"
    workspace.mkdir()
    catalog = tmp_path / "catalog"
    catalog.write_text("catalog")
    command = client_command(Path("/chosen-client"), catalog, workspace, "chosen", "high")
    assert "agents.enabled=false" in command
    assert "features.code_mode.enabled=false" in command
    assert "--ignore-rules" in command
    assert "--ignore-user-config" in command
    assert 'web_search="disabled"' in command
    assert "--unshare-pid" in command
    assert "--unshare-net" not in command
    assert str(Path.cwd()) not in command
    assert (workspace / "instructions.txt").read_text() == INSTRUCTIONS
    assert json.loads((workspace / "schema.json").read_text()) == SCHEMA
    assert (workspace / "catalog.json").read_text() == "catalog"
    captured = client_command(
        Path("/chosen-client"),
        catalog,
        workspace,
        "chosen",
        "high",
        capture_url="http://127.0.0.1:1234/v1",
    )
    assert 'model_provider="capture"' in captured


def fake_client(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    events: list[dict[str, object]],
    *,
    exitcode: int = 0,
    delay: bool = False,
) -> tuple[Path, Path]:
    home = root / "home"
    (home / ".codex").mkdir(parents=True)
    (home / ".codex/auth.json").write_text("synthetic account fixture")
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.setenv("PRIVATE_GENERATOR_SECRET", "hidden")
    secret = root / "private-generator"
    secret.write_text("hidden")
    binary = root / "client"
    code = f"""#!/usr/bin/python3
import json, os, sys, time
from pathlib import Path
assert not Path({str(secret)!r}).exists()
assert 'PRIVATE_GENERATOR_SECRET' not in os.environ
assert not Path({str(Path.cwd())!r}).exists()
assert Path({str(home / ".codex/auth.json")!r}).read_text() == 'synthetic account fixture'
sys.stdin.read()
for event in {events!r}:
    print(json.dumps(event), flush=True)
if {delay!r}: time.sleep(20)
sys.exit({exitcode})
"""
    binary.write_text(code)
    binary.chmod(0o700)
    catalog = root / "catalog"
    catalog.write_text("{}")
    return binary, catalog


def test_real_client_namespace_and_json_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answer = {"action": "run", "code": "print(3)", "note": "inspect"}
    events: list[dict[str, object]] = [
        {"type": "item.completed", "item": {"type": "reasoning"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(answer)}},
    ]
    binary, catalog = fake_client(tmp_path, monkeypatch, events)
    assert invoke(binary, catalog, "neutral prompt", tmp_path / "output") == answer
    assert (tmp_path / "output/prompt.txt").read_text() == "neutral prompt"


@pytest.mark.parametrize(
    ("event", "message"),
    [
        ({"type": "item.completed", "item": {"type": "command_execution"}}, "native tool"),
        ({"type": "turn.failed"}, "turn failed"),
        ({"type": "error"}, "turn failed"),
        ({"type": "turn.completed"}, "no final"),
    ],
)
def test_native_activity_and_failed_or_empty_turns_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, event: dict[str, object], message: str
) -> None:
    binary, catalog = fake_client(tmp_path, monkeypatch, [event])
    with pytest.raises(ValueError, match=message):
        invoke(binary, catalog, "neutral", tmp_path / "output")
    assert (tmp_path / "output/events.jsonl").is_file()


@pytest.mark.parametrize(
    "answer",
    [
        [],
        {},
        {"action": "unknown", "code": "", "note": ""},
        {"action": "run", "code": 1, "note": ""},
        {"action": "run", "code": "x" * 100_001, "note": ""},
        {"action": "run", "code": "", "note": None},
    ],
)
def test_model_action_schema_is_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: object
) -> None:
    events: list[dict[str, object]] = [
        {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(answer)}}
    ]
    binary, catalog = fake_client(tmp_path, monkeypatch, events)
    with pytest.raises(ValueError, match="invalid model action"):
        invoke(binary, catalog, "neutral", tmp_path / "output")


def test_client_error_timeout_and_capture_are_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary, catalog = fake_client(tmp_path, monkeypatch, [], exitcode=2)
    with pytest.raises(ValueError, match="client failed"):
        invoke(binary, catalog, "neutral", tmp_path / "failed")
    binary.write_text("#!/usr/bin/python3\nimport time\ntime.sleep(20)\n")
    with pytest.raises(subprocess.TimeoutExpired):
        invoke(binary, catalog, "neutral", tmp_path / "timeout", timeout=0.2)
    binary.write_text('#!/usr/bin/python3\nprint(\'{"type":"turn.completed"}\')\n')
    assert (
        invoke(binary, catalog, "neutral", tmp_path / "capture", capture_url="http://unused") == {}
    )
