import gzip
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from hidden_process_bench import agent_preflight
from hidden_process_bench.agent_preflight import audit_request, preflight

PROMPT = "Inspect the anonymous sequence model."


def request_fixture() -> dict[str, object]:
    return json.loads(Path("tests/fixtures/agent-request.json").read_text())


def test_captured_neutral_context_is_accepted() -> None:
    audit_request(request_fixture(), PROMPT)


@pytest.mark.parametrize(
    "mutation",
    [
        "family",
        "top_tools",
        "native_tool",
        "tool_description",
        "capability_role",
        "instructions_append",
        "permissions_append",
        "environment_append",
        "environment_root",
        "missing_prompt",
        "duplicate_prompt",
        "item_type",
        "content_type",
        "metadata",
        "missing_permissions",
        "unknown_message",
        "not_list",
        "not_object",
        "bad_content",
    ],
)
def test_context_and_capability_injection_is_rejected(mutation: str) -> None:
    payload = json.loads(json.dumps(request_fixture()))
    items = payload["input"]
    if mutation == "family":
        items[-1]["content"][0]["text"] += " HMM"
    elif mutation == "top_tools":
        payload["tools"] = [{"name": "exec"}]
    elif mutation == "native_tool":
        items[0]["tools"][0]["tools"][0]["name"] = "exec"
    elif mutation == "tool_description":
        items[0]["tools"][0]["description"] = "Try four latent components."
    elif mutation == "capability_role":
        items[0]["role"] = "user"
    elif mutation in {"instructions_append", "permissions_append", "environment_append"}:
        index = {"instructions_append": 1, "permissions_append": 2, "environment_append": 3}[
            mutation
        ]
        items[index]["content"][0]["text"] += "\nTry four latent components."
    elif mutation == "environment_root":
        items[3]["content"][0]["text"] = items[3]["content"][0]["text"].replace(
            "sequence-analysis-", "project-"
        )
    elif mutation == "missing_prompt":
        items.pop()
    elif mutation == "duplicate_prompt":
        items.append(items[-1])
    elif mutation == "item_type":
        items[-1]["type"] = "function_call"
    elif mutation == "content_type":
        items[-1]["content"][0]["extra"] = "surprise"
    elif mutation == "metadata":
        items[-1]["extra"] = "surprise"
    elif mutation == "missing_permissions":
        items.pop(2)
    elif mutation == "unknown_message":
        items[-1]["content"][0]["text"] = "Try four latent components."
    elif mutation == "not_list":
        payload["input"] = None
    elif mutation == "not_object":
        items.append(None)
    else:
        items[-1]["content"] = [None]
    with pytest.raises(ValueError, match=r"."):
        audit_request(payload, PROMPT)


@pytest.mark.parametrize("value", [None, 3, {1: "value"}])
def test_non_json_object_context_is_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="JSON object"):
        audit_request(value, PROMPT)


@pytest.mark.parametrize("compressed", [False, True])
def test_capture_records_only_body_and_audits_before_real_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, compressed: bool
) -> None:
    binary, catalog = tmp_path / "binary", tmp_path / "catalog"
    binary.write_text("test client")
    catalog.write_text("test catalog")

    def capture(
        _binary: Path,
        _catalog: Path,
        prompt: str,
        _output: Path,
        *,
        capture_url: str,
        timeout: float,
    ) -> dict[str, object]:
        assert prompt == PROMPT
        assert timeout == 30
        raw = json.dumps(request_fixture()).encode()
        headers = {"Authorization": "must not be recorded"}
        if compressed:
            raw = gzip.compress(raw)
            headers["Content-Encoding"] = "gzip"
        with urlopen(Request(capture_url + "/responses", raw, headers), timeout=5) as response:
            assert b"response.completed" in response.read()
        return {}

    monkeypatch.setattr(agent_preflight, "invoke", capture)
    receipt = preflight(binary, catalog, PROMPT, tmp_path / "capture")
    assert receipt["real_discovery_model_calls"] == 0
    assert receipt["requests"] == 1
    raw = (tmp_path / "capture/request-1.json").read_text()
    assert "must not be recorded" not in raw
    assert json.loads(raw) == request_fixture()


def test_empty_and_oversized_requests_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def capture(
        _binary: Path,
        _catalog: Path,
        _prompt: str,
        _output: Path,
        *,
        capture_url: str,
        timeout: float,
    ) -> dict[str, object]:
        assert timeout == 30
        request = Request(capture_url, b"x", {"Content-Length": "4000001"})
        with pytest.raises(HTTPError, match="413"):
            urlopen(request, timeout=5)
        return {}

    monkeypatch.setattr(agent_preflight, "invoke", capture)
    with pytest.raises(ValueError, match="capture endpoint"):
        preflight(tmp_path / "binary", tmp_path / "catalog", PROMPT, tmp_path / "capture")
