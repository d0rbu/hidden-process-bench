"""Fresh, tool-restricted Codex calls; credentials stay with the trusted client.

The model returns JSON code for a separate numerical sandbox. It has no native
filesystem, command, network, delegation, memory, skill or connector tools.
"""

import json
import os
import signal
import subprocess
import tempfile
from contextlib import suppress
from pathlib import Path

from hidden_process_bench.experiment import write_json

INSTRUCTIONS = """You analyze an anonymous neural sequence model. Infer a compact executable
account of the sequence-generating mechanism represented by its predictions. Choose
the representation yourself. No generating family or latent dimension is provided.
Use only evidence obtained from the supplied challenge through the numerical sandbox.
Do not use native tools or ask the user questions. Return only the required JSON.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["run", "submit"]},
        "code": {"type": "string"},
        "note": {"type": "string"},
    },
    "required": ["action", "code", "note"],
    "additionalProperties": False,
}


def restricted_catalog(source: Path, model: str) -> dict[str, object]:
    payload = json.loads(source.read_text())
    selected = next((item.copy() for item in payload["models"] if item["slug"] == model), None)
    if selected is None:
        raise ValueError("requested model absent from catalog")
    selected.update(
        {
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
    )
    return {"models": [selected]}


def _toml(value: object) -> str:
    if isinstance(value, dict):
        return "{" + ", ".join(str(k) + "=" + _toml(v) for k, v in value.items()) + "}"
    return json.dumps(value)


def client_command(
    binary: Path,
    catalog: Path,
    workspace: Path,
    model: str,
    effort: str,
    *,
    capture_url: str | None = None,
) -> list[str]:
    """A sparse client filesystem retains account auth but no project or home data."""
    settings: dict[str, object] = {
        "model_instructions_file": str(workspace / "instructions.txt"),
        "model_catalog_json": str(workspace / "catalog.json"),
        "model_reasoning_effort": effort,
        "project_doc_max_bytes": 0,
        "approval_policy": "never",
        "web_search": "disabled",
        "agents.enabled": False,
        "features.code_mode.enabled": False,
        "skills.bundled.enabled": False,
        "skills.include_instructions": False,
        "memories.use_memories": False,
        "memories.generate_memories": False,
        "suppress_unstable_features_warning": True,
    }
    if capture_url is not None:
        settings.update(
            {
                "model_provider": "capture",
                "model_providers.capture": {
                    "name": "capture",
                    "base_url": capture_url,
                    "requires_openai_auth": False,
                    "wire_api": "responses",
                    "request_max_retries": 0,
                    "stream_max_retries": 0,
                },
            }
        )
    (workspace / "instructions.txt").write_text(INSTRUCTIONS)
    (workspace / "catalog.json").write_bytes(catalog.read_bytes())
    write_json(workspace / "schema.json", SCHEMA)
    command = [
        "prlimit",
        "--fsize=16777216",
        "--core=0",
        "bwrap",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
    ]
    for path in ("/usr", "/lib", "/lib64", "/etc/ssl/certs", "/etc/resolv.conf", "/etc/hosts"):
        command += ["--ro-bind", path, path]
    auth = Path.home() / ".codex/auth.json"
    command += [
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        "/home",
        "--dir",
        str(auth.parent),
        "--ro-bind",
        str(auth),
        str(auth),
        "--ro-bind",
        str(binary),
        "/client",
        "--ro-bind",
        str(workspace),
        str(workspace),
        "--chdir",
        str(workspace),
        "/client",
        "exec",
        "--ignore-user-config",
        "--ignore-rules",
        "--ephemeral",
        "--skip-git-repo-check",
        "--json",
        "--sandbox",
        "read-only",
        "--cd",
        str(workspace),
        "--model",
        model,
        "--output-schema",
        str(workspace / "schema.json"),
    ]
    for key, value in settings.items():
        command += ["-c", key + "=" + _toml(value)]
    for feature in (
        "shell_tool",
        "view_image",
        "apps",
        "plugins",
        "remote_plugin",
        "memories",
        "shell_snapshot",
        "code_mode_host",
        "skill_search",
        "goals",
        "sleep_tool",
        "image_generation",
        "browser_use",
        "browser_use_external",
        "computer_use",
        "hooks",
    ):
        command += ["--disable", feature]
    return [*command, "--enable", "skip_host_skill_discovery", "-"]


def invoke(
    binary: Path,
    catalog: Path,
    prompt: str,
    output: Path,
    *,
    model: str = "gpt-6.1-sol",
    effort: str = "xhigh",
    timeout: float = 180,
    capture_url: str | None = None,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=False)
    (output / "prompt.txt").write_text(prompt)
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "HOME", "USER", "LOGNAME", "LANG", "TERM"}
    }
    with tempfile.TemporaryDirectory(prefix="sequence-analysis-") as temporary:
        command = client_command(
            binary, catalog, Path(temporary), model, effort, capture_url=capture_url
        )
        with (
            (output / "events.jsonl").open("w") as stdout,
            (output / "stderr.txt").open("w") as stderr,
        ):
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                text=True,
                stdout=stdout,
                stderr=stderr,
                env=environment,
                start_new_session=True,
            )
            try:
                process.communicate(prompt, timeout=timeout)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                raise
    if process.returncode:
        raise ValueError("model client failed; see retained stderr")
    messages = []
    for line in (output / "events.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event.get("type") == "item.completed":
            item = event["item"]
            if item["type"] == "agent_message":
                messages.append(item["text"])
            elif item["type"] != "reasoning":
                raise ValueError("unexpected native tool activity: " + item["type"])
        if event.get("type") in ("turn.failed", "error"):
            raise ValueError("model turn failed; see retained events")
    if capture_url is not None:
        return {}
    if not messages:
        raise ValueError("model client produced no final JSON action")
    answer = json.loads(messages[-1])
    if (
        not isinstance(answer, dict)
        or set(answer) != {"action", "code", "note"}
        or answer["action"] not in ("run", "submit")
        or not isinstance(answer["code"], str)
        or len(answer["code"]) > 100_000
        or not isinstance(answer["note"], str)
    ):
        raise ValueError("invalid model action")
    return answer
