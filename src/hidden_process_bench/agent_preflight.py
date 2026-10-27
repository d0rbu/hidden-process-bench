"""Capture a synthetic request and reject context/tool leaks before agent trials."""

import gzip
import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import cast, override

from hidden_process_bench.agent_client import INSTRUCTIONS, invoke
from hidden_process_bench.experiment import write_json

PERMISSIONS = """<permissions instructions>
Filesystem sandboxing defines which files can be read or written. `sandbox_mode` is `read-only`: The sandbox only permits reading files. Network access is restricted.
Approval policy is currently never. Do not provide the `sandbox_permissions` for any reason, commands will be rejected.
</permissions instructions>"""

ENVIRONMENT = """<environment_context>
  <cwd>{directory}</cwd>
  <shell>bash</shell>
  <current_date>{date}</current_date>
  <timezone>Etc/UTC</timezone>
  <filesystem><workspace_roots><root>{directory}</root></workspace_roots><permission_profile type="managed"><file_system type="restricted"><entry access="read"><special>:root</special></entry></file_system></permission_profile></filesystem>
</environment_context>"""

# Exact definition from stock Codex 0.164.0; changing its description is a context change.
INERT_TOOLS_SHA256 = "528216bf775bf0170a86ac9007fcd3cd74a944ef56b68f2deb379a6972168a98"


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("context item must be a JSON object")
    return cast(dict[str, object], value)


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("context must be a list")
    return cast(list[object], value)


def _environment(text: str) -> bool:
    directory = re.search(r"<cwd>(/tmp/sequence-analysis-[a-z0-9_]+)</cwd>", text)
    date = re.search(r"<current_date>([0-9]{4}-[0-9]{2}-[0-9]{2})</current_date>", text)
    return bool(
        directory and date and text == ENVIRONMENT.format(directory=directory[1], date=date[1])
    )


def audit_request(payload: object, prompt: str) -> None:
    payload = _object(payload)
    if payload.get("tools"):
        raise ValueError("unexpected top-level tools")
    serialized = json.dumps(payload).lower()
    if any(
        term in serialized
        for term in (
            "hidden-process-bench",
            "hmm",
            "stack_rule",
            "lag_rule",
            "process.py",
            "<subagents>",
            "memory_summary",
        )
    ):
        raise ValueError("non-neutral context")
    own = prompts = permissions = environments = capabilities = 0
    for raw in _list(payload.get("input")):
        item = _object(raw)
        if item.get("type") == "additional_tools":
            # This client exposes an unavailable question tool even in codex exec.
            # It has no data access; native tool activity is rejected by invoke().
            if set(item) != {"type", "id", "role", "tools"} or item["role"] != "developer":
                raise ValueError("unexpected capability metadata")
            tools = json.dumps(item["tools"], sort_keys=True, separators=(",", ":"))
            if hashlib.sha256(tools.encode()).hexdigest() != INERT_TOOLS_SHA256:
                raise ValueError("unexpected native tool capability")
            capabilities += 1
            continue
        if item.get("type") != "message" or set(item) - {"type", "id", "role", "content"}:
            raise ValueError("unexpected initial context item")
        for raw_content in _list(item["content"]):
            content = _object(raw_content)
            text = content.get("text")
            if set(content) != {"type", "text"} or not isinstance(text, str):
                raise ValueError("unexpected context content")
            role = item["role"]
            if role == "developer" and text in (INSTRUCTIONS, INSTRUCTIONS.rstrip("\n")):
                own += 1
            elif role == "user" and text == prompt:
                prompts += 1
            elif role == "developer" and text == PERMISSIONS:
                permissions += 1
            elif role == "user" and _environment(text):
                environments += 1
            else:
                raise ValueError("unexpected model-visible instructions")
    if (own, prompts, permissions, environments, capabilities) != (1, 1, 1, 1, 1):
        raise ValueError("missing or duplicated neutral prompt")


def preflight(binary: Path, catalog: Path, prompt: str, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=False)
    requests: list[dict[str, object]] = []

    class Capture(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            size = int(self.headers["Content-Length"])
            if not 0 < size <= 4_000_000:
                self.send_error(413)
                return
            raw = self.rfile.read(size)
            if self.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
            payload = json.loads(raw)
            requests.append(payload)
            write_json(output / f"request-{len(requests)}.json", payload)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for event in (
                {
                    "type": "response.created",
                    "response": {
                        "id": "resp_preflight",
                        "object": "response",
                        "status": "in_progress",
                        "output": [],
                    },
                },
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_preflight",
                        "object": "response",
                        "status": "completed",
                        "output": [],
                        "usage": {"input_tokens": 1, "output_tokens": 0, "total_tokens": 1},
                    },
                },
            ):
                self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())

        @override
        def log_message(self, format: str, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Capture)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        invoke(
            binary,
            catalog,
            prompt,
            output / "client",
            capture_url=f"http://127.0.0.1:{server.server_port}/v1",
            timeout=30,
        )
        if not requests:
            raise ValueError("client did not reach the capture endpoint")
        for payload in requests:
            audit_request(payload, prompt)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    receipt = {
        "status": "passed",
        "real_discovery_model_calls": 0,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
        "instructions_sha256": hashlib.sha256(INSTRUCTIONS.encode()).hexdigest(),
        "allowed_native_tools": [
            "request_user_input; unavailable in noninteractive mode; activity rejected"
        ],
        "requests": len(requests),
    }
    write_json(output / "preflight.json", receipt)
    return receipt
