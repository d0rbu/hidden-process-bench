"""Linux namespace and cgroup isolation for family-neutral executable methods.

Only anonymous public snapshots are mounted. Submitted files are returned through
a bounded pipe; the executable never gets a writable host directory.
"""

import base64
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal
from uuid import uuid4

import numpy as np

from hidden_process_bench.config import check_count, check_positive
from hidden_process_bench.experiment import write_json
from hidden_process_bench.submissions import copy_bundle, relative_file


@dataclass(frozen=True)
class Limits:
    wall_seconds: float = 30
    cpu_seconds: int = 20
    memory_mb: int = 1024
    output_bytes: int = 8_000_000

    def __post_init__(self) -> None:
        check_positive(self.wall_seconds)
        for value in (self.cpu_seconds, self.memory_mb, self.output_bytes):
            check_count(value)


@dataclass(frozen=True)
class Execution:
    status: Literal["success", "failed", "timeout", "sandbox_unavailable"]
    returncode: int | None
    seconds: float
    scope: str


def _unpack(wire: Path, output: Path, max_bytes: int) -> None:
    payload = json.loads(wire.read_text())
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("stdout"), str)
        or not isinstance(payload.get("files"), dict)
        or len(payload["files"]) > 256
    ):
        raise ValueError("invalid sandbox output envelope")
    total = len(payload["stdout"].encode())
    files = []
    for name, encoded in payload["files"].items():
        path = relative_file(name)
        if not isinstance(encoded, str):
            raise ValueError("generated files must use base64 strings")
        data = base64.b64decode(encoded, validate=True)
        total += len(data)
        if total > max_bytes:
            raise ValueError("sandbox output exceeds byte budget")
        files.append((path, data))
    if total > max_bytes:
        raise ValueError("sandbox stdout exceeds byte budget")
    for path, data in files:
        target = output / "generated" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (output / "stdout.txt").write_text(payload["stdout"])


def run_isolated(
    program: Path,
    entry: str,
    request: dict[str, object],
    output: Path,
    *,
    challenge: Path | None = None,
    limits: Limits | None = None,
) -> Execution:
    """Run JSON-stdin code with only stdlib/NumPy; no network, home, project or truth.

    Missing isolation fails closed. Inference mounts no challenge or teacher weights.
    Receipts retain logs, limits, source hashes and failed/incomplete execution status.
    """
    limits = Limits() if limits is None else limits
    entry_path = relative_file(entry)
    output.mkdir(parents=True, exist_ok=False)
    (output / "generated").mkdir()
    (output / "stdout.txt").touch()
    scope = "discovery-" + uuid4().hex + ".scope"
    started = perf_counter()
    missing = [
        name
        for name in ("bwrap", "prlimit", "systemd-run", "systemctl")
        if shutil.which(name) is None
    ]
    if sys.platform != "linux" or missing:
        (output / "stderr.txt").write_text(
            "Linux isolation unavailable: " + ", ".join(missing) + "\n"
        )
        result = Execution("sandbox_unavailable", None, perf_counter() - started, scope)
        write_json(output / "execution.json", {**asdict(result), "limits": asdict(limits)})
        return result
    with tempfile.TemporaryDirectory(prefix="process-runtime-") as temp:
        staging = Path(temp)
        source_hashes = copy_bundle(program, staging / "program")
        if entry_path.as_posix() not in source_hashes:
            raise ValueError("entry point is not in the supplied program bundle")
        challenge_hashes = (
            copy_bundle(challenge, staging / "challenge", 64_000_000)
            if challenge is not None
            else {}
        )
        shutil.copyfile(Path(__file__).with_name("isolated_worker.py"), staging / "worker.py")
        command = [
            "systemd-run",
            "--user",
            "--scope",
            "--quiet",
            "--collect",
            "--unit=" + scope,
            "--property=MemoryMax=" + str(limits.memory_mb * 1024 * 1024),
            "--property=MemorySwapMax=0",
            "--property=TasksMax=32",
            "--property=CPUQuota=200%",
            "prlimit",
            "--cpu=" + str(limits.cpu_seconds),
            "--fsize=" + str(8 * limits.output_bytes + 65536),
            "--nofile=64",
            "--core=0",
            "bwrap",
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--cap-drop",
            "ALL",
            "--clearenv",
            "--hostname",
            "process-runtime",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/lib",
            "/lib",
            "--ro-bind",
            sys.base_prefix,
            "/runtime",
            "--ro-bind",
            str(Path(np.__file__).parent),
            "/deps/numpy",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--tmpfs",
            "/output",
            "--ro-bind",
            str(staging / "program"),
            "/program",
            "--ro-bind",
            str(staging / "worker.py"),
            "/runner.py",
            "--chdir",
            "/program",
        ]
        if Path("/lib64").exists():
            command += ["--ro-bind", "/lib64", "/lib64"]
        numpy_libraries = Path(np.__file__).parent.parent / "numpy.libs"
        if numpy_libraries.exists():
            command += ["--ro-bind", str(numpy_libraries), "/deps/numpy.libs"]
        if challenge is not None:
            command += ["--ro-bind", str(staging / "challenge"), "/challenge"]
        for name, value in {
            "PATH": "/runtime/bin:/usr/bin",
            "HOME": "/tmp",
            "TMPDIR": "/tmp",
            "LANG": "C.UTF-8",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }.items():
            command += ["--setenv", name, value]
        command += [
            "/runtime/bin/python3.13",
            "-I",
            "-S",
            "/runner.py",
            entry_path.as_posix(),
            str(limits.output_bytes),
        ]
        data = json.dumps(request, allow_nan=False).encode()
        with (
            (output / "wire.json").open("wb") as stdout,
            (output / "stderr.txt").open("wb") as stderr,
        ):
            process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, start_new_session=True
            )
            try:
                process.communicate(data, timeout=limits.wall_seconds)
                status = "success" if process.returncode == 0 else "failed"
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                status = "timeout"
            finally:
                subprocess.run(
                    ["systemctl", "--user", "kill", "--kill-who=all", "--signal=KILL", scope],
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
        if status == "success":
            try:
                _unpack(output / "wire.json", output, limits.output_bytes)
            except (ValueError, OSError, RecursionError) as error:
                status = "failed"
                with (output / "stderr.txt").open("a") as stream:
                    stream.write("\nRejected output: " + str(error) + "\n")
        result = Execution(status, process.returncode, perf_counter() - started, scope)
        write_json(
            output / "execution.json",
            {
                **asdict(result),
                "limits": asdict(limits),
                "program_sha256": source_hashes,
                "challenge_sha256": challenge_hashes,
                "request_bytes": len(data),
                "runtime_python": sys.version.split()[0],
                "runtime_numpy": np.__version__,
            },
        )
        return result
