"""Standalone sandbox worker: standard library only, no evaluator imports.

The host mounts this file by itself. It serializes bounded output from tmpfs so
submitted code never gets a writable host directory, even during extraction.
"""

import base64
import contextlib
import json
import runpy
import sys
from pathlib import Path


def execute(entry: Path, output: Path, capture: Path, max_bytes: int) -> dict[str, object]:
    with capture.open("w") as stream, contextlib.redirect_stdout(stream):
        try:
            runpy.run_path(str(entry), run_name="__main__")
        except SystemExit as exit_status:
            if exit_status.code not in (None, 0):
                raise
    total = capture.stat().st_size
    files = {}
    for count, path in enumerate(output.rglob("*")):
        if count >= 512 or path.is_symlink():
            raise ValueError("invalid or excessive generated files")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("special output files are forbidden")
        total += path.stat().st_size
        if total > max_bytes or len(files) >= 256:
            raise ValueError("generated output exceeds byte/file limit")
        files[path.relative_to(output).as_posix()] = base64.b64encode(path.read_bytes()).decode()
    if total > max_bytes:
        raise ValueError("stdout exceeds byte limit")
    return {"stdout": capture.read_text(errors="replace"), "files": files}


def main() -> None:
    # -I -S prevents site initialization and importing the host's editable project.
    sys.path[:0] = ["/deps", "/program"]
    entry = Path("/program") / sys.argv[1]
    max_bytes = int(sys.argv[2])
    sys.argv = [str(entry)]
    result = execute(entry, Path("/output"), Path("/tmp/captured-stdout"), max_bytes)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
