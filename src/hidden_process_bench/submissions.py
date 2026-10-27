"""Family-neutral executable submissions, frozen before evaluator feedback."""

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from hidden_process_bench.config import check_count
from hidden_process_bench.experiment import write_json


def relative_file(value: object) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise ValueError("file names must be nonempty relative POSIX paths")
    path = Path(value)
    if not path.parts or path.is_absolute() or ".." in path.parts or len(path.parts) > 8:
        raise ValueError("file names must remain inside the submission")
    return path


def file_manifest(root: Path, *, max_bytes: int = 16_000_000) -> dict[str, str]:
    """Reject links/special files; bound the entire executable and its parameters."""
    check_count(max_bytes)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("bundle must be a real directory")
    result = {}
    total = 0
    for count, path in enumerate(root.rglob("*")):
        if count >= 512:
            raise ValueError("executable bundle exceeds its directory budget")
        name = path.relative_to(root)
        relative_file(name.as_posix())
        if path.is_symlink():
            raise ValueError("links are forbidden in executable bundles")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("special files are forbidden in executable bundles")
        total += path.stat().st_size
        if total > max_bytes or len(result) >= 256:
            raise ValueError("executable bundle exceeds its file/byte budget")
        result[name.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(result.items()))


def copy_bundle(source: Path, destination: Path, max_bytes: int = 16_000_000) -> dict[str, str]:
    before = file_manifest(source, max_bytes=max_bytes)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    shutil.copytree(source, destination)
    after = file_manifest(destination, max_bytes=max_bytes)
    if before != after:
        raise ValueError("source changed while its executable bundle was frozen")
    return after


@dataclass(frozen=True)
class FrozenSubmission:
    path: Path
    receipt_sha256: str


def freeze_submission(source: Path, output: Path) -> FrozenSubmission:
    """The method chooses the representation; no process family or order is required."""
    files = file_manifest(source)
    if "submission.json" not in files:
        raise ValueError("submission.json is required")
    payload = json.loads((source / "submission.json").read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported executable submission schema")
    required = {
        "schema_version",
        "predictor",
        "mechanism",
        "hypothesis",
        "assumptions",
        "resources",
    }
    if set(payload) != required:
        raise ValueError(
            "submission needs predictor, mechanism, hypothesis, assumptions and resources"
        )
    for field in ("hypothesis", "assumptions"):
        if not isinstance(payload[field], str) or not payload[field].strip():
            raise ValueError(f"{field} must be a nonempty explanation")
    if not isinstance(payload["resources"], dict):
        raise ValueError("resources must declare fitting work and access used")
    for field in ("predictor", "mechanism"):
        name = relative_file(payload[field]).as_posix()
        if name not in files:
            raise ValueError(f"{field} must name a submitted file")
    if not payload["predictor"].endswith(".py"):
        raise ValueError("the prototype predictor must be a Python entry point")
    if "frozen.json" in files:
        raise ValueError("frozen.json is reserved for evaluator receipts")
    frozen_files = copy_bundle(source, output)
    receipt = {
        "schema_version": 1,
        "files": frozen_files,
        "submission": payload,
        "structure_assessment": "pending_independent_review",
    }
    write_json(output / "frozen.json", receipt)
    file_manifest(output)  # Include the receipt itself in the runnable bundle budget.
    return FrozenSubmission(
        output, hashlib.sha256((output / "frozen.json").read_bytes()).hexdigest()
    )


def verify_submission(frozen: FrozenSubmission) -> dict[str, object]:
    root = frozen.path
    raw = (root / "frozen.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != frozen.receipt_sha256:
        raise ValueError("frozen receipt differs from its registered hash")
    receipt = json.loads(raw)
    if not isinstance(receipt, dict):
        raise ValueError("invalid frozen receipt")
    actual = file_manifest(root)
    actual.pop("frozen.json")
    if receipt.get("schema_version") != 1 or actual != receipt.get("files"):
        raise ValueError("frozen executable submission was modified")
    return receipt
