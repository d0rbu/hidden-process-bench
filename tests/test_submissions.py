import hashlib
import json
import os
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hidden_process_bench import submissions
from hidden_process_bench.experiment import write_json
from hidden_process_bench.submissions import (
    FrozenSubmission,
    copy_bundle,
    file_manifest,
    freeze_submission,
    relative_file,
    verify_submission,
)


def make_submission(path: Path, code: str = 'print("predictor")') -> Path:
    path.mkdir()
    (path / "predict.py").write_text(code)
    (path / "mechanism.txt").write_text("A constant distribution, no hidden variables.")
    write_json(
        path / "submission.json",
        {
            "schema_version": 1,
            "predictor": "predict.py",
            "mechanism": "mechanism.txt",
            "hypothesis": "Independent observations with constant probabilities.",
            "assumptions": "This is only a declared baseline, not a discovery claim.",
            "resources": {"queries": 0},
        },
    )
    return path


def test_generic_submission_freezes_code_and_receipt_independently(tmp_path: Path) -> None:
    source = make_submission(tmp_path / "source")
    frozen = freeze_submission(source, tmp_path / "frozen")
    receipt = verify_submission(frozen)
    assert receipt["structure_assessment"] == "pending_independent_review"
    assert receipt["files"] == file_manifest(source)
    (frozen.path / "predict.py").write_text("changed")
    with pytest.raises(ValueError, match="modified"):
        verify_submission(frozen)
    # Even editing the manifest along with the predictor cannot change the registered hash.
    (frozen.path / "frozen.json").write_text("{}")
    with pytest.raises(ValueError, match="registered hash"):
        verify_submission(frozen)


@pytest.mark.parametrize("name", [None, 2, "", ".", "/etc/passwd", "../truth", "a\\b", "a/" * 9])
def test_file_names_cannot_escape_bundle(name: object) -> None:
    with pytest.raises(ValueError, match="file names"):
        relative_file(name)


def test_manifest_rejects_links_special_files_and_excessive_bundles(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="real directory"):
        file_manifest(tmp_path / "absent")
    root = tmp_path / "bundle"
    root.mkdir()
    link = root / "link"
    link.symlink_to(tmp_path)
    with pytest.raises(ValueError, match="links"):
        file_manifest(root)
    link.unlink()
    fifo = root / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="special"):
        file_manifest(root)
    fifo.unlink()
    (root / "large").write_bytes(b"abcdef")
    with pytest.raises(ValueError, match="byte budget"):
        file_manifest(root, max_bytes=5)
    (root / "large").unlink()
    for index in range(257):
        (root / str(index)).touch()
    with pytest.raises(ValueError, match="file/byte"):
        file_manifest(root)
    for file in root.iterdir():
        file.unlink()
    for index in range(513):
        (root / str(index)).mkdir()
    with pytest.raises(ValueError, match="directory budget"):
        file_manifest(root)


def test_copy_checks_existing_destination_and_source_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "nested").mkdir()
    (source / "nested/a").write_text("before")
    with pytest.raises(FileExistsError):
        copy_bundle(source, source)
    original = submissions.shutil.copytree

    def changed(src: Path, dst: Path) -> None:
        monkeypatch.setattr(submissions.shutil, "copytree", original)
        original(src, dst)
        (dst / "nested/a").write_text("after")

    monkeypatch.setattr(submissions.shutil, "copytree", changed)
    with pytest.raises(ValueError, match="changed"):
        copy_bundle(source, tmp_path / "destination")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": 2}, "unsupported"),
        ({"extra": True}, "submission needs"),
        ({"hypothesis": " "}, "nonempty"),
        ({"assumptions": 3}, "nonempty"),
        ({"resources": []}, "resources"),
        ({"mechanism": "missing"}, "submitted file"),
        ({"predictor": "mechanism.txt"}, "Python"),
    ],
)
def test_freezing_rejects_invalid_schema(
    tmp_path: Path, change: dict[str, object], message: str
) -> None:
    root = make_submission(tmp_path / "source")
    payload = json.loads((root / "submission.json").read_text())
    payload.update(change)
    write_json(root / "submission.json", payload)
    with pytest.raises(ValueError, match=message):
        freeze_submission(root, tmp_path / "frozen")


def test_freezing_rejects_missing_or_reserved_receipts(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(ValueError, match=r"submission\.json"):
        freeze_submission(root, tmp_path / "unused")
    root = make_submission(tmp_path / "source")
    (root / "frozen.json").write_text("{}")
    with pytest.raises(ValueError, match="reserved"):
        freeze_submission(root, tmp_path / "unused")
    (root / "frozen.json").unlink()
    (root / "submission.json").write_text("[]")
    with pytest.raises(ValueError, match="unsupported"):
        freeze_submission(root, tmp_path / "unused")


def test_registered_but_incompatible_receipt_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "bad"
    root.mkdir()
    raw = b'{"schema_version": 2, "files": {}}'
    (root / "frozen.json").write_bytes(raw)
    frozen = FrozenSubmission(root, hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="modified"):
        verify_submission(frozen)


def test_nonobject_registered_receipt_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "frozen.json").write_bytes(b"[]")
    frozen = FrozenSubmission(tmp_path, hashlib.sha256(b"[]").hexdigest())
    with pytest.raises(ValueError, match="invalid frozen receipt"):
        verify_submission(frozen)


@pytest.mark.property
@settings(max_examples=60, deadline=None)
@given(st.text(max_size=100))
def test_accepted_names_resolve_inside_bundle(name: str) -> None:
    try:
        relative = relative_file(name)
    except ValueError:
        return
    root = Path("/tmp/fixed-discovery-bundle")
    assert (root / relative).resolve().is_relative_to(root)
