from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.check_markdown_links import check_document, main, markdown_files


@pytest.fixture
def docs_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("# Project\n", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize(
    "target",
    [
        "target.md",
        "./target.md",
        "/docs/target.md",
        "target.md?plain=1#section",
        "../README.md",
        "../docs/",
        "/",
        "my%20file.md",
        "<my file.md>",
        "caf%C3%A9.md",
        "notes(v1).md",
    ],
)
def test_existing_local_targets(docs_repo: Path, target: str) -> None:
    for name in ("target.md", "my file.md", "café.md", "notes(v1).md"):
        (docs_repo / "docs" / name).write_text("# Target\n", encoding="utf-8")
    document = docs_repo / "docs" / "page.md"
    document.write_text(f"[link]({target})\n", encoding="utf-8")

    assert check_document(document, docs_repo) == []


def test_inline_reference_and_image_links_report_all_missing_targets(docs_repo: Path) -> None:
    document = docs_repo / "README.md"
    document.write_text(
        "[inline](missing.md) [reference][guide] ![image](missing.png)\n\n"
        "[guide]: docs/missing.md\n",
        encoding="utf-8",
    )

    assert check_document(document, docs_repo) == [
        "README.md: missing local target: 'missing.md'",
        "README.md: missing local target: 'docs/missing.md'",
        "README.md: missing local target: 'missing.png'",
    ]


def test_nested_image_links_are_checked(docs_repo: Path) -> None:
    document = docs_repo / "README.md"
    document.write_text("[![image](missing.png)](README.md)\n", encoding="utf-8")

    assert check_document(document, docs_repo) == ["README.md: missing local target: 'missing.png'"]


@pytest.mark.parametrize(
    "target",
    [
        "https://example.invalid/missing",
        "https://[broken]",
        "http://example.invalid/missing",
        "mailto:example@example.invalid",
        "//example.invalid/missing",
        "#section",
        "",
    ],
)
def test_external_urls_and_fragment_only_links_are_not_fetched(
    docs_repo: Path, target: str
) -> None:
    document = docs_repo / "README.md"
    document.write_text(f"[link]({target})\n", encoding="utf-8")

    assert check_document(document, docs_repo) == []


def test_code_examples_are_not_links(docs_repo: Path) -> None:
    document = docs_repo / "README.md"
    document.write_text(
        "`[inline](missing.md)`\n\n```markdown\n[block](missing.md)\n```\n"
        "\n    [indented](missing.md)\n",
        encoding="utf-8",
    )

    assert check_document(document, docs_repo) == []


@pytest.mark.parametrize("target", ["../outside.md", "external.md"])
def test_existing_files_outside_repository_are_rejected(docs_repo: Path, target: str) -> None:
    outside = docs_repo.parent / "outside.md"
    outside.write_text("# Outside\n", encoding="utf-8")
    (docs_repo / "external.md").symlink_to(outside)
    document = docs_repo / "README.md"
    document.write_text(f"[link]({target})\n", encoding="utf-8")

    assert check_document(document, docs_repo) == [
        f"README.md: local target leaves the repository: {target!r}"
    ]


def test_invalid_utf8_produces_a_diagnostic(docs_repo: Path) -> None:
    document = docs_repo / "README.md"
    document.write_bytes(b"\xff")

    errors = check_document(document, docs_repo)

    assert len(errors) == 1
    assert "README.md: cannot read Markdown" in errors[0]


def test_discovery_includes_new_docs_and_skips_ignored_or_deleted_docs(docs_repo: Path) -> None:
    (docs_repo / ".gitignore").write_text("generated/\n", encoding="utf-8")
    (docs_repo / "generated").mkdir()
    (docs_repo / "generated" / "ignored.md").write_text("[bad](missing.md)\n", encoding="utf-8")
    new_doc = docs_repo / "docs" / "NEW.MD"
    new_doc.write_text("# New\n", encoding="utf-8")
    removed_doc = docs_repo / "removed.md"
    removed_doc.write_text("# Removed\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(docs_repo), "add", "README.md", "removed.md"], check=True)
    removed_doc.unlink()

    assert markdown_files(docs_repo) == [docs_repo / "README.md", new_doc]


def test_renaming_a_target_rechecks_unchanged_docs(
    docs_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = docs_repo / "docs" / "page.md"
    target.write_text("# Page\n", encoding="utf-8")
    (docs_repo / "README.md").write_text("[page](docs/page.md)\n", encoding="utf-8")
    assert main(docs_repo) == 0
    assert "2 Markdown files" in capsys.readouterr().out

    target.rename(docs_repo / "docs" / "renamed.md")

    assert main(docs_repo) == 1
    assert "README.md: missing local target: 'docs/page.md'" in capsys.readouterr().out


def test_repository_root_is_independent_of_working_directory(
    docs_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(docs_repo / "docs")

    assert main(docs_repo) == 0
