"""Check local Markdown file and image targets without making network requests."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt
from markdown_it.token import Token


def markdown_files(root: Path) -> list[Path]:
    """Include tracked and untracked Markdown, respecting Git's ignore rules."""
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(
        {
            root / name
            for name in result.stdout.split("\0")
            if name.lower().endswith(".md") and (root / name).is_file()
        }
    )


def _walk_tokens(tokens: Iterable[Token]) -> Iterator[Token]:
    for token in tokens:
        yield token
        yield from _walk_tokens(token.children or ())


def check_document(document: Path, root: Path) -> list[str]:
    """Return diagnostics for missing targets, repository escapes, or unreadable files."""
    label = document.relative_to(root)
    try:
        content = document.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return [f"{label}: cannot read Markdown: {error}"]

    errors: list[str] = []
    for token in _walk_tokens(MarkdownIt("commonmark").parse(content)):
        target = token.attrGet("href") or token.attrGet("src")
        if not isinstance(target, str):
            continue
        url = urlsplit(target)
        if url.scheme or url.netloc or not url.path:
            continue
        path = unquote(url.path)
        # GitHub resolves a leading slash relative to the repository root.
        resolved = root / path.lstrip("/") if path.startswith("/") else document.parent / path
        resolved = resolved.resolve()
        if not resolved.is_relative_to(root.resolve()):
            errors.append(f"{label}: local target leaves the repository: {target!r}")
        elif not resolved.exists():
            errors.append(f"{label}: missing local target: {target!r}")
    return errors


def main(root: Path) -> int:
    root = root.resolve()
    documents = markdown_files(root)
    errors = [error for document in documents for error in check_document(document, root)]
    if errors:
        print("\n".join(errors))
        return 1
    print(f"Checked local targets in {len(documents)} Markdown files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(__file__).resolve().parents[1]))
