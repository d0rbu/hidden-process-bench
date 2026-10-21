# List the available development commands.
default:
    @just --list

# Install the locked environment and the pre-push hook (never a commit hook).
setup:
    uv sync --locked
    uv run --locked pre-commit install --hook-type pre-push --install-hooks

# Run the complete local gate before pushing a PR branch.
check:
    uv run --locked pre-commit run --hook-stage pre-push --all-files --show-diff-on-failure

# Verify that project metadata and the lockfile agree.
lock-check:
    uv lock --check

# Check Python style, imports, and annotations.
lint:
    uv run --locked ruff check .

# Apply safe lint fixes and format Python files.
fmt:
    uv run --locked ruff check --fix .
    uv run --locked ruff format .

# Check formatting without changing files.
format-check:
    uv run --locked ruff format --check .

# Check static types.
typecheck:
    uv run --locked ty check

# Find unused and unreachable Python code.
deadcode:
    uv run --locked vulture

# Run tests with coverage; additional pytest arguments are passed through.
[positional-arguments]
test *args:
    uv run --locked python -m pytest "$@"

# Generate a browsable coverage report under htmlcov/.
coverage:
    uv run --locked python -m pytest --cov-report=term-missing --cov-report=html

# Validate local Markdown link and image targets across the repository.
docs-check:
    uv run --locked python -m scripts.check_markdown_links
