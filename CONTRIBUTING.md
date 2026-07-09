# Contributing to Sift

Contributions are welcome. Sift is a local first file search for KDE that runs
on the CPU unless you tell it otherwise.

## Dev setup

```bash
uv venv --python 3.12
uv pip install -e ".[dev]" sqlite-vec
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

The test suite uses fake embedders and transcribers, so it runs fast and needs
no model downloads or GPU. Heavy ML dependencies are optional extras (semantic,
image, asr, ocr, gui, kde).

## Principles

- CPU only by default. Every feature must work without a GPU, and a GPU is
  something the user turns on themselves.
- Local and private. No cloud calls, no telemetry.
- Graceful degradation. A missing optional dependency disables a feature with a
  warning instead of crashing.
- Do not hog the desktop. Indexing is throttled (see `sift bench`).

## Before opening a PR

- `ruff check`, `ruff format --check`, and `pytest` pass.
- New behavior has a test (use the fake model pattern in `tests/`).
- User visible changes update the README.

CI runs the same three commands against Python 3.11, 3.12, and 3.13. `main` is
protected, so work on a branch and open a pull request.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Security
issues go through [SECURITY.md](SECURITY.md), not the public issue tracker.
