# Contributing to pyCaRM

Thanks for your interest in contributing! This document covers how to get a
development environment running, how to run the test suite and linter, and
the conventions used for branches and pull requests.

## Development setup

Clone the repository and install it in editable mode with the development
extras:

```bash
git clone https://github.com/BETALAB-team/pyCaRM.git
cd pyCaRM
pip install -e ".[dev]"
```

Optional extras you may also want, depending on what you're working on:

```bash
pip install -e ".[docs]"      # Sphinx + theme, needed to build the documentation
pip install -e ".[coolprop]"  # CoolProp, needed for fluid-property calculations
```

Extras can be combined, e.g. `pip install -e ".[dev,coolprop]"`.

## Running the tests

The test suite uses `pytest` and lives under `tests/`:

```bash
pytest
```

If you're adding a new feature or fixing a bug, add or update a test under
`tests/` that would fail without your change.

## Linting

Lint checks use `ruff` against `carm/` and `tests/` (this is also what CI
runs):

```bash
pip install ruff
ruff check carm/ tests/
```

`examples/` is intentionally excluded from linting.

## Branch and pull request conventions

- Fork the repository (or create a branch, if you have write access) off
  `main`.
- Use a short, descriptive branch name, e.g. `fix/solver-picard-loop` or
  `feat/coaxial-grout-model`.
- Keep pull requests focused on a single change; unrelated cleanups should
  be their own PR.
- Make sure `pytest` and `ruff check carm/ tests/` both pass locally before
  opening a PR — CI runs both across the supported OS/Python matrix.
- Describe *what* changed and *why* in the PR description; link any related
  issue.
- New behavior should come with test coverage; bug fixes should include a
  regression test that fails before the fix and passes after it.

## Building the documentation

```bash
pip install -e ".[docs]"
cd docs
make html
```

The built HTML lands in `docs/build/html/`.

## Reporting issues

Use the issue templates under `.github/ISSUE_TEMPLATE/` when filing a bug
report or feature request — they help make sure we get the information we
need to help.
