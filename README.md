# drisk

Composable tools for quick Monte Carlo modelling, with an emphasis on distribution elicitation and model composition.

Drisk provides a compact, notebook-friendly API for probability distributions, elicitation workflows, and composable Monte Carlo models.

## Usage

```python
import drisk as dr

# Elicit a distribution from intuitive inputs.
dist = dr.LogitNormal.elicit(low=0.1, high=0.25)

# Draw Monte Carlo samples.
samples = dist.sample(1_000)
```

## Developer setup

This repository uses:

- [mise](https://mise.jdx.dev/) to install local development tools and run common project tasks
- [uv](https://docs.astral.sh/uv/) for Python dependency management and command execution

### 1. Install tools with mise

Install `mise` if you do not already have it, then install the tools pinned in `.mise.toml`:

```bash
mise install
```

This installs the project Python version, `uv`, `ruff`, and other helper tools for the repository.

### 2. Install Python dependencies with uv

Sync the project environment from `pyproject.toml` and `uv.lock`:

```bash
uv sync --all-groups
```

Use `uv run ...` for commands that should execute inside the managed environment:

```bash
uv run pytest
uv run ruff check .
uv run python -c "import drisk as dr; print(dr.Normal.elicit(0, 1))"
```

When adding dependencies, use `uv add` so `pyproject.toml` and `uv.lock` stay in sync:

```bash
uv add numpy
uv add --dev pytest
uv add --group docs jupyter
```

### 3. Run project tasks with mise

List available tasks:

```bash
mise tasks
```

Common tasks:

```bash
mise run test      # run the test suite
mise run lint      # run ruff checks
mise run format    # format the codebase
mise run check     # run lint and tests
mise run docs      # preview the Quarto documentation site
```

## Project layout

```text
src/drisk/
  distributions/
    base.py                  # shared distribution interfaces
    types.py                 # shared typing helpers
    univariate/
      base.py                # univariate distribution interface
      continuous/            # continuous domain interfaces + implementations
  random.py                  # seed/RNG helpers

tests/                       # pytest suite
docs/                        # Quarto documentation site
```
