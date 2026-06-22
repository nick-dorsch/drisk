# drisk

Composable tools for quick Monte Carlo modelling, with an emphasis on distribution
elicitation and model composition.

`drisk` provides a compact, notebook-friendly API for probability distributions,
elicitation workflows, correlated sampling, decision trees, sensitivity analysis, and
composable Monte Carlo models.

> `drisk` is currently in alpha. APIs may change before a stable release.

## Links

- Documentation: <https://nick-dorsch.github.io/drisk/>
- Repository: <https://github.com/nick-dorsch/drisk>
- Issues: <https://github.com/nick-dorsch/drisk/issues>

## Installation

Install from PyPI:

```bash
pip install drisk
```

Or with `uv`:

```bash
uv add drisk
```

`drisk` requires Python 3.12 or newer.

## Quick start

```python
import drisk as dr

# Elicit a distribution from intuitive inputs.
dist = dr.LogitNormal.elicit(lower=0.1, upper=0.25)

# Draw Monte Carlo samples.
samples = dist.sample(1_000)
```

## Features

- Continuous distributions such as `Normal`, `LogNormal`, `LogitNormal`, `Beta`, `PERT`, `Gamma`, and `Exponential`
- Discrete distributions such as `Bernoulli`, `Binomial`, `Poisson`, `Geometric`, and `NegativeBinomial`
- Distribution elicitation, fitting, sampling, plotting, and serialization-friendly parameters
- Mixture distributions with `UvMixture`
- Correlation matrix helpers and Gaussian / Student-t copulas
- Composable Monte Carlo models via `MCModel`, `MCOperation`, and `where`
- Decision tree modelling with chance, decision, and outcome nodes
- One-at-a-time sensitivity analysis helpers

## Documentation

The documentation site includes examples and API reference pages:

<https://nick-dorsch.github.io/drisk/>

## Developer setup

This repository uses:

- [mise](https://mise.jdx.dev/) to install local development tools and run common project tasks
- [uv](https://docs.astral.sh/uv/) for Python dependency management and command execution

### 1. Install tools with mise

Install `mise` if you do not already have it, then install the tools pinned in
`.mise.toml`:

```bash
mise install
```

This installs the project Python version, `uv`, `ruff`, and other helper tools for the
repository.

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
  copulas/                  # copula interfaces and implementations
  correlations/             # correlation matrix helpers
  decision/                 # decision tree nodes, branches, and trees
  distributions/            # distribution interfaces and implementations
  models/                   # composable Monte Carlo model helpers
  sensitivity/              # sensitivity analysis utilities
  arithmetic.py             # arithmetic/composition helpers
  random.py                 # seed/RNG helpers
  summary.py                # summary helpers

tests/                      # pytest suite
docs/                       # Quarto documentation site
```

## License

`drisk` is distributed under the MIT License. See [LICENSE](LICENSE) for details.
