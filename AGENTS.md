# AGENTS.md

Guidance for coding agents working in this repository.

## Project overview

`drisk` is a Python package for quick Monte Carlo modelling, with an emphasis on distribution elicitation and composable models. The package uses a `src/` layout and is intended to provide a notebook-friendly API:

```python
import drisk as dr

dist = dr.LogitNormal.elicit(0.1, 0.25)
samples = dist.sample(1000)
```

## Repository conventions

- Source code lives under `src/drisk/`.
- Tests live under `tests/`.
- Public user-facing APIs should generally be exported from `src/drisk/__init__.py` so notebook users can use `import drisk as dr`.
- Keep abstractions close to their package scope:
  - Top-level distribution interfaces in `src/drisk/distributions/`.
  - Univariate interfaces in `src/drisk/distributions/univariate/`.
  - Univariate continuous interfaces and implementations in `src/drisk/distributions/univariate/continuous/`.
- Domain semantics should be represented with abstract base classes / marker classes, e.g. `UvUnitBoundedContinuous`, not only with unions.
- Distribution `dist_type` values should be plain string literals, not enums.
- All concrete distributions should implement the `Distribution` API:
  - `sample(...)`
  - `rvs(...)` inherited as an alias of `sample(...)`
  - `elicit(...)`
  - `fit(...)`
  - `plot(...)`
- Elicitation methods should populate `elicitation_params` with the user-provided elicitation arguments.

## Dependency and environment management

Use `uv` for all Python dependency management and script execution.

Common commands:

```bash
uv lock
uv run pytest
uv run ruff check .
uv run ruff format .
uv run python -c "import drisk as dr; print(dr.Normal.elicit(0, 1))"
```

Do not use bare `python`, `pip`, or globally installed tools when working in this repo. Prefer `uv run ...` so commands execute in the project environment.

## Formatting and linting

This repo uses `ruff` for linting and formatting.

Before committing or handing off changes, run:

```bash
uv run ruff check .
uv run ruff format .
uv run pytest
```

If lint fixes are safe and mechanical, use:

```bash
uv run ruff check . --fix
uv run ruff format .
```

## Testing

Run the full test suite with:

```bash
uv run pytest
```

Add or update tests for behavior changes, especially for:

- Distribution elicitation
- Sampling shape and bounds
- Domain marker class inheritance
- Serialization-relevant fields like `dist_type`, `params`, and `elicitation_params`

## Git hygiene

- Keep commits focused and descriptive.
- Run tests and ruff before committing.
- Do not commit generated caches such as `__pycache__/`, `.pytest_cache/`, build artifacts, or virtual environments.
