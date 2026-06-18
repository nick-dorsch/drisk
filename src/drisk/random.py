"""Random number generation helpers for Drisk."""

from types import ModuleType
from typing import Any

import numpy as np

# Public type accepted by rvs(seed=...).
type SeedLike = int | np.random.Generator | np.random.RandomState | None


def get_rng(seed: SeedLike = None) -> Any:
    """
    Return a NumPy-compatible random number generator.

    ``None`` returns the module-level ``np.random`` generator, preserving
    compatibility with ``np.random.seed(...)``. Integer seeds create a fresh
    ``np.random.Generator``. Existing generators are returned unchanged so their
    state advances across calls.
    """
    if seed is None:
        return np.random
    if isinstance(seed, ModuleType | np.random.Generator | np.random.RandomState):
        return seed
    return np.random.default_rng(seed)


def get_scipy_random_state(
    seed: SeedLike = None,
) -> np.random.Generator | np.random.RandomState | None:
    """Return a value suitable for scipy.stats ``random_state=`` parameters."""
    if seed is None or isinstance(seed, ModuleType):
        return None
    return get_rng(seed)
