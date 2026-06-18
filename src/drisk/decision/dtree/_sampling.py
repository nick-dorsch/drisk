"""Sampling helpers for decision trees."""

from __future__ import annotations

import numpy as np

from drisk.distributions import Distribution
from drisk.models import MCModel
from drisk.random import SeedLike, get_rng

from ._types import ValueLike
from .branches import ChanceBranch, DecisionBranch


def branch_expected_value(
    branch: DecisionBranch | ChanceBranch,
    *,
    size: int | tuple[int, ...] | None = None,
    seed: SeedLike = None,
) -> float:
    rng = get_rng(seed)
    return value_expected_value(
        branch.value, size=size, seed=rng
    ) + branch.node.expected_value(
        size=size,
        seed=rng,
    )


def value_expected_value(
    value: ValueLike,
    *,
    size: int | tuple[int, ...] | None = None,
    seed: SeedLike = None,
) -> float:
    samples = np.ravel(
        sample_value(value, size=10_000 if size is None else size, seed=seed)
    )
    return float(np.mean(samples))


def sample_branch(
    branch: DecisionBranch | ChanceBranch,
    *,
    size: int | tuple[int, ...],
    seed: SeedLike = None,
) -> np.ndarray:
    rng = get_rng(seed)
    return sample_value(branch.value, size=size, seed=rng) + branch.node.sample(
        size=size, seed=rng
    )


def sample_value(
    value: ValueLike,
    *,
    size: int | tuple[int, ...],
    seed: SeedLike = None,
) -> np.ndarray:
    from .tree import DTree

    if isinstance(value, Distribution | MCModel):
        return np.asarray(value.sample(size=size, seed=seed), dtype=float)
    if isinstance(value, DTree):
        return np.asarray(value.sample(size=size, seed=seed), dtype=float)
    return np.broadcast_to(np.asarray(value, dtype=float), size)
