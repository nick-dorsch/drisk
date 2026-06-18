"""Chance nodes."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import field_validator

from drisk.random import SeedLike, get_rng

from .._coercion import coerce_chance_branches
from .._sampling import branch_expected_value, sample_branch
from ..chance_branch import ChanceBranch
from .base import DTreeNode


class ChanceNode(DTreeNode):
    """Chance node that follows branches according to their probabilities."""

    node_type: Literal["chance"] = "chance"
    branches: tuple[ChanceBranch, ...]

    def __init__(
        self,
        name: str | None = None,
        branches: Any = None,
        **data: Any,
    ) -> None:
        if name is not None and "name" not in data:
            data["name"] = name
        if branches is not None and "branches" not in data:
            data["branches"] = branches
        super().__init__(**data)

    @field_validator("branches", mode="before")
    @classmethod
    def coerce_branches(cls, branches: Any) -> tuple[ChanceBranch, ...]:
        return tuple(coerce_chance_branches(branches))

    @field_validator("branches")
    @classmethod
    def validate_branches(
        cls, branches: tuple[ChanceBranch, ...]
    ) -> tuple[ChanceBranch, ...]:
        if not branches:
            raise ValueError("ChanceNode requires at least one branch")
        probability_sum = sum(branch.probability for branch in branches)
        if not np.isclose(probability_sum, 1.0):
            raise ValueError(
                f"ChanceNode branch probabilities must sum to 1, got {probability_sum}"
            )
        return branches

    def expected_value(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
    ) -> float:
        return float(
            sum(
                branch.probability * branch_expected_value(branch, size=size, seed=seed)
                for branch in self.branches
            )
        )

    def sample(
        self,
        size: int | tuple[int, ...] = 1,
        *,
        seed: SeedLike = None,
    ) -> np.ndarray:
        rng = get_rng(seed)
        probabilities = [branch.probability for branch in self.branches]
        choices = rng.choice(len(self.branches), size=size, p=probabilities)
        branch_samples = [
            sample_branch(branch, size=size, seed=rng) for branch in self.branches
        ]
        result = np.empty(np.shape(choices), dtype=float)
        for i, samples in enumerate(branch_samples):
            result[choices == i] = np.asarray(samples, dtype=float)[choices == i]
        return result

    def rollback_rows(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
    ) -> list[dict[str, Any]]:
        rows = [
            {
                "node": self.name or "chance",
                "node_type": self.node_type,
                "branch": branch.name,
                "probability": branch.probability,
                "expected_value": branch_expected_value(branch, size=size, seed=seed),
                "selected": None,
            }
            for branch in self.branches
        ]
        for branch in self.branches:
            rows.extend(branch.node.rollback_rows(size=size, seed=seed))
        return rows
