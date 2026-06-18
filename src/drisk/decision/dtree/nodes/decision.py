"""Decision nodes."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import field_validator

from drisk.random import SeedLike

from .._coercion import coerce_decision_branches
from .._sampling import branch_expected_value, sample_branch
from ..decision_branch import DecisionBranch
from .base import DTreeNode


class DecisionNode(DTreeNode):
    """Decision node that selects the branch with the highest expected value."""

    node_type: Literal["decision"] = "decision"
    branches: tuple[DecisionBranch, ...]

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
    def coerce_branches(cls, branches: Any) -> tuple[DecisionBranch, ...]:
        return tuple(coerce_decision_branches(branches))

    @field_validator("branches")
    @classmethod
    def validate_branches(
        cls, branches: tuple[DecisionBranch, ...]
    ) -> tuple[DecisionBranch, ...]:
        if not branches:
            raise ValueError("DecisionNode requires at least one branch")
        return branches

    def selected_branch(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
    ) -> DecisionBranch:
        values = [
            branch_expected_value(branch, size=size, seed=seed)
            for branch in self.branches
        ]
        return self.branches[int(np.argmax(values))]

    def expected_value(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
    ) -> float:
        return branch_expected_value(
            self.selected_branch(size=size, seed=seed), size=size, seed=seed
        )

    def sample(
        self,
        size: int | tuple[int, ...] = 1,
        *,
        seed: SeedLike = None,
    ) -> np.ndarray:
        branch = self.selected_branch(size=size, seed=seed)
        return sample_branch(branch, size=size, seed=seed)

    def rollback_rows(
        self,
        *,
        size: int | tuple[int, ...] | None = None,
        seed: SeedLike = None,
    ) -> list[dict[str, Any]]:
        values = [
            branch_expected_value(branch, size=size, seed=seed)
            for branch in self.branches
        ]
        selected_index = int(np.argmax(values))
        rows = [
            {
                "node": self.name or "decision",
                "node_type": self.node_type,
                "branch": branch.name,
                "probability": None,
                "expected_value": value,
                "selected": i == selected_index,
            }
            for i, (branch, value) in enumerate(zip(self.branches, values, strict=True))
        ]
        for branch in self.branches:
            rows.extend(branch.node.rollback_rows(size=size, seed=seed))
        return rows
