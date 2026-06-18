"""Coercion helpers for decision tree inputs."""

from __future__ import annotations

from typing import Any

from .chance_branch import ChanceBranch
from .decision_branch import DecisionBranch


def coerce_decision_branches(branches: Any) -> list[DecisionBranch]:
    from .nodes.factory import as_node

    if isinstance(branches, dict):
        return [
            DecisionBranch(name=str(name), node=as_node(value))
            for name, value in branches.items()
        ]
    return [
        branch
        if isinstance(branch, DecisionBranch)
        else DecisionBranch.model_validate(branch)
        for branch in branches
    ]


def coerce_chance_branches(branches: Any) -> list[ChanceBranch]:
    from .nodes.factory import as_node

    if isinstance(branches, dict):
        coerced = []
        for name, spec in branches.items():
            if isinstance(spec, tuple) and len(spec) == 2:
                probability, value = spec
                coerced.append(
                    ChanceBranch(
                        name=str(name), probability=probability, node=as_node(value)
                    )
                )
            else:
                coerced.append(ChanceBranch.model_validate({"name": name, **spec}))
        return coerced
    return [
        branch
        if isinstance(branch, ChanceBranch)
        else ChanceBranch.model_validate(branch)
        for branch in branches
    ]
