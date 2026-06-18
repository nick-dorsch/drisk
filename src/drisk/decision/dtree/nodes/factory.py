"""Decision tree node factory helpers."""

from __future__ import annotations

from typing import Any

from .base import DTreeNode


def as_node(value: Any) -> DTreeNode:
    """Return ``value`` as a decision tree node."""
    from .chance import ChanceNode
    from .decision import DecisionNode
    from .outcome import OutcomeNode

    if isinstance(value, DTreeNode):
        return value
    if isinstance(value, dict) and "node_type" in value:
        node_type = value["node_type"]
        if node_type == "decision":
            return DecisionNode.model_validate(value)
        if node_type == "chance":
            return ChanceNode.model_validate(value)
        if node_type == "outcome":
            return OutcomeNode.model_validate(value)
        raise ValueError(f"Unknown decision tree node_type: {node_type}")
    return OutcomeNode(value=value)
