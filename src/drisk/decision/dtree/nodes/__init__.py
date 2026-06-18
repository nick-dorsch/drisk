"""Decision tree node classes."""

from typing import Any

__all__ = ["ChanceNode", "DecisionNode", "DTreeNode", "OutcomeNode", "as_node"]


def __getattr__(name: str) -> Any:
    if name == "DTreeNode":
        from .base import DTreeNode

        return DTreeNode
    if name == "ChanceNode":
        from .chance import ChanceNode

        return ChanceNode
    if name == "DecisionNode":
        from .decision import DecisionNode

        return DecisionNode
    if name == "OutcomeNode":
        from .outcome import OutcomeNode

        return OutcomeNode
    if name == "as_node":
        from .factory import as_node

        return as_node
    raise AttributeError(name)
