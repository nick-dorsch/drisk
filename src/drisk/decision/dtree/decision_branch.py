"""Decision branches."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, SerializeAsAny, model_validator

from ._types import ValueLike
from .nodes.base import DTreeNode


class DecisionBranch(BaseModel):
    """A branch leaving a decision node."""

    name: str
    node: SerializeAsAny[DTreeNode]
    value: ValueLike = 0

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def coerce_node(cls, data: Any) -> Any:
        if isinstance(data, dict) and "node" in data:
            from .nodes.factory import as_node

            copied = dict(data)
            copied["node"] = as_node(copied["node"])
            return copied
        return data
