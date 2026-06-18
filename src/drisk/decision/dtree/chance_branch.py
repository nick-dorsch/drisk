"""Chance branches."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, model_validator

from ._types import ValueLike
from .nodes.base import DTreeNode


class ChanceBranch(BaseModel):
    """A probability-weighted branch leaving a chance node."""

    name: str
    probability: float = Field(ge=0)
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
