"""Arithmetic helpers for composable Monte Carlo expressions."""

from __future__ import annotations

from typing import Any


class ArithmeticMixin:
    """Mixin that turns arithmetic into lazy Monte Carlo model expressions."""

    def __add__(self, other: Any) -> Any:
        """Create a lazy model for ``self + other``."""
        from drisk.models import MCModel, MCOperation

        return MCModel.from_operation(MCOperation.ADD, self, other)

    def __radd__(self, other: Any) -> Any:
        """Create a lazy model for ``other + self``."""
        from drisk.models import MCModel, MCOperation

        return MCModel.from_operation(MCOperation.ADD, other, self)

    def __sub__(self, other: Any) -> Any:
        """Create a lazy model for ``self - other``."""
        from drisk.models import MCModel, MCOperation

        return MCModel.from_operation(MCOperation.SUBTRACT, self, other)

    def __rsub__(self, other: Any) -> Any:
        """Create a lazy model for ``other - self``."""
        from drisk.models import MCModel, MCOperation

        return MCModel.from_operation(MCOperation.SUBTRACT, other, self)

    def __mul__(self, other: Any) -> Any:
        """Create a lazy model for ``self * other``."""
        from drisk.models import MCModel, MCOperation

        return MCModel.from_operation(MCOperation.MULTIPLY, self, other)

    def __rmul__(self, other: Any) -> Any:
        """Create a lazy model for ``other * self``."""
        from drisk.models import MCModel, MCOperation

        return MCModel.from_operation(MCOperation.MULTIPLY, other, self)

    def __truediv__(self, other: Any) -> Any:
        """Create a lazy model for ``self / other``."""
        from drisk.models import MCModel, MCOperation

        return MCModel.from_operation(MCOperation.DIVIDE, self, other)

    def __rtruediv__(self, other: Any) -> Any:
        """Create a lazy model for ``other / self``."""
        from drisk.models import MCModel, MCOperation

        return MCModel.from_operation(MCOperation.DIVIDE, other, self)

    def __pow__(self, other: Any) -> Any:
        """Create a lazy model for ``self ** other``."""
        from drisk.models import MCModel, MCOperation

        return MCModel.from_operation(MCOperation.POWER, self, other)

    def __rpow__(self, other: Any) -> Any:
        """Create a lazy model for ``other ** self``."""
        from drisk.models import MCModel, MCOperation

        return MCModel.from_operation(MCOperation.POWER, other, self)

    def __lt__(self, other: Any) -> Any:
        """Create a lazy model for ``self < other``."""
        from drisk.models import MCModel, MCOperation

        return MCModel.from_operation(MCOperation.LESS, self, other)

    def __le__(self, other: Any) -> Any:
        """Create a lazy model for ``self <= other``."""
        from drisk.models import MCModel, MCOperation

        return MCModel.from_operation(MCOperation.LESS_EQUAL, self, other)

    def __gt__(self, other: Any) -> Any:
        """Create a lazy model for ``self > other``."""
        from drisk.models import MCModel, MCOperation

        return MCModel.from_operation(MCOperation.GREATER, self, other)

    def __ge__(self, other: Any) -> Any:
        """Create a lazy model for ``self >= other``."""
        from drisk.models import MCModel, MCOperation

        return MCModel.from_operation(MCOperation.GREATER_EQUAL, self, other)

    def __neg__(self) -> Any:
        """Create a lazy model for ``-self``."""
        from drisk.models import MCModel, MCOperation

        return MCModel.from_operation(MCOperation.NEGATIVE, self)

    def __pos__(self) -> Any:
        """Create a lazy model for ``+self``."""
        from drisk.models import MCModel, MCOperation

        return MCModel.from_operation(MCOperation.POSITIVE, self)

    def __abs__(self) -> Any:
        """Create a lazy model for ``abs(self)``."""
        from drisk.models import MCModel, MCOperation

        return MCModel.from_operation(MCOperation.ABS, self)
