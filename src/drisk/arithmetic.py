"""Arithmetic helpers for composable Monte Carlo expressions."""

from __future__ import annotations

from typing import Any

import numpy as np


class ArithmeticMixin:
    """Mixin that turns arithmetic into lazy Monte Carlo model expressions."""

    def __add__(self, other: Any) -> Any:
        """Create a lazy model for ``self + other``."""
        from drisk.models import MCModel

        return MCModel.from_operation(np.add, self, other)

    def __radd__(self, other: Any) -> Any:
        """Create a lazy model for ``other + self``."""
        from drisk.models import MCModel

        return MCModel.from_operation(np.add, other, self)

    def __sub__(self, other: Any) -> Any:
        """Create a lazy model for ``self - other``."""
        from drisk.models import MCModel

        return MCModel.from_operation(np.subtract, self, other)

    def __rsub__(self, other: Any) -> Any:
        """Create a lazy model for ``other - self``."""
        from drisk.models import MCModel

        return MCModel.from_operation(np.subtract, other, self)

    def __mul__(self, other: Any) -> Any:
        """Create a lazy model for ``self * other``."""
        from drisk.models import MCModel

        return MCModel.from_operation(np.multiply, self, other)

    def __rmul__(self, other: Any) -> Any:
        """Create a lazy model for ``other * self``."""
        from drisk.models import MCModel

        return MCModel.from_operation(np.multiply, other, self)

    def __truediv__(self, other: Any) -> Any:
        """Create a lazy model for ``self / other``."""
        from drisk.models import MCModel

        return MCModel.from_operation(np.divide, self, other)

    def __rtruediv__(self, other: Any) -> Any:
        """Create a lazy model for ``other / self``."""
        from drisk.models import MCModel

        return MCModel.from_operation(np.divide, other, self)

    def __pow__(self, other: Any) -> Any:
        """Create a lazy model for ``self ** other``."""
        from drisk.models import MCModel

        return MCModel.from_operation(np.power, self, other)

    def __rpow__(self, other: Any) -> Any:
        """Create a lazy model for ``other ** self``."""
        from drisk.models import MCModel

        return MCModel.from_operation(np.power, other, self)

    def __neg__(self) -> Any:
        """Create a lazy model for ``-self``."""
        from drisk.models import MCModel

        return MCModel.from_operation(np.negative, self)

    def __pos__(self) -> Any:
        """Create a lazy model for ``+self``."""
        from drisk.models import MCModel

        return MCModel.from_operation(np.positive, self)

    def __abs__(self) -> Any:
        """Create a lazy model for ``abs(self)``."""
        from drisk.models import MCModel

        return MCModel.from_operation(np.abs, self)
