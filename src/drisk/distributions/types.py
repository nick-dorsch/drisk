"""Distribution type discriminators and shared typing helpers."""

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np


class DataFrameLike(Protocol):
    """Structural type for pandas/polars-style tabular data."""

    def to_numpy(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Return the tabular data as a NumPy array."""
        ...


# Input data accepted by distribution fitting APIs.
type ArrayLike = np.ndarray | Sequence[Any] | DataFrameLike
