"""Copula models for jointly sampling marginal distributions."""

from .base import Copula
from .gaussian import GaussianCopula
from .student_t import StudentTCopula

__all__ = ["Copula", "GaussianCopula", "StudentTCopula"]
