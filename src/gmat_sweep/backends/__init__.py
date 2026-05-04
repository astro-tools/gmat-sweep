"""Execution backends behind a single Pool abstraction."""

from __future__ import annotations

from gmat_sweep.backends.base import Pool
from gmat_sweep.backends.joblib import LocalJoblibPool

__all__ = ["LocalJoblibPool", "Pool"]
