"""Execution backends behind a single Pool abstraction.

:class:`gmat_sweep.backends.joblib.LocalJoblibPool` is the always-available
default. :class:`DaskPool` and :class:`RayPool` live behind the optional
``[dask]`` and ``[ray]`` extras and are exposed lazily through this
package's :func:`__getattr__` — accessing them when the underlying extra
is not installed raises :class:`AttributeError` with an install hint, so a
minimal install never imports ``distributed`` or ``ray`` at package-import
time.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from gmat_sweep.backends.base import Pool
from gmat_sweep.backends.joblib import LocalJoblibPool

if TYPE_CHECKING:
    from gmat_sweep.backends.dask import DaskPool
    from gmat_sweep.backends.ray import RayPool

__all__ = ["DaskPool", "LocalJoblibPool", "Pool", "RayPool"]


_OPTIONAL_BACKENDS = {
    "DaskPool": ("distributed", "gmat_sweep.backends.dask", "dask"),
    "RayPool": ("ray", "gmat_sweep.backends.ray", "ray"),
}


def __getattr__(name: str) -> Any:
    entry = _OPTIONAL_BACKENDS.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    probe_module, backend_module, extra = entry
    try:
        importlib.import_module(probe_module)
    except ImportError as exc:
        raise AttributeError(
            f"{name} requires the [{extra}] extra: pip install gmat-sweep[{extra}]"
        ) from exc
    return getattr(importlib.import_module(backend_module), name)
