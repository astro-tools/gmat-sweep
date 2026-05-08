"""Execution backends behind a single Pool abstraction.

:class:`gmat_sweep.backends.joblib.LocalJoblibPool` is the always-available
default. :class:`DaskPool`, :class:`RayPool`, :class:`KubernetesJobPool`,
and :class:`MPIPool` live behind the optional ``[dask]``, ``[ray]``,
``[k8s]``, and ``[mpi]`` extras and are exposed lazily through this
package's :func:`__getattr__` — accessing them when the underlying extra
is not installed raises :class:`AttributeError` with an install hint, so
a minimal install never imports ``distributed``, ``ray``, ``kubernetes``,
or ``mpi4py`` at package-import time.
"""

from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING, Any

# Disable Ray's auto-uv runtime_env hook before any `import ray` happens.
#
# When the driver runs under `uv run` (CI under `uv run pytest`, or any local
# `uv`-based dev shell), Ray detects the parent `uv run` process and silently
# rewrites `runtime_env` to relaunch each worker with `uv run python ...` from
# a packaged copy of the project's working directory. uv then rebuilds a fresh
# worker venv from the project's *base* dependencies — no extras — so the
# worker's `import ray` raises `ModuleNotFoundError` and `ray.init()` retries
# 5 times before raising `RaySystemError`.
#
# The constant Ray reads (`ray._private.ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV`)
# is evaluated once at the time `ray` is imported, so the env var must be set
# before any `import ray`. Setting it here covers both import paths to
# :class:`gmat_sweep.backends.ray.RayPool`: the package's lazy ``__getattr__``
# below imports ``ray`` for its install probe, and a direct
# ``from gmat_sweep.backends.ray import RayPool`` triggers this module first.
#
# `setdefault` respects an explicit user opt-in — set
# ``RAY_ENABLE_UV_RUN_RUNTIME_ENV=1`` in the shell to re-enable the hook.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

from gmat_sweep.backends.base import Pool
from gmat_sweep.backends.joblib import LocalJoblibPool

if TYPE_CHECKING:
    from gmat_sweep.backends.dask import DaskPool
    from gmat_sweep.backends.kubernetes import KubernetesJobPool
    from gmat_sweep.backends.mpi import MPIPool
    from gmat_sweep.backends.ray import RayPool

__all__ = ["DaskPool", "KubernetesJobPool", "LocalJoblibPool", "MPIPool", "Pool", "RayPool"]


_OPTIONAL_BACKENDS = {
    "DaskPool": ("distributed", "gmat_sweep.backends.dask", "dask"),
    "KubernetesJobPool": ("kubernetes", "gmat_sweep.backends.kubernetes", "k8s"),
    "MPIPool": ("mpi4py", "gmat_sweep.backends.mpi", "mpi"),
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
