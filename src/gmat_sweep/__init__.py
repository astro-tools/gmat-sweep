"""gmat-sweep: parameter sweeps and Monte Carlo dispersions over GMAT missions in parallel.

Public symbols are exposed via :pep:`562` lazy attribute access. Submodules that
transitively pull in :mod:`pandas`, :mod:`pyarrow`, or :mod:`tqdm` are only
imported on first reference, so ``import gmat_sweep`` (and the
``gmat-sweep --help`` cold path) does not pay their import cost. The lazy
surface is still discoverable through :func:`dir` and resolvable by static
type checkers via the :data:`typing.TYPE_CHECKING` re-export block below.
"""

from __future__ import annotations

import logging
import sys
import types
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from gmat_sweep.errors import (
    BackendError,
    GmatSweepError,
    ManifestCorruptError,
    RunFailed,
    SweepConfigError,
)

if TYPE_CHECKING:
    # Re-exported for static type checkers. The actual imports happen lazily
    # via __getattr__ — never at module load time — so heavy submodules
    # (aggregate, api, sweep, sensitivity, plotting) are not pulled on
    # `import gmat_sweep`.
    from gmat_sweep.aggregate import (
        lazy_contacts,
        lazy_ephemerides,
        lazy_fused_reports,
        lazy_multiindex,
        mc_convergence,
        sweep_diff,
        sweep_summary,
    )
    from gmat_sweep.api import (
        latin_hypercube,
        latin_hypercube_extend,
        monte_carlo,
        monte_carlo_extend,
        sweep,
    )
    from gmat_sweep.backends import LocalJoblibPool, Pool
    from gmat_sweep.grids import (
        expand_grid_to_run_specs,
        expand_latin_hypercube_to_run_specs,
        expand_monte_carlo_extension_to_run_specs,
        expand_monte_carlo_to_run_specs,
        expand_samples_to_run_specs,
        full_factorial,
        latin_hypercube_samples,
    )
    from gmat_sweep.manifest import (
        MANIFEST_SCHEMA_VERSION,
        Manifest,
        ManifestEntry,
        canonical_script_sha256,
    )
    from gmat_sweep.sensitivity import sobol_analyze, sobol_sample
    from gmat_sweep.spec import RunOutcome, RunSpec, SweepSpec
    from gmat_sweep.sweep import Sweep


try:
    __version__ = version("gmat-sweep")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Default the gmat-sweep logger to WARNING. Sweep orchestration emits
# diagnostic INFO records on every per-run completion that would otherwise
# spam the parent process. Only set the level if the user has not configured
# it before importing gmat-sweep, so explicit `logging.getLogger("gmat_sweep")`
# config wins.
_logger = logging.getLogger(__name__)
if _logger.level == logging.NOTSET:
    _logger.setLevel(logging.WARNING)


_LAZY_ATTRS: dict[str, str] = {
    "lazy_contacts": "gmat_sweep.aggregate",
    "lazy_ephemerides": "gmat_sweep.aggregate",
    "lazy_fused_reports": "gmat_sweep.aggregate",
    "lazy_multiindex": "gmat_sweep.aggregate",
    "mc_convergence": "gmat_sweep.aggregate",
    "sweep_diff": "gmat_sweep.aggregate",
    "sweep_summary": "gmat_sweep.aggregate",
    "latin_hypercube": "gmat_sweep.api",
    "latin_hypercube_extend": "gmat_sweep.api",
    "monte_carlo": "gmat_sweep.api",
    "monte_carlo_extend": "gmat_sweep.api",
    "sweep": "gmat_sweep.api",
    "LocalJoblibPool": "gmat_sweep.backends",
    "Pool": "gmat_sweep.backends",
    "expand_grid_to_run_specs": "gmat_sweep.grids",
    "expand_latin_hypercube_to_run_specs": "gmat_sweep.grids",
    "expand_monte_carlo_extension_to_run_specs": "gmat_sweep.grids",
    "expand_monte_carlo_to_run_specs": "gmat_sweep.grids",
    "expand_samples_to_run_specs": "gmat_sweep.grids",
    "full_factorial": "gmat_sweep.grids",
    "latin_hypercube_samples": "gmat_sweep.grids",
    "MANIFEST_SCHEMA_VERSION": "gmat_sweep.manifest",
    "Manifest": "gmat_sweep.manifest",
    "ManifestEntry": "gmat_sweep.manifest",
    "canonical_script_sha256": "gmat_sweep.manifest",
    "sobol_analyze": "gmat_sweep.sensitivity",
    "sobol_sample": "gmat_sweep.sensitivity",
    "RunOutcome": "gmat_sweep.spec",
    "RunSpec": "gmat_sweep.spec",
    "SweepSpec": "gmat_sweep.spec",
    "Sweep": "gmat_sweep.sweep",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY_ATTRS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    attr = getattr(import_module(module_path), name)
    globals()[name] = attr
    return attr


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_ATTRS.keys())


class _GmatSweepModule(types.ModuleType):
    """Module subclass that repairs the ``gmat_sweep.sweep`` name collision.

    The public API exposes ``gmat_sweep.sweep`` as a function (re-exported
    from :mod:`gmat_sweep.api`), but there's also a submodule named
    ``gmat_sweep.sweep`` that defines :class:`Sweep`. Whenever that submodule
    is imported — via :pep:`562` lazy access of :class:`Sweep` here, via
    ``from gmat_sweep.sweep import Sweep`` in downstream code, or via any
    other path — Python writes the submodule into :data:`gmat_sweep.__dict__`
    at the ``sweep`` key, shadowing the function.

    The old eager :mod:`gmat_sweep.__init__` papered over this by reassigning
    ``sweep`` to the function at the bottom of the module. With lazy loading
    we can't run that reassignment unconditionally — it would force the
    :mod:`gmat_sweep.api` import (and the pandas / pyarrow / tqdm chain it
    pulls) on every ``import gmat_sweep``. Instead we detect the shadow on
    attribute access and resolve it on demand. The added cost is one
    :func:`isinstance` check per attribute read on the package.
    """

    def __getattribute__(self, name: str) -> Any:
        value = super().__getattribute__(name)
        if name == "sweep" and isinstance(value, types.ModuleType):
            # Submodule shadow detected. Resolve to the public function from
            # gmat_sweep.api and cache it in __dict__ so subsequent reads
            # short-circuit through normal attribute lookup.
            from gmat_sweep.api import sweep as _fn

            super().__setattr__("sweep", _fn)
            return _fn
        return value


sys.modules[__name__].__class__ = _GmatSweepModule


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "BackendError",
    "GmatSweepError",
    "LocalJoblibPool",
    "Manifest",
    "ManifestCorruptError",
    "ManifestEntry",
    "Pool",
    "RunFailed",
    "RunOutcome",
    "RunSpec",
    "Sweep",
    "SweepConfigError",
    "SweepSpec",
    "__version__",
    "canonical_script_sha256",
    "expand_grid_to_run_specs",
    "expand_latin_hypercube_to_run_specs",
    "expand_monte_carlo_extension_to_run_specs",
    "expand_monte_carlo_to_run_specs",
    "expand_samples_to_run_specs",
    "full_factorial",
    "latin_hypercube",
    "latin_hypercube_extend",
    "latin_hypercube_samples",
    "lazy_contacts",
    "lazy_ephemerides",
    "lazy_fused_reports",
    "lazy_multiindex",
    "mc_convergence",
    "monte_carlo",
    "monte_carlo_extend",
    "sobol_analyze",
    "sobol_sample",
    "sweep",
    "sweep_diff",
    "sweep_summary",
]
