"""gmat-sweep: parameter sweeps and Monte Carlo dispersions over GMAT missions in parallel."""

import logging
from importlib.metadata import PackageNotFoundError, version

from gmat_sweep.aggregate import lazy_contacts, lazy_ephemerides, lazy_multiindex
from gmat_sweep.api import latin_hypercube, monte_carlo, sweep
from gmat_sweep.backends import Pool
from gmat_sweep.errors import (
    BackendError,
    GmatSweepError,
    ManifestCorruptError,
    RunFailed,
    SweepConfigError,
)
from gmat_sweep.grids import (
    expand_grid_to_run_specs,
    expand_latin_hypercube_to_run_specs,
    expand_monte_carlo_to_run_specs,
    expand_samples_to_run_specs,
    full_factorial,
    latin_hypercube_samples,
)
from gmat_sweep.manifest import Manifest, ManifestEntry, canonical_script_sha256
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

__all__ = [
    "BackendError",
    "GmatSweepError",
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
    "expand_monte_carlo_to_run_specs",
    "expand_samples_to_run_specs",
    "full_factorial",
    "latin_hypercube",
    "latin_hypercube_samples",
    "lazy_contacts",
    "lazy_ephemerides",
    "lazy_multiindex",
    "monte_carlo",
    "sweep",
]
