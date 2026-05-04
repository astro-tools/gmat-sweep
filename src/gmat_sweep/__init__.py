"""gmat-sweep: parameter sweeps and Monte Carlo dispersions over GMAT missions in parallel."""

from importlib.metadata import PackageNotFoundError, version

from gmat_sweep.errors import (
    BackendError,
    GmatSweepError,
    ManifestCorruptError,
    RunFailed,
    SweepConfigError,
)
from gmat_sweep.grids import expand_grid_to_run_specs, full_factorial
from gmat_sweep.spec import RunOutcome, RunSpec, SweepSpec

try:
    __version__ = version("gmat-sweep")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "BackendError",
    "GmatSweepError",
    "ManifestCorruptError",
    "RunFailed",
    "RunOutcome",
    "RunSpec",
    "SweepConfigError",
    "SweepSpec",
    "__version__",
    "expand_grid_to_run_specs",
    "full_factorial",
]
