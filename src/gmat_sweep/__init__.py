"""gmat-sweep: parameter sweeps and Monte Carlo dispersions over GMAT missions in parallel."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("gmat-sweep")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
