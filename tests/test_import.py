"""Smoke test: the package imports and exposes a non-empty version string."""

from __future__ import annotations

import gmat_sweep
import gmat_sweep.distributions  # v0.2 module; imported here so coverage sees the stub.


def test_import() -> None:
    assert isinstance(gmat_sweep.__version__, str)
    assert gmat_sweep.__version__


def test_distributions_module_is_importable() -> None:
    """v0.2 module — currently a docstring stub. Importable so the coverage gate is meaningful."""
    assert gmat_sweep.distributions.__name__ == "gmat_sweep.distributions"
