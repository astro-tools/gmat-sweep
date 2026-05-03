"""Smoke test: the package imports and exposes a non-empty version string."""

from __future__ import annotations

import gmat_sweep


def test_import() -> None:
    assert isinstance(gmat_sweep.__version__, str)
    assert gmat_sweep.__version__
