"""Tests for the gmat_sweep.backends package's lazy __getattr__ and import-time env setup.

The package's ``__init__`` does two things worth pinning at unit level:

1. A lazy ``__getattr__`` that imports :class:`DaskPool` and :class:`RayPool`
   only when the user asks for them. With the matching extra installed, the
   attribute access returns the class. Without it, the access raises
   :class:`AttributeError` whose message names the extra so a missing-extras
   error includes a copy-paste install command.

2. ``RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`` is set via ``os.environ.setdefault`` at
   import time so a driver started under ``uv run`` does not silently rebuild
   each Ray worker's venv from the project's *base* dependencies (see #76).
   ``setdefault`` is the load-bearing detail — a user who explicitly sets the
   env var to ``"1"`` (re-enabling Ray's auto-`uv` hook) must have their
   choice preserved.
"""

from __future__ import annotations

import importlib
import sys

import pytest

import gmat_sweep.backends as backends_pkg


def test_unknown_attribute_raises_attribute_error() -> None:
    """``gmat_sweep.backends.<unknown>`` raises ``AttributeError`` with the
    standard ``module ... has no attribute ...`` message."""
    with pytest.raises(AttributeError) as ei:
        backends_pkg.NoSuchPool  # noqa: B018 - intentional attribute access  # type: ignore[attr-defined]
    assert "gmat_sweep.backends" in str(ei.value)
    assert "NoSuchPool" in str(ei.value)


def test_dask_pool_attribute_error_when_distributed_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``distributed`` cannot be imported, ``gmat_sweep.backends.DaskPool``
    raises :class:`AttributeError` whose message names the ``[dask]`` extra."""
    # ``setitem(... None)`` makes ``importlib.import_module("distributed")``
    # raise ``ImportError`` — same trick the dask backend's lazy-import test
    # uses (see tests/test_backends_dask.py).
    monkeypatch.setitem(sys.modules, "distributed", None)
    with pytest.raises(AttributeError) as ei:
        backends_pkg.DaskPool  # noqa: B018 - intentional attribute access
    msg = str(ei.value)
    assert "DaskPool" in msg
    assert "[dask]" in msg
    assert "pip install gmat-sweep[dask]" in msg


def test_ray_pool_attribute_error_when_ray_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``ray`` cannot be imported, ``gmat_sweep.backends.RayPool`` raises
    :class:`AttributeError` whose message names the ``[ray]`` extra."""
    monkeypatch.setitem(sys.modules, "ray", None)
    with pytest.raises(AttributeError) as ei:
        backends_pkg.RayPool  # noqa: B018 - intentional attribute access
    msg = str(ei.value)
    assert "RayPool" in msg
    assert "[ray]" in msg
    assert "pip install gmat-sweep[ray]" in msg


def test_dask_pool_attribute_returns_class_when_extra_installed() -> None:
    """The happy path: with ``distributed`` importable, the attribute access
    returns the :class:`DaskPool` class itself."""
    pytest.importorskip("distributed")
    from gmat_sweep.backends.dask import DaskPool as direct_cls

    assert backends_pkg.DaskPool is direct_cls


def test_ray_pool_attribute_returns_class_when_extra_installed() -> None:
    """The happy path: with ``ray`` importable, the attribute access returns
    the :class:`RayPool` class itself."""
    pytest.importorskip("ray")
    from gmat_sweep.backends.ray import RayPool as direct_cls

    assert backends_pkg.RayPool is direct_cls


def test_ray_runtime_env_var_set_to_zero_when_unset_at_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``RAY_ENABLE_UV_RUN_RUNTIME_ENV`` unset, importing
    :mod:`gmat_sweep.backends` sets it to ``"0"``."""
    monkeypatch.delenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", raising=False)
    importlib.reload(backends_pkg)
    import os

    assert os.environ.get("RAY_ENABLE_UV_RUN_RUNTIME_ENV") == "0"


def test_ray_runtime_env_var_preserves_explicit_user_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the user has set ``RAY_ENABLE_UV_RUN_RUNTIME_ENV=1`` explicitly,
    importing :mod:`gmat_sweep.backends` does not overwrite it. ``setdefault``
    is the load-bearing detail — flipping to unconditional ``__setitem__``
    would silently undo a deliberate opt-in."""
    monkeypatch.setenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "1")
    importlib.reload(backends_pkg)
    import os

    assert os.environ.get("RAY_ENABLE_UV_RUN_RUNTIME_ENV") == "1"
