"""Smoke test: the package imports and exposes a non-empty version string."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

import gmat_sweep
import gmat_sweep.distributions  # v0.2 module; imported here so coverage sees the stub.


def test_import() -> None:
    assert isinstance(gmat_sweep.__version__, str)
    assert gmat_sweep.__version__


def test_distributions_module_is_importable() -> None:
    """v0.2 module — currently a docstring stub. Importable so the coverage gate is meaningful."""
    assert gmat_sweep.distributions.__name__ == "gmat_sweep.distributions"


def test_cold_start_does_not_load_heavy_dependencies() -> None:
    """``import gmat_sweep`` must not pull pandas, pyarrow, or tqdm.

    These dominate the cold-start cost of ``gmat-sweep --help`` (300-800 ms
    on a warm filesystem) and only one path through the package actually
    needs them. The lazy attribute-access dispatch in ``gmat_sweep/__init__``
    pulls each heavy submodule on first reference.

    Runs in a child interpreter so a prior test's eager attribute access
    doesn't pollute ``sys.modules``.
    """
    script = textwrap.dedent(
        """
        import sys
        import gmat_sweep  # noqa: F401
        forbidden = {"pandas", "pyarrow", "pyarrow.dataset", "tqdm", "tqdm.auto"}
        leaked = sorted(forbidden & set(sys.modules))
        if leaked:
            raise SystemExit(f"leaked modules: {leaked}")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"cold start leaked dependencies: {result.stderr}"


def test_lazy_attribute_access_resolves_sweep() -> None:
    """``gmat_sweep.sweep`` must still resolve via PEP 562 ``__getattr__``."""
    fn = gmat_sweep.sweep
    assert callable(fn)


def test_lazy_attribute_access_raises_for_unknown_name() -> None:
    """Unknown attributes still raise ``AttributeError``."""
    with pytest.raises(AttributeError, match="no_such_symbol"):
        gmat_sweep.no_such_symbol  # noqa: B018
