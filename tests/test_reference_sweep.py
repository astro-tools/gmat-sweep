"""16-run SMA grid against the LEO fixture, compared to a committed golden Parquet.

This is the canary test: a non-trivial sweep run end-to-end through real GMAT
and frozen against a checked-in expected output. A diff means *something*
about the stack changed — a force-model default, a propagator step, a
parser column rename, the worker's time-column synthesis, the aggregator's
frame shape — and someone needs to look at it on purpose.

Regenerating the golden when behaviour changes deliberately
-----------------------------------------------------------

If you've made a change you *expect* to alter the reference output, regenerate
the golden file by setting an environment variable and re-running this single
test:

    GMAT_SWEEP_REGEN_GOLDEN=1 uv run pytest tests/test_reference_sweep.py -m integration

The test re-runs the sweep, writes the result to ``tests/data/golden/sma_16.parquet``
(overwriting whatever was there), and skips the comparison. Commit the new
Parquet alongside the change that motivated it. Without the env var the test
asserts as normal and a diff fails the build.

Cross-OS comparison
-------------------

Floating-point determinism across operating systems is approximate, not exact.
The comparison uses :func:`pandas.testing.assert_frame_equal` with tight
``rtol`` / ``atol`` so a real behaviour change still breaks the build but
sub-ULP CPU drift between Linux and Windows runners doesn't.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

import gmat_sweep
from gmat_sweep.backends.joblib import LocalJoblibPool

pytestmark = pytest.mark.integration

pytest.importorskip("gmat_run")


_GRID = {"Sat.SMA": list(np.linspace(7000.0, 7300.0, 16))}
_GOLDEN_PATH = Path(__file__).parent / "data" / "golden" / "sma_16.parquet"
_REGEN_ENV_VAR = "GMAT_SWEEP_REGEN_GOLDEN"


def _normalise_for_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Reset MultiIndex to columns so Parquet round-trip is clean."""
    return cast(pd.DataFrame, df.reset_index())


def _frames_equal(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_exact=False,
        rtol=1e-9,
        atol=1e-6,
    )


def test_reference_sma_sweep_matches_golden_parquet(leo_basic_script: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    df = gmat_sweep.sweep(leo_basic_script, grid=_GRID, backend=LocalJoblibPool(workers=2), out=out)

    flat = _normalise_for_comparison(df)

    if os.environ.get(_REGEN_ENV_VAR) == "1":
        _GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        flat.to_parquet(_GOLDEN_PATH)
        pytest.skip(
            f"regenerated {_GOLDEN_PATH} from this run; commit the new file "
            f"and unset {_REGEN_ENV_VAR} to re-enable the comparison."
        )

    if not _GOLDEN_PATH.exists():
        pytest.fail(
            f"golden file not found at {_GOLDEN_PATH}. "
            f"Run with {_REGEN_ENV_VAR}=1 to seed it from the current sweep "
            "and commit the result."
        )

    expected = pd.read_parquet(_GOLDEN_PATH)
    _frames_equal(flat, expected)
