"""Per-run DataFrame parity between ``sweep()`` and a direct ``Mission`` call.

For each run in a 4-run reference sweep, the per-run slice of the aggregated
DataFrame must equal what a direct :class:`gmat_run.Mission` call with the same
overrides would produce — after column-wise type coercion. This is the
end-to-end correctness check the issue's first acceptance bullet calls out.

"After column-wise type coercion" means: the aggregator adds a synthesised
``time`` column (copy of the first datetime column) and a ``__status`` column,
and reindexes by ``(run_id, time)``. The direct call produces the bare
ReportFile frame with no ``time`` / ``__status`` and an integer index. The
test strips the aggregator's additions and compares the underlying data
columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

import gmat_sweep
from gmat_sweep.backends.joblib import LocalJoblibPool

pytestmark = pytest.mark.integration

gmat_run = pytest.importorskip("gmat_run")


_GRID = {"Sat.SMA": [7050.0, 7100.0, 7150.0, 7200.0]}


@pytest.fixture(scope="module")
def four_run_sweep_df(
    leo_basic_script: Path, tmp_path_factory: pytest.TempPathFactory
) -> pd.DataFrame:
    """Run the reference 4-run sweep once and reuse it across the module."""
    out = tmp_path_factory.mktemp("four-run-sweep")
    return gmat_sweep.sweep(
        leo_basic_script, grid=_GRID, backend=LocalJoblibPool(max_workers=2), out=out
    )


def _direct_run(script: Path, overrides: dict[str, Any], working_dir: Path) -> pd.DataFrame:
    """Mirror what ``gmat_sweep.worker.run_one`` does, minus the Parquet write."""
    mission = gmat_run.Mission.load(script)
    for key, value in overrides.items():
        mission[key] = value
    results = mission.run(working_dir=working_dir)
    (frame,) = results.reports.values()  # fixture has exactly one ReportFile
    return cast(pd.DataFrame, frame.reset_index(drop=True))


def _strip_aggregator_columns(slice_df: pd.DataFrame) -> pd.DataFrame:
    """Remove the columns the aggregator synthesises so the frame matches a direct mission.run().

    ``.xs(run_id, level="run_id")`` leaves ``time`` as the surviving index level;
    drop it (the user's original epoch column ``Sat.UTCGregorian`` carries the
    same data and round-trips into the comparison frame), then drop the
    aggregator-only ``__status`` column and renumber the index.
    """
    return cast(
        pd.DataFrame,
        slice_df.reset_index(level="time", drop=True)
        .drop(columns=["__status"])
        .reset_index(drop=True),
    )


@pytest.mark.parametrize("run_id", [0, 1, 2, 3])
def test_per_run_frame_matches_direct_mission_call(
    leo_basic_script: Path,
    four_run_sweep_df: pd.DataFrame,
    tmp_path: Path,
    run_id: int,
) -> None:
    sma = _GRID["Sat.SMA"][run_id]
    swept = _strip_aggregator_columns(
        cast(pd.DataFrame, four_run_sweep_df.xs(run_id, level="run_id"))
    )

    direct = _direct_run(
        leo_basic_script,
        overrides={"Sat.SMA": sma},
        working_dir=tmp_path / f"direct-{run_id}",
    )

    pd.testing.assert_frame_equal(
        swept,
        direct,
        check_exact=False,
        rtol=1e-9,
        atol=1e-6,
    )


def test_run_ids_cover_full_grid(four_run_sweep_df: pd.DataFrame) -> None:
    run_ids = sorted(four_run_sweep_df.index.get_level_values("run_id").unique().tolist())
    assert run_ids == list(range(len(_GRID["Sat.SMA"])))


def test_status_column_is_ok_for_every_run(four_run_sweep_df: pd.DataFrame) -> None:
    assert (four_run_sweep_df["__status"] == "ok").all()
