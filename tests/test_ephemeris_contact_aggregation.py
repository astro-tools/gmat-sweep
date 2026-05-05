"""End-to-end ephemeris + contact aggregation — v0.2 validation suite (issue #40).

Drives a real sweep through the public ``api.sweep()`` against a fake ``gmat_run``
that emits ``EphemerisFile`` and ``ContactLocator`` outputs alongside the report,
then assembles the per-kind frames via ``lazy_ephemerides`` / ``lazy_contacts``
loaded from the on-disk manifest. Pins the worker → manifest → aggregator
pipeline for the two non-report output kinds — ``test_aggregate.py`` exercises
the aggregators on hand-crafted Parquets, but only an end-to-end run catches
regressions in the worker's column handling for ephemerides and the
``interval_id`` synthesis for contacts.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from gmat_sweep.aggregate import lazy_contacts, lazy_ephemerides
from gmat_sweep.api import sweep
from gmat_sweep.errors import SweepConfigError
from gmat_sweep.manifest import Manifest
from tests.conftest import FakeGmatRun, FakeMission, FakeResults


def _write_script(tmp_path: Path) -> Path:
    path = tmp_path / "mission.script"
    path.write_text("% GMAT mission\nCreate Spacecraft Sat;\n", encoding="utf-8")
    return path


def _report_frame() -> pd.DataFrame:
    return pd.DataFrame({"time": pd.to_datetime(["2026-05-04T00:00:00"]), "x": [1.0]})


def _ephemeris_frame_with_epoch_column() -> pd.DataFrame:
    """Ephemeris frame keyed on ``Epoch`` — gmat-run's actual column name.

    ``worker._synthesize_time_column`` copies (does not rename) the first
    datetime column to ``time`` so the user's original column is preserved
    through the aggregation. Using ``Epoch`` here lets the column-name
    round-trip test assert that contract end-to-end.
    """
    return pd.DataFrame(
        {
            "Epoch": pd.to_datetime(["2026-05-04T00:00:00", "2026-05-04T00:01:00"]),
            "X": [7000.0, 7001.0],
            "Y": [0.0, 1.0],
        }
    )


def _contact_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Start": pd.to_datetime(["2026-05-04T00:00:00", "2026-05-04T01:00:00"]),
            "Duration_s": [60.0, 90.0],
        }
    )


def _install_ephem_contact_loader(
    fake_gmat_run: FakeGmatRun,
    *,
    eph_kinds: dict[str, pd.DataFrame] | None = None,
    contacts: dict[str, pd.DataFrame] | None = None,
    setitem_hook: Any = None,
) -> None:
    """Install a loader whose ``run()`` returns reports + ephemeris + contacts."""
    eph = eph_kinds if eph_kinds is not None else {"SatEphem": _ephemeris_frame_with_epoch_column()}
    con = contacts if contacts is not None else {"GroundContact": _contact_frame()}

    def _load(path: Any, **_: Any) -> FakeMission:
        mission = FakeMission(script_path=Path(path), setitem_hook=setitem_hook)

        def _run(**_kwargs: Any) -> FakeResults:
            return FakeResults(
                reports={"R": _report_frame()},
                ephemerides={k: v.copy() for k, v in eph.items()},
                contacts={k: v.copy() for k, v in con.items()},
            )

        mission.run_hook = _run
        fake_gmat_run.last_mission = mission
        return mission

    fake_gmat_run.module.Mission = SimpleNamespace(load=_load)  # type: ignore[attr-defined]


# ---- 4-run sweep, both kinds aggregated ----------------------------------


def test_4_run_sweep_aggregates_ephemeris_and_contact_into_indexed_frames(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    _install_ephem_contact_loader(fake_gmat_run)

    sweep(
        script,
        grid={"Sat.SMA": [7000.0, 7100.0, 7200.0, 7300.0]},
        workers=1,
        out=out,
        progress=False,
    )

    manifest = Manifest.load(out / "manifest.jsonl")
    eph_df = lazy_ephemerides(manifest, out)
    contacts_df = lazy_contacts(manifest, out)

    assert eph_df.index.names == ["run_id", "time"]
    assert sorted(eph_df.index.get_level_values("run_id").unique().tolist()) == [0, 1, 2, 3]

    assert contacts_df.index.names == ["run_id", "interval_id"]
    assert sorted(contacts_df.index.get_level_values("run_id").unique().tolist()) == [0, 1, 2, 3]
    # The contact frame had 2 intervals per run, so interval_id is {0, 1}.
    assert sorted(contacts_df.xs(0, level="run_id").index.tolist()) == [0, 1]


# ---- failed run materialises as NaN row in both aggregators --------------


def test_failed_run_lands_as_nan_row_in_both_aggregators(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"

    def _setitem(_key: str, value: Any) -> None:
        if value == 7100.0:
            raise ValueError("rejected by GMAT")

    _install_ephem_contact_loader(fake_gmat_run, setitem_hook=_setitem)

    sweep(
        script,
        grid={"Sat.SMA": [7000.0, 7100.0, 7200.0, 7300.0]},
        workers=1,
        out=out,
        progress=False,
    )

    manifest = Manifest.load(out / "manifest.jsonl")

    eph_df = lazy_ephemerides(manifest, out)
    eph_failed = eph_df.xs(1, level="run_id")
    assert (eph_failed["__status"] == "failed").all()
    assert eph_failed.drop(columns=["__status"]).isna().all().all()

    contacts_df = lazy_contacts(manifest, out)
    con_failed = contacts_df.xs(1, level="run_id")
    assert (con_failed["__status"] == "failed").all()
    assert con_failed.drop(columns=["__status"]).isna().all().all()


# ---- multi-ephemeris with name=None raises -------------------------------


def test_multi_ephemeris_with_name_none_raises_listing_available_names(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    _install_ephem_contact_loader(
        fake_gmat_run,
        eph_kinds={
            "SatEphem": _ephemeris_frame_with_epoch_column(),
            "GroundEphem": _ephemeris_frame_with_epoch_column(),
        },
    )

    sweep(script, grid={"Sat.SMA": [7000.0, 7100.0]}, workers=1, out=out, progress=False)

    manifest = Manifest.load(out / "manifest.jsonl")
    with pytest.raises(SweepConfigError, match=r"GroundEphem.*SatEphem"):
        lazy_ephemerides(manifest, out)


# ---- worker → aggregator column-name round-trip --------------------------


def test_user_ephemeris_columns_survive_round_trip_through_aggregator(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    """The worker copies (does not rename) the first datetime column to
    ``time`` for the MultiIndex; the user's original ``Epoch`` column must
    survive untouched in the aggregated frame, alongside any data columns.
    Without this, a regression that started renaming the column would silently
    strip user data.
    """
    script = _write_script(tmp_path)
    out = tmp_path / "out"
    _install_ephem_contact_loader(fake_gmat_run)

    sweep(script, grid={"Sat.SMA": [7000.0, 7100.0]}, workers=1, out=out, progress=False)

    manifest = Manifest.load(out / "manifest.jsonl")
    eph_df = lazy_ephemerides(manifest, out)

    # The "time" column was synthesised by the worker and consumed as the
    # MultiIndex secondary level, so it should not appear as a data column.
    # The user's original Epoch + X + Y columns must still be present.
    assert set(eph_df.columns) >= {"Epoch", "X", "Y"}
    assert "time" not in eph_df.columns
    # Epoch values match the input (modulo Parquet datetime round-trip).
    eph_run_0 = eph_df.xs(0, level="run_id").reset_index(drop=True)
    expected_epoch = pd.to_datetime(["2026-05-04T00:00:00", "2026-05-04T00:01:00"])
    pd.testing.assert_index_equal(
        pd.Index(eph_run_0["Epoch"].tolist()), pd.Index(expected_epoch.tolist())
    )
