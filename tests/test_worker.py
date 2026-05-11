"""Tests for gmat_sweep.worker.run_one — happy path, every failure mode, never raises."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from gmat_sweep.spec import RunSpec
from gmat_sweep.worker import run_one
from tests.conftest import FakeGmatRun, FakeGmatRunError, FakeResults


def _make_spec(
    *,
    output_dir: Path,
    run_id: int = 0,
    overrides: dict[str, Any] | None = None,
    run_options: dict[str, Any] | None = None,
    seed: int | None = None,
    script_name: str = "mission.script",
) -> RunSpec:
    return RunSpec(
        script_path=Path(f"/missions/{script_name}"),
        overrides=overrides if overrides is not None else {},
        output_dir=output_dir,
        run_id=run_id,
        seed=seed,
        run_options=run_options if run_options is not None else {},
    )


# ---- happy path ----------------------------------------------------------


def test_run_one_ok_writes_parquet_and_returns_ok(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    df_a = pd.DataFrame({"t": [0.0, 1.0], "x": [1.0, 2.0]})
    df_b = pd.DataFrame({"t": [0.0], "y": [42.0]})

    def _run(**_: Any) -> FakeResults:
        return FakeResults(reports={"R1": df_a, "R2": df_b}, log="GMAT engine log line\n")

    fake_gmat_run.install_loader(run_hook=_run)

    out_dir = tmp_path / "run-0"
    spec = _make_spec(
        output_dir=out_dir,
        overrides={"Sat.SMA": 7000.0, "Sat.ECC": 0.01},
        run_options={"overwrite": True},
    )
    outcome = run_one(spec)

    assert outcome.status == "ok"
    assert outcome.run_id == 0
    assert outcome.stderr is None
    assert outcome.duration_s >= 0
    assert outcome.started_at <= outcome.ended_at
    assert set(outcome.output_paths) == {"report__R1", "report__R2"}

    # Parquet round-trip.
    pd.testing.assert_frame_equal(pd.read_parquet(outcome.output_paths["report__R1"]), df_a)
    pd.testing.assert_frame_equal(pd.read_parquet(outcome.output_paths["report__R2"]), df_b)

    # Worker log present and contains override + engine-log lines.
    log_text = (out_dir / "worker.log").read_text(encoding="utf-8")
    assert "Sat.SMA" in log_text
    assert "GMAT engine log line" in log_text
    assert "status=ok" in log_text

    # mission.run got the working_dir and the forwarded run_options.
    assert fake_gmat_run.last_mission is not None
    assert fake_gmat_run.last_mission.run_kwargs_log == [
        {"working_dir": out_dir, "overwrite": True}
    ]
    # Overrides were applied in spec order.
    assert fake_gmat_run.last_mission.overrides_log == [
        ("Sat.SMA", 7000.0),
        ("Sat.ECC", 0.01),
    ]


def test_run_one_ok_with_no_reports_returns_empty_output_paths(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    spec = _make_spec(output_dir=tmp_path / "run-0")
    outcome = run_one(spec)

    assert outcome.status == "ok"
    assert outcome.output_paths == {}
    assert outcome.stderr is None
    assert (tmp_path / "run-0" / "worker.log").exists()


def test_run_one_creates_output_dir_when_missing(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    nested = tmp_path / "a" / "b" / "run-0"
    assert not nested.exists()
    outcome = run_one(_make_spec(output_dir=nested))

    assert outcome.status == "ok"
    assert nested.is_dir()


def test_run_one_logs_seed_when_set(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    spec = _make_spec(output_dir=tmp_path / "run-0", seed=1729)
    run_one(spec)

    log_text = (tmp_path / "run-0" / "worker.log").read_text(encoding="utf-8")
    assert "seed=1729" in log_text


# ---- failure modes -------------------------------------------------------


def test_run_one_failed_when_override_raises(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    def _setitem(key: str, _value: Any) -> None:
        if key == "Sat.ECC":
            raise ValueError(f"GMAT rejected write to {key!r}: ECC > 1")

    fake_gmat_run.install_loader(setitem_hook=_setitem)

    out_dir = tmp_path / "run-0"
    outcome = run_one(_make_spec(output_dir=out_dir, overrides={"Sat.SMA": 7000.0, "Sat.ECC": 1.5}))

    assert outcome.status == "failed"
    assert outcome.stderr is not None
    assert "Sat.ECC" in outcome.stderr
    assert "Traceback" in outcome.stderr
    assert outcome.output_paths == {}
    log_text = (out_dir / "worker.log").read_text(encoding="utf-8")
    assert "status=failed" in log_text


def test_run_one_failed_when_load_raises(tmp_path: Path, fake_gmat_run: FakeGmatRun) -> None:
    fake_gmat_run.install_failing_loader(FileNotFoundError("/missions/nope.script does not exist"))

    out_dir = tmp_path / "run-0"
    outcome = run_one(_make_spec(output_dir=out_dir, script_name="nope.script"))

    assert outcome.status == "failed"
    assert outcome.stderr is not None
    assert "nope.script" in outcome.stderr


def test_run_one_failed_when_run_raises_includes_engine_log(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    engine_log = "GMAT: detailed engine error trace"

    def _run(**_: Any) -> FakeResults:
        raise FakeGmatRunError("RunScript returned status -1", log=engine_log)

    fake_gmat_run.install_loader(run_hook=_run)

    outcome = run_one(_make_spec(output_dir=tmp_path / "run-0"))

    assert outcome.status == "failed"
    assert outcome.stderr is not None
    assert "RunScript returned status -1" in outcome.stderr
    assert engine_log in outcome.stderr


def test_run_one_failed_when_gmat_run_import_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Block the import without using the fake_gmat_run fixture: install a
    # meta-path finder that raises ImportError for "gmat_run".
    monkeypatch.delitem(sys.modules, "gmat_run", raising=False)

    class _BlockingFinder:
        def find_spec(self, name: str, _path: Any = None, _target: Any = None) -> None:
            if name == "gmat_run":
                raise ImportError("simulated gmat_run install failure")
            return None

    monkeypatch.setattr(sys, "meta_path", [_BlockingFinder(), *sys.meta_path])

    out_dir = tmp_path / "run-0"
    outcome = run_one(_make_spec(output_dir=out_dir))

    assert outcome.status == "failed"
    assert outcome.stderr is not None
    assert "gmat_run" in outcome.stderr


def test_run_one_failed_when_parquet_write_fails(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    df = pd.DataFrame({"t": [0.0]})

    def _run(**_: Any) -> FakeResults:
        return FakeResults(reports={"R1": df})

    fake_gmat_run.install_loader(run_hook=_run)

    out_dir = tmp_path / "run-0"
    out_dir.mkdir()
    # Block the parquet write by pre-creating report__R1.parquet as a
    # *directory* — df.to_parquet then fails with
    # IsADirectoryError / PermissionError.
    (out_dir / "report__R1.parquet").mkdir()

    outcome = run_one(_make_spec(output_dir=out_dir))

    assert outcome.status == "failed"
    assert outcome.stderr is not None
    assert "Traceback" in outcome.stderr


# ---- contract: never raises ----------------------------------------------


@pytest.mark.parametrize(
    "scenario",
    ["load_raises", "setitem_raises", "run_raises", "import_blocked"],
)
def test_run_one_never_raises_for_known_failure_modes(
    scenario: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    if scenario == "import_blocked":
        monkeypatch.delitem(sys.modules, "gmat_run", raising=False)

        class _BlockingFinder:
            def find_spec(self, name: str, _path: Any = None, _target: Any = None) -> None:
                if name == "gmat_run":
                    raise ImportError("blocked")
                return None

        monkeypatch.setattr(sys, "meta_path", [_BlockingFinder(), *sys.meta_path])
    else:
        fake = request.getfixturevalue("fake_gmat_run")
        if scenario == "load_raises":
            fake.install_failing_loader(RuntimeError("boom"))
        elif scenario == "setitem_raises":

            def _setitem(*_a: Any, **_k: Any) -> None:
                raise RuntimeError("nope")

            fake.install_loader(setitem_hook=_setitem)
        elif scenario == "run_raises":

            def _run(**_: Any) -> FakeResults:
                raise RuntimeError("explosion")

            fake.install_loader(run_hook=_run)

    spec = _make_spec(output_dir=tmp_path / "run-0", overrides={"x": 1})
    # The contract: this call returns; it does not raise.
    outcome = run_one(spec)
    assert outcome.status == "failed"
    assert outcome.stderr


def test_run_one_failed_when_filehandler_init_raises(
    tmp_path: Path, fake_gmat_run: FakeGmatRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A FileHandler init failure must be folded into RunOutcome.failed, not raised.

    The bug fixed in #134 (item 4): mkdir succeeded but FileHandler.__init__
    raised (permission denied, EROFS, ENOSPC), and the exception escaped
    run_one, violating the module's "never raises" contract.
    """
    import logging

    original = logging.FileHandler

    def _exploding_filehandler(*args: Any, **kwargs: Any) -> logging.FileHandler:
        raise PermissionError("simulated FileHandler init failure")

    monkeypatch.setattr(logging, "FileHandler", _exploding_filehandler)
    try:
        spec = _make_spec(output_dir=tmp_path / "run-0")
        outcome = run_one(spec)
    finally:
        monkeypatch.setattr(logging, "FileHandler", original)

    assert outcome.status == "failed"
    assert outcome.stderr is not None
    assert "PermissionError" in outcome.stderr
    assert "simulated FileHandler init failure" in outcome.stderr


# ---- time-column synthesis ----------------------------------------------
#
# gmat-run's ReportFile parser names columns after the GMAT field (e.g.
# "Sat.UTCGregorian"); aggregate.lazy_multiindex needs a column literally
# called "time". The worker copies the first datetime column to "time"
# before to_parquet so the user's original column names survive into the
# aggregated DataFrame.


def test_run_one_synthesizes_time_column_from_first_datetime(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    df = pd.DataFrame(
        {
            "Sat.UTCGregorian": pd.to_datetime(["2026-05-04T00:00:00", "2026-05-04T00:00:30"]),
            "Sat.SMA": [7000.0, 7000.0],
        }
    )

    def _run(**_: Any) -> FakeResults:
        return FakeResults(reports={"R": df})

    fake_gmat_run.install_loader(run_hook=_run)
    out_dir = tmp_path / "run-0"
    outcome = run_one(_make_spec(output_dir=out_dir))

    assert outcome.status == "ok"
    written = pd.read_parquet(outcome.output_paths["report__R"])
    assert "time" in written.columns
    assert "Sat.UTCGregorian" in written.columns
    pd.testing.assert_series_equal(written["time"], written["Sat.UTCGregorian"], check_names=False)


def test_run_one_leaves_existing_time_column_untouched(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    explicit_time = pd.to_datetime(["2026-05-04T00:00:00"])
    other_dt = pd.to_datetime(["2099-01-01T00:00:00"])
    df = pd.DataFrame({"time": explicit_time, "Sat.UTCGregorian": other_dt, "x": [1.0]})

    def _run(**_: Any) -> FakeResults:
        return FakeResults(reports={"R": df})

    fake_gmat_run.install_loader(run_hook=_run)
    outcome = run_one(_make_spec(output_dir=tmp_path / "run-0"))

    written = pd.read_parquet(outcome.output_paths["report__R"])
    pd.testing.assert_series_equal(
        written["time"], pd.Series(explicit_time, name="time"), check_names=False
    )


def test_run_one_with_no_datetime_columns_writes_unchanged(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    # No datetime column anywhere → nothing to synthesize. The Parquet matches
    # the input frame verbatim; aggregate.lazy_multiindex's existing ValueError
    # ("missing the 'time' column") still fires for this user, by design.
    df = pd.DataFrame({"t_seconds": [0.0, 1.0], "x": [1.0, 2.0]})

    def _run(**_: Any) -> FakeResults:
        return FakeResults(reports={"R": df})

    fake_gmat_run.install_loader(run_hook=_run)
    outcome = run_one(_make_spec(output_dir=tmp_path / "run-0"))

    written = pd.read_parquet(outcome.output_paths["report__R"])
    assert "time" not in written.columns
    pd.testing.assert_frame_equal(written, df)


# ---- ephemeris writes ---------------------------------------------------
#
# EphemerisFile DataFrames carry their epoch in a column named "Epoch" (OEM,
# STK, and SPK parsers all surface it that way). The worker reuses
# _synthesize_time_column so the same (run_id, time) aggregator works against
# ephemeris parquets without renaming the user's columns.


def test_run_one_writes_ephemeris_parquet_with_synthesized_time(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    eph = pd.DataFrame(
        {
            "Epoch": pd.to_datetime(["2026-05-04T00:00:00", "2026-05-04T00:01:00"]),
            "X": [7000.0, 7001.0],
        }
    )

    def _run(**_: Any) -> FakeResults:
        return FakeResults(ephemerides={"SatEphem": eph})

    fake_gmat_run.install_loader(run_hook=_run)
    out_dir = tmp_path / "run-0"
    outcome = run_one(_make_spec(output_dir=out_dir))

    assert outcome.status == "ok"
    assert set(outcome.output_paths) == {"ephemeris__SatEphem"}
    written = pd.read_parquet(outcome.output_paths["ephemeris__SatEphem"])
    assert "time" in written.columns
    assert "Epoch" in written.columns
    pd.testing.assert_series_equal(written["time"], written["Epoch"], check_names=False)


# ---- contact writes -----------------------------------------------------
#
# ContactLocator DataFrames have variable schemas (Legacy vs. five tabular
# variants) and are intervals, not point samples. The worker assigns a fresh
# integer interval_id column so the contacts aggregator can use
# (run_id, interval_id) as the MultiIndex.


def test_run_one_writes_contact_parquet_with_interval_id(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    starts = pd.to_datetime(["2026-05-04T00:00:00", "2026-05-04T01:00:00"])
    stops = pd.to_datetime(["2026-05-04T00:05:00", "2026-05-04T01:08:00"])
    contact = pd.DataFrame({"Start": starts, "Stop": stops, "Observer": ["GS1", "GS1"]})

    def _run(**_: Any) -> FakeResults:
        return FakeResults(contacts={"GroundContact": contact})

    fake_gmat_run.install_loader(run_hook=_run)
    out_dir = tmp_path / "run-0"
    outcome = run_one(_make_spec(output_dir=out_dir))

    assert outcome.status == "ok"
    assert set(outcome.output_paths) == {"contact__GroundContact"}
    written = pd.read_parquet(outcome.output_paths["contact__GroundContact"])
    assert list(written["interval_id"]) == [0, 1]
    assert list(written.columns) == ["Start", "Stop", "Observer", "interval_id"]


def test_run_one_writes_all_three_kinds_side_by_side(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    report = pd.DataFrame({"time": pd.to_datetime(["2026-05-04T00:00:00"]), "x": [1.0]})
    eph = pd.DataFrame({"Epoch": pd.to_datetime(["2026-05-04T00:00:00"]), "X": [7000.0]})
    contact = pd.DataFrame({"Start": pd.to_datetime(["2026-05-04T00:00:00"]), "Duration": [60.0]})

    def _run(**_: Any) -> FakeResults:
        return FakeResults(reports={"R": report}, ephemerides={"E": eph}, contacts={"C": contact})

    fake_gmat_run.install_loader(run_hook=_run)
    out_dir = tmp_path / "run-0"
    outcome = run_one(_make_spec(output_dir=out_dir))

    assert outcome.status == "ok"
    assert set(outcome.output_paths) == {"report__R", "ephemeris__E", "contact__C"}
    assert (out_dir / "report__R.parquet").exists()
    assert (out_dir / "ephemeris__E.parquet").exists()
    assert (out_dir / "contact__C.parquet").exists()


# ---- defensive: a stale handler on the same logger does not bleed across runs ----


def test_run_one_does_not_bleed_handlers_between_runs(
    tmp_path: Path, fake_gmat_run: FakeGmatRun
) -> None:
    import logging

    spec_a = _make_spec(output_dir=tmp_path / "run-0", run_id=0)
    spec_b = _make_spec(output_dir=tmp_path / "run-1", run_id=1)
    run_one(spec_a)
    run_one(spec_b)
    run_one(spec_a)

    # Same logger name across the two run_id=0 calls; we should not have
    # accumulated multiple file handlers attached to it.
    logger = logging.getLogger("gmat_sweep.worker.run_0")
    assert logger.handlers == []
