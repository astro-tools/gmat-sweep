"""Tests for gmat_sweep.aggregate — lazy multi-indexed result assembly from Parquet."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from gmat_sweep.aggregate import (
    lazy_contacts,
    lazy_ephemerides,
    lazy_fused_reports,
    lazy_multiindex,
)
from gmat_sweep.errors import SweepConfigError
from gmat_sweep.manifest import Manifest, ManifestEntry


def _utc(year: int, month: int, day: int, h: int = 0, m: int = 0, s: int = 0) -> datetime:
    return datetime(year, month, day, h, m, s, tzinfo=timezone.utc)


def _make_manifest(entries: list[ManifestEntry]) -> Manifest:
    return Manifest(
        script_sha256="a" * 64,
        gmat_sweep_version="0.1.0",
        gmat_run_version="0.4.0",
        gmat_install_version="R2026a",
        python_version="3.12.3",
        os_platform="Linux-6.6.0",
        sweep_seed=None,
        parameter_spec={},
        run_count=len(entries),
        entries=entries,
    )


def _ok_entry(
    run_id: int,
    parquet_path: Path,
    *,
    key: str = "report__ReportFile1",
    extra_paths: dict[str, Path] | None = None,
) -> ManifestEntry:
    paths = {key: parquet_path}
    if extra_paths:
        paths.update(extra_paths)
    return ManifestEntry(
        run_id=run_id,
        overrides={},
        status="ok",
        output_paths=paths,
        started_at=_utc(2026, 5, 4),
        ended_at=_utc(2026, 5, 4, 0, 0, 1),
        duration_s=1.0,
        stderr=None,
        log_path=None,
    )


def _nonok_entry(run_id: int, status: str) -> ManifestEntry:
    return ManifestEntry(
        run_id=run_id,
        overrides={},
        status=status,  # type: ignore[arg-type]
        output_paths={},
        started_at=_utc(2026, 5, 4),
        ended_at=_utc(2026, 5, 4, 0, 0, 1),
        duration_s=1.0,
        stderr="boom" if status == "failed" else None,
        log_path=None,
    )


def _write_run_parquet(
    output_dir: Path,
    run_id: int,
    *,
    n_rows: int = 3,
    basename: str = "report__ReportFile1",
) -> Path:
    path = output_dir / f"run-{run_id}" / f"{basename}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [f"2026-05-04T00:00:0{i}" for i in range(n_rows)],
            ),
            "x": [run_id * 10 + i for i in range(n_rows)],
            "y": [run_id * 100 + i * 2 for i in range(n_rows)],
        }
    )
    df.to_parquet(path)
    return path


def _write_contact_parquet(
    output_dir: Path,
    run_id: int,
    *,
    n_intervals: int = 2,
    basename: str = "contact__GroundContact",
) -> Path:
    path = output_dir / f"run-{run_id}" / f"{basename}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "Start": pd.to_datetime(
                [f"2026-05-04T00:0{i}:00" for i in range(n_intervals)],
            ),
            "Duration": [60.0 + i for i in range(n_intervals)],
            "interval_id": list(range(n_intervals)),
        }
    )
    df.to_parquet(path)
    return path


# ---- happy paths ----------------------------------------------------------


def test_lazy_multiindex_16_run_all_ok(tmp_path: Path) -> None:
    paths = [_write_run_parquet(tmp_path, i) for i in range(16)]
    manifest = _make_manifest([_ok_entry(i, p) for i, p in enumerate(paths)])

    df = lazy_multiindex(manifest, tmp_path)

    assert df.index.names == ["run_id", "time"]
    run_ids = sorted(df.index.get_level_values("run_id").unique().tolist())
    assert run_ids == list(range(16))
    assert df.index.get_level_values("time").dtype == "datetime64[ns]"
    assert (df["__status"] == "ok").all()
    assert len(df) == 16 * 3
    assert {"x", "y", "__status"} == set(df.columns)


def test_lazy_multiindex_15_ok_plus_one_failed(tmp_path: Path) -> None:
    paths = [_write_run_parquet(tmp_path, i, n_rows=2) for i in range(15)]
    entries: list[ManifestEntry] = [_ok_entry(i, p) for i, p in enumerate(paths)]
    entries.append(_nonok_entry(15, "failed"))

    df = lazy_multiindex(_make_manifest(entries), tmp_path)

    assert len(df) == 15 * 2 + 1
    failed_rows = df.xs(15, level="run_id")
    assert len(failed_rows) == 1
    assert (failed_rows["__status"] == "failed").all()
    assert failed_rows[["x", "y"]].isna().all().all()
    assert (df.loc[df["__status"] == "ok"]).shape[0] == 15 * 2


def test_lazy_multiindex_skipped_run(tmp_path: Path) -> None:
    p = _write_run_parquet(tmp_path, 0, n_rows=2)
    manifest = _make_manifest([_ok_entry(0, p), _nonok_entry(1, "skipped")])

    df = lazy_multiindex(manifest, tmp_path)

    skipped = df.xs(1, level="run_id")
    assert (skipped["__status"] == "skipped").all()
    assert skipped[["x", "y"]].isna().all().all()


def test_lazy_multiindex_spool_false_matches_spool_true(tmp_path: Path) -> None:
    paths = [_write_run_parquet(tmp_path, i, n_rows=3) for i in range(4)]
    manifest = _make_manifest([_ok_entry(i, p) for i, p in enumerate(paths)])

    streamed = lazy_multiindex(manifest, tmp_path, spool=True)
    eager = lazy_multiindex(manifest, tmp_path, spool=False)

    pd.testing.assert_frame_equal(streamed, eager)


def test_lazy_multiindex_relative_paths_resolve_against_output_dir(tmp_path: Path) -> None:
    abs_path = _write_run_parquet(tmp_path, 0, n_rows=2)
    rel_path = abs_path.relative_to(tmp_path)
    entry = ManifestEntry(
        run_id=0,
        overrides={},
        status="ok",
        output_paths={"report__ReportFile1": rel_path},
        started_at=_utc(2026, 5, 4),
        ended_at=_utc(2026, 5, 4, 0, 0, 1),
        duration_s=1.0,
        stderr=None,
        log_path=None,
    )

    df = lazy_multiindex(_make_manifest([entry]), tmp_path)
    assert len(df) == 2


# ---- empty / degenerate cases --------------------------------------------


def test_lazy_multiindex_empty_manifest(tmp_path: Path) -> None:
    df = lazy_multiindex(_make_manifest([]), tmp_path)

    assert df.empty
    assert df.index.names == ["run_id", "time"]
    assert "__status" in df.columns
    assert df.index.get_level_values("time").dtype == "datetime64[ns]"


def test_lazy_multiindex_all_failed(tmp_path: Path) -> None:
    entries = [_nonok_entry(i, "failed") for i in range(3)]

    df = lazy_multiindex(_make_manifest(entries), tmp_path)

    assert len(df) == 3
    assert (df["__status"] == "failed").all()
    assert list(df.columns) == ["__status"]


# ---- error paths ---------------------------------------------------------


def test_lazy_multiindex_two_reports_dispatch_via_name(tmp_path: Path) -> None:
    # Each of 4 runs produces ReportFile1 + ReportFile2 with different schemas.
    entries: list[ManifestEntry] = []
    for run_id in range(4):
        p1 = _write_run_parquet(tmp_path, run_id, n_rows=2, basename="report__ReportFile1")
        p2 = _write_run_parquet(tmp_path, run_id, n_rows=3, basename="report__ReportFile2")
        entries.append(
            _ok_entry(
                run_id,
                p1,
                key="report__ReportFile1",
                extra_paths={"report__ReportFile2": p2},
            )
        )
    manifest = _make_manifest(entries)

    df_a = lazy_multiindex(manifest, tmp_path, name="ReportFile1")
    df_b = lazy_multiindex(manifest, tmp_path, name="ReportFile2")

    assert len(df_a) == 4 * 2
    assert len(df_b) == 4 * 3
    assert sorted(df_a.index.get_level_values("run_id").unique().tolist()) == [0, 1, 2, 3]
    assert sorted(df_b.index.get_level_values("run_id").unique().tolist()) == [0, 1, 2, 3]


def test_lazy_multiindex_two_reports_name_none_raises_listing_names(tmp_path: Path) -> None:
    p1 = _write_run_parquet(tmp_path, 0, basename="report__ReportFile1")
    p2 = _write_run_parquet(tmp_path, 0, basename="report__ReportFile2")
    entry = _ok_entry(0, p1, key="report__ReportFile1", extra_paths={"report__ReportFile2": p2})

    with pytest.raises(SweepConfigError, match=r"ReportFile1.*ReportFile2"):
        lazy_multiindex(_make_manifest([entry]), tmp_path)


def test_lazy_multiindex_unknown_name_raises_listing_available(tmp_path: Path) -> None:
    p = _write_run_parquet(tmp_path, 0)
    with pytest.raises(SweepConfigError, match=r"no report output named 'Nope'"):
        lazy_multiindex(_make_manifest([_ok_entry(0, p)]), tmp_path, name="Nope")


def test_lazy_multiindex_ok_entry_with_no_outputs_raises(tmp_path: Path) -> None:
    entry = ManifestEntry(
        run_id=0,
        overrides={},
        status="ok",
        output_paths={},
        started_at=_utc(2026, 5, 4),
        ended_at=_utc(2026, 5, 4, 0, 0, 1),
        duration_s=1.0,
        stderr=None,
        log_path=None,
    )

    with pytest.raises(ValueError, match="no output_paths"):
        lazy_multiindex(_make_manifest([entry]), tmp_path)


def test_lazy_multiindex_missing_time_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "run-0" / "report__ReportFile1.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame({"x": [1, 2, 3]}).to_parquet(path)

    with pytest.raises(ValueError, match="time"):
        lazy_multiindex(_make_manifest([_ok_entry(0, path)]), tmp_path)


# ---- ephemeris aggregation -----------------------------------------------
#
# lazy_ephemerides shares the (run_id, time) index machinery with
# lazy_multiindex; tests focus on the dispatch contract (prefix filter,
# name= selector) rather than re-pinning the index assembly.


def test_lazy_ephemerides_single_ephemeris_picked_automatically(tmp_path: Path) -> None:
    paths = [_write_run_parquet(tmp_path, i, basename="ephemeris__SatEphem") for i in range(4)]
    entries = [_ok_entry(i, p, key="ephemeris__SatEphem") for i, p in enumerate(paths)]

    df = lazy_ephemerides(_make_manifest(entries), tmp_path)

    assert df.index.names == ["run_id", "time"]
    assert sorted(df.index.get_level_values("run_id").unique().tolist()) == [0, 1, 2, 3]
    assert (df["__status"] == "ok").all()


def test_lazy_ephemerides_name_none_raises_when_two_present(tmp_path: Path) -> None:
    p1 = _write_run_parquet(tmp_path, 0, basename="ephemeris__SatEphem")
    p2 = _write_run_parquet(tmp_path, 0, basename="ephemeris__GroundEphem")
    entry = _ok_entry(0, p1, key="ephemeris__SatEphem", extra_paths={"ephemeris__GroundEphem": p2})

    with pytest.raises(SweepConfigError, match=r"GroundEphem.*SatEphem"):
        lazy_ephemerides(_make_manifest([entry]), tmp_path)


def test_lazy_ephemerides_failed_run_materialises_as_nan_row(tmp_path: Path) -> None:
    # 3 ok runs (with both report and ephemeris outputs) + 1 failed run.
    entries: list[ManifestEntry] = []
    for run_id in range(3):
        report = _write_run_parquet(tmp_path, run_id, basename="report__R")
        eph = _write_run_parquet(tmp_path, run_id, n_rows=2, basename="ephemeris__SatEphem")
        entries.append(
            _ok_entry(
                run_id,
                report,
                key="report__R",
                extra_paths={"ephemeris__SatEphem": eph},
            )
        )
    entries.append(_nonok_entry(3, "failed"))

    df = lazy_ephemerides(_make_manifest(entries), tmp_path)

    assert len(df) == 3 * 2 + 1
    failed_rows = df.xs(3, level="run_id")
    assert len(failed_rows) == 1
    assert (failed_rows["__status"] == "failed").all()
    assert failed_rows.drop(columns=["__status"]).isna().all().all()


def test_lazy_ephemerides_ok_run_without_ephemeris_lands_as_status_ok_nan_row(
    tmp_path: Path,
) -> None:
    # Run 0 produced a report only; run 1 produced both. lazy_ephemerides should
    # surface run 0 as one "ok" NaN row (not failed) since the run itself was
    # successful — it just didn't produce an ephemeris.
    p_report = _write_run_parquet(tmp_path, 0, basename="report__R")
    p_eph_1 = _write_run_parquet(tmp_path, 1, n_rows=2, basename="ephemeris__SatEphem")
    p_report_1 = _write_run_parquet(tmp_path, 1, basename="report__R")
    entries = [
        _ok_entry(0, p_report, key="report__R"),
        _ok_entry(1, p_report_1, key="report__R", extra_paths={"ephemeris__SatEphem": p_eph_1}),
    ]

    df = lazy_ephemerides(_make_manifest(entries), tmp_path)

    run_0_rows = df.xs(0, level="run_id")
    assert len(run_0_rows) == 1
    assert (run_0_rows["__status"] == "ok").all()


# ---- contact aggregation ------------------------------------------------


def test_lazy_contacts_single_contact_picked_automatically(tmp_path: Path) -> None:
    paths = [_write_contact_parquet(tmp_path, i, n_intervals=2) for i in range(3)]
    entries = [_ok_entry(i, p, key="contact__GroundContact") for i, p in enumerate(paths)]

    df = lazy_contacts(_make_manifest(entries), tmp_path)

    assert df.index.names == ["run_id", "interval_id"]
    assert len(df) == 3 * 2
    assert sorted(df.index.get_level_values("run_id").unique().tolist()) == [0, 1, 2]
    assert sorted(df.xs(0, level="run_id").index.tolist()) == [0, 1]
    assert (df["__status"] == "ok").all()


def test_lazy_contacts_name_none_raises_when_two_present(tmp_path: Path) -> None:
    p1 = _write_contact_parquet(tmp_path, 0, basename="contact__Ground1")
    p2 = _write_contact_parquet(tmp_path, 0, basename="contact__Ground2")
    entry = _ok_entry(0, p1, key="contact__Ground1", extra_paths={"contact__Ground2": p2})

    with pytest.raises(SweepConfigError, match=r"Ground1.*Ground2"):
        lazy_contacts(_make_manifest([entry]), tmp_path)


def test_lazy_contacts_failed_run_materialises_as_nan_row_with_na_interval(
    tmp_path: Path,
) -> None:
    paths = [_write_contact_parquet(tmp_path, i, n_intervals=2) for i in range(3)]
    entries: list[ManifestEntry] = [
        _ok_entry(i, p, key="contact__GroundContact") for i, p in enumerate(paths)
    ]
    entries.append(_nonok_entry(3, "failed"))

    df = lazy_contacts(_make_manifest(entries), tmp_path)

    assert len(df) == 3 * 2 + 1
    failed_rows = df.xs(3, level="run_id")
    assert len(failed_rows) == 1
    assert (failed_rows["__status"] == "failed").all()
    # Failed-row interval_id is pd.NA in the nullable Int64 level.
    assert bool(pd.isna(failed_rows.index[0]))


def _write_timed_parquet(
    output_dir: Path,
    run_id: int,
    basename: str,
    timestamps: list[str],
    *,
    value_offset: int = 0,
) -> Path:
    """Helper: write a per-run report parquet with caller-controlled timestamps."""
    path = output_dir / f"run-{run_id}" / f"{basename}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(timestamps),
            "x": [value_offset + i for i in range(len(timestamps))],
        }
    )
    df.to_parquet(path)
    return path


def _ok_two_reports_entry(
    run_id: int,
    path_a: Path,
    path_b: Path,
) -> ManifestEntry:
    return _ok_entry(run_id, path_a, key="report__A", extra_paths={"report__B": path_b})


# ---- fused multi-report aggregation --------------------------------------


def test_lazy_fused_reports_exact_tolerance_inner_join(tmp_path: Path) -> None:
    times = [f"2026-05-04T00:00:0{i}" for i in range(3)]
    entries: list[ManifestEntry] = []
    for run_id in range(2):
        pa = _write_timed_parquet(tmp_path, run_id, "report__A", times, value_offset=run_id * 100)
        pb = _write_timed_parquet(tmp_path, run_id, "report__B", times, value_offset=run_id * 200)
        entries.append(_ok_two_reports_entry(run_id, pa, pb))

    df = lazy_fused_reports(_make_manifest(entries), tmp_path, ["A", "B"], tolerance="exact")

    assert df.index.names == ["run_id", "time"]
    assert isinstance(df.columns, pd.MultiIndex)
    assert df.columns.names == ["report", "field"]
    assert {("A", "x"), ("B", "x"), ("A", "__status"), ("B", "__status"), ("__status", "")} <= set(
        df.columns
    )
    assert len(df) == 2 * 3
    assert (df[("A", "__status")] == "ok").all()
    assert (df[("B", "__status")] == "ok").all()
    assert (df[("__status", "")] == "ok").all()


def test_lazy_fused_reports_numeric_tolerance_matches_within_window(tmp_path: Path) -> None:
    # Anchor (A) at 1-Hz over :00..:04. B at every 10s starting :00. With
    # tolerance=2s and merge_asof's default backward direction, only A times
    # within 2s after the most recent B time get a B match. A=:00..:02 match
    # B=:00; A=:03..:04 are >2s past B=:00 and B=:10 is in the future, so they
    # land NaN on the B side.
    a_times = [f"2026-05-04T00:00:0{i}" for i in range(5)]
    b_times = ["2026-05-04T00:00:00", "2026-05-04T00:00:10"]
    pa = _write_timed_parquet(tmp_path, 0, "report__A", a_times)
    pb = _write_timed_parquet(tmp_path, 0, "report__B", b_times, value_offset=99)
    entry = _ok_two_reports_entry(0, pa, pb)

    df = lazy_fused_reports(
        _make_manifest([entry]), tmp_path, ["A", "B"], tolerance=pd.Timedelta(seconds=2)
    )

    assert len(df) == 5
    # First three anchor rows hit B=:00 (value_offset=99 so x=99 there).
    matched = df[("B", "x")].notna()
    assert matched.tolist() == [True, True, True, False, False]
    assert (df.loc[matched, ("B", "x")] == 99).all()


def test_lazy_fused_reports_no_overlap_within_tolerance(tmp_path: Path) -> None:
    # A times: :00, :01, :02. B time: :30. tolerance=1s → no rows match B.
    a_times = [f"2026-05-04T00:00:0{i}" for i in range(3)]
    b_times = ["2026-05-04T00:00:30"]
    pa = _write_timed_parquet(tmp_path, 0, "report__A", a_times)
    pb = _write_timed_parquet(tmp_path, 0, "report__B", b_times, value_offset=42)
    entry = _ok_two_reports_entry(0, pa, pb)

    df = lazy_fused_reports(
        _make_manifest([entry]), tmp_path, ["A", "B"], tolerance=pd.Timedelta(seconds=1)
    )

    assert len(df) == 3
    assert df[("A", "x")].notna().all()
    assert df[("B", "x")].isna().all()


def test_lazy_fused_reports_anchor_failed_lands_as_one_nat_row(tmp_path: Path) -> None:
    # Run 0 has both A and B; run 1 is ok but only produced B (anchor missing).
    times = ["2026-05-04T00:00:00", "2026-05-04T00:00:01"]
    pa0 = _write_timed_parquet(tmp_path, 0, "report__A", times)
    pb0 = _write_timed_parquet(tmp_path, 0, "report__B", times, value_offset=10)
    pb1 = _write_timed_parquet(tmp_path, 1, "report__B", times, value_offset=100)
    entries = [
        _ok_two_reports_entry(0, pa0, pb0),
        ManifestEntry(
            run_id=1,
            overrides={},
            status="ok",
            output_paths={"report__B": pb1},
            started_at=_utc(2026, 5, 4),
            ended_at=_utc(2026, 5, 4, 0, 0, 1),
            duration_s=1.0,
            stderr=None,
            log_path=None,
        ),
    ]

    df = lazy_fused_reports(_make_manifest(entries), tmp_path, ["A", "B"], tolerance="exact")

    run_1 = df.xs(1, level="run_id")
    assert len(run_1) == 1
    assert bool(pd.isna(run_1.index[0]))
    assert run_1[("A", "x")].isna().all()
    assert run_1[("B", "x")].isna().all()
    # Per-report status mirrors lazy_multiindex's per-report contract:
    # both A and B report "ok" because the run itself was ok.
    assert (run_1[("A", "__status")] == "ok").all()
    assert (run_1[("B", "__status")] == "ok").all()
    assert (run_1[("__status", "")] == "ok").all()


def test_lazy_fused_reports_failed_run_per_report_status_propagates(tmp_path: Path) -> None:
    times = ["2026-05-04T00:00:00", "2026-05-04T00:00:01"]
    pa = _write_timed_parquet(tmp_path, 0, "report__A", times)
    pb = _write_timed_parquet(tmp_path, 0, "report__B", times, value_offset=10)
    entries = [
        _ok_two_reports_entry(0, pa, pb),
        _nonok_entry(1, "failed"),
    ]

    df = lazy_fused_reports(_make_manifest(entries), tmp_path, ["A", "B"], tolerance="exact")

    run_1 = df.xs(1, level="run_id")
    assert len(run_1) == 1
    assert (run_1[("A", "__status")] == "failed").all()
    assert (run_1[("B", "__status")] == "failed").all()
    assert (run_1[("__status", "")] == "failed").all()


def test_lazy_fused_reports_right_report_failed_anchor_ok(tmp_path: Path) -> None:
    # Run 0: both reports ok. Run 1: A ok, B missing (manifest entry has no B key).
    times = ["2026-05-04T00:00:00", "2026-05-04T00:00:01"]
    pa0 = _write_timed_parquet(tmp_path, 0, "report__A", times)
    pb0 = _write_timed_parquet(tmp_path, 0, "report__B", times, value_offset=10)
    pa1 = _write_timed_parquet(tmp_path, 1, "report__A", times, value_offset=100)
    entries = [
        _ok_two_reports_entry(0, pa0, pb0),
        _ok_entry(1, pa1, key="report__A"),
    ]

    df = lazy_fused_reports(_make_manifest(entries), tmp_path, ["A", "B"], tolerance="exact")

    # Run 1: anchor has data → 2 rows. B's data NaN because B was missing.
    run_1 = df.xs(1, level="run_id")
    assert len(run_1) == 2
    assert run_1[("A", "x")].notna().all()
    assert run_1[("B", "x")].isna().all()
    # B's per-report status is "ok" — the run was ok, just no B output.
    assert (run_1[("B", "__status")] == "ok").all()


def test_lazy_fused_reports_three_reports_chain_merge(tmp_path: Path) -> None:
    times = [f"2026-05-04T00:00:0{i}" for i in range(2)]
    pa = _write_timed_parquet(tmp_path, 0, "report__A", times, value_offset=0)
    pb = _write_timed_parquet(tmp_path, 0, "report__B", times, value_offset=10)
    pc = _write_timed_parquet(tmp_path, 0, "report__C", times, value_offset=20)
    entry = _ok_entry(
        0,
        pa,
        key="report__A",
        extra_paths={"report__B": pb, "report__C": pc},
    )

    df = lazy_fused_reports(_make_manifest([entry]), tmp_path, ["A", "B", "C"], tolerance="exact")

    assert len(df) == 2
    assert {("A", "x"), ("B", "x"), ("C", "x")} <= set(df.columns)
    assert df[("A", "x")].tolist() == [0, 1]
    assert df[("B", "x")].tolist() == [10, 11]
    assert df[("C", "x")].tolist() == [20, 21]


def test_lazy_fused_reports_empty_manifest(tmp_path: Path) -> None:
    df = lazy_fused_reports(_make_manifest([]), tmp_path, ["A", "B"], tolerance="exact")

    assert df.empty
    assert df.index.names == ["run_id", "time"]
    assert isinstance(df.columns, pd.MultiIndex)
    assert ("__status", "") in df.columns


def test_lazy_fused_reports_single_name_raises(tmp_path: Path) -> None:
    with pytest.raises(SweepConfigError, match=r"at least 2 report names"):
        lazy_fused_reports(_make_manifest([]), tmp_path, ["A"], tolerance="exact")


def test_lazy_fused_reports_duplicate_names_raises(tmp_path: Path) -> None:
    with pytest.raises(SweepConfigError, match=r"unique report names"):
        lazy_fused_reports(_make_manifest([]), tmp_path, ["A", "A"], tolerance="exact")


def test_lazy_contacts_three_kinds_aggregate_independently(tmp_path: Path) -> None:
    # End-to-end acceptance: a sweep producing report + ephemeris + contact
    # yields three independent multi-indexed frames.
    entries: list[ManifestEntry] = []
    for run_id in range(4):
        p_report = _write_run_parquet(tmp_path, run_id, basename="report__R")
        p_eph = _write_run_parquet(tmp_path, run_id, basename="ephemeris__E")
        p_contact = _write_contact_parquet(tmp_path, run_id, n_intervals=2)
        entries.append(
            _ok_entry(
                run_id,
                p_report,
                key="report__R",
                extra_paths={
                    "ephemeris__E": p_eph,
                    "contact__GroundContact": p_contact,
                },
            )
        )
    manifest = _make_manifest(entries)

    reports_df = lazy_multiindex(manifest, tmp_path)
    eph_df = lazy_ephemerides(manifest, tmp_path)
    contacts_df = lazy_contacts(manifest, tmp_path)

    assert reports_df.index.names == ["run_id", "time"]
    assert eph_df.index.names == ["run_id", "time"]
    assert contacts_df.index.names == ["run_id", "interval_id"]
    assert len(reports_df) == 4 * 3
    assert len(eph_df) == 4 * 3
    assert len(contacts_df) == 4 * 2
