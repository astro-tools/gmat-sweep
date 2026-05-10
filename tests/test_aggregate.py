"""Tests for gmat_sweep.aggregate — lazy multi-indexed result assembly from Parquet."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import pytest

from gmat_sweep.aggregate import (
    lazy_contacts,
    lazy_ephemerides,
    lazy_fused_reports,
    lazy_multiindex,
    mc_convergence,
    sweep_diff,
    sweep_summary,
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


def test_lazy_multiindex_1000_run_peak_memory_bounded(tmp_path: Path) -> None:
    """A 1000-run aggregate keeps Python-tracked peak allocation bounded.

    Regression guard for #130: the previous ``_read_ok_runs`` accumulated
    one ``pandas.DataFrame`` per fragment in a list before
    ``pd.concat``-ing them, so peak memory scaled linearly with run count.
    The streaming ``pa.concat_tables`` path materialises pandas once at
    the end; peak stays on the order of the final frame.
    """
    import tracemalloc

    n_runs = 1000
    paths = [_write_run_parquet(tmp_path, i, n_rows=3) for i in range(n_runs)]
    manifest = _make_manifest([_ok_entry(i, p) for i, p in enumerate(paths)])

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        df = lazy_multiindex(manifest, tmp_path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    final_size = int(df.memory_usage(deep=True).sum())
    # Measured: the streaming pa.concat_tables path peaks at ~8x the final
    # frame's pandas memory; the old "append a pandas frame per fragment +
    # pd.concat at the end" path peaked at ~50-70x on the same workload.
    # A 20x ceiling separates the two with margin in both directions.
    assert peak < 20 * final_size, (
        f"peak tracemalloc allocation = {peak} bytes vs. final frame "
        f"= {final_size} bytes (ratio {peak / final_size:.1f}x)"
    )


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


# ---------------------------------------------------------------------------
# sweep_summary
# ---------------------------------------------------------------------------


def _summary_df(
    *,
    n_runs: int = 100,
    n_steps: int = 4,
    statuses: dict[int, str] | None = None,
) -> pd.DataFrame:
    """A ``(run_id, time)``-MultiIndexed test frame with two data columns."""
    statuses = statuses or {}
    rows: list[dict[str, object]] = []
    times = pd.to_datetime([f"2026-05-04T00:00:0{i}" for i in range(n_steps)])
    for run_id in range(n_runs):
        status = statuses.get(run_id, "ok")
        if status != "ok":
            rows.append(
                {
                    "run_id": run_id,
                    "time": pd.NaT,
                    "x": float("nan"),
                    "y": float("nan"),
                    "__status": status,
                }
            )
            continue
        for step, t in enumerate(times):
            rows.append(
                {
                    "run_id": run_id,
                    "time": t,
                    "x": float(run_id) + step * 0.1,
                    "y": float(run_id) * 2.0 - step,
                    "__status": "ok",
                }
            )
    return cast(pd.DataFrame, pd.DataFrame(rows).set_index(["run_id", "time"]))


def test_sweep_summary_default_shape_and_multiindex_columns() -> None:
    df = _summary_df(n_runs=10, n_steps=3)
    summary = sweep_summary(df)

    # Row index keyed on the unique time steps from the input.
    assert summary.index.name == "time"
    assert len(summary) == 3

    # Column index is a 2-level (statistic, field) MultiIndex with default
    # mean + std + 5/50/95 quantiles times the 2 data columns (x, y).
    assert isinstance(summary.columns, pd.MultiIndex)
    assert summary.columns.names == ["statistic", "field"]
    statistics = list(dict.fromkeys(summary.columns.get_level_values(0)))
    assert statistics == ["mean", "std", "q0.05", "q0.5", "q0.95"]
    fields = list(dict.fromkeys(summary.columns.get_level_values(1)))
    assert fields == ["x", "y"]


def test_sweep_summary_quantile_matches_hand_rolled_groupby() -> None:
    """DoD criterion: (time, q=0.5) slice equals df.groupby('time').quantile(0.5)."""
    df = _summary_df(n_runs=100, n_steps=4)
    summary = sweep_summary(df)

    # Drop __status to mirror sweep_summary's internal data-only view.
    expected = df.drop(columns=["__status"]).groupby(level="time").quantile(0.5)
    got = cast(pd.DataFrame, summary["q0.5"])
    pd.testing.assert_frame_equal(got, expected, check_names=False)


def test_sweep_summary_by_run_id_collapses_across_time() -> None:
    df = _summary_df(n_runs=4, n_steps=5)
    summary = sweep_summary(df, by="run_id", q=(), include=("mean",))

    assert summary.index.name == "run_id"
    assert list(summary.index) == [0, 1, 2, 3]
    # Each run's mean(x) is run_id + mean(0..n_steps-1)*0.1.
    expected_x = pd.Series(
        [float(rid) + 0.2 for rid in range(4)],
        index=pd.Index([0, 1, 2, 3], name="run_id"),
        name=("mean", "x"),
    )
    pd.testing.assert_series_equal(summary[("mean", "x")], expected_x)


def test_sweep_summary_count_ok_counts_non_nan_values() -> None:
    df = _summary_df(n_runs=5, n_steps=3)
    summary = sweep_summary(df, q=(), include=("count_ok",))

    # Every ok row contributes; per time, count == 5 across all 5 runs.
    assert (summary[("count_ok", "x")] == 5).all()
    assert (summary[("count_ok", "y")] == 5).all()


def test_sweep_summary_dropna_true_excludes_failed_and_skipped_runs() -> None:
    df = _summary_df(n_runs=6, n_steps=2, statuses={1: "failed", 4: "skipped"})
    summary = sweep_summary(df, q=(), include=("count_ok",))

    # 4 ok runs (0, 2, 3, 5) — failed/skipped runs (NaT marker rows) excluded.
    assert summary.index.name == "time"
    assert len(summary) == 2
    assert (summary[("count_ok", "x")] == 4).all()


def test_sweep_summary_dropna_false_keeps_marker_rows() -> None:
    df = _summary_df(n_runs=3, n_steps=2, statuses={2: "failed"})
    summary = sweep_summary(df, q=(), include=("count_ok",), dropna=False)

    # The NaT marker row from the failed run lands as a NaT-keyed group.
    assert bool(pd.isna(summary.index).any())
    # The two real time steps still see 2 ok contributions each.
    real_rows = summary.loc[summary.index.notna()]
    assert (real_rows[("count_ok", "x")] == 2).all()


def test_sweep_summary_custom_q_and_include() -> None:
    df = _summary_df(n_runs=20, n_steps=3)
    summary = sweep_summary(
        df,
        q=(0.25, 0.75),
        include=("min", "max"),
    )
    statistics = list(dict.fromkeys(summary.columns.get_level_values(0)))
    assert statistics == ["min", "max", "q0.25", "q0.75"]


def test_sweep_summary_no_status_column_works() -> None:
    """A frame without __status should aggregate cleanly — dropna is a no-op."""
    df = _summary_df(n_runs=4, n_steps=2).drop(columns=["__status"])
    summary = sweep_summary(df, q=(0.5,), include=())
    assert list(summary.columns.get_level_values(0)) == ["q0.5", "q0.5"]
    assert len(summary) == 2


def test_sweep_summary_rejects_unsupported_by() -> None:
    df = _summary_df(n_runs=2, n_steps=2)
    with pytest.raises(SweepConfigError, match=r"by=.*not supported"):
        sweep_summary(df, by="run")


def test_sweep_summary_rejects_q_outside_open_interval() -> None:
    df = _summary_df(n_runs=2, n_steps=2)
    with pytest.raises(SweepConfigError, match=r"open interval"):
        sweep_summary(df, q=(0.0, 0.5))


def test_sweep_summary_rejects_unknown_statistic() -> None:
    df = _summary_df(n_runs=2, n_steps=2)
    with pytest.raises(SweepConfigError, match=r"unknown statistic"):
        sweep_summary(df, include=("median",))


def test_sweep_summary_rejects_missing_index_level() -> None:
    df = _summary_df(n_runs=2, n_steps=2).reset_index().set_index("run_id")
    with pytest.raises(SweepConfigError, match=r"does not have a 'time' level"):
        sweep_summary(df)


def test_sweep_summary_rejects_duplicate_q() -> None:
    df = _summary_df(n_runs=2, n_steps=2)
    with pytest.raises(SweepConfigError, match=r"q must not contain duplicates"):
        sweep_summary(df, q=(0.5, 0.5))


def test_sweep_summary_rejects_duplicate_include() -> None:
    df = _summary_df(n_runs=2, n_steps=2)
    with pytest.raises(SweepConfigError, match=r"include must not contain duplicates"):
        sweep_summary(df, include=("mean", "mean"))


# ---------------------------------------------------------------------------
# mc_convergence
# ---------------------------------------------------------------------------


def _conv_df(
    *,
    n_runs: int,
    n_steps: int = 1,
    statuses: dict[int, str] | None = None,
    seed: int = 42,
    sigma: float = 1.0,
) -> pd.DataFrame:
    """A ``(run_id, time)``-MultiIndexed test frame for mc_convergence.

    Each ok run has ``n_steps`` rows; the per-run final ``miss`` value is
    drawn from ``N(0, sigma^2)`` under a fixed numpy generator. Earlier
    rows linearly ramp from 0 to the terminal value, so callable metrics
    that pull the last row see the same draw as ``terminal_only=True``.
    """
    statuses = statuses or {}
    rng = np.random.default_rng(seed)
    terminal = rng.normal(loc=0.0, scale=sigma, size=n_runs)
    rows: list[dict[str, object]] = []
    times = pd.to_datetime([f"2026-05-04T00:00:0{i}" for i in range(n_steps)])
    for run_id in range(n_runs):
        status = statuses.get(run_id, "ok")
        if status != "ok":
            rows.append(
                {
                    "run_id": run_id,
                    "time": pd.NaT,
                    "miss": float("nan"),
                    "__status": status,
                }
            )
            continue
        for step, t in enumerate(times):
            ramp = (step + 1) / n_steps
            rows.append(
                {
                    "run_id": run_id,
                    "time": t,
                    "miss": float(terminal[run_id]) * ramp,
                    "__status": "ok",
                }
            )
    return cast(pd.DataFrame, pd.DataFrame(rows).set_index(["run_id", "time"]))


def test_mc_convergence_terminal_only_columns_and_shape() -> None:
    df = _conv_df(n_runs=10)
    conv = mc_convergence(df, "miss", terminal_only=True)

    assert list(conv.columns) == ["n", "running_mean", "running_std", "se_mean"]
    assert len(conv) == 10
    assert list(conv["n"]) == list(range(1, 11))
    # n=1 has no sample variance under ddof=1.
    assert pd.isna(conv.loc[0, "running_std"])
    assert pd.isna(conv.loc[0, "se_mean"])


def test_mc_convergence_matches_hand_rolled_cumulative_stats() -> None:
    df = _conv_df(n_runs=20)
    conv = mc_convergence(df, "miss", terminal_only=True)

    terminal = df.groupby(level="run_id")["miss"].last().to_numpy()
    for k in (1, 2, 5, 10, 20):
        prefix = terminal[:k]
        expected_mean = float(prefix.mean())
        row = conv.loc[k - 1]
        assert row["running_mean"] == pytest.approx(expected_mean)
        if k == 1:
            assert pd.isna(row["running_std"])
            assert pd.isna(row["se_mean"])
        else:
            expected_std = float(prefix.std(ddof=1))
            assert row["running_std"] == pytest.approx(expected_std)
            assert row["se_mean"] == pytest.approx(expected_std / np.sqrt(k))


def test_mc_convergence_se_curve_matches_sigma_over_sqrt_n() -> None:
    """Definition-of-done check: ``se_mean(n) ~ sigma / sqrt(n)`` to within MC noise."""
    sigma = 2.5
    df = _conv_df(n_runs=2000, sigma=sigma, seed=20260509)
    conv = mc_convergence(df, "miss", terminal_only=True)

    # The terminal-row SE should land within ~10% of the analytic ceiling
    # at n=2000 (large enough to suppress the small-sample noise).
    final = conv.iloc[-1]
    analytic = sigma / np.sqrt(int(final["n"]))
    assert abs(float(final["se_mean"]) - analytic) / analytic < 0.10

    # The SE curve's late-n decay rate should track 1/sqrt(n): compare two
    # samples at n=500 and n=2000 — the ratio should be close to 2.
    def se_at(n: int) -> float:
        return float(conv.loc[conv["n"] == n, "se_mean"].iloc[0])

    ratio = se_at(500) / se_at(2000)
    assert 1.7 < ratio < 2.3


def test_mc_convergence_running_std_stable_at_km_scale() -> None:
    """Welford regression: km-magnitude metrics produce the right std (not 0).

    Older sum-of-squares variance suffered catastrophic cancellation when
    the mean was large compared to the std — ``arr * arr ~ 1e8`` for
    km-magnitude samples, the per-step subtraction below the float64 ULP
    drove variance slightly negative, and ``np.clip(0)`` reported zero.
    """
    n_runs = 5000
    sigma = 50.0
    mu = 7.1e3
    rng = np.random.default_rng(20260510)
    terminal = rng.normal(loc=mu, scale=sigma, size=n_runs)
    df = pd.DataFrame(
        {
            "run_id": range(n_runs),
            "time": pd.to_datetime(["2026-05-04T00:00:00"] * n_runs),
            "miss": terminal,
            "__status": ["ok"] * n_runs,
        }
    ).set_index(["run_id", "time"])

    conv = mc_convergence(df, "miss", terminal_only=True)

    expected_std = float(np.std(terminal, ddof=1))
    actual_std = float(conv["running_std"].iloc[-1])
    rel_error = abs(actual_std - expected_std) / expected_std
    assert rel_error < 1e-6, (
        f"running_std at n={n_runs} = {actual_std}, expected ~{expected_std} "
        f"(relative error {rel_error:.3e})"
    )


def test_mc_convergence_drops_failed_and_skipped_runs() -> None:
    df = _conv_df(n_runs=8, statuses={2: "failed", 5: "skipped"})
    conv = mc_convergence(df, "miss", terminal_only=True)

    # 6 ok runs survive after status-filtering.
    assert len(conv) == 6
    assert list(conv["n"]) == [1, 2, 3, 4, 5, 6]


def test_mc_convergence_callable_metric() -> None:
    df = _conv_df(n_runs=5)
    # Callable that returns the final-time absolute value of "miss".
    conv = mc_convergence(df, lambda sub: float(abs(sub["miss"].iloc[-1])))

    expected = df.groupby(level="run_id")["miss"].last().abs().to_numpy()
    np.testing.assert_allclose(conv["running_mean"].iloc[-1], expected.mean())
    assert list(conv.columns) == ["n", "running_mean", "running_std", "se_mean"]


def test_mc_convergence_callable_metric_receives_time_indexed_subframe() -> None:
    df = _conv_df(n_runs=3, n_steps=2)
    seen_indices: list[Any] = []

    def metric(sub: pd.DataFrame) -> float:
        seen_indices.append(list(sub.index.names))
        return float(sub["miss"].iloc[-1])

    mc_convergence(df, metric)
    # Each sub-frame should have run_id stripped — only the time level remains.
    assert all(names == ["time"] for names in seen_indices)


def test_mc_convergence_callable_metric_bad_return_raises() -> None:
    df = _conv_df(n_runs=3)

    def bad_metric(sub: pd.DataFrame) -> Any:  # returns a Series, not a float
        return sub["miss"]

    with pytest.raises(SweepConfigError, match=r"numeric scalar"):
        mc_convergence(df, bad_metric)


def test_mc_convergence_terminal_only_false_emits_per_time_block() -> None:
    df = _conv_df(n_runs=4, n_steps=3)
    conv = mc_convergence(df, "miss", terminal_only=False)

    assert list(conv.columns) == ["time", "n", "running_mean", "running_std", "se_mean"]
    # 3 unique time steps x 4 prefix sizes = 12 rows.
    assert len(conv) == 12
    # Per-time blocks each carry n=1..4 in order.
    for _t, block in conv.groupby("time"):
        assert list(block["n"]) == [1, 2, 3, 4]
    # The terminal time slice should reproduce the terminal_only=True curve.
    terminal_time = conv["time"].iloc[-1]
    last_block = conv.loc[conv["time"] == terminal_time].reset_index(drop=True)
    terminal_conv = mc_convergence(df, "miss", terminal_only=True)
    pd.testing.assert_series_equal(
        last_block["running_mean"], terminal_conv["running_mean"], check_names=False
    )


def test_mc_convergence_rejects_unknown_column() -> None:
    df = _conv_df(n_runs=3)
    with pytest.raises(SweepConfigError, match=r"not a column of df"):
        mc_convergence(df, "nope")


def test_mc_convergence_rejects_missing_run_id_index() -> None:
    df = _conv_df(n_runs=3).reset_index().set_index("time")
    with pytest.raises(SweepConfigError, match=r"must have a 'run_id' level"):
        mc_convergence(df, "miss")


def test_mc_convergence_works_without_status_column() -> None:
    df = _conv_df(n_runs=4).drop(columns=["__status"])
    conv = mc_convergence(df, "miss", terminal_only=True)
    assert len(conv) == 4
    assert list(conv["n"]) == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# sweep_diff
# ---------------------------------------------------------------------------


def _diff_df(
    *,
    n_runs: int = 3,
    n_steps: int = 2,
    sma_offset: float = 0.0,
    statuses: dict[int, str] | None = None,
    include_status: bool = True,
) -> pd.DataFrame:
    """A ``(run_id, time)``-MultiIndexed sweep frame with two numeric columns.

    Each ok run carries ``Sat.SMA = 7000 + run_id + sma_offset + step*0.01``
    and ``Sat.X = (run_id + 1) * 10.0 + step`` (kept strictly positive so
    self-diff of `__rel` is well-defined — ``0/0`` is NaN, not zero).
    Non-ok runs collapse to one NaT-time, NaN-data marker row, matching
    the rest of this test module's convention.
    """
    statuses = statuses or {}
    rows: list[dict[str, object]] = []
    times = pd.to_datetime([f"2026-05-09T00:00:0{i}" for i in range(n_steps)])
    for run_id in range(n_runs):
        status = statuses.get(run_id, "ok")
        if status != "ok":
            row: dict[str, object] = {
                "run_id": run_id,
                "time": pd.NaT,
                "Sat.SMA": float("nan"),
                "Sat.X": float("nan"),
            }
            if include_status:
                row["__status"] = status
            rows.append(row)
            continue
        for step, t in enumerate(times):
            row = {
                "run_id": run_id,
                "time": t,
                "Sat.SMA": 7000.0 + run_id + sma_offset + step * 0.01,
                "Sat.X": float(run_id + 1) * 10.0 + step,
            }
            if include_status:
                row["__status"] = "ok"
            rows.append(row)
    return cast(pd.DataFrame, pd.DataFrame(rows).set_index(["run_id", "time"]))


def test_sweep_diff_self_diff_is_zero_everywhere() -> None:
    """DoD criterion 1: a sweep diffed against a copy of itself is zero."""
    df = _diff_df(n_runs=4, n_steps=3)
    diff = sweep_diff(df, df.copy())

    # Every numeric column gets a __diff and a __rel under how="both".
    assert set(diff.columns) >= {"Sat.SMA__diff", "Sat.SMA__rel", "Sat.X__diff", "Sat.X__rel"}
    np.testing.assert_array_equal(diff["Sat.SMA__diff"].to_numpy(), 0.0)
    np.testing.assert_array_equal(diff["Sat.X__diff"].to_numpy(), 0.0)
    np.testing.assert_array_equal(diff["Sat.SMA__rel"].to_numpy(), 0.0)
    np.testing.assert_array_equal(diff["Sat.X__rel"].to_numpy(), 0.0)


def test_sweep_diff_perturbed_sma_produces_nonzero_diff() -> None:
    """DoD criterion 2: SMA=7000 vs SMA=7050 produces a non-zero __diff column."""
    a = _diff_df(n_runs=3, n_steps=2)
    b = _diff_df(n_runs=3, n_steps=2, sma_offset=50.0)
    diff = sweep_diff(a, b)

    # Sat.SMA shifted by exactly +50 per row; Sat.X unchanged.
    np.testing.assert_array_equal(diff["Sat.SMA__diff"].to_numpy(), 50.0)
    np.testing.assert_array_equal(diff["Sat.X__diff"].to_numpy(), 0.0)


def test_sweep_diff_default_how_is_both_with_interleaved_columns() -> None:
    a = _diff_df(n_runs=2, n_steps=2)
    b = _diff_df(n_runs=2, n_steps=2, sma_offset=1.0)
    diff = sweep_diff(a, b)

    # Default how="both" interleaves diff/rel per source column.
    cols = [c for c in diff.columns if c != "__status_diff"]
    assert cols == ["Sat.SMA__diff", "Sat.SMA__rel", "Sat.X__diff", "Sat.X__rel"]


def test_sweep_diff_how_absolute_emits_only_diff_columns() -> None:
    a = _diff_df(n_runs=2, n_steps=2)
    b = _diff_df(n_runs=2, n_steps=2, sma_offset=1.0)
    diff = sweep_diff(a, b, how="absolute")

    cols = [c for c in diff.columns if c != "__status_diff"]
    assert cols == ["Sat.SMA__diff", "Sat.X__diff"]


def test_sweep_diff_how_relative_emits_only_rel_columns() -> None:
    a = _diff_df(n_runs=2, n_steps=2)
    b = _diff_df(n_runs=2, n_steps=2, sma_offset=1.0)
    diff = sweep_diff(a, b, how="relative")

    cols = [c for c in diff.columns if c != "__status_diff"]
    assert cols == ["Sat.SMA__rel", "Sat.X__rel"]


def test_sweep_diff_relative_division_by_zero_is_nan() -> None:
    """``a == 0`` rows in the baseline produce NaN in __rel, not inf."""
    times = pd.to_datetime(["2026-05-09T00:00:00"])
    a = pd.DataFrame({"run_id": [0], "time": times, "x": [0.0], "__status": ["ok"]}).set_index(
        ["run_id", "time"]
    )
    b = pd.DataFrame({"run_id": [0], "time": times, "x": [5.0], "__status": ["ok"]}).set_index(
        ["run_id", "time"]
    )

    diff = sweep_diff(a, b, how="relative")
    assert pd.isna(diff["x__rel"].iloc[0])


def test_sweep_diff_on_run_id_collapses_time_level_to_terminal_row() -> None:
    a = _diff_df(n_runs=3, n_steps=4)
    b = _diff_df(n_runs=3, n_steps=4, sma_offset=10.0)
    diff = sweep_diff(a, b, on="run_id")

    # Output is indexed by run_id only.
    assert diff.index.name == "run_id"
    assert list(diff.index) == [0, 1, 2]

    # Terminal-row Sat.SMA differs by exactly the offset (constant across runs).
    np.testing.assert_array_equal(diff["Sat.SMA__diff"].to_numpy(), 10.0)


def test_sweep_diff_tolerance_float_masks_below_threshold_diffs() -> None:
    a = _diff_df(n_runs=3, n_steps=2)
    # Shift Sat.X by exactly 1, Sat.SMA by 0.005 (under 0.01 tolerance).
    b = _diff_df(n_runs=3, n_steps=2, sma_offset=0.005)
    b = b.assign(**{"Sat.X": b["Sat.X"] + 1.0})

    diff = sweep_diff(a, b, tolerance=0.01)

    # Sat.SMA diff (0.005) is below the cutoff → masked to NaN everywhere.
    assert diff["Sat.SMA__diff"].isna().all()
    assert diff["Sat.SMA__rel"].isna().all()
    # Sat.X diff (1.0) is above the cutoff → preserved.
    np.testing.assert_array_equal(diff["Sat.X__diff"].to_numpy(), 1.0)


def test_sweep_diff_tolerance_callable_applies_per_column_cutoff() -> None:
    a = _diff_df(n_runs=2, n_steps=2)
    b = _diff_df(n_runs=2, n_steps=2, sma_offset=2.0)
    b = b.assign(**{"Sat.X": b["Sat.X"] + 2.0})

    # Mask Sat.SMA below 5.0 (so diff=2 is masked) but only mask Sat.X below
    # 1.0 (so diff=2 survives).
    cutoffs = {"Sat.SMA": 5.0, "Sat.X": 1.0}
    diff = sweep_diff(a, b, tolerance=lambda col: cutoffs[col])

    assert diff["Sat.SMA__diff"].isna().all()
    np.testing.assert_array_equal(diff["Sat.X__diff"].to_numpy(), 2.0)


def test_sweep_diff_status_diff_encodes_pair_when_either_side_not_ok() -> None:
    a = _diff_df(n_runs=4, n_steps=2, statuses={1: "failed"})
    b = _diff_df(n_runs=4, n_steps=2, statuses={2: "skipped"})
    diff = sweep_diff(a, b, on="run_id")

    # run 0 / 3 are ok on both sides → "ok"
    # run 1 failed on a, ok on b → "failed/ok"
    # run 2 ok on a, skipped on b → "ok/skipped"
    assert diff.loc[0, "__status_diff"] == "ok"
    assert diff.loc[1, "__status_diff"] == "failed/ok"
    assert diff.loc[2, "__status_diff"] == "ok/skipped"
    assert diff.loc[3, "__status_diff"] == "ok"


def test_sweep_diff_status_diff_omitted_when_neither_side_has_status() -> None:
    a = _diff_df(n_runs=2, n_steps=2, include_status=False)
    b = _diff_df(n_runs=2, n_steps=2, sma_offset=1.0, include_status=False)
    diff = sweep_diff(a, b)

    assert "__status_diff" not in diff.columns


def test_sweep_diff_status_diff_treats_missing_side_as_ok() -> None:
    a = _diff_df(n_runs=3, n_steps=2, statuses={1: "failed"})
    b = _diff_df(n_runs=3, n_steps=2, sma_offset=1.0, include_status=False)
    diff = sweep_diff(a, b, on="run_id")

    # b has no __status — treat its side as "ok" for every run.
    assert diff.loc[0, "__status_diff"] == "ok"
    assert diff.loc[1, "__status_diff"] == "failed/ok"
    assert diff.loc[2, "__status_diff"] == "ok"


def test_sweep_diff_drops_non_shared_and_non_numeric_columns() -> None:
    a = _diff_df(n_runs=2, n_steps=2)
    b = _diff_df(n_runs=2, n_steps=2, sma_offset=1.0)
    # Add an a-only numeric column and a shared non-numeric column.
    a = a.assign(extra=1.0, label="aaa")
    b = b.assign(label="bbb")

    diff = sweep_diff(a, b)

    # extra is a-only → dropped. label is shared but non-numeric → dropped.
    cols = [c for c in diff.columns if c != "__status_diff"]
    assert all("extra" not in c and "label" not in c for c in cols)
    assert "Sat.SMA__diff" in cols
    assert "Sat.X__diff" in cols


def test_sweep_diff_index_intersection_drops_keys_present_on_only_one_side() -> None:
    a = _diff_df(n_runs=4, n_steps=1)
    # Drop run_id=3 from b — sweep_diff should align on the intersection.
    b = _diff_df(n_runs=4, n_steps=1, sma_offset=1.0).drop(index=3, level="run_id")

    diff = sweep_diff(a, b)

    surviving_run_ids = sorted({rid for rid, _ in diff.index})
    assert surviving_run_ids == [0, 1, 2]


def test_sweep_diff_rejects_unknown_how() -> None:
    a = _diff_df(n_runs=2, n_steps=2)
    with pytest.raises(SweepConfigError, match=r"how=.*not supported"):
        sweep_diff(a, a.copy(), how="median")


def test_sweep_diff_rejects_unsupported_on() -> None:
    a = _diff_df(n_runs=2, n_steps=2)
    with pytest.raises(SweepConfigError, match=r"on=.*not supported"):
        sweep_diff(a, a.copy(), on="time")


def test_sweep_diff_rejects_mismatched_index_level_names() -> None:
    a = _diff_df(n_runs=2, n_steps=2)
    # Rename b's secondary level to break the index-level-name match.
    b = a.copy()
    b.index = b.index.set_names(["run_id", "interval_id"])

    with pytest.raises(SweepConfigError, match=r"same index level names"):
        sweep_diff(a, b)


def test_sweep_diff_rejects_on_run_id_when_index_has_no_run_id_level() -> None:
    df = _diff_df(n_runs=2, n_steps=2).reset_index().set_index("time")
    with pytest.raises(SweepConfigError, match=r"requires a 'run_id' index level"):
        sweep_diff(df, df.copy(), on="run_id")


# ---------------------------------------------------------------------------
# engine="polars" — opt-in polars output across the DataFrame-returning surface
# ---------------------------------------------------------------------------

# pytest.importorskip skips this section's tests when the [polars] extra is not
# installed; the TYPE_CHECKING import gives mypy something to resolve when the
# typecheck CI cell runs without the extra (the polars override in pyproject
# treats the symbol as Any).
if TYPE_CHECKING:
    import polars as pl
else:
    pl = pytest.importorskip("polars")


def _assert_polars_matches_pandas_flat(
    pdf: pd.DataFrame,
    plf: pl.DataFrame,
    *,
    numeric_check_col: str,
) -> None:
    """Assert a polars-engine result matches its pandas-engine equivalent.

    The MultiIndex (if any) is flattened to leading columns under the polars
    engine, so we compare row count, column set (after the flatten), and at
    least one numeric column to text precision — the issue's stated DoD.
    Pandas ``NaN``/``NaT``/``pd.NA`` and polars ``null`` all collapse to
    ``None`` for the per-element comparison.
    """
    import math

    def _normalise(v: Any) -> Any:
        if v is None or v is pd.NaT or v is pd.NA:
            return None
        if isinstance(v, float) and math.isnan(v):
            return None
        return v

    flat = pdf.reset_index() if pdf.index.nlevels > 1 or pdf.index.name else pdf
    assert plf.height == len(flat)
    assert set(plf.columns) == set(flat.columns)
    pl_col = [_normalise(v) for v in plf.get_column(numeric_check_col).to_list()]
    pd_col = [_normalise(v) for v in flat[numeric_check_col].tolist()]
    assert pl_col == pd_col


def test_lazy_multiindex_engine_polars_flattens_index_and_matches_pandas(
    tmp_path: Path,
) -> None:
    paths = [_write_run_parquet(tmp_path, i, n_rows=2) for i in range(4)]
    entries: list[ManifestEntry] = [_ok_entry(i, p) for i, p in enumerate(paths)]
    entries.append(_nonok_entry(4, "failed"))
    manifest = _make_manifest(entries)

    pdf = lazy_multiindex(manifest, tmp_path)
    plf = lazy_multiindex(manifest, tmp_path, engine="polars")

    assert isinstance(plf, pl.DataFrame)
    assert plf.columns[:2] == ["run_id", "time"]
    _assert_polars_matches_pandas_flat(pdf, plf, numeric_check_col="x")


def test_lazy_ephemerides_engine_polars_matches_pandas(tmp_path: Path) -> None:
    paths = [
        _write_run_parquet(tmp_path, i, n_rows=2, basename="ephemeris__Eph1") for i in range(3)
    ]
    manifest = _make_manifest([_ok_entry(i, p, key="ephemeris__Eph1") for i, p in enumerate(paths)])

    pdf = lazy_ephemerides(manifest, tmp_path)
    plf = lazy_ephemerides(manifest, tmp_path, engine="polars")

    assert isinstance(plf, pl.DataFrame)
    _assert_polars_matches_pandas_flat(pdf, plf, numeric_check_col="y")


def test_lazy_contacts_engine_polars_preserves_int64_nullable_interval_id(
    tmp_path: Path,
) -> None:
    p0 = _write_contact_parquet(tmp_path, 0, n_intervals=2)
    p1 = _write_contact_parquet(tmp_path, 1, n_intervals=1)
    entries: list[ManifestEntry] = [
        _ok_entry(0, p0, key="contact__GroundContact"),
        _ok_entry(1, p1, key="contact__GroundContact"),
        _nonok_entry(2, "failed"),
    ]
    manifest = _make_manifest(entries)

    pdf = lazy_contacts(manifest, tmp_path)
    plf = lazy_contacts(manifest, tmp_path, engine="polars")

    assert isinstance(plf, pl.DataFrame)
    assert plf.columns[:2] == ["run_id", "interval_id"]
    # The failed run's interval_id must be null in polars (round-tripped from
    # pandas Int64 + pd.NA), not 0 or some other sentinel.
    failed_row = plf.filter(pl.col("run_id") == 2)
    assert failed_row.height == 1
    assert failed_row["interval_id"][0] is None
    _assert_polars_matches_pandas_flat(pdf, plf, numeric_check_col="Duration")


def test_mc_convergence_engine_polars_matches_pandas() -> None:
    df = _conv_df(n_runs=12)
    pdf = mc_convergence(df, "miss", terminal_only=True)
    plf = mc_convergence(df, "miss", terminal_only=True, engine="polars")

    assert isinstance(plf, pl.DataFrame)
    _assert_polars_matches_pandas_flat(pdf, plf, numeric_check_col="running_mean")


def test_sweep_diff_engine_polars_matches_pandas() -> None:
    a = _diff_df(n_runs=3, n_steps=2)
    b = _diff_df(n_runs=3, n_steps=2, sma_offset=50.0)
    pdf = sweep_diff(a, b)
    plf = sweep_diff(a, b, engine="polars")

    assert isinstance(plf, pl.DataFrame)
    assert plf.columns[:2] == ["run_id", "time"]
    _assert_polars_matches_pandas_flat(pdf, plf, numeric_check_col="Sat.SMA__diff")


@pytest.mark.parametrize(
    "call",
    [
        lambda m, d: lazy_multiindex(m, d, engine="bogus"),
        lambda m, d: lazy_ephemerides(m, d, engine="bogus"),
        lambda m, d: lazy_contacts(m, d, engine="bogus"),
    ],
)
def test_lazy_aggregators_reject_unknown_engine(
    tmp_path: Path,
    call: Any,
) -> None:
    manifest = _make_manifest([])
    with pytest.raises(SweepConfigError, match=r"engine='bogus' is not supported"):
        call(manifest, tmp_path)


def test_mc_convergence_rejects_unknown_engine() -> None:
    df = _conv_df(n_runs=2)
    with pytest.raises(SweepConfigError, match=r"engine='bogus' is not supported"):
        mc_convergence(df, "miss", engine="bogus")


def test_sweep_diff_rejects_unknown_engine() -> None:
    a = _diff_df(n_runs=2, n_steps=2)
    with pytest.raises(SweepConfigError, match=r"engine='bogus' is not supported"):
        sweep_diff(a, a.copy(), engine="bogus")


def test_polars_missing_extra_raises_install_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When polars is not importable, the ImportError carries the install hint."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "polars":
            raise ImportError("No module named 'polars'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    p = _write_run_parquet(tmp_path, 0, n_rows=1)
    manifest = _make_manifest([_ok_entry(0, p)])
    with pytest.raises(ImportError, match=r"pip install gmat-sweep\[polars\]"):
        lazy_multiindex(manifest, tmp_path, engine="polars")
