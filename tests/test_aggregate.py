"""Tests for gmat_sweep.aggregate — lazy multi-indexed result assembly from Parquet."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from gmat_sweep.aggregate import lazy_multiindex
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


def _ok_entry(run_id: int, parquet_path: Path) -> ManifestEntry:
    return ManifestEntry(
        run_id=run_id,
        overrides={},
        status="ok",
        output_paths={"ReportFile1": parquet_path},
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


def _write_run_parquet(output_dir: Path, run_id: int, *, n_rows: int = 3) -> Path:
    path = output_dir / f"run-{run_id}" / "ReportFile1.parquet"
    path.parent.mkdir(parents=True)
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
        output_paths={"ReportFile1": rel_path},
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


def test_lazy_multiindex_multi_report_run_raises(tmp_path: Path) -> None:
    p1 = _write_run_parquet(tmp_path, 0, n_rows=2)
    p2 = tmp_path / "run-0" / "ReportFile2.parquet"
    pd.DataFrame({"time": pd.to_datetime(["2026-05-04T00:00:00"]), "x": [0]}).to_parquet(p2)

    entry = ManifestEntry(
        run_id=0,
        overrides={},
        status="ok",
        output_paths={"ReportFile1": p1, "ReportFile2": p2},
        started_at=_utc(2026, 5, 4),
        ended_at=_utc(2026, 5, 4, 0, 0, 1),
        duration_s=1.0,
        stderr=None,
        log_path=None,
    )

    with pytest.raises(NotImplementedError, match=r"v0\.2"):
        lazy_multiindex(_make_manifest([entry]), tmp_path)


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
    path = tmp_path / "run-0" / "ReportFile1.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame({"x": [1, 2, 3]}).to_parquet(path)

    with pytest.raises(ValueError, match="time"):
        lazy_multiindex(_make_manifest([_ok_entry(0, path)]), tmp_path)
