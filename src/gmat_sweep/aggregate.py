"""Lazy assembly of multi-indexed parent DataFrames from per-run Parquet files.

Three public entry points cover the three GMAT output kinds gmat-run surfaces:

- :func:`lazy_multiindex` — ``ReportFile`` outputs, ``(run_id, time)`` index.
- :func:`lazy_ephemerides` — ``EphemerisFile`` outputs, ``(run_id, time)`` index.
- :func:`lazy_contacts` — ``ContactLocator`` outputs, ``(run_id, interval_id)``
  index.

Each walks a :class:`gmat_sweep.manifest.Manifest`, dispatches to per-run
Parquet files by the ``<kind>__<name>`` key prefix the worker writes, and
stitches the per-run frames into one multi-indexed
:class:`pandas.DataFrame`. Failed and skipped runs (and ``ok`` runs that
did not produce the requested output kind) are materialised as one-row,
NaN-filled slices so the caller sees a complete row per run rather than
having to reconcile a ``DataFrame`` against the manifest by hand.

When a sweep produces multiple outputs of the same kind (e.g. two
``ReportFile`` resources), pass ``name=`` to pick one. With a single
output of that kind, ``name=None`` resolves it automatically; with two or
more, ``name=None`` raises :class:`gmat_sweep.errors.SweepConfigError`
listing the available names.

The default report/ephemeris path streams each run's record batches
through pandas one fragment at a time so peak conversion memory is one
batch, not one full sweep. ``spool=False`` flips to an eager
fragment-at-a-time read for small sweeps where the streaming overhead is
not worth it; the result DataFrame is identical.

:func:`sweep_summary` collapses the parent DataFrame across runs into a
per-``time`` (or per-``run_id``) statistics frame — the canonical input
for "median +/- 95% band over time" dispersion plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from gmat_sweep.errors import SweepConfigError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import TypeAlias

    import polars as pl

    from gmat_sweep.manifest import Manifest, ManifestEntry

    DataFrame: TypeAlias = pd.DataFrame | pl.DataFrame

__all__ = [
    "lazy_contacts",
    "lazy_ephemerides",
    "lazy_fused_reports",
    "lazy_multiindex",
    "mc_convergence",
    "sweep_diff",
    "sweep_summary",
]

_VALID_BY: tuple[str, ...] = ("time", "run_id")
_VALID_INCLUDE: tuple[str, ...] = ("mean", "std", "min", "max", "count_ok")
_VALID_HOW: tuple[str, ...] = ("absolute", "relative", "both")
_VALID_ENGINES: tuple[str, ...] = ("pandas", "polars")


def _check_engine(engine: str) -> None:
    if engine not in _VALID_ENGINES:
        raise SweepConfigError(
            f"engine={engine!r} is not supported; pass one of {list(_VALID_ENGINES)}"
        )


def _to_polars(df: pd.DataFrame) -> pl.DataFrame:
    """Flatten a pandas-engine sweep DataFrame into a polars DataFrame.

    The ``(run_id, <secondary>)`` MultiIndex becomes two leading columns;
    row order is preserved (the input is already ``sort_index``-ed). Polars
    is imported lazily so the default ``engine="pandas"`` path does not
    require the ``[polars]`` extra.
    """
    try:
        import polars as pl
    except ImportError as exc:
        raise ImportError(
            "engine='polars' requires the optional polars extra; "
            "install with: pip install gmat-sweep[polars]"
        ) from exc

    flat = df.reset_index() if df.index.nlevels > 0 and df.index.names != [None] else df
    return pl.from_pandas(flat)


_RUN_ID_COL = "run_id"
_STATUS_COL = "__status"
_STATUS_DIFF_COL = "__status_diff"
_TIME_COL = "time"
_INTERVAL_ID_COL = "interval_id"

# Secondary-index dtype + per-entry "missing" sentinel, keyed by the column
# name the aggregator uses as the second MultiIndex level. Adding a new
# secondary-index kind is a one-line edit here.
_SECONDARY_INDEX_DTYPE: dict[str, str] = {
    _TIME_COL: "datetime64[ns]",
    _INTERVAL_ID_COL: "Int64",
}
_SECONDARY_INDEX_MISSING: dict[str, Any] = {
    _TIME_COL: pd.NaT,
    _INTERVAL_ID_COL: pd.NA,
}


@overload
def lazy_multiindex(
    manifest: Manifest,
    output_dir: Path,
    *,
    name: str | None = ...,
    spool: bool = ...,
    engine: Literal["pandas"] = ...,
) -> pd.DataFrame: ...
@overload
def lazy_multiindex(
    manifest: Manifest,
    output_dir: Path,
    *,
    name: str | None = ...,
    spool: bool = ...,
    engine: Literal["polars"],
) -> pl.DataFrame: ...
@overload
def lazy_multiindex(
    manifest: Manifest,
    output_dir: Path,
    *,
    name: str | None = ...,
    spool: bool = ...,
    engine: str,
) -> DataFrame: ...
def lazy_multiindex(
    manifest: Manifest,
    output_dir: Path,
    *,
    name: str | None = None,
    spool: bool = True,
    engine: str = "pandas",
) -> DataFrame:
    """Assemble the ``(run_id, time)``-indexed report DataFrame from a sweep's outputs.

    Iterates ``manifest.entries`` in order. For each ``ok`` entry the
    Parquet listed in :attr:`ManifestEntry.output_paths` under the
    ``report__<name>`` key is read via :mod:`pyarrow.dataset` and tagged
    with its ``run_id``. For each ``failed`` or ``skipped`` entry — and
    for any ``ok`` entry that did not produce the requested report — one
    NaN-filled row is materialised with ``time = NaT`` and ``__status``
    set to the run-level status (``"failed"`` / ``"skipped"`` for non-ok
    runs, ``"ok"`` for ok runs missing this report).

    Relative paths in ``output_paths`` are resolved against ``output_dir``;
    absolute paths are used as-is.

    Parameters
    ----------
    manifest
        The sweep manifest. Drives both the set of runs and their status.
    output_dir
        Sweep output root. Used to anchor any relative paths recorded in
        the manifest.
    name
        Report resource name to aggregate. ``None`` (default) picks the
        sole report if exactly one report is present across the sweep.
        Sweeps that produced multiple reports per run must pass ``name=``
        explicitly; the call raises
        :class:`gmat_sweep.errors.SweepConfigError` listing the available
        names otherwise.
    spool
        ``True`` (default) streams each run's record batches into pandas
        one batch at a time. ``False`` reads each run's Parquet eagerly
        in one shot — simpler control flow, higher peak memory.
    engine
        ``"pandas"`` (default) returns a ``(run_id, time)``-MultiIndexed
        :class:`pandas.DataFrame`. ``"polars"`` returns a
        :class:`polars.DataFrame` whose ``(run_id, time)`` MultiIndex is
        flattened into two leading sorted columns; row count and the
        non-index column set match the pandas-engine equivalent. Requires
        the ``[polars]`` extra; an :class:`ImportError` with the install
        hint is raised when polars is not importable.

    Raises
    ------
    SweepConfigError
        ``name=None`` was passed but the sweep produced more than one
        report (the exception message lists the available names), the
        explicitly-named report does not appear in any ok run's outputs,
        or ``engine`` is neither ``"pandas"`` nor ``"polars"``.
    ValueError
        An ``ok`` entry has no ``output_paths`` at all (a run that ran
        successfully must have produced something), or a per-run Parquet
        is missing the ``time`` column required for the index.
    """
    _check_engine(engine)
    df = _aggregate(
        manifest,
        output_dir,
        kind="report",
        secondary_index=_TIME_COL,
        name=name,
        spool=spool,
    )
    return _to_polars(df) if engine == "polars" else df


@overload
def lazy_ephemerides(
    manifest: Manifest,
    output_dir: Path,
    *,
    name: str | None = ...,
    spool: bool = ...,
    engine: Literal["pandas"] = ...,
) -> pd.DataFrame: ...
@overload
def lazy_ephemerides(
    manifest: Manifest,
    output_dir: Path,
    *,
    name: str | None = ...,
    spool: bool = ...,
    engine: Literal["polars"],
) -> pl.DataFrame: ...
@overload
def lazy_ephemerides(
    manifest: Manifest,
    output_dir: Path,
    *,
    name: str | None = ...,
    spool: bool = ...,
    engine: str,
) -> DataFrame: ...
def lazy_ephemerides(
    manifest: Manifest,
    output_dir: Path,
    *,
    name: str | None = None,
    spool: bool = True,
    engine: str = "pandas",
) -> DataFrame:
    """Assemble the ``(run_id, time)``-indexed ephemeris DataFrame from a sweep's outputs.

    Mirrors :func:`lazy_multiindex` but dispatches on ``ephemeris__<name>``
    keys instead of ``report__<name>``. The worker copies the first
    datetime column of each ephemeris frame (``Epoch`` for OEM, STK, and
    SPK formats) to a column named ``time`` before writing Parquet, so
    the same ``(run_id, time)`` index machinery applies.

    See :func:`lazy_multiindex` for parameter and exception semantics,
    including the ``engine`` knob.
    """
    _check_engine(engine)
    df = _aggregate(
        manifest,
        output_dir,
        kind="ephemeris",
        secondary_index=_TIME_COL,
        name=name,
        spool=spool,
    )
    return _to_polars(df) if engine == "polars" else df


@overload
def lazy_contacts(
    manifest: Manifest,
    output_dir: Path,
    *,
    name: str | None = ...,
    engine: Literal["pandas"] = ...,
) -> pd.DataFrame: ...
@overload
def lazy_contacts(
    manifest: Manifest,
    output_dir: Path,
    *,
    name: str | None = ...,
    engine: Literal["polars"],
) -> pl.DataFrame: ...
@overload
def lazy_contacts(
    manifest: Manifest,
    output_dir: Path,
    *,
    name: str | None = ...,
    engine: str,
) -> DataFrame: ...
def lazy_contacts(
    manifest: Manifest,
    output_dir: Path,
    *,
    name: str | None = None,
    engine: str = "pandas",
) -> DataFrame:
    """Assemble the ``(run_id, interval_id)``-indexed contact DataFrame from a sweep's outputs.

    Mirrors :func:`lazy_multiindex` but dispatches on ``contact__<name>``
    keys and uses ``interval_id`` — the per-run row position the worker
    assigns at write time, ``0..K-1`` per run — as the secondary index
    level. ``ContactLocator`` outputs are typically tiny (one row per
    visibility interval), so there is no ``spool`` knob; reads are
    fragment-at-a-time eager.

    Failed, skipped, and report-only ``ok`` runs materialise as one row
    with ``interval_id = pd.NA`` (cast as the nullable ``Int64`` dtype so
    integer interval indices and missing values share one level). Under
    ``engine="polars"`` the nullable ``Int64`` round-trips into a polars
    ``Int64`` column with ``null`` for the missing slots.

    See :func:`lazy_multiindex` for the rest of the parameter and
    exception semantics, including the ``engine`` knob.
    """
    _check_engine(engine)
    df = _aggregate(
        manifest,
        output_dir,
        kind="contact",
        secondary_index=_INTERVAL_ID_COL,
        name=name,
        spool=False,
    )
    return _to_polars(df) if engine == "polars" else df


def _aggregate(
    manifest: Manifest,
    output_dir: Path,
    *,
    kind: str,
    secondary_index: str,
    name: str | None,
    spool: bool,
) -> pd.DataFrame:
    prefix = f"{kind}__"
    index_names = (_RUN_ID_COL, secondary_index)

    names_seen: set[str] = set()
    for entry in manifest.entries:
        if entry.status != "ok":
            continue
        if not entry.output_paths:
            raise ValueError(
                f"manifest entry for run_id={entry.run_id} has status='ok' but no output_paths"
            )
        for key in entry.output_paths:
            if key.startswith(prefix):
                names_seen.add(key[len(prefix) :])

    if name is None:
        if len(names_seen) > 1:
            raise SweepConfigError(
                f"sweep produced {len(names_seen)} {kind} outputs "
                f"({sorted(names_seen)}); pass name= to select one"
            )
        if len(names_seen) == 1:
            name = next(iter(names_seen))
        # 0 names → name stays None; every ok run is treated as "did not
        # produce this kind" and lands as a NaN row, matching the
        # all-failed contract.
    elif name not in names_seen and any(e.status == "ok" for e in manifest.entries):
        raise SweepConfigError(
            f"no {kind} output named {name!r} in this sweep; available: {sorted(names_seen)}"
        )

    target_key = f"{prefix}{name}" if name is not None else None

    paths_with_run_id: list[tuple[Path, int]] = []
    nonok_entries: list[ManifestEntry] = []
    for entry in manifest.entries:
        if entry.status == "ok" and target_key is not None and target_key in entry.output_paths:
            raw_path = entry.output_paths[target_key]
            resolved = raw_path if raw_path.is_absolute() else output_dir / raw_path
            paths_with_run_id.append((resolved, entry.run_id))
        else:
            nonok_entries.append(entry)

    ok_df = _read_ok_runs(
        paths_with_run_id,
        secondary_index=secondary_index,
        index_names=index_names,
        spool=spool,
    )
    data_columns = [c for c in ok_df.columns if c != _STATUS_COL] if not ok_df.empty else []
    nonok_df = _materialise_nonok(
        nonok_entries,
        data_columns=data_columns,
        secondary_index=secondary_index,
        index_names=index_names,
    )

    parts = [df for df in (ok_df, nonok_df) if not df.empty]
    if not parts:
        return _empty_frame(secondary_index=secondary_index, index_names=index_names)

    return cast(pd.DataFrame, pd.concat(parts, axis=0).sort_index())


def _read_ok_runs(
    paths_with_run_id: Sequence[tuple[Path, int]],
    *,
    secondary_index: str,
    index_names: tuple[str, str],
    spool: bool,
) -> pd.DataFrame:
    if not paths_with_run_id:
        return pd.DataFrame()

    # pyarrow normalises filesystem paths to forward-slash form on every
    # platform, so we key path -> run_id in POSIX form and look up
    # fragment.path the same way. Without this, str(WindowsPath(...)) on
    # Windows produces backslash keys that never match what pyarrow returns.
    paths_str = [Path(p).as_posix() for p, _ in paths_with_run_id]
    path_to_run_id = {Path(p).as_posix(): run_id for p, run_id in paths_with_run_id}

    dataset = ds.dataset(paths_str, format="parquet")  # type: ignore[no-untyped-call]

    # Stay in Arrow for the per-fragment loop and materialise pandas once at
    # the end. pa.concat_tables shares buffers across input tables, so peak
    # memory is one pandas frame's worth of the full ok-row set — not two
    # (the per-fragment frames + the merged frame) as the older
    # accumulate-and-pd.concat path produced. See #130.
    tables: list[pa.Table] = []
    for fragment in dataset.get_fragments():
        run_id = path_to_run_id[Path(fragment.path).as_posix()]
        if spool:
            for batch in fragment.to_batches():
                tables.append(_augment_with_run_id(pa.Table.from_batches([batch]), run_id))
        else:
            tables.append(_augment_with_run_id(fragment.to_table(), run_id))

    if not tables:
        return pd.DataFrame()

    merged_table = pa.concat_tables(tables, promote_options="default")
    if secondary_index not in merged_table.column_names:
        raise ValueError(
            f"per-run Parquet output is missing the {secondary_index!r} column "
            f"required for the (run_id, {secondary_index}) MultiIndex"
        )
    merged = cast(pd.DataFrame, merged_table.to_pandas())
    # Cast the secondary index to the per-kind canonical dtype: datetime64[ns]
    # for time (so NaT is the missing sentinel for non-ok rows), Int64 nullable
    # for interval_id (so pd.NA can share the same level as ok integer rows).
    merged[secondary_index] = _coerce_secondary_index(merged[secondary_index], secondary_index)
    merged[_STATUS_COL] = "ok"
    return cast(pd.DataFrame, merged.set_index(list(index_names)))


def _coerce_secondary_index(series: pd.Series, secondary_index: str) -> pd.Series:
    # pandas-stubs overloads .astype on literal dtype strings, so the dynamic
    # lookup on _SECONDARY_INDEX_DTYPE has to widen to Any to satisfy the
    # checker. Runtime behaviour matches every static overload.
    dtype: Any = _SECONDARY_INDEX_DTYPE[secondary_index]
    if secondary_index == _TIME_COL:
        return pd.to_datetime(series).astype(dtype)
    return series.astype(dtype)


def _augment_with_run_id(table: pa.Table, run_id: int) -> pa.Table:
    n_rows = table.num_rows
    run_id_col = pa.array([run_id] * n_rows, type=pa.int64())
    return cast(pa.Table, table.append_column(_RUN_ID_COL, run_id_col))


def _materialise_nonok(
    entries: Sequence[ManifestEntry],
    *,
    data_columns: Sequence[str],
    secondary_index: str,
    index_names: tuple[str, str],
) -> pd.DataFrame:
    if not entries:
        return pd.DataFrame()

    rows: dict[str, Any] = {col: [np.nan] * len(entries) for col in data_columns}
    rows[_RUN_ID_COL] = [e.run_id for e in entries]
    rows[secondary_index] = [_SECONDARY_INDEX_MISSING[secondary_index]] * len(entries)
    rows[_STATUS_COL] = [e.status for e in entries]

    df = pd.DataFrame(rows)
    df[secondary_index] = _coerce_secondary_index(df[secondary_index], secondary_index)
    return cast(pd.DataFrame, df.set_index(list(index_names)))


def _empty_frame(*, secondary_index: str, index_names: tuple[str, str]) -> pd.DataFrame:
    secondary_series = pd.Series([], dtype=_SECONDARY_INDEX_DTYPE[secondary_index])
    return pd.DataFrame(
        {_STATUS_COL: pd.Series([], dtype="object")},
        index=pd.MultiIndex.from_arrays(
            [pd.Series([], dtype="int64"), secondary_series],
            names=list(index_names),
        ),
    )


def lazy_fused_reports(
    manifest: Manifest,
    output_dir: Path,
    names: Sequence[str],
    *,
    tolerance: str | pd.Timedelta,
    spool: bool = True,
) -> pd.DataFrame:
    """Fuse N ``ReportFile`` outputs per run into one wide ``(run_id, time)``-indexed DataFrame.

    Each report is read via :func:`lazy_multiindex` and the per-run slices
    are stitched together into a single frame whose columns form a
    two-level :class:`pandas.MultiIndex` keyed by ``(report_name, column)``.
    The first name in ``names`` is the merge anchor: subsequent reports
    are joined to it per ``run_id`` via :func:`pandas.merge_asof` (with
    the user-supplied ``tolerance``), or via an inner join on ``time``
    when ``tolerance="exact"`` — appropriate when every report shares a
    step setting.

    Parameters
    ----------
    manifest
        The sweep manifest.
    output_dir
        Sweep output root, used to anchor relative paths in the manifest.
    names
        The ``ReportFile`` resource names to fuse, in merge order. The
        first name is the anchor (left side of every merge); later names
        are joined onto it. Must contain at least two unique names — for a
        single report use :func:`lazy_multiindex` with ``name=...``.
    tolerance
        Required. Either the literal string ``"exact"`` (collapses to an
        inner join on ``time`` per run, appropriate when reports share a
        step setting) or any value :func:`pandas.merge_asof` accepts as a
        ``tolerance`` argument — typically a :class:`pandas.Timedelta`.
    spool
        Forwarded to each underlying :func:`lazy_multiindex` call.
        ``True`` (default) streams per-run Parquet a batch at a time;
        ``False`` reads each fragment in one shot.

    Returns
    -------
    pandas.DataFrame
        Row index: ``(run_id, time)`` :class:`MultiIndex`. Column index:
        a two-level :class:`MultiIndex` ``("report", "field")``. For each
        report, data columns appear at ``(report_name, column)`` and the
        per-report status (preserving the per-report contract from
        :func:`lazy_multiindex`) at ``(report_name, "__status")``. A
        run-level status sits at ``("__status", "")``.

    Notes
    -----
    The anchor selection is asymmetric. A run whose anchor failed
    (``__status != "ok"``) — or whose anchor parquet was missing — lands
    as a single ``time=NaT`` row with all data ``NaN`` and the per-report
    ``__status`` of every report preserved. The other reports' data for
    that run is not surfaced. Pick the report most likely to be present
    as the first entry of ``names``.

    Raises
    ------
    SweepConfigError
        ``names`` has fewer than two entries, contains duplicates, or any
        entry does not match a ``ReportFile`` resource in the sweep
        (raised by the underlying :func:`lazy_multiindex` call).
    """
    if len(names) < 2:
        raise SweepConfigError(
            f"lazy_fused_reports requires at least 2 report names; got {len(names)}. "
            "Use lazy_multiindex(name=...) for a single report."
        )
    if len(set(names)) != len(names):
        raise SweepConfigError(
            f"lazy_fused_reports requires unique report names; got duplicates in {list(names)}"
        )

    asof_tolerance: pd.Timedelta | None
    if isinstance(tolerance, str):
        if tolerance != "exact":
            raise SweepConfigError(
                f"tolerance must be the literal string 'exact' or a pd.Timedelta; got {tolerance!r}"
            )
        asof_tolerance = None
    else:
        asof_tolerance = tolerance

    per_report: list[pd.DataFrame] = [
        lazy_multiindex(manifest, output_dir, name=name, spool=spool) for name in names
    ]

    # Internal flat labels avoid string-prefix collisions with arbitrary
    # GMAT column names. The flat → (report, field) mapping is applied
    # once at the very end to build the column MultiIndex.
    flat_to_tuple: dict[str, tuple[str, str]] = {}
    rename_per_report: list[dict[str, str]] = []
    for ridx, (name, df) in enumerate(zip(names, per_report, strict=True)):
        rename: dict[str, str] = {}
        for cidx, col in enumerate(df.columns):
            flat = f"_s{ridx}" if col == _STATUS_COL else f"_d{ridx}_{cidx}"
            rename[col] = flat
            flat_to_tuple[flat] = (name, col)
        rename_per_report.append(rename)

    flat_per_report: list[pd.DataFrame] = [
        df.reset_index().rename(columns=rename_per_report[i]) for i, df in enumerate(per_report)
    ]

    # One groupby(run_id) per report — O(N) total — instead of a fresh
    # boolean mask per (report, run_id) pair, which was O(N · runs) and
    # showed up as a quadratic hot spot on large sweeps (#130). pandas'
    # group iteration is cheap; we materialise the per-run slice lazily.
    grouped_per_report: list[dict[int, pd.DataFrame]] = [
        {
            int(rid): cast(pd.DataFrame, sub.drop(columns=[_RUN_ID_COL]))
            for rid, sub in df.groupby(_RUN_ID_COL, sort=False)
        }
        for df in flat_per_report
    ]
    empty_slices: list[pd.DataFrame] = [
        flat_per_report[i].iloc[0:0].drop(columns=[_RUN_ID_COL]) for i in range(len(per_report))
    ]

    run_status: dict[int, str] = {e.run_id: e.status for e in manifest.entries}
    run_ids: list[int] = [e.run_id for e in manifest.entries]

    parts: list[pd.DataFrame] = []
    for run_id in run_ids:
        per_run = [grouped_per_report[i].get(run_id, empty_slices[i]) for i in range(len(per_report))]
        anchor = per_run[0]
        anchor_has_data = not anchor.empty and bool(anchor[_TIME_COL].notna().any())
        if anchor_has_data:
            fused = _merge_run_fused(
                per_run, tolerance=asof_tolerance, rename_per_report=rename_per_report
            )
        else:
            fused = _anchor_failed_row_fused(
                per_run,
                rename_per_report=rename_per_report,
                run_level_status=run_status[run_id],
            )
        fused[_RUN_ID_COL] = run_id
        fused[_STATUS_COL] = run_status[run_id]
        parts.append(fused)

    if not parts:
        return _empty_fused_frame(flat_to_tuple)

    result = pd.concat(parts, axis=0, ignore_index=True, sort=False)
    result[_TIME_COL] = pd.to_datetime(result[_TIME_COL]).astype("datetime64[ns]")
    result = result.set_index([_RUN_ID_COL, _TIME_COL]).sort_index()

    new_cols: list[tuple[str, str]] = [
        (_STATUS_COL, "") if col == _STATUS_COL else flat_to_tuple[col] for col in result.columns
    ]
    result.columns = pd.MultiIndex.from_tuples(new_cols, names=["report", "field"])
    return cast(pd.DataFrame, result)


def _merge_run_fused(
    per_run: Sequence[pd.DataFrame],
    *,
    tolerance: pd.Timedelta | None,
    rename_per_report: Sequence[dict[str, str]],
) -> pd.DataFrame:
    # Anchor has ok-data here; later reports join onto it. ``tolerance=None``
    # is the "exact" sentinel from lazy_fused_reports — collapses the merge
    # to an inner join on time. Reports with only a NaN-marker row for this
    # run contribute one row's worth of NaN data + their NaN-marker per-report
    # status.
    anchor = cast(
        pd.DataFrame,
        per_run[0].dropna(subset=[_TIME_COL]).sort_values(_TIME_COL).reset_index(drop=True),
    )

    for i in range(1, len(per_run)):
        right = per_run[i]
        right_has_data = not right.empty and bool(right[_TIME_COL].notna().any())
        if right_has_data:
            right_ok = cast(
                pd.DataFrame,
                right.dropna(subset=[_TIME_COL]).sort_values(_TIME_COL).reset_index(drop=True),
            )
            if tolerance is None:
                anchor = anchor.merge(right_ok, on=_TIME_COL, how="inner")
            else:
                # pd.merge_asof silently produces wrong answers when its
                # inputs aren't sorted on the asof key; we sort both sides
                # explicitly above, but assert the contract so future
                # refactors here surface as a clear failure rather than
                # silent data corruption.
                if not anchor[_TIME_COL].is_monotonic_increasing:
                    raise AssertionError(
                        "lazy_fused_reports: anchor frame is not sorted on "
                        f"{_TIME_COL!r} before pd.merge_asof — required for "
                        "asof tolerance matching"
                    )
                anchor = pd.merge_asof(anchor, right_ok, on=_TIME_COL, tolerance=tolerance)
        else:
            for orig_col, flat in rename_per_report[i].items():
                if orig_col == _STATUS_COL:
                    anchor[flat] = right[flat].iloc[0] if not right.empty else pd.NA
                else:
                    anchor[flat] = np.nan
    return anchor


def _anchor_failed_row_fused(
    per_run: Sequence[pd.DataFrame],
    *,
    rename_per_report: Sequence[dict[str, str]],
    run_level_status: str,
) -> pd.DataFrame:
    # Anchor has no ok-data for this run; emit one NaT-time row carrying
    # every report's per-report status. Data columns are NaN regardless of
    # whether non-anchor reports happened to have data — the asymmetry is
    # documented on lazy_fused_reports.
    row: dict[str, Any] = {_TIME_COL: pd.NaT}
    for i, df in enumerate(per_run):
        for orig_col, flat in rename_per_report[i].items():
            if orig_col == _STATUS_COL:
                if not df.empty:
                    row[flat] = df[flat].iloc[0]
                else:
                    row[flat] = run_level_status
            else:
                row[flat] = np.nan
    return pd.DataFrame([row])


def _empty_fused_frame(flat_to_tuple: dict[str, tuple[str, str]]) -> pd.DataFrame:
    cols = [*flat_to_tuple.values(), (_STATUS_COL, "")]
    return pd.DataFrame(
        {col: pd.Series([], dtype="object") for col in cols},
        index=pd.MultiIndex.from_arrays(
            [pd.Series([], dtype="int64"), pd.Series([], dtype="datetime64[ns]")],
            names=[_RUN_ID_COL, _TIME_COL],
        ),
        columns=pd.MultiIndex.from_tuples(cols, names=["report", "field"]),
    )


def sweep_summary(
    df: pd.DataFrame,
    *,
    by: str = _TIME_COL,
    q: Sequence[float] = (0.05, 0.5, 0.95),
    include: Sequence[str] = ("mean", "std"),
    dropna: bool = True,
) -> pd.DataFrame:
    """Summarise a sweep DataFrame across runs at each ``by`` key.

    Turns a ``(run_id, time)``-MultiIndexed DataFrame — as returned by
    :func:`lazy_multiindex`, :func:`gmat_sweep.sweep`,
    :func:`gmat_sweep.monte_carlo`, or :func:`gmat_sweep.latin_hypercube`
    — into a per-``by`` statistics frame: one row per unique ``by``
    value, one column per ``(statistic, original-column)`` pair under a
    two-level :class:`pandas.MultiIndex`. The default
    ``q=(0.05, 0.5, 0.95)`` matches the standard 5/50/95 dispersion bands
    and feeds directly into :func:`gmat_sweep.plotting.sweep_band_plot`.

    Parameters
    ----------
    df
        Input DataFrame indexed by ``(run_id, time)`` (or any other
        2-level :class:`pandas.MultiIndex` whose levels include the
        requested ``by`` key). A ``__status`` column, if present, is
        treated as a per-run status flag and excluded from the statistic
        columns.
    by
        Index level to group on. ``"time"`` (default) collapses across
        runs at each time step. ``"run_id"`` collapses across time
        steps within each run. Other values raise
        :class:`gmat_sweep.errors.SweepConfigError` — categorical or
        arbitrary keys are out of scope in this release.
    q
        Quantiles to compute. Each entry must be a float in the open
        interval ``(0, 1)``. The default returns the 5th, 50th, and
        95th percentiles. Pass an empty tuple to skip quantiles.
    include
        Non-quantile statistics to compute, in the order they appear in
        the output's column-level ``"statistic"`` index. Allowed values:
        ``"mean"``, ``"std"``, ``"min"``, ``"max"``, ``"count_ok"``.
        ``"count_ok"`` is the per-group count of non-NaN values in each
        data column. Pass an empty tuple to skip the non-quantile stats
        and emit only quantiles.
    dropna
        ``True`` (default) drops rows whose ``__status != "ok"`` before
        aggregating, so failed and skipped runs are excluded from every
        statistic. ``False`` keeps them — their NaT/NaN marker rows
        contribute a NaT (or run-id) group to the output, mostly NaN.
        When ``df`` has no ``__status`` column the flag has no effect.

    Returns
    -------
    pandas.DataFrame
        Row index: the unique values of ``df.index.get_level_values(by)``.
        Column index: a two-level :class:`pandas.MultiIndex`
        ``("statistic", "field")``. Statistic labels are exactly the
        entries of ``include`` plus ``f"q{q_val}"`` (e.g. ``"q0.05"``,
        ``"q0.5"``, ``"q0.95"``) for each requested quantile, in the
        order ``include`` then ``q``. ``field`` carries the original
        data-column names.

    Raises
    ------
    SweepConfigError
        ``by`` is not ``"time"`` or ``"run_id"``; any ``q_val`` falls
        outside ``(0, 1)``; ``include`` contains an unknown statistic;
        or ``by`` is not an index level of ``df``.

    Examples
    --------
    >>> import pandas as pd
    >>> from gmat_sweep import sweep_summary
    >>> # df is a (run_id, time)-MultiIndexed sweep DataFrame
    >>> summary = sweep_summary(df)  # doctest: +SKIP
    >>> summary[("q0.5", "Sat.X")]  # median Sat.X across runs at each time  # doctest: +SKIP
    """
    if by not in _VALID_BY:
        raise SweepConfigError(
            f"sweep_summary: by={by!r} is not supported in this release; "
            f"pass one of {list(_VALID_BY)}"
        )

    bad_q = [val for val in q if not (0.0 < float(val) < 1.0)]
    if bad_q:
        raise SweepConfigError(
            f"sweep_summary: q values must lie in the open interval (0, 1); got {bad_q}"
        )

    bad_include = [s for s in include if s not in _VALID_INCLUDE]
    if bad_include:
        raise SweepConfigError(
            f"sweep_summary: unknown statistic(s) in include={list(include)}: "
            f"{bad_include}; allowed: {list(_VALID_INCLUDE)}"
        )

    # Duplicate quantiles or include entries would collide in the
    # MultiIndex column key built via pd.concat({label: ...}) below and
    # produce a non-unique columns axis. Reject up front instead.
    dup_q = sorted({val for val in q if list(q).count(val) > 1})
    if dup_q:
        raise SweepConfigError(
            f"sweep_summary: q must not contain duplicates; got duplicate(s) {dup_q}"
        )
    dup_include = sorted({s for s in include if list(include).count(s) > 1})
    if dup_include:
        raise SweepConfigError(
            f"sweep_summary: include must not contain duplicates; "
            f"got duplicate(s) {dup_include}"
        )

    if df.index.nlevels < 2 or by not in (df.index.names or []):
        raise SweepConfigError(
            f"sweep_summary: df.index does not have a {by!r} level "
            f"(got names={list(df.index.names or [])})"
        )

    working = df
    if dropna and _STATUS_COL in working.columns:
        working = working.loc[working[_STATUS_COL] == "ok"]

    data = working.drop(columns=[_STATUS_COL], errors="ignore")
    data_columns = list(data.columns)

    grouped = data.groupby(level=by, dropna=dropna)

    stat_blocks: list[tuple[str, pd.DataFrame]] = []
    for stat in include:
        if stat == "count_ok":
            block = cast(pd.DataFrame, grouped.count())
        else:
            block = cast(pd.DataFrame, grouped.agg(stat))
        stat_blocks.append((stat, block.reindex(columns=data_columns)))

    for q_val in q:
        block = cast(pd.DataFrame, grouped.quantile(float(q_val)))
        stat_blocks.append((f"q{q_val}", block.reindex(columns=data_columns)))

    if not stat_blocks:
        # No statistics requested — return an empty-column frame still keyed
        # by the unique `by` values so callers can attach their own columns.
        empty_index = data.index.get_level_values(by).unique()
        return pd.DataFrame(
            index=empty_index,
            columns=pd.MultiIndex.from_arrays([[], []], names=["statistic", "field"]),
        )

    result = pd.concat(
        {label: block for label, block in stat_blocks},
        axis=1,
        names=["statistic", "field"],
    )
    return result


_RUNNING_COLUMNS: tuple[str, ...] = ("running_mean", "running_std", "se_mean")


@overload
def mc_convergence(
    df: pd.DataFrame,
    metric: str | Callable[[pd.DataFrame], float],
    *,
    terminal_only: bool = ...,
    engine: Literal["pandas"] = ...,
) -> pd.DataFrame: ...
@overload
def mc_convergence(
    df: pd.DataFrame,
    metric: str | Callable[[pd.DataFrame], float],
    *,
    terminal_only: bool = ...,
    engine: Literal["polars"],
) -> pl.DataFrame: ...
@overload
def mc_convergence(
    df: pd.DataFrame,
    metric: str | Callable[[pd.DataFrame], float],
    *,
    terminal_only: bool = ...,
    engine: str,
) -> DataFrame: ...
def mc_convergence(
    df: pd.DataFrame,
    metric: str | Callable[[pd.DataFrame], float],
    *,
    terminal_only: bool = False,
    engine: str = "pandas",
) -> DataFrame:
    """Diagnose Monte Carlo convergence: running mean / std / SE of the mean over run-id prefixes.

    Reduces ``df`` to a per-run scalar (or per-run-per-time scalar) under
    ``metric`` and reports cumulative statistics across the first
    ``n = 1..N`` runs in ``run_id`` order. Standard error of the mean is
    ``running_std / sqrt(n)`` with a sample standard deviation
    (``ddof=1``); ``n=1`` rows therefore carry ``NaN`` for both
    ``running_std`` and ``se_mean``. Failed and skipped runs are dropped
    via ``__status`` before the prefix scan, so the ``n`` axis reflects
    successful runs only.

    Parameters
    ----------
    df
        ``(run_id, time)``-MultiIndexed DataFrame as returned by
        :func:`gmat_sweep.sweep`, :func:`gmat_sweep.monte_carlo`, or
        :func:`gmat_sweep.latin_hypercube`. A ``__status`` column, if
        present, is used to drop non-ok runs.
    metric
        Either a column name in ``df`` or a callable
        ``(per_run_subframe) -> float``. The callable is invoked once per
        ``run_id`` with that run's ``time``-indexed slice (``__status``
        dropped) and must return a single float — useful for derived
        metrics like final-step miss distance.
    terminal_only
        ``True`` collapses the time index by taking ``.last()`` per run
        for column-name metrics — the canonical "did the dispersion of
        the final state converge?" view. ``False`` (default) keeps every
        time step and emits one running curve per ``time``. Ignored for
        callable metrics, which already return one scalar per run.
    engine
        ``"pandas"`` (default) returns a :class:`pandas.DataFrame` with a
        plain :class:`~pandas.RangeIndex`. ``"polars"`` returns the same
        flat-column frame as a :class:`polars.DataFrame`. Requires the
        ``[polars]`` extra; same semantics as :func:`lazy_multiindex`.

    Returns
    -------
    pandas.DataFrame or polars.DataFrame
        Long-form frame with columns ``n``, ``running_mean``,
        ``running_std``, ``se_mean``. When ``terminal_only=False`` and
        ``metric`` is a column name, an additional leading ``time``
        column carries the per-time grouping. The frame is sorted by
        ``time`` (when present) then ``n`` ascending.

    Raises
    ------
    SweepConfigError
        ``df.index`` is not a MultiIndex with a ``run_id`` level; the
        column-name ``metric`` is not in ``df``; the callable
        ``metric`` does not return a numeric scalar; or ``engine`` is
        neither ``"pandas"`` nor ``"polars"``.

    Examples
    --------
    >>> from gmat_sweep import mc_convergence
    >>> conv = mc_convergence(df, "MissDistance", terminal_only=True)  # doctest: +SKIP
    >>> conv.tail()  # doctest: +SKIP
            n  running_mean  running_std       se_mean
    995   996      ...           ...           ...
    """
    _check_engine(engine)
    if df.index.nlevels < 2 or _RUN_ID_COL not in (df.index.names or []):
        raise SweepConfigError(
            f"mc_convergence: df.index must have a {_RUN_ID_COL!r} level "
            f"(got names={list(df.index.names or [])})"
        )

    working = df
    if _STATUS_COL in working.columns:
        working = working.loc[working[_STATUS_COL] == "ok"]
    working = working.drop(columns=[_STATUS_COL], errors="ignore")

    if callable(metric):
        per_run = _reduce_callable_metric(working, metric)
        result = _running_stats(per_run.to_numpy(), n_offset=1)
    elif metric not in working.columns:
        raise SweepConfigError(
            f"mc_convergence: metric={metric!r} is not a column of df and is not callable"
        )
    elif terminal_only:
        per_run_series = working.groupby(level=_RUN_ID_COL)[metric].last().sort_index()
        result = _running_stats(per_run_series.to_numpy(), n_offset=1)
    else:
        result = _running_stats_per_time(working[metric])

    return _to_polars(result) if engine == "polars" else result


def _reduce_callable_metric(
    df: pd.DataFrame,
    metric: Callable[[pd.DataFrame], float],
) -> pd.Series[Any]:
    values: dict[int, float] = {}
    for run_id, sub in df.groupby(level=_RUN_ID_COL, sort=True):
        # Drop the run_id index level so the user-facing subframe is
        # a plain time-indexed DataFrame.
        sub_local = cast(pd.DataFrame, sub.droplevel(_RUN_ID_COL))
        result = metric(sub_local)
        try:
            values[int(cast(int, run_id))] = float(result)
        except (TypeError, ValueError) as exc:
            raise SweepConfigError(
                f"mc_convergence: callable metric must return a numeric scalar; "
                f"run_id={run_id} returned {type(result).__name__}"
            ) from exc
    return pd.Series(values, name="metric").sort_index()


def _running_stats(values: Any, *, n_offset: int) -> pd.DataFrame:
    """Running mean / sample std / SE over the first k entries for k = 1..len(values).

    Uses Welford's online recurrence so the variance update is numerically
    stable for samples drawn from a distribution whose mean is large
    relative to its std (e.g. km-magnitude position metrics with
    metre-magnitude dispersion, where the older cumulative
    sum-of-squares identity catastrophically cancels and reports zero
    std).
    """
    arr = np.asarray(values, dtype=float)
    n_total = arr.size
    if n_total == 0:
        empty: dict[str, pd.Series[Any]] = {"n": pd.Series([], dtype="int64")}
        for col in _RUNNING_COLUMNS:
            empty[col] = pd.Series([], dtype=float)
        return pd.DataFrame(empty)

    running_mean = np.empty(n_total, dtype=float)
    running_std = np.empty(n_total, dtype=float)
    mean = 0.0
    m2 = 0.0
    for i in range(n_total):
        x = float(arr[i])
        n = i + 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        m2 += delta * delta2
        running_mean[i] = mean
        running_std[i] = np.sqrt(m2 / (n - 1)) if n > 1 else np.nan

    counts = np.arange(1, n_total + 1, dtype=float)
    ks = np.arange(n_offset, n_offset + n_total, dtype=np.int64)
    se_mean = running_std / np.sqrt(counts)
    return pd.DataFrame(
        {
            "n": ks,
            "running_mean": running_mean,
            "running_std": running_std,
            "se_mean": se_mean,
        }
    )


@overload
def sweep_diff(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    on: str | None = ...,
    how: str = ...,
    tolerance: float | Callable[[str], float] | None = ...,
    engine: Literal["pandas"] = ...,
) -> pd.DataFrame: ...
@overload
def sweep_diff(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    on: str | None = ...,
    how: str = ...,
    tolerance: float | Callable[[str], float] | None = ...,
    engine: Literal["polars"],
) -> pl.DataFrame: ...
@overload
def sweep_diff(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    on: str | None = ...,
    how: str = ...,
    tolerance: float | Callable[[str], float] | None = ...,
    engine: str,
) -> DataFrame: ...
def sweep_diff(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    on: str | None = None,
    how: str = "both",
    tolerance: float | Callable[[str], float] | None = None,
    engine: str = "pandas",
) -> DataFrame:
    """Pairwise compare two same-shape sweep DataFrames into a diff frame.

    Aligns ``df_a`` and ``df_b`` on the intersection of their indexes,
    picks the numeric columns shared between them, and emits per-column
    ``<col>__diff = b - a`` and/or ``<col>__rel = (b - a) / a`` columns
    ready for plotting against the rest of the sweep helpers.

    Parameters
    ----------
    df_a, df_b
        Sweep DataFrames to compare. Both must share the same index level
        names (e.g. both ``(run_id, time)``); otherwise
        :class:`gmat_sweep.errors.SweepConfigError` is raised. Index keys
        present in only one side are silently dropped from the output.
    on
        ``None`` (default) compares row-by-row on the existing index.
        ``"run_id"`` collapses each side via
        ``groupby(level="run_id").last()`` first — the per-run final-step
        view, useful for "did the dispersion of the final state change?"
        comparisons. Other values are not supported.
    how
        ``"absolute"`` emits only ``<col>__diff = b - a``. ``"relative"``
        emits only ``<col>__rel = (b - a) / a`` (entries where ``a == 0``
        land as NaN). ``"both"`` (default) emits both, interleaved per
        source column.
    tolerance
        ``None`` (default) emits raw diffs. A ``float`` masks every diff
        whose absolute value is strictly below the cutoff to NaN — both
        the ``__diff`` and the matching ``__rel`` entry are masked at the
        same positions, so the output highlights only the meaningful
        changes. A callable is invoked once per source column as
        ``tolerance(col_name) -> float`` to produce a per-column cutoff,
        which is the right shape when the data columns carry mixed units
        (e.g. ``Sat.X`` in km vs. ``Sat.VX`` in km/s).
    engine
        ``"pandas"`` (default) returns a :class:`pandas.DataFrame` with
        the same index shape as the inputs. ``"polars"`` returns a
        :class:`polars.DataFrame` whose index levels are flattened into
        leading columns. Requires the ``[polars]`` extra; same semantics
        as :func:`lazy_multiindex`.

    Returns
    -------
    pandas.DataFrame or polars.DataFrame
        Same index as the (aligned) inputs (flattened to leading columns
        under ``engine="polars"``). One column per
        ``(<source-column>, suffix)`` pair, where suffix is ``__diff`` or
        ``__rel`` per ``how``. When at least one side carries a
        ``__status`` column, an extra trailing ``__status_diff`` column
        encodes the per-row status pair: ``"ok"`` when both sides are
        ``"ok"``, otherwise ``"<a_status>/<b_status>"`` (e.g.
        ``"failed/ok"``, ``"ok/skipped"``).

    Raises
    ------
    SweepConfigError
        ``how`` is not one of ``"absolute"``, ``"relative"``, ``"both"``;
        ``on`` is neither ``None`` nor ``"run_id"``; ``df_a`` and ``df_b``
        do not share the same index level names; ``on="run_id"`` was
        passed against a frame whose index has no ``run_id`` level; or
        ``engine`` is neither ``"pandas"`` nor ``"polars"``.

    Examples
    --------
    >>> from gmat_sweep import sweep, sweep_diff
    >>> baseline = sweep("mission.script", grid={"Sat.SMA": [7000.0]}, out=...)  # doctest: +SKIP
    >>> perturbed = sweep("mission.script", grid={"Sat.SMA": [7050.0]}, out=...)  # doctest: +SKIP
    >>> diff = sweep_diff(baseline, perturbed, on="run_id")  # doctest: +SKIP
    >>> diff[["Sat.SMA__diff", "Sat.SMA__rel"]]  # doctest: +SKIP
    """
    _check_engine(engine)
    if how not in _VALID_HOW:
        raise SweepConfigError(
            f"sweep_diff: how={how!r} is not supported; pass one of {list(_VALID_HOW)}"
        )

    if on is not None and on != _RUN_ID_COL:
        raise SweepConfigError(
            f"sweep_diff: on={on!r} is not supported in this release; "
            f"pass on=None or on={_RUN_ID_COL!r}"
        )

    a_names = list(df_a.index.names or [])
    b_names = list(df_b.index.names or [])
    if a_names != b_names:
        raise SweepConfigError(
            f"sweep_diff: df_a and df_b must share the same index level names; "
            f"got {a_names} vs {b_names}"
        )

    if on == _RUN_ID_COL:
        if _RUN_ID_COL not in a_names:
            raise SweepConfigError(
                f"sweep_diff: on={_RUN_ID_COL!r} requires a {_RUN_ID_COL!r} "
                f"index level; got {a_names}"
            )
        if df_a.index.nlevels > 1:
            # sort_index() before groupby().last() so the "final row per run"
            # is the row with the largest secondary-index value, not whichever
            # row happens to come last under the input's storage order. The
            # raw groupby().last() inherited that order silently.
            df_a = df_a.sort_index().groupby(level=_RUN_ID_COL).last()
            df_b = df_b.sort_index().groupby(level=_RUN_ID_COL).last()

    a_status = df_a[_STATUS_COL] if _STATUS_COL in df_a.columns else None
    b_status = df_b[_STATUS_COL] if _STATUS_COL in df_b.columns else None
    a_data = df_a.drop(columns=[_STATUS_COL], errors="ignore")
    b_data = df_b.drop(columns=[_STATUS_COL], errors="ignore")

    shared_cols = [c for c in a_data.columns if c in b_data.columns]
    numeric_cols = [
        c
        for c in shared_cols
        if pd.api.types.is_numeric_dtype(a_data[c]) and pd.api.types.is_numeric_dtype(b_data[c])
    ]

    common_index = df_a.index.intersection(df_b.index, sort=False)
    a_data = a_data.loc[common_index, numeric_cols]
    b_data = b_data.loc[common_index, numeric_cols]

    diff_block = b_data.subtract(a_data)
    if how in ("relative", "both"):
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_block = diff_block.divide(a_data)
        rel_block = rel_block.replace([np.inf, -np.inf], np.nan)
    else:
        rel_block = None

    if tolerance is not None:
        for col in numeric_cols:
            cutoff = float(tolerance(col)) if callable(tolerance) else float(tolerance)
            mask = diff_block[col].abs() < cutoff
            if how in ("absolute", "both"):
                diff_block.loc[mask, col] = np.nan
            if rel_block is not None:
                rel_block.loc[mask, col] = np.nan

    diff_named = diff_block.rename(columns={c: f"{c}__diff" for c in numeric_cols})
    rel_named = (
        rel_block.rename(columns={c: f"{c}__rel" for c in numeric_cols})
        if rel_block is not None
        else None
    )

    if how == "absolute":
        result = diff_named
    elif how == "relative":
        assert rel_named is not None
        result = rel_named
    else:
        assert rel_named is not None
        interleaved: list[str] = []
        for c in numeric_cols:
            interleaved.append(f"{c}__diff")
            interleaved.append(f"{c}__rel")
        result = pd.concat([diff_named, rel_named], axis=1)[interleaved]

    if a_status is not None or b_status is not None:
        a_aligned = (
            a_status.reindex(common_index).fillna("ok").astype(str)
            if a_status is not None
            else pd.Series("ok", index=common_index, dtype=object)
        )
        b_aligned = (
            b_status.reindex(common_index).fillna("ok").astype(str)
            if b_status is not None
            else pd.Series("ok", index=common_index, dtype=object)
        )
        both_ok = (a_aligned == "ok") & (b_aligned == "ok")
        status_diff = (a_aligned + "/" + b_aligned).astype(object)
        status_diff[both_ok] = "ok"
        result[_STATUS_DIFF_COL] = status_diff

    sorted_result = cast(pd.DataFrame, result.sort_index())
    return _to_polars(sorted_result) if engine == "polars" else sorted_result


def _running_stats_per_time(series: pd.Series[Any]) -> pd.DataFrame:
    # series is indexed by (run_id, time). Sort by run_id within each time
    # group so the prefix scan is well-defined, then apply _running_stats
    # to each group. pandas-stubs types reorder_levels' first arg as
    # ``list[int]``, so cast to Any for the level-name path.
    levels: Any = [_TIME_COL, _RUN_ID_COL]
    sorted_series = series.reorder_levels(levels).sort_index()

    parts: list[pd.DataFrame] = []
    for time_value, group in sorted_series.groupby(level=_TIME_COL, sort=True):
        block = _running_stats(group.to_numpy(), n_offset=1)
        block.insert(0, _TIME_COL, time_value)
        parts.append(block)

    if not parts:
        empty = pd.DataFrame({col: pd.Series([], dtype=float) for col in _RUNNING_COLUMNS})
        empty.insert(0, "n", pd.Series([], dtype="int64"))
        empty.insert(0, _TIME_COL, pd.Series([], dtype="datetime64[ns]"))
        return empty

    # Pin output ordering explicitly — the contract advertised in
    # mc_convergence's docstring ('sorted by time then n ascending')
    # should hold under refactors to the upstream groupby/concat path.
    return pd.concat(parts, axis=0, ignore_index=True).sort_values(
        [_TIME_COL, "n"], ignore_index=True
    )
