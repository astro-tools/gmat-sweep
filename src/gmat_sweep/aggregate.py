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
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from gmat_sweep.errors import SweepConfigError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gmat_sweep.manifest import Manifest, ManifestEntry

__all__ = ["lazy_contacts", "lazy_ephemerides", "lazy_multiindex"]


_RUN_ID_COL = "run_id"
_STATUS_COL = "__status"
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


def lazy_multiindex(
    manifest: Manifest,
    output_dir: Path,
    *,
    name: str | None = None,
    spool: bool = True,
) -> pd.DataFrame:
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

    Raises
    ------
    SweepConfigError
        ``name=None`` was passed but the sweep produced more than one
        report (the exception message lists the available names), or the
        explicitly-named report does not appear in any ok run's outputs.
    ValueError
        An ``ok`` entry has no ``output_paths`` at all (a run that ran
        successfully must have produced something), or a per-run Parquet
        is missing the ``time`` column required for the index.
    """
    return _aggregate(
        manifest,
        output_dir,
        kind="report",
        secondary_index=_TIME_COL,
        name=name,
        spool=spool,
    )


def lazy_ephemerides(
    manifest: Manifest,
    output_dir: Path,
    *,
    name: str | None = None,
    spool: bool = True,
) -> pd.DataFrame:
    """Assemble the ``(run_id, time)``-indexed ephemeris DataFrame from a sweep's outputs.

    Mirrors :func:`lazy_multiindex` but dispatches on ``ephemeris__<name>``
    keys instead of ``report__<name>``. The worker copies the first
    datetime column of each ephemeris frame (``Epoch`` for OEM, STK, and
    SPK formats) to a column named ``time`` before writing Parquet, so
    the same ``(run_id, time)`` index machinery applies.

    See :func:`lazy_multiindex` for parameter and exception semantics.
    """
    return _aggregate(
        manifest,
        output_dir,
        kind="ephemeris",
        secondary_index=_TIME_COL,
        name=name,
        spool=spool,
    )


def lazy_contacts(
    manifest: Manifest,
    output_dir: Path,
    *,
    name: str | None = None,
) -> pd.DataFrame:
    """Assemble the ``(run_id, interval_id)``-indexed contact DataFrame from a sweep's outputs.

    Mirrors :func:`lazy_multiindex` but dispatches on ``contact__<name>``
    keys and uses ``interval_id`` — the per-run row position the worker
    assigns at write time, ``0..K-1`` per run — as the secondary index
    level. ``ContactLocator`` outputs are typically tiny (one row per
    visibility interval), so there is no ``spool`` knob; reads are
    fragment-at-a-time eager.

    Failed, skipped, and report-only ``ok`` runs materialise as one row
    with ``interval_id = pd.NA`` (cast as the nullable ``Int64`` dtype so
    integer interval indices and missing values share one level).

    See :func:`lazy_multiindex` for the rest of the parameter and
    exception semantics.
    """
    return _aggregate(
        manifest,
        output_dir,
        kind="contact",
        secondary_index=_INTERVAL_ID_COL,
        name=name,
        spool=False,
    )


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

    frames: list[pd.DataFrame] = []
    for fragment in dataset.get_fragments():
        run_id = path_to_run_id[Path(fragment.path).as_posix()]
        if spool:
            for batch in fragment.to_batches():
                frames.append(_batch_to_pandas(batch, run_id))
        else:
            table = fragment.to_table()
            frames.append(_batch_to_pandas(table, run_id))

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if secondary_index not in merged.columns:
        raise ValueError(
            f"per-run Parquet output is missing the {secondary_index!r} column "
            f"required for the (run_id, {secondary_index}) MultiIndex"
        )
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


def _batch_to_pandas(batch_or_table: Any, run_id: int) -> pd.DataFrame:
    n_rows = batch_or_table.num_rows
    run_id_col = pa.array([run_id] * n_rows, type=pa.int64())
    augmented = batch_or_table.append_column(_RUN_ID_COL, run_id_col)
    return cast(pd.DataFrame, augmented.to_pandas())


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
