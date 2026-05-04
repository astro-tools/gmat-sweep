"""Lazy assembly of the multi-indexed parent DataFrame from per-run Parquet files.

The single public entry point is :func:`lazy_multiindex`. It walks a
:class:`gmat_sweep.manifest.Manifest`, reads each successful run's Parquet
output through :mod:`pyarrow.dataset`, and stitches the per-run frames
into one ``(run_id, time)``-indexed :class:`pandas.DataFrame`. Failed and
skipped runs are materialised as one-row, NaN-filled slices so the
caller sees a complete row per run rather than having to reconcile a
``DataFrame`` against the manifest by hand.

The default path streams each run's record batches through pandas one
fragment at a time so peak conversion memory is one batch, not one full
sweep. ``spool=False`` flips to an eager fragment-at-a-time read for
small sweeps where the streaming overhead is not worth it; the result
DataFrame is identical.

v0.1 assumes one Parquet output per ``ok`` run. Multi-report runs raise
:class:`NotImplementedError` and defer to the v0.2 reshape helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from gmat_sweep.manifest import Manifest, ManifestEntry

__all__ = ["lazy_multiindex"]


_INDEX_NAMES: tuple[str, str] = ("run_id", "time")
_STATUS_COL = "__status"
_TIME_COL = "time"


def lazy_multiindex(
    manifest: Manifest,
    output_dir: Path,
    *,
    spool: bool = True,
) -> pd.DataFrame:
    """Assemble the public ``(run_id, time)``-indexed DataFrame from a sweep's outputs.

    Iterates ``manifest.entries`` in order. For each ``ok`` entry the
    sole Parquet listed in :attr:`ManifestEntry.output_paths` is read via
    :mod:`pyarrow.dataset` and tagged with its ``run_id``. For each
    ``failed`` or ``skipped`` entry one NaN-filled row is materialised
    with ``time = NaT`` and ``__status`` set to the run-level status.
    Relative paths in ``output_paths`` are resolved against ``output_dir``;
    absolute paths are used as-is.

    The returned frame has a ``(run_id, time)`` :class:`pandas.MultiIndex`,
    a ``time`` level coerced to ``datetime64[ns]``, and a ``__status``
    column populated for every row.

    Parameters
    ----------
    manifest
        The sweep manifest. Drives both the set of runs and their status.
    output_dir
        Sweep output root. Used to anchor any relative paths recorded in
        the manifest.
    spool
        ``True`` (default) streams each run's record batches into pandas
        one batch at a time. ``False`` reads each run's Parquet eagerly
        in one shot — simpler control flow, higher peak memory.

    Raises
    ------
    NotImplementedError
        If an ``ok`` run has more than one Parquet output. v0.1 assumes a
        single ReportFile per run; multi-report aggregation is deferred
        to the v0.2 reshape helpers.
    ValueError
        If an ``ok`` entry has no ``output_paths``, or if a per-run
        Parquet is missing the ``time`` column required for the index.
    """
    paths_with_run_id: list[tuple[Path, int]] = []
    nonok_entries: list[ManifestEntry] = []
    for entry in manifest.entries:
        if entry.status == "ok":
            if len(entry.output_paths) == 0:
                raise ValueError(
                    f"manifest entry for run_id={entry.run_id} has status='ok' but no output_paths"
                )
            if len(entry.output_paths) > 1:
                raise NotImplementedError(
                    f"run_id={entry.run_id} produced {len(entry.output_paths)} Parquet "
                    "outputs; v0.1 lazy_multiindex assumes one ReportFile per run. "
                    "Multi-report aggregation is deferred to the v0.2 reshape helpers."
                )
            ((_name, raw_path),) = entry.output_paths.items()
            resolved = raw_path if raw_path.is_absolute() else output_dir / raw_path
            paths_with_run_id.append((resolved, entry.run_id))
        else:
            nonok_entries.append(entry)

    ok_df = _read_ok_runs(paths_with_run_id, spool=spool)
    data_columns = [c for c in ok_df.columns if c != _STATUS_COL] if not ok_df.empty else []
    nonok_df = _materialise_nonok(nonok_entries, data_columns=data_columns)

    parts = [df for df in (ok_df, nonok_df) if not df.empty]
    if not parts:
        return _empty_frame()

    return cast(pd.DataFrame, pd.concat(parts, axis=0).sort_index())


def _read_ok_runs(
    paths_with_run_id: Sequence[tuple[Path, int]],
    *,
    spool: bool,
) -> pd.DataFrame:
    if not paths_with_run_id:
        return pd.DataFrame()

    paths_str = [str(p) for p, _ in paths_with_run_id]
    path_to_run_id = {str(p): run_id for p, run_id in paths_with_run_id}

    dataset = ds.dataset(paths_str, format="parquet")  # type: ignore[no-untyped-call]

    frames: list[pd.DataFrame] = []
    for fragment in dataset.get_fragments():
        run_id = path_to_run_id[fragment.path]
        if spool:
            for batch in fragment.to_batches():
                frames.append(_batch_to_pandas(batch, run_id))
        else:
            table = fragment.to_table()
            frames.append(_batch_to_pandas(table, run_id))

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    if _TIME_COL not in merged.columns:
        raise ValueError(
            "per-run Parquet output is missing the 'time' column required for the "
            "(run_id, time) MultiIndex"
        )
    merged[_TIME_COL] = pd.to_datetime(merged[_TIME_COL]).astype("datetime64[ns]")
    merged[_STATUS_COL] = "ok"
    return cast(pd.DataFrame, merged.set_index(list(_INDEX_NAMES)))


def _batch_to_pandas(batch_or_table: Any, run_id: int) -> pd.DataFrame:
    n_rows = batch_or_table.num_rows
    run_id_col = pa.array([run_id] * n_rows, type=pa.int64())
    augmented = batch_or_table.append_column("run_id", run_id_col)
    return cast(pd.DataFrame, augmented.to_pandas())


def _materialise_nonok(
    entries: Sequence[ManifestEntry],
    *,
    data_columns: Sequence[str],
) -> pd.DataFrame:
    if not entries:
        return pd.DataFrame()

    rows: dict[str, list[Any]] = {col: [np.nan] * len(entries) for col in data_columns}
    rows["run_id"] = [e.run_id for e in entries]
    rows[_TIME_COL] = [pd.NaT] * len(entries)
    rows[_STATUS_COL] = [e.status for e in entries]

    df = pd.DataFrame(rows)
    df[_TIME_COL] = pd.to_datetime(df[_TIME_COL]).astype("datetime64[ns]")
    return cast(pd.DataFrame, df.set_index(list(_INDEX_NAMES)))


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {_STATUS_COL: pd.Series([], dtype="object")},
        index=pd.MultiIndex.from_arrays(
            [
                pd.Series([], dtype="int64"),
                pd.Series([], dtype="datetime64[ns]"),
            ],
            names=list(_INDEX_NAMES),
        ),
    )
