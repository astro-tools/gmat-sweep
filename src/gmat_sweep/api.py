"""Public entry points: sweep, monte_carlo, latin_hypercube.

v0.1 ships :func:`sweep` only — the full-factorial parameter-grid path. The
Monte Carlo and Latin hypercube wrappers land in v0.2 alongside
:mod:`gmat_sweep.distributions`.
"""

from __future__ import annotations

import tempfile
import weakref
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gmat_sweep.backends.joblib import LocalJoblibPool
from gmat_sweep.grids import expand_grid_to_run_specs
from gmat_sweep.sweep import Sweep

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["sweep"]

# Manifest filename inside the sweep's output directory. Picked to match the
# JSON Lines format suffix for grep-friendliness; downstream consumers
# (resume, CLI show) load it back with :meth:`Manifest.load`.
_MANIFEST_FILENAME = "manifest.jsonl"


def sweep(
    mission: str | Path,
    *,
    grid: Mapping[str, Iterable[Any]],
    workers: int = -1,
    out: Path | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Run a full-factorial parameter sweep over a GMAT mission.

    Builds the cartesian product of ``grid`` into one :class:`RunSpec` per
    combination, dispatches them through a fresh :class:`LocalJoblibPool`,
    appends each completion to a JSON Lines manifest under ``out``, and
    returns the aggregated ``(run_id, time)``-MultiIndexed
    :class:`pandas.DataFrame`.

    Parameters
    ----------
    mission:
        Path to the GMAT ``.script`` file every run loads.
    grid:
        Mapping from dotted-path field name (e.g. ``"Sat.SMA"``) to the
        sequence of values to sweep. Iterables are materialised once at call
        time so callers may pass generators.
    workers:
        Number of subprocess workers. ``-1`` (default) uses every available
        core; positive integers cap the pool. Forwarded to
        :class:`LocalJoblibPool`.
    out:
        Sweep output directory. ``None`` (default) creates a fresh
        :class:`tempfile.TemporaryDirectory` whose lifetime is tied to the
        returned DataFrame — the temp dir survives until the caller drops the
        DataFrame, mirroring the :meth:`gmat_run.Mission.run` Results lifetime
        trick. Pass an explicit path to keep the per-run Parquet files and
        the manifest after the call returns.
    seed:
        Optional integer recorded on the manifest header. v0.1 does not
        consume it; reserved for v0.2 Monte Carlo runs.

    Returns
    -------
    pandas.DataFrame
        ``(run_id, time)``-MultiIndexed frame produced by
        :func:`gmat_sweep.aggregate.lazy_multiindex`. Failed and skipped runs
        appear as one NaN-filled row with ``__status`` set accordingly — a
        single bad run does not abort the sweep or raise from this call.
    """
    mission_path = Path(mission)

    # Materialise grid values once: expand_grid_to_run_specs would do it for
    # the cartesian product, but we also need the materialised dict for the
    # manifest header (generators don't survive json.dumps), so do it up
    # front and reuse the same object.
    materialised_grid: dict[str, list[Any]] = {k: list(v) for k, v in grid.items()}

    tempdir: tempfile.TemporaryDirectory[str] | None
    if out is None:
        tempdir = tempfile.TemporaryDirectory(prefix="gmat-sweep-")
        output_dir = Path(tempdir.name)
    else:
        tempdir = None
        output_dir = Path(out)
        output_dir.mkdir(parents=True, exist_ok=True)

    runs = expand_grid_to_run_specs(materialised_grid, mission_path, output_dir)
    manifest_path = output_dir / _MANIFEST_FILENAME

    with LocalJoblibPool(workers=workers) as pool:
        df = (
            Sweep(
                runs=runs,
                backend=pool,
                manifest_path=manifest_path,
                output_dir=output_dir,
                script_path=mission_path,
                parameter_spec=materialised_grid,
                sweep_seed=seed,
            )
            .run()
            .to_dataframe()
        )

    if tempdir is not None:
        # Defer temp-dir cleanup until the user drops the DataFrame so any
        # downstream Parquet read against a path recorded in the manifest
        # still finds the file. lazy_multiindex materialises every per-run
        # frame into memory before returning, so the DataFrame itself is
        # self-contained — but the manifest path references on disk only
        # remain valid until the temp dir is removed. weakref.finalize keeps
        # the TemporaryDirectory alive (its bound cleanup method is the
        # callback) until the DataFrame is collected.
        weakref.finalize(df, tempdir.cleanup)

    return df
