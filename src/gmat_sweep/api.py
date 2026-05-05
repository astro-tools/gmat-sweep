"""Public entry points: sweep, monte_carlo, latin_hypercube."""

from __future__ import annotations

import tempfile
import weakref
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gmat_sweep.backends.joblib import LocalJoblibPool
from gmat_sweep.distributions import DistSpec, _serialise_perturb
from gmat_sweep.errors import SweepConfigError
from gmat_sweep.grids import (
    expand_grid_to_run_specs,
    expand_latin_hypercube_to_run_specs,
    expand_monte_carlo_to_run_specs,
    expand_samples_to_run_specs,
)
from gmat_sweep.spec import RunSpec
from gmat_sweep.sweep import Sweep

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["latin_hypercube", "monte_carlo", "sweep"]

# Manifest filename inside the sweep's output directory. Picked to match the
# JSON Lines format suffix for grep-friendliness; downstream consumers
# (resume, CLI show) load it back with :meth:`Manifest.load`.
_MANIFEST_FILENAME = "manifest.jsonl"


def sweep(
    mission: str | Path,
    *,
    grid: Mapping[str, Iterable[Any]] | None = None,
    samples: pd.DataFrame | None = None,
    workers: int = -1,
    out: Path | None = None,
    seed: int | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Run a parameter sweep over a GMAT mission.

    Two mutually exclusive run-set shapes are accepted: pass ``grid=`` for a
    full-factorial cartesian product, or ``samples=`` for an explicit
    pre-built DataFrame (one run per row, columns are dotted-path field
    names). Exactly one must be provided. The expanded run set is dispatched
    through a fresh :class:`LocalJoblibPool`, each completion is appended to
    a JSON Lines manifest under ``out``, and the aggregated
    ``(run_id, time)``-MultiIndexed :class:`pandas.DataFrame` is returned.

    Parameters
    ----------
    mission:
        Path to the GMAT ``.script`` file every run loads.
    grid:
        Mapping from dotted-path field name (e.g. ``"Sat.SMA"``) to the
        sequence of values to sweep. Iterables are materialised once at call
        time so callers may pass generators. Mutually exclusive with
        ``samples``.
    samples:
        :class:`pandas.DataFrame` whose columns are dotted-path field names
        and whose rows are the run set. The default :class:`pandas.RangeIndex`
        becomes ``run_id``; non-default indices raise. Mutually exclusive with
        ``grid``. Use this when you have already built a sample design (Latin
        hypercube, Halton/Sobol, custom) and want to hand it in directly.
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
        Optional integer recorded on the manifest header. ``sweep()`` itself
        does not consume it for grid or explicit-row sweeps; the value lives
        on the manifest for round-trip introspection. :func:`monte_carlo`
        and :func:`latin_hypercube` are the wrappers that consume a seed to
        derive their per-run draws.
    progress:
        ``True`` (default) draws a :mod:`tqdm` progress bar on stderr as
        runs complete. Set to ``False`` for non-interactive use (CI logs,
        notebooks committed with outputs) where the progress bar would
        otherwise be captured as noisy stderr snapshots.

    Returns
    -------
    pandas.DataFrame
        ``(run_id, time)``-MultiIndexed frame produced by
        :func:`gmat_sweep.aggregate.lazy_multiindex`. Failed and skipped runs
        appear as one NaN-filled row with ``__status`` set accordingly — a
        single bad run does not abort the sweep or raise from this call.

    Raises
    ------
    SweepConfigError
        If both ``grid`` and ``samples`` are passed, if neither is passed, or
        if either argument fails its own structural validation.
    """
    if grid is not None and samples is not None:
        raise SweepConfigError("sweep() accepts either grid= or samples=, not both")
    if grid is None and samples is None:
        raise SweepConfigError("sweep() requires one of grid= or samples=")

    mission_path = Path(mission)

    parameter_spec: dict[str, Any]
    build_runs: Callable[[Path], list[RunSpec]]
    if grid is not None:
        # Materialise grid values once: expand_grid_to_run_specs would do it
        # for the cartesian product, but we also need the materialised dict
        # for the manifest header (generators don't survive json.dumps), so
        # do it up front and reuse the same object.
        materialised_grid: dict[str, list[Any]] = {k: list(v) for k, v in grid.items()}
        # v1 schema tags every parameter_spec shape with a ``_kind``
        # discriminator so a downstream reader doesn't have to infer the
        # sweep kind from the keys present. v0.1 grid manifests omit the tag
        # and ``Manifest.load`` keeps loading them as if ``_kind="grid"``.
        parameter_spec = {"_kind": "grid", **materialised_grid}

        def build_runs(output_dir: Path) -> list[RunSpec]:
            return expand_grid_to_run_specs(materialised_grid, mission_path, output_dir)
    else:
        assert samples is not None  # narrowed by the XOR check above
        # ``rows`` is a list-of-lists in column order so the round-trip is a
        # one-line pd.DataFrame(...) call.
        parameter_spec = {
            "_kind": "explicit",
            "columns": [str(c) for c in samples.columns],
            "rows": samples.values.tolist(),
        }

        def build_runs(output_dir: Path) -> list[RunSpec]:
            return expand_samples_to_run_specs(samples, mission_path, output_dir)

    return _run_sweep(
        mission_path=mission_path,
        build_runs=build_runs,
        parameter_spec=parameter_spec,
        sweep_seed=seed,
        workers=workers,
        out=out,
        progress=progress,
    )


def monte_carlo(
    mission: str | Path,
    *,
    n: int,
    perturb: Mapping[str, DistSpec],
    seed: int | None = None,
    workers: int = -1,
    out: Path | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Run a Monte Carlo dispersion sweep over a GMAT mission.

    Builds an explicit-row run set of ``n`` runs by independently sampling
    each ``perturb`` parameter from its own distribution. Per-parameter
    sub-seeds are derived from the parameter *name* via
    :func:`derive_param_seed <gmat_sweep.distributions.derive_param_seed>`,
    so adding a perturbed parameter to an existing sweep does not change
    the draws of any other parameter at any ``run_id``.

    Parameters
    ----------
    mission:
        Path to the GMAT ``.script`` file every run loads.
    n:
        Number of stochastic runs. Must be ``>= 1``.
    perturb:
        Mapping from dotted-path field name to a distribution spec. Each
        value is one of the three shorthand tuples (``("normal", mu,
        sigma)``, ``("uniform", lo, hi)``, ``("lognormal", mu, sigma)``) or
        any pre-frozen :class:`scipy.stats._distn_infrastructure.rv_frozen`.
        See :data:`gmat_sweep.distributions.DistSpec` for the full surface.
    seed:
        Optional integer parent seed. ``None`` falls back to OS entropy and
        is **not** reproducible. With an integer seed two calls at the same
        ``(mission, n, perturb, seed)`` produce bit-equal DataFrames.
    workers:
        Number of subprocess workers; same semantics as :func:`sweep`.
    out:
        Sweep output directory; same semantics as :func:`sweep`.
    progress:
        Whether to draw the :mod:`tqdm` progress bar; same semantics as
        :func:`sweep`.

    Returns
    -------
    pandas.DataFrame
        ``(run_id, time)``-MultiIndexed frame, one row per (run, time-step)
        pair, with ``run_id`` cardinality ``n``. A failed run lands as one
        NaN row with ``__status="failed"`` — same contract as :func:`sweep`.

    Raises
    ------
    SweepConfigError
        If ``perturb`` is empty, ``n < 1``, or any parameter spec is
        ill-formed.
    """
    mission_path = Path(mission)
    parameter_spec: dict[str, Any] = {
        "_kind": "monte_carlo",
        "perturb": _serialise_perturb(perturb),
        "n": n,
        "seed": seed,
    }

    def build_runs(output_dir: Path) -> list[RunSpec]:
        return expand_monte_carlo_to_run_specs(
            perturb, n=n, seed=seed, script_path=mission_path, output_dir=output_dir
        )

    return _run_sweep(
        mission_path=mission_path,
        build_runs=build_runs,
        parameter_spec=parameter_spec,
        sweep_seed=seed,
        workers=workers,
        out=out,
        progress=progress,
    )


def latin_hypercube(
    mission: str | Path,
    *,
    n: int,
    perturb: Mapping[str, DistSpec],
    seed: int | None = None,
    workers: int = -1,
    out: Path | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Run a Latin hypercube sweep over a GMAT mission.

    Backed by :class:`scipy.stats.qmc.LatinHypercube`: draws ``n`` unit-cube
    points stratified across each of ``len(perturb)`` axes and maps each
    column through the user's distribution via ``rv.ppf(...)``. Latin
    hypercube sampling typically beats plain Monte Carlo when ``n`` is
    small relative to the problem's dimensionality, because the coverage
    of each axis is enforced by construction.

    Parameters
    ----------
    mission:
        Path to the GMAT ``.script`` file every run loads.
    n:
        Number of Latin hypercube points. Must be ``>= 1``.
    perturb:
        Mapping from dotted-path field name to a distribution spec — same
        accepted shapes as :func:`monte_carlo`.
    seed:
        Optional integer seed forwarded to
        :class:`scipy.stats.qmc.LatinHypercube`. ``None`` falls back to OS
        entropy and is **not** reproducible. With an integer seed two calls
        at the same ``(mission, n, perturb, seed)`` produce bit-equal
        DataFrames.
    workers:
        Same semantics as :func:`sweep`.
    out:
        Same semantics as :func:`sweep`.
    progress:
        Same semantics as :func:`sweep`.

    Returns
    -------
    pandas.DataFrame
        ``(run_id, time)``-MultiIndexed frame with ``run_id`` cardinality
        ``n``. Same failure-as-row contract as :func:`sweep`.

    Raises
    ------
    SweepConfigError
        If ``perturb`` is empty, ``n < 1``, or any parameter spec is
        ill-formed.
    """
    mission_path = Path(mission)
    parameter_spec: dict[str, Any] = {
        "_kind": "latin_hypercube",
        "perturb": _serialise_perturb(perturb),
        "n": n,
        "seed": seed,
    }

    def build_runs(output_dir: Path) -> list[RunSpec]:
        return expand_latin_hypercube_to_run_specs(
            perturb, n=n, seed=seed, script_path=mission_path, output_dir=output_dir
        )

    return _run_sweep(
        mission_path=mission_path,
        build_runs=build_runs,
        parameter_spec=parameter_spec,
        sweep_seed=seed,
        workers=workers,
        out=out,
        progress=progress,
    )


def _run_sweep(
    *,
    mission_path: Path,
    build_runs: Callable[[Path], list[RunSpec]],
    parameter_spec: dict[str, Any],
    sweep_seed: int | None,
    workers: int,
    out: Path | None,
    progress: bool,
) -> pd.DataFrame:
    """Shared orchestration for the public entry points.

    Resolves the output directory (creating a sweep-scoped temp dir when
    ``out`` is ``None``), builds the run set against that directory, runs
    them through a fresh :class:`LocalJoblibPool`, and returns the
    aggregated DataFrame. When a temp dir was created its cleanup is
    deferred to the moment the returned DataFrame is garbage-collected.
    """
    tempdir: tempfile.TemporaryDirectory[str] | None
    if out is None:
        tempdir = tempfile.TemporaryDirectory(prefix="gmat-sweep-")
        output_dir = Path(tempdir.name)
    else:
        tempdir = None
        output_dir = Path(out)
        output_dir.mkdir(parents=True, exist_ok=True)

    runs = build_runs(output_dir)
    manifest_path = output_dir / _MANIFEST_FILENAME

    with LocalJoblibPool(workers=workers) as pool:
        df = (
            Sweep(
                runs=runs,
                backend=pool,
                manifest_path=manifest_path,
                output_dir=output_dir,
                script_path=mission_path,
                parameter_spec=parameter_spec,
                sweep_seed=sweep_seed,
                progress=progress,
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
