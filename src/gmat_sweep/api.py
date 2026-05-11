"""Public entry points: sweep, monte_carlo, latin_hypercube."""

from __future__ import annotations

import contextlib
import tempfile
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

from gmat_sweep.backends.base import Pool
from gmat_sweep.backends.joblib import LocalJoblibPool
from gmat_sweep.distributions import DistSpec, _serialise_perturb
from gmat_sweep.errors import SweepConfigError
from gmat_sweep.grids import (
    expand_latin_hypercube_to_run_specs,
    expand_monte_carlo_to_run_specs,
    expand_samples_to_run_specs,
    full_factorial_size,
    iter_grid_run_specs,
)
from gmat_sweep.spec import RunSpec
from gmat_sweep.sweep import Sweep

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    from gmat_sweep.aggregate import DataFrame

__all__ = [
    "latin_hypercube",
    "monte_carlo",
    "monte_carlo_extend",
    "sweep",
]

# Manifest filename inside the sweep's output directory. Picked to match the
# JSON Lines format suffix for grep-friendliness; downstream consumers
# (resume, CLI show) load it back with :meth:`Manifest.load`.
_MANIFEST_FILENAME = "manifest.jsonl"


@overload
def sweep(
    mission: str | Path,
    *,
    grid: Mapping[str, Iterable[Any]] | None = ...,
    samples: pd.DataFrame | None = ...,
    backend: Pool | None = ...,
    out: str | Path | None = ...,
    seed: int | None = ...,
    progress: bool = ...,
    fsync_each: bool = ...,
    fsync_batch: int = ...,
    engine: Literal["pandas"] = ...,
) -> pd.DataFrame: ...
@overload
def sweep(
    mission: str | Path,
    *,
    grid: Mapping[str, Iterable[Any]] | None = ...,
    samples: pd.DataFrame | None = ...,
    backend: Pool | None = ...,
    out: str | Path | None = ...,
    seed: int | None = ...,
    progress: bool = ...,
    fsync_each: bool = ...,
    fsync_batch: int = ...,
    engine: Literal["polars"],
) -> pl.DataFrame: ...
@overload
def sweep(
    mission: str | Path,
    *,
    grid: Mapping[str, Iterable[Any]] | None = ...,
    samples: pd.DataFrame | None = ...,
    backend: Pool | None = ...,
    out: str | Path | None = ...,
    seed: int | None = ...,
    progress: bool = ...,
    fsync_each: bool = ...,
    fsync_batch: int = ...,
    engine: str,
) -> DataFrame: ...
def sweep(
    mission: str | Path,
    *,
    grid: Mapping[str, Iterable[Any]] | None = None,
    samples: pd.DataFrame | None = None,
    backend: Pool | None = None,
    out: str | Path | None = None,
    seed: int | None = None,
    progress: bool = True,
    fsync_each: bool = True,
    fsync_batch: int = 50,
    engine: str = "pandas",
) -> DataFrame:
    """Run a parameter sweep over a GMAT mission.

    Two mutually exclusive run-set shapes are accepted: pass ``grid=`` for a
    full-factorial cartesian product, or ``samples=`` for an explicit
    pre-built DataFrame (one run per row, columns are dotted-path field
    names). Exactly one must be provided. The expanded run set is dispatched
    through ``backend`` (defaulting to a fresh :class:`LocalJoblibPool` over
    every available core), each completion is appended to a JSON Lines
    manifest under ``out``, and the aggregated ``(run_id, time)``-MultiIndexed
    :class:`pandas.DataFrame` is returned.

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
    backend:
        A constructed :class:`Pool` to dispatch runs through. ``None``
        (default) constructs a fresh :class:`LocalJoblibPool` over every
        available core for the duration of the call and closes it on the way
        out. Pass an explicit pool to cap parallelism
        (``LocalJoblibPool(max_workers=4)``), to use a different execution
        backend (Dask, Ray, a custom subclass), or to share one pool across
        several sweeps — when supplied, the caller owns the pool's lifecycle
        and ``sweep()`` does not call :meth:`Pool.close`.
    out:
        Sweep output directory. ``None`` (default) creates a fresh
        :class:`tempfile.TemporaryDirectory` whose lifetime is tied to the
        returned DataFrame — the temp dir survives until the caller drops the
        DataFrame, mirroring the :meth:`gmat_run.Mission.run` Results lifetime
        trick. Pass an explicit path to keep the per-run Parquet files and
        the manifest after the call returns. Relative paths are resolved to
        absolute before per-run directories are created, so manifest entries
        and the ``working_dir`` handed to GMAT do not depend on the caller's
        CWD or on GMAT's installed ``OUTPUT_PATH``.
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
    fsync_each:
        ``True`` (default) fsyncs the manifest after every appended
        entry — strict per-run durability, matching the v0.3 behaviour.
        Set to ``False`` to amortise the fsync cost across batches of
        ``fsync_batch`` entries; useful for sub-second runs at large
        counts where the per-entry fsync would otherwise dominate the
        driver thread. Forwarded verbatim to :class:`Sweep`; see
        :class:`Sweep` for the full durability tradeoff.
    fsync_batch:
        Fsync interval (in entries) when ``fsync_each`` is ``False``.
        Default ``50``. Ignored when ``fsync_each`` is ``True``.
    engine:
        ``"pandas"`` (default) returns a ``(run_id, time)``-MultiIndexed
        :class:`pandas.DataFrame`. ``"polars"`` returns a
        :class:`polars.DataFrame` with the MultiIndex flattened into two
        leading sorted columns; row count and the non-index column set
        match the pandas-engine equivalent. Requires the ``[polars]``
        extra; an :class:`ImportError` with the install hint is raised
        when polars is not importable. See
        :func:`gmat_sweep.aggregate.lazy_multiindex` for the full
        engine-knob contract.

    Returns
    -------
    pandas.DataFrame or polars.DataFrame
        ``(run_id, time)``-MultiIndexed frame produced by
        :func:`gmat_sweep.aggregate.lazy_multiindex` (or its polars-engine
        equivalent). Failed and skipped runs appear as one NaN-filled row
        with ``__status`` set accordingly — a single bad run does not
        abort the sweep or raise from this call.

    Raises
    ------
    SweepConfigError
        If both ``grid`` and ``samples`` are passed, if neither is passed,
        if either argument fails its own structural validation, or if
        ``engine`` is neither ``"pandas"`` nor ``"polars"``.
    """
    if grid is not None and samples is not None:
        raise SweepConfigError("sweep() accepts either grid= or samples=, not both")
    if grid is None and samples is None:
        raise SweepConfigError("sweep() requires one of grid= or samples=")

    mission_path = Path(mission)

    parameter_spec: dict[str, Any]
    build_runs: Callable[[Path], Iterable[RunSpec]]
    expected_run_count: int | None = None
    if grid is not None:
        # Materialise grid values once: iter_grid_run_specs would do it for
        # the cartesian product, but we also need the materialised dict for
        # the manifest header (generators don't survive json.dumps), so do
        # it up front and reuse the same object.
        materialised_grid: dict[str, list[Any]] = {k: list(v) for k, v in grid.items()}
        # Every parameter_spec shape carries a ``_kind`` discriminator so a
        # downstream reader doesn't have to infer the sweep kind from the
        # keys present. Older grid manifests omit the tag, and
        # ``Manifest.load`` keeps loading them as if ``_kind="grid"``.
        parameter_spec = {"_kind": "grid", **materialised_grid}
        expected_run_count = full_factorial_size(materialised_grid)

        def build_runs(output_dir: Path) -> Iterable[RunSpec]:
            # Streaming: the cartesian product is generated lazily and the
            # pool's bounded imap dispatcher caps in-flight specs to
            # ~4 * workers. A 10⁵-row factorial never materialises in full.
            return iter_grid_run_specs(materialised_grid, mission_path, output_dir)
    else:
        assert samples is not None  # narrowed by the XOR check above
        # ``rows`` is a list-of-lists in column order so the round-trip is a
        # one-line pd.DataFrame(...) call.
        parameter_spec = {
            "_kind": "explicit",
            "columns": [str(c) for c in samples.columns],
            "rows": samples.values.tolist(),
        }

        def build_runs(output_dir: Path) -> Iterable[RunSpec]:
            return expand_samples_to_run_specs(samples, mission_path, output_dir)

    return _run_sweep(
        mission_path=mission_path,
        build_runs=build_runs,
        parameter_spec=parameter_spec,
        sweep_seed=seed,
        backend=backend,
        out=out,
        progress=progress,
        engine=engine,
        fsync_each=fsync_each,
        fsync_batch=fsync_batch,
        expected_run_count=expected_run_count,
    )


@overload
def monte_carlo(
    mission: str | Path,
    *,
    n: int,
    perturb: Mapping[str, DistSpec],
    seed: int | None = ...,
    backend: Pool | None = ...,
    out: str | Path | None = ...,
    progress: bool = ...,
    fsync_each: bool = ...,
    fsync_batch: int = ...,
    engine: Literal["pandas"] = ...,
) -> pd.DataFrame: ...
@overload
def monte_carlo(
    mission: str | Path,
    *,
    n: int,
    perturb: Mapping[str, DistSpec],
    seed: int | None = ...,
    backend: Pool | None = ...,
    out: str | Path | None = ...,
    progress: bool = ...,
    fsync_each: bool = ...,
    fsync_batch: int = ...,
    engine: Literal["polars"],
) -> pl.DataFrame: ...
@overload
def monte_carlo(
    mission: str | Path,
    *,
    n: int,
    perturb: Mapping[str, DistSpec],
    seed: int | None = ...,
    backend: Pool | None = ...,
    out: str | Path | None = ...,
    progress: bool = ...,
    fsync_each: bool = ...,
    fsync_batch: int = ...,
    engine: str,
) -> DataFrame: ...
def monte_carlo(
    mission: str | Path,
    *,
    n: int,
    perturb: Mapping[str, DistSpec],
    seed: int | None = None,
    backend: Pool | None = None,
    out: str | Path | None = None,
    progress: bool = True,
    fsync_each: bool = True,
    fsync_batch: int = 50,
    engine: str = "pandas",
) -> DataFrame:
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
    backend:
        Execution backend; same semantics as :func:`sweep`.
    out:
        Sweep output directory; same semantics as :func:`sweep`.
    progress:
        Whether to draw the :mod:`tqdm` progress bar; same semantics as
        :func:`sweep`.
    engine:
        Output engine; same semantics as :func:`sweep`.

    Returns
    -------
    pandas.DataFrame or polars.DataFrame
        ``(run_id, time)``-MultiIndexed frame (or polars flat-key
        equivalent under ``engine="polars"``), one row per (run,
        time-step) pair, with ``run_id`` cardinality ``n``. A failed run
        lands as one NaN row with ``__status="failed"`` — same contract
        as :func:`sweep`.

    Raises
    ------
    SweepConfigError
        If ``perturb`` is empty, ``n < 1``, any parameter spec is
        ill-formed, or ``engine`` is neither ``"pandas"`` nor
        ``"polars"``.
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
        backend=backend,
        out=out,
        progress=progress,
        engine=engine,
        fsync_each=fsync_each,
        fsync_batch=fsync_batch,
    )


@overload
def latin_hypercube(
    mission: str | Path,
    *,
    n: int,
    perturb: Mapping[str, DistSpec],
    seed: int | None = ...,
    backend: Pool | None = ...,
    out: str | Path | None = ...,
    progress: bool = ...,
    fsync_each: bool = ...,
    fsync_batch: int = ...,
    engine: Literal["pandas"] = ...,
) -> pd.DataFrame: ...
@overload
def latin_hypercube(
    mission: str | Path,
    *,
    n: int,
    perturb: Mapping[str, DistSpec],
    seed: int | None = ...,
    backend: Pool | None = ...,
    out: str | Path | None = ...,
    progress: bool = ...,
    fsync_each: bool = ...,
    fsync_batch: int = ...,
    engine: Literal["polars"],
) -> pl.DataFrame: ...
@overload
def latin_hypercube(
    mission: str | Path,
    *,
    n: int,
    perturb: Mapping[str, DistSpec],
    seed: int | None = ...,
    backend: Pool | None = ...,
    out: str | Path | None = ...,
    progress: bool = ...,
    fsync_each: bool = ...,
    fsync_batch: int = ...,
    engine: str,
) -> DataFrame: ...
def latin_hypercube(
    mission: str | Path,
    *,
    n: int,
    perturb: Mapping[str, DistSpec],
    seed: int | None = None,
    backend: Pool | None = None,
    out: str | Path | None = None,
    progress: bool = True,
    fsync_each: bool = True,
    fsync_batch: int = 50,
    engine: str = "pandas",
) -> DataFrame:
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
    backend:
        Same semantics as :func:`sweep`.
    out:
        Same semantics as :func:`sweep`.
    progress:
        Same semantics as :func:`sweep`.
    engine:
        Same semantics as :func:`sweep`.

    Returns
    -------
    pandas.DataFrame or polars.DataFrame
        ``(run_id, time)``-MultiIndexed frame (or polars flat-key
        equivalent under ``engine="polars"``) with ``run_id`` cardinality
        ``n``. Same failure-as-row contract as :func:`sweep`.

    Raises
    ------
    SweepConfigError
        If ``perturb`` is empty, ``n < 1``, any parameter spec is
        ill-formed, or ``engine`` is neither ``"pandas"`` nor
        ``"polars"``.
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
        backend=backend,
        out=out,
        progress=progress,
        engine=engine,
        fsync_each=fsync_each,
        fsync_batch=fsync_batch,
    )


@overload
def monte_carlo_extend(
    manifest: str | Path,
    script: str | Path,
    *,
    n: int,
    backend: Pool | None = ...,
    allow_script_drift: bool = ...,
    progress: bool = ...,
    fsync_each: bool = ...,
    fsync_batch: int = ...,
    engine: Literal["pandas"] = ...,
) -> pd.DataFrame: ...
@overload
def monte_carlo_extend(
    manifest: str | Path,
    script: str | Path,
    *,
    n: int,
    backend: Pool | None = ...,
    allow_script_drift: bool = ...,
    progress: bool = ...,
    fsync_each: bool = ...,
    fsync_batch: int = ...,
    engine: Literal["polars"],
) -> pl.DataFrame: ...
@overload
def monte_carlo_extend(
    manifest: str | Path,
    script: str | Path,
    *,
    n: int,
    backend: Pool | None = ...,
    allow_script_drift: bool = ...,
    progress: bool = ...,
    fsync_each: bool = ...,
    fsync_batch: int = ...,
    engine: str,
) -> DataFrame: ...
def monte_carlo_extend(
    manifest: str | Path,
    script: str | Path,
    *,
    n: int,
    backend: Pool | None = None,
    allow_script_drift: bool = False,
    progress: bool = True,
    fsync_each: bool = True,
    fsync_batch: int = 50,
    engine: str = "pandas",
) -> DataFrame:
    """Append ``n`` more bit-deterministic Monte Carlo runs to an existing sweep.

    Loads the manifest written by a prior :func:`monte_carlo` call,
    dispatches ``n`` new runs at ``run_id`` range
    ``[old_n, old_n + n)`` (where ``old_n`` is the cumulative high-water
    mark including any prior extensions), and returns the aggregated
    DataFrame over **all** runs (original + every extension applied so
    far). Per-parameter draws at the new ``run_id``\\ s are bit-equal to
    the same indices of a fresh ``monte_carlo(n=old_n+n, seed=...)``
    call thanks to the position-determinism of
    :func:`numpy.random.SeedSequence.spawn`.

    The original ``perturb`` mapping and ``seed`` are read from the
    manifest's ``parameter_spec`` — the caller does not (and cannot)
    change them. Adding new perturbed parameters mid-sweep is not
    supported and would break determinism.

    Parameters
    ----------
    manifest:
        Path to the existing ``manifest.jsonl``. Its parent is the
        sweep's ``output_dir`` and must still exist on disk —
        successful runs' Parquet files are read from there as-is when
        the aggregated DataFrame is built.
    script:
        Path to the same GMAT ``.script`` the original sweep loaded.
        Its canonical SHA-256 must equal the manifest's
        ``script_sha256`` unless ``allow_script_drift`` is set —
        otherwise the original runs and the new ones would have loaded
        different scripts and the aggregated DataFrame would mix them.
    n:
        Number of additional runs to dispatch. Must be ``>= 1``.
    backend:
        Execution backend; same semantics as :func:`monte_carlo`.
    allow_script_drift:
        ``False`` (default) raises :class:`SweepConfigError` on a hash
        mismatch with both hashes in the message. ``True`` proceeds
        anyway and emits a :class:`RuntimeWarning`. Same surface as
        :meth:`gmat_sweep.Sweep.from_manifest`.
    progress:
        Whether to draw the :mod:`tqdm` progress bar over the new runs.
    engine:
        Output engine; same semantics as :func:`sweep`.

    Returns
    -------
    pandas.DataFrame or polars.DataFrame
        ``(run_id, time)``-MultiIndexed frame (or polars flat-key
        equivalent under ``engine="polars"``) whose ``run_id``
        cardinality is ``old_n + n``.

    Raises
    ------
    SweepConfigError
        If the manifest's ``parameter_spec._kind`` is not
        ``"monte_carlo"``, ``n < 1``, the script hash drifted with
        ``allow_script_drift=False``, the base sweep is incomplete
        (any ``run_id`` in ``[0, old_n)`` is failed or missing), or
        ``engine`` is neither ``"pandas"`` nor ``"polars"``.
    """
    # Local import to avoid a Sweep ↔ api import cycle at module load.
    from gmat_sweep.sweep import Sweep

    with _resolve_pool(backend) as pool:
        sweep_obj = Sweep.from_manifest(
            manifest,
            script,
            backend=pool,
            allow_script_drift=allow_script_drift,
            progress=progress,
            fsync_each=fsync_each,
            fsync_batch=fsync_batch,
        ).extend(n=n)
        return sweep_obj.to_dataframe(engine=engine)


def latin_hypercube_extend(
    manifest: str | Path,
    script: str | Path,
    *,
    n: int,
    backend: Pool | None = None,
    allow_script_drift: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """Refuse to extend a Latin hypercube sweep — no bit-equivalence semantics.

    Always raises :class:`SweepConfigError`. Extending an LH sweep
    changes the per-axis stratification of every existing sample (the
    ``n`` bins repartition under
    :class:`scipy.stats.qmc.LatinHypercube`), so there is no slice of
    a larger LH draw that reproduces the original ``n`` samples
    bit-for-bit. The function exists to **refuse cleanly** rather than
    silently produce wrong draws — its signature mirrors
    :func:`monte_carlo_extend` so a caller writing a generic
    "extend whatever sweep this is" wrapper gets a clear error rather
    than an :class:`AttributeError`.

    Not in :data:`__all__` and not in the user-facing docs: it is a typed
    refusal sentinel for advanced wrappers, not a feature. Import it
    directly from :mod:`gmat_sweep.api` if you need it.
    """
    raise SweepConfigError(
        "latin_hypercube_extend is unsupported: extending a Latin hypercube sweep "
        "changes the stratification of every existing sample, so the new draws "
        "would not be bit-equal to the originals. Use monte_carlo_extend on a "
        "Monte Carlo manifest, or run a fresh latin_hypercube(n=old_n + new) sweep."
    )


@contextlib.contextmanager
def _resolve_pool(backend: Pool | None) -> Iterator[Pool]:
    """Yield the pool to dispatch through, owning lifecycle iff we built it.

    ``backend is None`` constructs a default :class:`LocalJoblibPool` and
    closes it on exit; a user-supplied pool is yielded as-is and the caller
    keeps lifecycle ownership.
    """
    if backend is None:
        with LocalJoblibPool() as pool:
            yield pool
    else:
        yield backend


def _run_sweep(
    *,
    mission_path: Path,
    build_runs: Callable[[Path], Iterable[RunSpec]],
    parameter_spec: dict[str, Any],
    sweep_seed: int | None,
    backend: Pool | None,
    out: str | Path | None,
    progress: bool,
    engine: str,
    fsync_each: bool = True,
    fsync_batch: int = 50,
    expected_run_count: int | None = None,
) -> DataFrame:
    """Shared orchestration for the public entry points.

    Resolves the output directory (creating a sweep-scoped temp dir when
    ``out`` is ``None``), builds the run set against that directory, runs
    them through ``backend`` (or a default :class:`LocalJoblibPool` when
    ``backend is None``), and returns the aggregated DataFrame. When a temp
    dir was created its cleanup is deferred to the moment the returned
    DataFrame is garbage-collected.
    """
    tempdir: tempfile.TemporaryDirectory[str] | None
    if out is None:
        tempdir = tempfile.TemporaryDirectory(prefix="gmat-sweep-")
        output_dir = Path(tempdir.name)
    else:
        tempdir = None
        # `output_dir` becomes each `RunSpec.output_dir`, which is forwarded to
        # `gmat_run.Mission.run(working_dir=...)` and ultimately to the GMAT
        # API. GMAT resolves a relative `working_dir` against its installed
        # `OUTPUT_PATH` (e.g. `/opt/gmat/output/` in the canonical container
        # image), which is rarely what the caller wants. Resolve to absolute
        # so per-run paths land where the user pointed regardless of GMAT's
        # configured output root.
        output_dir = Path(out).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    runs = build_runs(output_dir)
    manifest_path = output_dir / _MANIFEST_FILENAME

    with _resolve_pool(backend) as pool:
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
                fsync_each=fsync_each,
                fsync_batch=fsync_batch,
                expected_run_count=expected_run_count,
            )
            .run()
            .to_dataframe(engine=engine)
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
