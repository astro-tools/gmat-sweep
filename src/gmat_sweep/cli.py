"""Console-script entry point for the ``gmat-sweep`` command.

The CLI is a thin wrapper over :mod:`gmat_sweep.api` and
:meth:`gmat_sweep.manifest.Manifest.load`. Argument parsing, grid-spec parsing,
and perturb-spec parsing live here; everything else delegates to the existing
public surface.

Exit codes
----------
``0``   success
``1``   any other :class:`gmat_sweep.errors.GmatSweepError`
``2``   :class:`gmat_sweep.errors.SweepConfigError` or argparse usage error
``3``   :class:`gmat_sweep.errors.ManifestCorruptError`
``4``   :class:`gmat_sweep.errors.BackendError`
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from gmat_sweep.errors import (
    BackendError,
    GmatSweepError,
    ManifestCorruptError,
    SweepConfigError,
)
from gmat_sweep.manifest import Manifest, ManifestEntry

# Heavy submodules (api, backends.{dask,ray,mpi}, joblib) are imported on
# first use — `gmat-sweep --help` should not pull pandas / pyarrow / tqdm /
# joblib just to print usage text. ``_cmd_*`` functions import from
# ``gmat_sweep.api`` directly inside their bodies; backend pool classes are
# resolved through :func:`_import_backend_class`, which keeps them
# monkey-patchable in tests.

# Backend pool class slots. ``None`` means "load lazily on first access";
# tests that do ``monkeypatch.setattr(cli, "MPIPool", fake)`` set their
# preferred class here and :func:`_import_backend_class` picks it up.
DaskPool: Any = None
LocalJoblibPool: Any = None
MPIPool: Any = None
RayPool: Any = None

# Public-API function slots. Same pattern as the backend pool slots — kept at
# module level so ``monkeypatch.setattr("gmat_sweep.cli.sweep", fake)`` still
# works without forcing ``gmat_sweep.api`` (and its tqdm / pandas / pyarrow
# transitive imports) on every ``gmat-sweep --help`` invocation.
sweep: Any = None
monte_carlo: Any = None
latin_hypercube: Any = None
monte_carlo_extend: Any = None

_BACKEND_MODULES: dict[str, str] = {
    "DaskPool": "gmat_sweep.backends.dask",
    "LocalJoblibPool": "gmat_sweep.backends.joblib",
    "MPIPool": "gmat_sweep.backends.mpi",
    "RayPool": "gmat_sweep.backends.ray",
}

_API_FUNCS = ("sweep", "monte_carlo", "latin_hypercube", "monte_carlo_extend")


def _import_backend_class(class_name: str) -> Any:
    """Return the named backend pool class, importing on first call."""
    existing = globals().get(class_name)
    if existing is not None:
        return existing
    from importlib import import_module

    cls = getattr(import_module(_BACKEND_MODULES[class_name]), class_name)
    globals()[class_name] = cls
    return cls


def _import_api_func(name: str) -> Any:
    """Return the named :mod:`gmat_sweep.api` function, importing on first call."""
    existing = globals().get(name)
    if existing is not None:
        return existing
    from importlib import import_module

    fn = getattr(import_module("gmat_sweep.api"), name)
    globals()[name] = fn
    return fn


if TYPE_CHECKING:
    import pandas as pd

    from gmat_sweep.backends.base import Pool
    from gmat_sweep.distributions import DistSpec

__all__ = ["main"]

EXIT_OK = 0
EXIT_GENERIC = 1
EXIT_CONFIG = 2
EXIT_MANIFEST = 3
EXIT_BACKEND = 4

_PERTURB_TAGS = ("normal", "uniform", "lognormal")

_DETAIL_HEADERS = ("run_id", "status", "duration_s", "stderr_summary", "log_path")
_STDERR_SUMMARY_WIDTH = 60
_EMPTY_CELL = "—"
_STATUS_BUCKETS: dict[str, int] = {"failed": 0, "skipped": 1, "ok": 2}


def _parse_grid_value(token: str) -> int | float | str:
    """Coerce a single explicit-list token via int → float → str fallback.

    Non-finite floats (``nan``, ``inf``, ``-inf``) are rejected even though
    Python's :func:`float` parses them: a NaN axis carries no signal, and an
    infinite override would silently produce an unrunnable spec. They surface
    as :class:`SweepConfigError` here instead of corrupting the run set.
    """
    try:
        return int(token)
    except ValueError:
        pass
    try:
        as_float = float(token)
    except ValueError:
        return token
    if not math.isfinite(as_float):
        raise SweepConfigError(f"non-finite numeric value is not allowed: {token!r}")
    return as_float


def _parse_grid_spec(spec: str) -> tuple[str, list[Any]]:
    """Parse one ``--grid`` argument into ``(dotted_path, values)``.

    Two forms:

    * ``name=lo:hi:count`` — ``count`` evenly spaced points from ``lo`` to
      ``hi`` inclusive (``numpy.linspace``). ``lo`` and ``hi`` are floats;
      ``count`` is an integer ≥ 2.
    * ``name=v1,v2,v3`` — explicit values, each coerced via int → float → str.

    Empty, malformed, or numerically invalid specs raise
    :class:`SweepConfigError`.
    """
    if "=" not in spec:
        raise SweepConfigError(
            f"grid spec must contain '=': {spec!r} "
            "(expected 'name=lo:hi:count' or 'name=v1,v2,...')"
        )
    name, _, rhs = spec.partition("=")
    name = name.strip()
    if not name:
        raise SweepConfigError(f"grid spec is missing a name: {spec!r}")
    if not rhs:
        raise SweepConfigError(f"grid spec for {name!r} has no values: {spec!r}")

    if ":" in rhs:
        parts = rhs.split(":")
        if len(parts) != 3:
            raise SweepConfigError(
                f"linspace grid spec for {name!r} must have three colon-separated parts "
                f"'lo:hi:count', got {rhs!r}"
            )
        lo_str, hi_str, count_str = parts
        try:
            lo = float(lo_str)
            hi = float(hi_str)
        except ValueError as exc:
            raise SweepConfigError(
                f"linspace bounds for {name!r} must be numeric, got {lo_str!r}:{hi_str!r}"
            ) from exc
        if not math.isfinite(lo) or not math.isfinite(hi):
            raise SweepConfigError(
                f"linspace bounds for {name!r} must be finite, got {lo_str!r}:{hi_str!r}"
            )
        try:
            count = int(count_str)
        except ValueError as exc:
            raise SweepConfigError(
                f"linspace count for {name!r} must be an integer, got {count_str!r}"
            ) from exc
        if count < 2:
            raise SweepConfigError(f"linspace count for {name!r} must be >= 2, got {count}")
        import numpy as np

        return name, np.linspace(lo, hi, count).tolist()

    tokens = rhs.split(",")
    if any(t == "" for t in tokens):
        raise SweepConfigError(f"explicit grid spec for {name!r} has an empty value: {rhs!r}")
    return name, [_parse_grid_value(t) for t in tokens]


def _parse_perturb_spec(spec: str) -> tuple[str, DistSpec]:
    """Parse one ``--perturb`` argument into ``(dotted_path, DistSpec)``.

    Three shorthand forms — ``name=normal:mu:sigma``,
    ``name=uniform:lo:hi``, and ``name=lognormal:mu:sigma`` — mirroring the
    tuple specs accepted by :func:`gmat_sweep.monte_carlo` and
    :func:`gmat_sweep.latin_hypercube`. The unknown-tag check happens here so
    a typo surfaces at parse time; numeric validation (``sigma > 0``,
    ``hi > lo``, finiteness) is left to
    :func:`gmat_sweep.distributions.to_rv_frozen`, which raises the same
    :class:`SweepConfigError` downstream.
    """
    if "=" not in spec:
        raise SweepConfigError(
            f"perturb spec must contain '=': {spec!r} "
            "(expected 'name=normal:mu:sigma', 'name=uniform:lo:hi', "
            "or 'name=lognormal:mu:sigma')"
        )
    name, _, rhs = spec.partition("=")
    name = name.strip()
    if not name:
        raise SweepConfigError(f"perturb spec is missing a name: {spec!r}")
    if not rhs:
        raise SweepConfigError(f"perturb spec for {name!r} has no distribution: {spec!r}")

    parts = rhs.split(":")
    if len(parts) != 3:
        raise SweepConfigError(
            f"perturb spec for {name!r} must have three colon-separated parts "
            f"'tag:p1:p2', got {rhs!r}"
        )
    tag, p1_str, p2_str = parts
    if tag not in _PERTURB_TAGS:
        raise SweepConfigError(
            f"unknown perturb distribution tag {tag!r} for {name!r}; "
            f"expected one of {', '.join(_PERTURB_TAGS)}"
        )
    try:
        p1 = float(p1_str)
        p2 = float(p2_str)
    except ValueError as exc:
        raise SweepConfigError(
            f"perturb parameters for {name!r} must be numeric, got {p1_str!r}:{p2_str!r}"
        ) from exc
    return name, (tag, p1, p2)


def _parse_backend_arg(spec: str) -> tuple[str, int | float | str]:
    """Parse one ``--backend-arg`` token into ``(key, coerced_value)``.

    Same int → float → str coercion as :func:`_parse_grid_value`.
    """
    if "=" not in spec:
        raise SweepConfigError(f"--backend-arg must be 'KEY=VALUE': {spec!r}")
    name, _, rhs = spec.partition("=")
    name = name.strip()
    if not name:
        raise SweepConfigError(f"--backend-arg is missing a key: {spec!r}")
    if not rhs:
        raise SweepConfigError(f"--backend-arg for {name!r} has no value: {spec!r}")
    return name, _parse_grid_value(rhs)


def _build_pool(args: argparse.Namespace) -> Pool:
    """Construct the backend pool selected by ``--backend`` / ``--backend-arg``.

    The four sweep-running subcommands and ``resume`` route through here so
    the backend wiring lives in one place. ``--backend local`` (default)
    returns a :class:`LocalJoblibPool` and rejects any ``--backend-arg``;
    ``dask`` / ``ray`` map ``--workers`` onto ``n_workers`` / ``num_cpus``
    (with the default ``-1`` meaning "let the pool pick") and forward every
    ``--backend-arg`` as a kwarg to the pool constructor. ``mpi`` ignores
    ``--workers`` (rank count is set by the launcher or by
    ``--backend-arg max_workers=N``) and forwards every ``--backend-arg``
    to :class:`MPIPool`. Unknown kwargs surface as the pool's own
    :class:`BackendError`.
    """
    backend_kwargs: dict[str, Any] = {}
    for raw in args.backend_arg:
        key, value = _parse_backend_arg(raw)
        if key in backend_kwargs:
            raise SweepConfigError(f"--backend-arg for {key!r} given more than once")
        backend_kwargs[key] = value

    if args.backend == "local":
        if backend_kwargs:
            raise SweepConfigError(
                "--backend-arg is not supported with --backend local; "
                "the local pool only accepts --workers"
            )
        return cast("Pool", _import_backend_class("LocalJoblibPool")(max_workers=args.workers))
    workers = args.workers if args.workers > 0 else None
    if args.backend == "dask":
        return cast(
            "Pool",
            _import_backend_class("DaskPool")(n_workers=workers, **backend_kwargs),
        )
    if args.backend == "ray":
        return cast(
            "Pool",
            _import_backend_class("RayPool")(num_cpus=workers, **backend_kwargs),
        )
    if args.backend == "mpi":
        # MPIPool's rank count is fixed by the launcher (``mpirun -n …`` or the
        # ``--backend-arg max_workers=N`` escape hatch); --workers has no
        # meaning here. Silently ignoring the flag used to confuse users into
        # thinking their value had been honoured — fail loudly instead.
        if args.workers != -1:
            raise SweepConfigError(
                "--workers is not supported with --backend mpi; the rank count is "
                "fixed by the MPI launcher. Pass '--backend-arg max_workers=N' to "
                "cap MPIPool's in-flight set instead."
            )
        return cast("Pool", _import_backend_class("MPIPool")(**backend_kwargs))
    raise AssertionError(f"unreachable backend: {args.backend!r}")  # pragma: no cover


def _load_samples(path: Path) -> pd.DataFrame:
    """Load an explicit-row samples DataFrame from CSV or Parquet.

    Suffix-dispatched: ``.csv`` reads via :func:`pandas.read_csv`, ``.parquet``
    via :func:`pandas.read_parquet`. Any other suffix raises
    :class:`SweepConfigError`. A missing file also raises
    :class:`SweepConfigError` so the CLI's "bad config" exit code applies
    uniformly.
    """
    if not path.is_file():
        raise SweepConfigError(f"samples file not found: {path}")
    suffix = path.suffix.lower()
    import pandas as pd

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise SweepConfigError(
        f"samples file suffix {suffix!r} is not supported; expected '.csv' or '.parquet'"
    )


def _format_summary(manifest: Manifest, output_dir: Path) -> str:
    """One-line summary: ``N runs (A ok[, B failed][, C skipped]) in T.TT s — output: PATH``.

    Zero-count buckets are suppressed.
    """
    counts: dict[str, int] = {"ok": 0, "failed": 0, "skipped": 0}
    duration = 0.0
    for entry in manifest.entries:
        counts[entry.status] += 1
        duration += entry.duration_s

    parts = [f"{counts[k]} {k}" for k in ("ok", "failed", "skipped") if counts[k] > 0]
    breakdown = ", ".join(parts) if parts else "0 ok"
    n = len(manifest.entries)
    return f"{n} runs ({breakdown}) in {duration:.2f} s — output: {output_dir}"


def _cmd_run(args: argparse.Namespace) -> int:
    script = Path(args.script)
    if not script.is_file():
        print(f"gmat-sweep: script not found: {script}", file=sys.stderr)
        return EXIT_CONFIG

    grid: dict[str, list[Any]] = {}
    for raw in args.grid:
        name, values = _parse_grid_spec(raw)
        if name in grid:
            raise SweepConfigError(f"grid spec for {name!r} given more than once")
        grid[name] = values

    sweep = _import_api_func("sweep")

    out = Path(args.out)
    with _build_pool(args) as pool:
        sweep(
            script,
            grid=grid,
            backend=pool,
            out=out,
            fsync_each=args.fsync_each,
            fsync_batch=args.fsync_batch,
        )

    manifest = Manifest.load(out / "manifest.jsonl")
    print(_format_summary(manifest, out))
    return EXIT_OK


def _truncate_stderr_summary(stderr: str | None) -> str:
    """First line of ``stderr`` truncated to ``_STDERR_SUMMARY_WIDTH`` chars.

    Returns ``""`` for ``None`` or empty input. Strings longer than the window
    are cut so the visible result (text + ``"..."``) totals at most
    ``_STDERR_SUMMARY_WIDTH`` characters.
    """
    if not stderr:
        return ""
    first = stderr.splitlines()[0] if stderr.splitlines() else ""
    if len(first) <= _STDERR_SUMMARY_WIDTH:
        return first
    return first[: _STDERR_SUMMARY_WIDTH - 3] + "..."


def _detail_row(entry: ManifestEntry) -> tuple[str, str, str, str, str]:
    """Render one manifest entry as the five string cells of a detail-table row."""
    summary = _truncate_stderr_summary(entry.stderr) or _EMPTY_CELL
    log = _EMPTY_CELL if entry.log_path is None else str(entry.log_path)
    return (
        str(entry.run_id),
        entry.status,
        f"{entry.duration_s:.2f}",
        summary,
        log,
    )


def _format_detail_table(
    manifest: Manifest, output_dir: Path, *, status_filter: str | None = None
) -> str:
    """Render the per-run detail table plus the trailing one-line summary.

    Rows are sorted with ``failed`` first, then ``skipped``, then ``ok``;
    within each bucket by ``run_id`` ascending. ``status_filter`` (if given)
    keeps only rows of that status; the summary line still reflects the
    full manifest.
    """
    entries = manifest.entries
    if status_filter is not None:
        entries = [e for e in entries if e.status == status_filter]
    sorted_entries = sorted(entries, key=lambda e: (_STATUS_BUCKETS.get(e.status, 99), e.run_id))
    rows = [_detail_row(e) for e in sorted_entries]

    widths = [len(h) for h in _DETAIL_HEADERS]
    for row in rows:
        for i, cell in enumerate(row):
            if len(cell) > widths[i]:
                widths[i] = len(cell)

    def _format_row(cells: tuple[str, ...] | list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)).rstrip()

    lines = [_format_row(_DETAIL_HEADERS)]
    lines.extend(_format_row(row) for row in rows)
    lines.append(_format_summary(manifest, output_dir))
    return "\n".join(lines)


def _format_run_detail(entry: ManifestEntry) -> str:
    """Render one run's full record: header fields, full stderr, override dict."""
    header_fields = (
        ("status", entry.status),
        ("duration_s", f"{entry.duration_s:.2f}"),
        ("started_at", entry.started_at.isoformat()),
        ("ended_at", entry.ended_at.isoformat()),
        ("log_path", _EMPTY_CELL if entry.log_path is None else str(entry.log_path)),
    )
    label_width = max(len(label) for label, _ in header_fields)
    lines = [f"run_id: {entry.run_id}"]
    lines.extend(f"{label.ljust(label_width)}  {value}" for label, value in header_fields)

    lines.append("")
    lines.append("overrides:")
    if entry.overrides:
        key_width = max(len(k) for k in entry.overrides)
        for key in sorted(entry.overrides):
            lines.append(f"  {key.ljust(key_width)}  {entry.overrides[key]!r}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("stderr:")
    if entry.stderr:
        lines.extend(entry.stderr.splitlines() or [entry.stderr])
    else:
        lines.append("(no stderr)")
    return "\n".join(lines)


def _cmd_show(args: argparse.Namespace) -> int:
    # Validate flag shape before touching disk so a typo like
    # `gmat-sweep show --filter ok` fails immediately, not after a manifest
    # stat or load.
    if args.filter is not None and not args.detail:
        raise SweepConfigError("--filter requires --detail")

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"gmat-sweep: manifest not found: {manifest_path}", file=sys.stderr)
        return EXIT_MANIFEST

    manifest = Manifest.load(manifest_path)

    if args.run is not None:
        for entry in manifest.entries:
            if entry.run_id == args.run:
                print(_format_run_detail(entry))
                return EXIT_OK
        print(
            f"gmat-sweep: run_id {args.run} not found in manifest",
            file=sys.stderr,
        )
        return EXIT_MANIFEST

    if args.detail:
        print(_format_detail_table(manifest, manifest_path.parent, status_filter=args.filter))
        return EXIT_OK

    print(_format_summary(manifest, manifest_path.parent))
    return EXIT_OK


def _collect_perturb(raw_specs: list[str]) -> dict[str, DistSpec]:
    """Parse repeated ``--perturb`` flags into a single mapping; reject duplicates."""
    perturb: dict[str, DistSpec] = {}
    for raw in raw_specs:
        name, dist = _parse_perturb_spec(raw)
        if name in perturb:
            raise SweepConfigError(f"perturb spec for {name!r} given more than once")
        perturb[name] = dist
    return perturb


def _cmd_monte_carlo(args: argparse.Namespace) -> int:
    script = Path(args.script)
    if not script.is_file():
        print(f"gmat-sweep: script not found: {script}", file=sys.stderr)
        return EXIT_CONFIG

    perturb = _collect_perturb(args.perturb)
    monte_carlo = _import_api_func("monte_carlo")

    out = Path(args.out)
    with _build_pool(args) as pool:
        monte_carlo(
            script,
            n=args.n,
            perturb=perturb,
            seed=args.seed,
            backend=pool,
            out=out,
            fsync_each=args.fsync_each,
            fsync_batch=args.fsync_batch,
        )

    manifest = Manifest.load(out / "manifest.jsonl")
    print(_format_summary(manifest, out))
    return EXIT_OK


def _cmd_latin_hypercube(args: argparse.Namespace) -> int:
    script = Path(args.script)
    if not script.is_file():
        print(f"gmat-sweep: script not found: {script}", file=sys.stderr)
        return EXIT_CONFIG

    perturb = _collect_perturb(args.perturb)
    latin_hypercube = _import_api_func("latin_hypercube")

    out = Path(args.out)
    with _build_pool(args) as pool:
        latin_hypercube(
            script,
            n=args.n,
            perturb=perturb,
            seed=args.seed,
            backend=pool,
            out=out,
            fsync_each=args.fsync_each,
            fsync_batch=args.fsync_batch,
        )

    manifest = Manifest.load(out / "manifest.jsonl")
    print(_format_summary(manifest, out))
    return EXIT_OK


def _cmd_explicit(args: argparse.Namespace) -> int:
    script = Path(args.script)
    if not script.is_file():
        print(f"gmat-sweep: script not found: {script}", file=sys.stderr)
        return EXIT_CONFIG

    samples = _load_samples(Path(args.samples))
    sweep = _import_api_func("sweep")

    out = Path(args.out)
    with _build_pool(args) as pool:
        sweep(
            script,
            samples=samples,
            backend=pool,
            out=out,
            fsync_each=args.fsync_each,
            fsync_batch=args.fsync_batch,
        )

    manifest = Manifest.load(out / "manifest.jsonl")
    print(_format_summary(manifest, out))
    return EXIT_OK


def _cmd_archive(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"gmat-sweep: manifest not found: {manifest_path}", file=sys.stderr)
        return EXIT_MANIFEST
    script = Path(args.script)
    if not script.is_file():
        print(f"gmat-sweep: script not found: {script}", file=sys.stderr)
        return EXIT_CONFIG

    from gmat_sweep import __version__ as sweep_version
    from gmat_sweep.archive import _archive_sweep

    manifest = Manifest.load(manifest_path)
    out = Path(args.out)
    bundle = _archive_sweep(
        manifest=manifest,
        output_dir=manifest_path.parent,
        script_path=script,
        out=out,
        include_logs=args.include_logs,
        sweep_version=sweep_version,
        allow_script_drift=args.allow_script_drift,
    )
    print(_format_summary(manifest, manifest_path.parent))
    print(f"archive: {bundle}")
    return EXIT_OK


def _cmd_extend(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"gmat-sweep: manifest not found: {manifest_path}", file=sys.stderr)
        return EXIT_MANIFEST
    script = Path(args.script)
    if not script.is_file():
        print(f"gmat-sweep: script not found: {script}", file=sys.stderr)
        return EXIT_CONFIG

    monte_carlo_extend = _import_api_func("monte_carlo_extend")

    with _build_pool(args) as pool:
        monte_carlo_extend(
            manifest_path,
            script,
            n=args.n,
            backend=pool,
            allow_script_drift=args.allow_script_drift,
            fsync_each=args.fsync_each,
            fsync_batch=args.fsync_batch,
        )

    manifest = Manifest.load(manifest_path)
    print(_format_summary(manifest, manifest_path.parent))
    return EXIT_OK


def _cmd_resume(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"gmat-sweep: manifest not found: {manifest_path}", file=sys.stderr)
        return EXIT_MANIFEST
    script = Path(args.script)
    if not script.is_file():
        print(f"gmat-sweep: script not found: {script}", file=sys.stderr)
        return EXIT_CONFIG

    # Local import: only the resume path drives Sweep directly.
    from gmat_sweep.sweep import Sweep

    with _build_pool(args) as pool:
        sweep_obj = Sweep.from_manifest(
            manifest_path,
            script,
            backend=pool,
            allow_script_drift=args.allow_script_drift,
            fsync_each=args.fsync_each,
            fsync_batch=args.fsync_batch,
        ).resume()

    print(_format_summary(sweep_obj.to_manifest(), manifest_path.parent))
    return EXIT_OK


def _add_fsync_flags(subparser: argparse.ArgumentParser) -> None:
    """Attach ``--fsync-each`` / ``--no-fsync-each`` and ``--fsync-batch`` to a subparser."""
    subparser.add_argument(
        "--fsync-each",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fsync the manifest after every appended entry (default: --fsync-each). "
            "Pass --no-fsync-each to amortise fsyncs across batches of "
            "--fsync-batch entries — faster for sub-second runs at large counts, "
            "at the cost of losing up to FSYNC_BATCH-1 trailing entries on host "
            "crash. The resume flow re-runs the missing slice."
        ),
    )
    subparser.add_argument(
        "--fsync-batch",
        type=int,
        default=50,
        metavar="N",
        help=(
            "Fsync interval (in entries) when --no-fsync-each is set. "
            "Ignored when --fsync-each is in effect. Default: 50."
        ),
    )


_WORKERS_HELP = (
    "Number of subprocess workers. Default -1 uses every available core. "
    "Ignored on --backend mpi (the launcher fixes the rank count); pass "
    "--workers explicitly there and gmat-sweep exits 2."
)


def _add_workers_flag(subparser: argparse.ArgumentParser) -> None:
    """Attach the shared ``--workers`` flag to a sweep-running subparser."""
    subparser.add_argument(
        "--workers",
        type=int,
        default=-1,
        metavar="N",
        help=_WORKERS_HELP,
    )


def _add_backend_flag(subparser: argparse.ArgumentParser) -> None:
    """Attach ``--backend`` / ``--backend-arg`` to a sweep-running subparser."""
    subparser.add_argument(
        "--backend",
        choices=("local", "dask", "ray", "mpi"),
        default="local",
        metavar="NAME",
        help=(
            "Execution backend. 'local' (default) runs on this machine via "
            "joblib/loky workers. 'dask' requires the [dask] extra; 'ray' "
            "requires the [ray] extra; 'mpi' requires the [mpi] extra. "
            "Missing extras exit 4."
        ),
    )
    subparser.add_argument(
        "--backend-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra keyword forwarded to the dask/ray pool constructor "
            "(e.g. 'threads_per_worker=2', 'address=ray://host:port'). "
            "Repeat for multiple kwargs. Values coerced int → float → str. "
            "Not allowed with --backend local."
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gmat-sweep",
        description="Run parameter sweeps over a GMAT mission and inspect the resulting manifest.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    run = subparsers.add_parser(
        "run",
        help="Run a parameter sweep over a GMAT script.",
        description=(
            "Run a full-factorial parameter sweep. Each --grid flag adds one axis; "
            "multiple --grid flags combine into the cartesian product."
        ),
    )
    run.add_argument(
        "--grid",
        action="append",
        default=[],
        required=True,
        metavar="SPEC",
        help=(
            "Grid axis spec. Two forms: "
            "'name=lo:hi:count' for count evenly-spaced points (e.g. 'Sat.SMA=7000:8000:5'), "
            "or 'name=v1,v2,v3' for explicit values (e.g. 'Sat.DryMass=100,200,300'). "
            "Repeat --grid for additional axes; the cartesian product is run."
        ),
    )
    _add_workers_flag(run)
    run.add_argument(
        "--out",
        required=True,
        metavar="PATH",
        help="Output directory for per-run artefacts and manifest.jsonl.",
    )
    run.add_argument(
        "script",
        metavar="SCRIPT",
        help="Path to the GMAT .script file.",
    )
    _add_backend_flag(run)
    _add_fsync_flags(run)
    run.set_defaults(func=_cmd_run)

    show = subparsers.add_parser(
        "show",
        help="Print a one-line summary of a sweep manifest.",
        description=(
            "Load a manifest.jsonl file and print either a one-line summary "
            "(default), a per-run detail table (--detail), or one run's full "
            "record (--run N)."
        ),
    )
    show.add_argument(
        "manifest",
        metavar="MANIFEST",
        help="Path to a manifest.jsonl produced by 'gmat-sweep run'.",
    )
    show_mode = show.add_mutually_exclusive_group()
    show_mode.add_argument(
        "--detail",
        action="store_true",
        help=(
            "Print a per-run table (run_id, status, duration_s, stderr_summary, "
            "log_path) sorted failed → skipped → ok, then the one-line summary."
        ),
    )
    show_mode.add_argument(
        "--run",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Print run_id=N's full record: header fields, override dict, full "
            "stderr. Exits 3 if N is not in the manifest."
        ),
    )
    show.add_argument(
        "--filter",
        choices=("ok", "failed", "skipped"),
        default=None,
        metavar="STATUS",
        help="With --detail, restrict the table to runs of one status.",
    )
    show.set_defaults(func=_cmd_show)

    monte = subparsers.add_parser(
        "monte-carlo",
        help="Run a Monte Carlo dispersion sweep over a GMAT script.",
        description=(
            "Run n stochastic samples by independently sampling each --perturb "
            "parameter from its own distribution. Two runs at the same "
            "(--perturb, --n, --seed) produce bit-equal DataFrames; without "
            "--seed the draws fall back to OS entropy."
        ),
    )
    monte.add_argument(
        "--n",
        type=int,
        required=True,
        metavar="N",
        help="Number of stochastic runs (>= 1).",
    )
    monte.add_argument(
        "--perturb",
        action="append",
        default=[],
        required=True,
        metavar="SPEC",
        help=(
            "Perturb axis spec. Three forms: "
            "'name=normal:mu:sigma', 'name=uniform:lo:hi', 'name=lognormal:mu:sigma' "
            "(e.g. 'Sat.SMA=normal:7100:50'). Repeat --perturb for additional axes."
        ),
    )
    monte.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="SEED",
        help="Optional integer parent seed. Omit for OS entropy (non-reproducible).",
    )
    _add_workers_flag(monte)
    monte.add_argument(
        "--out",
        required=True,
        metavar="PATH",
        help="Output directory for per-run artefacts and manifest.jsonl.",
    )
    monte.add_argument(
        "script",
        metavar="SCRIPT",
        help="Path to the GMAT .script file.",
    )
    _add_backend_flag(monte)
    _add_fsync_flags(monte)
    monte.set_defaults(func=_cmd_monte_carlo)

    lhs = subparsers.add_parser(
        "latin-hypercube",
        help="Run a Latin hypercube sweep over a GMAT script.",
        description=(
            "Draw n Latin hypercube points stratified across each --perturb axis "
            "and map them through the user's distribution. Same --perturb syntax "
            "as 'monte-carlo'."
        ),
    )
    lhs.add_argument(
        "--n",
        type=int,
        required=True,
        metavar="N",
        help="Number of Latin hypercube points (>= 1).",
    )
    lhs.add_argument(
        "--perturb",
        action="append",
        default=[],
        required=True,
        metavar="SPEC",
        help=(
            "Perturb axis spec. Three forms: "
            "'name=normal:mu:sigma', 'name=uniform:lo:hi', 'name=lognormal:mu:sigma'. "
            "Repeat --perturb for additional axes."
        ),
    )
    lhs.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="SEED",
        help="Optional integer seed for the Latin hypercube sampler.",
    )
    _add_workers_flag(lhs)
    lhs.add_argument(
        "--out",
        required=True,
        metavar="PATH",
        help="Output directory for per-run artefacts and manifest.jsonl.",
    )
    lhs.add_argument(
        "script",
        metavar="SCRIPT",
        help="Path to the GMAT .script file.",
    )
    _add_backend_flag(lhs)
    _add_fsync_flags(lhs)
    lhs.set_defaults(func=_cmd_latin_hypercube)

    explicit = subparsers.add_parser(
        "explicit",
        help="Run an explicit-row sweep from a CSV or Parquet sample design.",
        description=(
            "Load --samples (CSV or Parquet) into a DataFrame and run one mission "
            "per row. Column names are dotted-path field names; the row index "
            "becomes run_id."
        ),
    )
    explicit.add_argument(
        "--samples",
        required=True,
        metavar="PATH",
        help="Path to a .csv or .parquet sample design.",
    )
    _add_workers_flag(explicit)
    explicit.add_argument(
        "--out",
        required=True,
        metavar="PATH",
        help="Output directory for per-run artefacts and manifest.jsonl.",
    )
    explicit.add_argument(
        "script",
        metavar="SCRIPT",
        help="Path to the GMAT .script file.",
    )
    _add_backend_flag(explicit)
    _add_fsync_flags(explicit)
    explicit.set_defaults(func=_cmd_explicit)

    extend = subparsers.add_parser(
        "extend",
        help="Append more bit-deterministic Monte Carlo runs to an existing sweep.",
        description=(
            "Load a Monte Carlo manifest.jsonl, dispatch N additional runs at "
            "run_id range [old_n, old_n + N), and reuse the manifest's seed and "
            "perturb mapping so the new draws are bit-equal to the same indices "
            "of a fresh monte_carlo(n=old_n + N) call. Refuses if the base "
            "sweep has any failed or missing runs in [0, old_n) — call "
            "'gmat-sweep resume' first to fill those in."
        ),
    )
    extend.add_argument(
        "manifest",
        metavar="MANIFEST",
        help="Path to a Monte Carlo manifest.jsonl produced by a prior sweep.",
    )
    extend.add_argument(
        "--n",
        type=int,
        required=True,
        metavar="N",
        help="Number of additional stochastic runs to append (>= 1).",
    )
    extend.add_argument(
        "--script",
        required=True,
        metavar="PATH",
        help=(
            "Path to the same GMAT .script the original sweep loaded. Its "
            "canonical SHA-256 must equal the manifest's script_sha256 unless "
            "--allow-script-drift is set."
        ),
    )
    _add_workers_flag(extend)
    extend.add_argument(
        "--allow-script-drift",
        action="store_true",
        help=(
            "Proceed even if the script's canonical hash differs from the manifest's. "
            "Emits a RuntimeWarning."
        ),
    )
    _add_backend_flag(extend)
    _add_fsync_flags(extend)
    extend.set_defaults(func=_cmd_extend)

    resume = subparsers.add_parser(
        "resume",
        help="Re-run only the failed and missing entries from an existing manifest.",
        description=(
            "Reload a manifest.jsonl, rebuild the original run iterable, and "
            "re-submit only the runs whose latest entry is 'failed' or that have "
            "no entry on disk yet. Successful runs' Parquet files are reused."
        ),
    )
    resume.add_argument(
        "manifest",
        metavar="MANIFEST",
        help="Path to a manifest.jsonl produced by a prior sweep.",
    )
    resume.add_argument(
        "--script",
        required=True,
        metavar="PATH",
        help=(
            "Path to the same GMAT .script the original sweep loaded. Its canonical "
            "SHA-256 must equal the manifest's script_sha256 unless --allow-script-drift "
            "is set."
        ),
    )
    _add_workers_flag(resume)
    resume.add_argument(
        "--allow-script-drift",
        action="store_true",
        help=(
            "Proceed even if the script's canonical hash differs from the manifest's. "
            "Emits a RuntimeWarning."
        ),
    )
    _add_backend_flag(resume)
    _add_fsync_flags(resume)
    resume.set_defaults(func=_cmd_resume)

    archive = subparsers.add_parser(
        "archive",
        help="Pack a finished sweep into a portable .zip for archival deposit.",
        description=(
            "Bundle a sweep's script, manifest, and per-run Parquet outputs into "
            "a single .zip suitable for Zenodo / JOSS deposit or internal handoff. "
            "Output paths in the bundled manifest are rewritten to be bundle-"
            "relative; a sha256sum-compatible MANIFEST.hash and a generated "
            "README.md describing how to resume the sweep are also included."
        ),
    )
    archive.add_argument(
        "manifest",
        metavar="MANIFEST",
        help="Path to a manifest.jsonl produced by a prior sweep.",
    )
    archive.add_argument(
        "--script",
        required=True,
        metavar="PATH",
        help=(
            "Path to the same GMAT .script the sweep loaded. Its canonical "
            "SHA-256 must equal the manifest's script_sha256 unless "
            "--allow-script-drift is set."
        ),
    )
    archive.add_argument(
        "--out",
        required=True,
        metavar="PATH",
        help="Destination .zip path. Parent directories are created on demand.",
    )
    archive.add_argument(
        "--include-logs",
        action="store_true",
        help=(
            "Bundle per-run worker.log files alongside the Parquet outputs. "
            "Default is to drop them so the archive stays small; the manifest's "
            "log_path field is set to null in that case."
        ),
    )
    archive.add_argument(
        "--allow-script-drift",
        action="store_true",
        help=(
            "Proceed even if the script's canonical hash differs from the "
            "manifest's. The bundle still records the manifest's original hash."
        ),
    )
    archive.set_defaults(func=_cmd_archive)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns the process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except SweepConfigError as exc:
        print(f"gmat-sweep: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except ManifestCorruptError as exc:
        location = f"{exc.path}:{exc.line_number}" if exc.line_number is not None else str(exc.path)
        print(f"gmat-sweep: {exc} ({location})", file=sys.stderr)
        return EXIT_MANIFEST
    except BackendError as exc:
        print(f"gmat-sweep: backend error: {exc}", file=sys.stderr)
        return EXIT_BACKEND
    except GmatSweepError as exc:
        print(f"gmat-sweep: {exc}", file=sys.stderr)
        return EXIT_GENERIC


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
