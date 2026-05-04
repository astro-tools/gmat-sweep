"""Console-script entry point: ``gmat-sweep run`` / ``gmat-sweep show``.

The CLI is a thin wrapper over :func:`gmat_sweep.api.sweep` and
:meth:`gmat_sweep.manifest.Manifest.load`. Argument parsing and grid-spec
parsing live here; everything else delegates to the existing public surface.

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
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from gmat_sweep.api import sweep
from gmat_sweep.errors import (
    BackendError,
    GmatSweepError,
    ManifestCorruptError,
    SweepConfigError,
)
from gmat_sweep.manifest import Manifest

__all__ = ["main"]

EXIT_OK = 0
EXIT_GENERIC = 1
EXIT_CONFIG = 2
EXIT_MANIFEST = 3
EXIT_BACKEND = 4


def _parse_grid_value(token: str) -> int | float | str:
    """Coerce a single explicit-list token via int → float → str fallback."""
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return token


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
        try:
            count = int(count_str)
        except ValueError as exc:
            raise SweepConfigError(
                f"linspace count for {name!r} must be an integer, got {count_str!r}"
            ) from exc
        if count < 2:
            raise SweepConfigError(f"linspace count for {name!r} must be >= 2, got {count}")
        return name, np.linspace(lo, hi, count).tolist()

    tokens = rhs.split(",")
    if any(t == "" for t in tokens):
        raise SweepConfigError(f"explicit grid spec for {name!r} has an empty value: {rhs!r}")
    return name, [_parse_grid_value(t) for t in tokens]


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

    out = Path(args.out)
    sweep(script, grid=grid, workers=args.workers, out=out)

    manifest = Manifest.load(out / "manifest.jsonl")
    print(_format_summary(manifest, out))
    return EXIT_OK


def _cmd_show(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"gmat-sweep: manifest not found: {manifest_path}", file=sys.stderr)
        return EXIT_MANIFEST
    manifest = Manifest.load(manifest_path)
    print(_format_summary(manifest, manifest_path.parent))
    return EXIT_OK


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
    run.add_argument(
        "--workers",
        type=int,
        default=-1,
        metavar="N",
        help="Number of subprocess workers. Default -1 uses every available core.",
    )
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
    run.set_defaults(func=_cmd_run)

    show = subparsers.add_parser(
        "show",
        help="Print a one-line summary of a sweep manifest.",
        description="Load a manifest.jsonl file and print a one-line summary.",
    )
    show.add_argument(
        "manifest",
        metavar="MANIFEST",
        help="Path to a manifest.jsonl produced by 'gmat-sweep run'.",
    )
    show.set_defaults(func=_cmd_show)

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
        print(f"gmat-sweep: {exc} ({exc.path})", file=sys.stderr)
        return EXIT_MANIFEST
    except BackendError as exc:
        print(f"gmat-sweep: backend error: {exc}", file=sys.stderr)
        return EXIT_BACKEND
    except GmatSweepError as exc:
        print(f"gmat-sweep: {exc}", file=sys.stderr)
        return EXIT_GENERIC


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
